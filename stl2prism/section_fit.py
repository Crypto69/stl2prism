"""Plane sections of a triangle mesh, and clean sketch curves fitted to them.

This is the engine behind "Create Mesh Section Sketch" + "Fit Curves to
Mesh Section" done by hand in Fusion: cut the mesh with a plane, chain the
crossing segments into closed loops, then redraw every loop as lines and
arcs where they hold the tolerance and as a fitted spline where they do
not (the domes of a controller shell, a scanned handle).

numpy only, on purpose: the same file is copied into the Fusion add-in,
which has no shapely or trimesh. The pipeline's sliced-loft mode uses the
plane cut too (`stl2prism.sliced_loft`), so every slice of a loft and every
sketch in the generated Fusion script comes from this one cutter.

Coordinates: `section_loops` returns 2-D points in an in-plane frame (u, v)
with `to_3d(xy)` = origin + u*x + v*y. Outer loops are counter-clockwise,
holes clockwise, holes attached to the outer loop that contains them.
"""
import numpy as np

from .profile_fit import (segment_polyline, snap_profile, solve_junctions,
                          try_full_circle)


# ---------------------------------------------------------------------------
# plane cut

def plane_basis(normal):
    """Orthonormal (u, v, n): n is the unit normal, u the in-plane direction
    closest to the world axis least aligned with n, v = n x u. Deterministic,
    so two slices with the same normal share a frame."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    k = int(np.argmin(np.abs(n)))
    a = np.zeros(3)
    a[k] = 1.0
    u = a - n * (a @ n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def cut_segments(V, F, origin, normal):
    """Every crossing of the plane with a triangle, as a segment between the
    two crossing points. Returns (P, S): P (m, 3) crossing points, one per
    crossed mesh edge, and S (k, 2) indices into P, one row per crossed
    triangle. Points come from the edge (sorted vertex pair), so two
    triangles sharing an edge share the point exactly and loops chain
    without any rounding."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    d = (V - np.asarray(origin, float)) @ n
    # a vertex exactly on the plane counts as the positive side, so every
    # crossed triangle has exactly two crossing edges
    side = d >= 0
    s = side[F]
    crossed = ~(s.all(1) | (~s).all(1))
    if not crossed.any():
        return np.zeros((0, 3)), np.zeros((0, 2), np.int64)
    T = F[crossed]
    st = s[crossed]
    # edges of each crossed triangle: (0,1) (1,2) (2,0); crossing when the
    # two ends differ in side
    E = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)   # (k, 3, 2)
    X = np.stack([st[:, 0] != st[:, 1], st[:, 1] != st[:, 2], st[:, 2] != st[:, 0]], axis=1)
    assert (X.sum(1) == 2).all()
    Ek = E[X].reshape(-1, 2, 2)                                        # (k, 2, 2)
    Ek = np.sort(Ek, axis=2)
    flat = Ek.reshape(-1, 2)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    a, b = uniq[:, 0], uniq[:, 1]
    da, db = d[a], d[b]
    t = da / (da - db)
    P = V[a] + (V[b] - V[a]) * t[:, None]
    S = inv.reshape(-1, 2)
    return P, S


def chain_loops(S):
    """Chain segments (rows of point-index pairs) into closed loops. Each
    point of a closed mesh section is used by exactly two segments; a point
    used once (an open, leaky mesh) starts an open chain that is returned
    as well, so nothing is silently dropped. Returns a list of index
    lists; a closed loop does not repeat its first index."""
    if len(S) == 0:
        return []
    nbrs = {}
    for k, (i, j) in enumerate(S):
        nbrs.setdefault(int(i), []).append((int(j), k))
        nbrs.setdefault(int(j), []).append((int(i), k))
    used = np.zeros(len(S), bool)
    loops = []
    # open chains first (their ends are the points with one segment)
    starts = [p for p, l in nbrs.items() if len(l) == 1] + list(nbrs)
    for start in starts:
        for (_, k0) in nbrs[start]:
            if used[k0]:
                continue
            loop = [start]
            cur, k = start, k0
            while True:
                used[k] = True
                i, j = S[k]
                nxt = int(j) if int(i) == cur else int(i)
                if nxt == start:
                    break
                loop.append(nxt)
                cur = nxt
                step = next(((q, kk) for (q, kk) in nbrs[cur] if not used[kk]), None)
                if step is None:
                    break
                k = step[1]
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def signed_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def point_in_loop(pt, xy):
    """Even-odd ray cast of one point against a closed 2-D loop."""
    x, y = pt
    x0, y0 = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    cond = (y0 > y) != (y1 > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        xs = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.sum(cond & (x < xs)) % 2)


def nest_loops(loops_xy):
    """Group closed 2-D loops into (outer, [holes]) by containment depth:
    even depth is material, odd depth a hole in the smallest containing
    even loop. Loops come back oriented: outers CCW, holes CW."""
    n = len(loops_xy)
    areas = [abs(signed_area(l)) for l in loops_xy]
    order = sorted(range(n), key=lambda i: -areas[i])       # big first
    parent = [-1] * n
    depth = [0] * n
    for idx, i in enumerate(order):
        pt = loops_xy[i][0]
        # the smallest loop that contains this one: candidates are bigger
        # loops (earlier in `order`), checked smallest first
        for j in reversed(order[:idx]):
            if areas[j] > areas[i] and point_in_loop(pt, loops_xy[j]):
                parent[i] = j
                depth[i] = depth[j] + 1
                break
    out = {}
    for i in range(n):
        if depth[i] % 2 == 0:
            xy = loops_xy[i]
            out[i] = (xy if signed_area(xy) > 0 else xy[::-1].copy(), [])
    for i in range(n):
        if depth[i] % 2 == 1 and parent[i] in out:
            xy = loops_xy[i]
            out[parent[i]][1].append(xy if signed_area(xy) < 0 else xy[::-1].copy())
    return [out[i] for i in sorted(out, key=lambda i: -areas[i])]


def section_loops(V, F, origin, normal, min_area=1e-6):
    """Cut the mesh (V, F) with the plane (origin, normal).

    Returns (loops, frame): loops is a list of (outer_xy, [hole_xy, ...])
    with 2-D points in the plane frame, outers CCW and holes CW, largest
    outer first; frame is (origin, u, v, n) with to_3d(xy) = origin + u*x
    + v*y. Loops of less than `min_area` (mm^2) are dropped as numerical
    dust. Open chains (a leaky mesh) are kept as loops too: closed by
    their chord, which is what any sketch would do."""
    u, v, n = plane_basis(normal)
    o = np.asarray(origin, float)
    P, S = cut_segments(V, F, o, n)
    xy_all = np.stack([(P - o) @ u, (P - o) @ v], axis=1) if len(P) else np.zeros((0, 2))
    loops = []
    for idx in chain_loops(S):
        xy = xy_all[idx]
        # drop consecutive duplicates (a crossing at a shared vertex)
        keep = np.ones(len(xy), bool)
        keep[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=1) > 1e-9
        xy = xy[keep]
        if len(xy) >= 3 and abs(signed_area(xy)) >= min_area:
            loops.append(xy)
    return nest_loops(loops), (o, u, v, n)


def to_3d(xy, frame):
    o, u, v, _ = frame
    xy = np.asarray(xy, float)
    return o + np.outer(xy[:, 0], u) + np.outer(xy[:, 1], v)


# ---------------------------------------------------------------------------
# curve fitting

def douglas_peucker(pts, tol):
    """Indices of the vertices kept by Douglas-Peucker at `tol` (open
    polyline; the ends are always kept)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n <= 2:
        return list(range(n))
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]
        L2 = float(seg @ seg)
        rel = pts[a + 1:b] - pts[a]
        # distance to the finite chord, not the infinite line: a hook that
        # doubles back along the chord's own line is 0 from the line and
        # must not be thinned away
        t = np.clip((rel @ seg) / L2, 0.0, 1.0) if L2 > 1e-24 else np.zeros(len(rel))
        dist = np.linalg.norm(rel - t[:, None] * seg, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[a + 1 + k] = True
            stack.append((a, a + 1 + k))
            stack.append((a + 1 + k, b))
    return list(np.nonzero(keep)[0])


def _chord(p):
    return float(np.linalg.norm(np.asarray(p['p1'], float) - np.asarray(p['p0'], float)))


def _span_len(p):
    pts = p.get('_pts')
    if pts is None or len(pts) < 2:
        return _chord(p)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _seg_dist_max(pts, a, b, chunk=256):
    """Worst distance from points (n, 2) to the segments a->b (m, 2),
    vectorised in chunks of points (n*m floats at a time)."""
    pts = np.asarray(pts, float)
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-18)
    worst = 0.0
    for i in range(0, len(pts), chunk):
        P = pts[i:i + chunk]
        rel = P[:, None, :] - a[None, :, :]                       # (c, m, 2)
        t = np.clip((rel * ab[None]).sum(2) / L2[None], 0.0, 1.0)  # (c, m)
        d = np.linalg.norm(rel - t[:, :, None] * ab[None], axis=2).min(1)
        worst = max(worst, float(d.max()) if len(d) else 0.0)
    return worst


def _open_polyline_dev(pts, q):
    """Worst distance from `pts` to the open polyline `q` (2-D)."""
    pts = np.asarray(pts, float)
    q = np.asarray(q, float)
    if len(q) < 2:
        return float(np.linalg.norm(pts - q[0], axis=1).max()) if len(q) else 0.0
    return _seg_dist_max(pts, q[:-1], q[1:])


def _prim_dev(p):
    """Worst distance of a primitive's own raw points from the finite
    primitive (segment or arc, not the infinite line or circle), so an
    endpoint that walked away from the data counts too."""
    pts = p.get('_pts')
    if pts is None or len(pts) < 2 or p['type'] == 'spline':
        return 0.0
    q = np.vstack([prim_points([p]), [np.asarray(p['p1'], float)]])
    return _open_polyline_dev(pts, q)


def _fragment_runs(prims, min_len, min_run, tol):
    """Cyclic runs of primitives that lines and arcs did not really fit: a
    run of >= min_run consecutive crumbs shorter than min_len (a curve
    chopped into pieces), or any run holding a primitive whose own raw
    points sit more than tol off it (a stretch the segmenter gave up on).
    Returns a list of index lists in cyclic order."""
    n = len(prims)
    bad = [_prim_dev(p) > tol for p in prims]
    weak = [bad[i] or _span_len(p) < min_len for i, p in enumerate(prims)]
    if all(weak):
        return [list(range(n))] if (n >= min_run or any(bad)) else []
    # start after a strong primitive so runs do not straddle the seam
    start = next(i for i in range(n) if not weak[i])
    runs, cur = [], []
    for k in range(1, n + 1):
        i = (start + k) % n
        if weak[i]:
            cur.append(i)
        else:
            if len(cur) >= min_run or any(bad[j] for j in cur):
                runs.append(cur)
            cur = []
    return runs


def _close_junctions(prims):
    """Make consecutive primitives share their endpoint exactly: the two
    ends are averaged (they already sit within a hair of each other, both
    being the raw section point the segmenter split at). Arc endpoints
    are then put back on their circle and the neighbour follows, so a
    three-point arc drawn from p0, mid, p1 is the fitted circle."""
    from .profile_fit import _project_arc_ends
    n = len(prims)
    if n < 2:
        return prims
    for i in range(n):
        a, b = prims[i], prims[(i + 1) % n]
        x = 0.5 * (np.asarray(a['p1'], float) + np.asarray(b['p0'], float))
        a['p1'] = x.copy()
        b['p0'] = x.copy()
    for p in prims:
        if p['type'] == 'arc':
            _project_arc_ends(p)
    for i in range(n):
        a, b = prims[i], prims[(i + 1) % n]
        if a['type'] == 'arc':
            b['p0'] = np.asarray(a['p1'], float).copy()
            if b['type'] == 'spline':
                b['pts'][0] = b['p0']
        elif b['type'] == 'arc':
            a['p1'] = np.asarray(b['p0'], float).copy()
            if a['type'] == 'spline':
                a['pts'][-1] = a['p1']
    return prims


def _replace_with_splines(prims, runs, tol):
    """Replace each fragment run by one spline primitive through the raw
    section points the fragments covered, thinned by Douglas-Peucker at
    tol/2. The spline starts and ends exactly on its neighbours' junction
    points, so the loop stays closed."""
    if not runs:
        return prims
    n = len(prims)
    drop = set(i for r in runs for i in r)
    first_of = {r[0]: r for r in runs}
    out = []
    # walk from the primitive after the last run's end, so every run is
    # met at its first index
    start = (runs[-1][-1] + 1) % n
    for k in range(n):
        i = (start + k) % n
        if i in first_of:
            r = first_of[i]
            raw = [np.asarray(prims[j]['_pts'], float) for j in r]
            pts = [raw[0]] + [q[1:] for q in raw[1:]]
            pts = np.vstack(pts)
            pts = pts[douglas_peucker(pts, tol / 2)]
            p0 = np.asarray(prims[r[0]]['p0'], float)
            p1 = np.asarray(prims[r[-1]]['p1'], float)
            pts[0], pts[-1] = p0, p1
            out.append({'type': 'spline', 'pts': pts, 'p0': p0.copy(), 'p1': p1.copy(),
                        '_pts': np.vstack(raw)})
        elif i not in drop:
            out.append(prims[i])
    return out


def fit_loop(xy, tol=0.08, mesh_pts=None, min_prim_mm=None, min_run=3, clean=False):
    """Redraw one closed 2-D loop as sketch primitives.

    Lines and arcs first (`profile_fit.segment_polyline`, the fitter the
    prismatic engine uses). Where that only manages a run of `min_run` or
    more crumbs shorter than `min_prim_mm` (default 8*tol, at least 0.6
    mm), or a primitive whose own points miss it by more than tol, the run
    becomes one {'type': 'spline', 'pts'} through the raw points, thinned
    at tol/2. Neighbours are then made to share their endpoints. A loop
    that is one circle within tol is returned as a full circle dict.

    `clean=True` also runs the prismatic engine's frame snapping and
    tangent junction solving (squared-up lines, true fillets). Off by
    default: on an organic section those moves cost more accuracy than
    they buy (measured on the RC-N2 controller: twice as many primitives
    off by more than tol). `mesh_pts` (optional, 2-D) refines arc fits on
    nearby mesh vertices, as the prismatic engine does.
    """
    from .profile_fit import refine_arcs_with_points
    pts = np.asarray(xy, float)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return []
    circ = try_full_circle(pts, tol)
    if circ:
        if mesh_pts is not None:
            refine_arcs_with_points([circ], mesh_pts, tol)
        return circ
    prims = segment_polyline(pts, tol=tol, closed=True)
    if not prims:
        return []
    if mesh_pts is not None:
        refine_arcs_with_points(prims, mesh_pts, tol)
    if clean:
        prims = snap_profile(prims)
        solve_junctions(prims, closed=True)
    if min_prim_mm is None:
        min_prim_mm = max(0.6, 8.0 * tol)
    runs = _fragment_runs(prims, min_prim_mm, min_run, tol)
    if runs and len(runs) == 1 and len(runs[0]) == len(prims):
        # the whole loop is one curve: a single closed spline
        raw = np.vstack([np.asarray(p['_pts'], float)[:-1] for p in prims])
        keep = douglas_peucker(np.vstack([raw, raw[:1]]), tol / 2)[:-1]
        return [{'type': 'spline', 'pts': raw[keep], 'p0': raw[keep][0].copy(),
                 'p1': raw[keep][0].copy(), 'closed': True, '_pts': raw}]
    return _close_junctions(_replace_with_splines(prims, runs, tol))


def prim_points(prims, arc_step_deg=6.0):
    """Dense 2-D polyline of a fitted loop (for deviation checks and
    previews). Full circles and closed splines come back closed."""
    if isinstance(prims, dict):                       # full circle
        c, r = np.asarray(prims['center'], float), float(prims['r'])
        a = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        return c + r * np.stack([np.cos(a), np.sin(a)], 1)
    out = []
    for p in prims:
        if p['type'] == 'line':
            out.append(np.asarray([p['p0']], float))
        elif p['type'] == 'arc':
            c, r = np.asarray(p['center'], float), float(p['r'])
            a0 = np.arctan2(p['p0'][1] - c[1], p['p0'][0] - c[0])
            a1 = np.arctan2(p['p1'][1] - c[1], p['p1'][0] - c[0])
            if p.get('ccw', True):
                while a1 <= a0:
                    a1 += 2 * np.pi
            else:
                while a1 >= a0:
                    a1 -= 2 * np.pi
            k = max(2, int(abs(a1 - a0) / np.radians(arc_step_deg)) + 1)
            a = np.linspace(a0, a1, k, endpoint=False)
            out.append(c + r * np.stack([np.cos(a), np.sin(a)], 1))
        else:
            out.append(np.asarray(p['pts'], float)[:-1] if not p.get('closed')
                       else np.asarray(p['pts'], float))
    return np.vstack(out) if out else np.zeros((0, 2))


def polyline_deviation(pts, ref):
    """Max distance from points `pts` to the closed polyline `ref` (2-D)."""
    ref = np.asarray(ref, float)
    return _seg_dist_max(pts, ref, np.roll(ref, -1, axis=0))


def fit_section(V, F, origin, normal, tol=0.08, mesh_pts=None):
    """Cut and fit in one go. Returns {'origin', 'u', 'v', 'normal',
    'loops': [(outer_prims, [hole_prims...]), ...], 'raw': the raw loops,
    'stats': {'loops', 'holes', 'lines', 'arcs', 'circles', 'splines',
    'dev_max'}} where dev_max is the worst distance from the raw section
    points to the fitted curves."""
    loops, frame = section_loops(V, F, origin, normal)
    fitted, stats = [], dict(loops=0, holes=0, lines=0, arcs=0, circles=0,
                             splines=0, dev_max=0.0)
    for outer, holes in loops:
        fo = fit_loop(outer, tol, mesh_pts)
        fh = [fit_loop(h, tol, mesh_pts) for h in holes]
        fitted.append((fo, fh))
        stats['loops'] += 1
        stats['holes'] += len(holes)
        for prims, raw in [(fo, outer)] + list(zip(fh, holes)):
            if isinstance(prims, dict):
                stats['circles'] += 1
            else:
                for p in prims:
                    stats[{'line': 'lines', 'arc': 'arcs', 'spline': 'splines'}[p['type']]] += 1
            if len(prims):
                dense = prim_points(prims)
                stats['dev_max'] = max(stats['dev_max'],
                                       polyline_deviation(raw[::max(1, len(raw) // 400)], dense))
    o, u, v, n = frame
    return {'origin': o, 'u': u, 'v': v, 'normal': n, 'loops': fitted,
            'raw': loops, 'stats': stats}


def lift_prims(prims, frame):
    """Fitted 2-D primitives of one loop -> the same primitives with 3-D
    points (mm) in `frame` = (origin, u, v, n): 'p0', 'p1', 'mid' (arcs),
    'pts' (splines), or {'center3', 'r'} for a full circle. What the Fusion
    sections script and the web preview draw."""
    def L(xy):
        return to_3d(np.asarray(xy, float).reshape(-1, 2), frame)
    if isinstance(prims, dict):
        return {'type': 'circle', 'center3': L(prims['center'])[0].tolist(), 'r': float(prims['r'])}
    out = []
    for p in prims:
        if p['type'] == 'line':
            out.append({'type': 'line', 'p0': L(p['p0'])[0].tolist(), 'p1': L(p['p1'])[0].tolist()})
        elif p['type'] == 'arc':
            from .rebuild import _arc_mid
            out.append({'type': 'arc', 'p0': L(p['p0'])[0].tolist(), 'p1': L(p['p1'])[0].tolist(),
                        'mid': L(_arc_mid(p))[0].tolist(), 'r': float(p['r'])})
        else:
            pts = np.asarray(p['pts'], float)
            if p.get('closed'):
                pts = np.vstack([pts, pts[:1]])
            out.append({'type': 'spline', 'pts': L(pts).tolist()})
    return out


def section_preview(V, F, origin, normal, tol=0.08):
    """One traced section for the web app and the sections script:
    {'origin', 'normal', 'loops': [(outer3d, [holes3d])], 'polylines':
    [[x, y, z]...] dense curves per loop for drawing, 'stats'}."""
    sec = fit_section(V, F, origin, normal, tol=tol)
    frame = (sec['origin'], sec['u'], sec['v'], sec['normal'])
    loops, polys = [], []
    for fo, fh in sec['loops']:
        loops.append((lift_prims(fo, frame), [lift_prims(h, frame) for h in fh]))
        for prims in [fo] + fh:
            if len(prims):
                q = prim_points(prims)
                polys.append(to_3d(np.vstack([q, q[:1]]), frame).tolist())
    return {'origin': np.asarray(sec['origin']).tolist(), 'normal': np.asarray(sec['normal']).tolist(),
            'loops': loops, 'polylines': polys, 'stats': sec['stats']}
