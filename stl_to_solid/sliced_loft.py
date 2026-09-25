"""Sliced loft: mesh -> stack of plane sections -> lofted B-spline solid.

The automated form of what one does by hand in Fusion (Create Mesh
Section Sketch at a spacing, Fit Curves to Mesh Section, Loft): slice the
body along one axis at `interval` mm, redraw every section outline as a
closed B-spline, and loft the stack into one smooth face per run of
sections. Runs break at flat faces perpendicular to the axis (a real step
becomes a real planar face, not a smear) and wherever the section topology
changes (loop count, hole count). Holes along the axis are lofted on their
own and cut. Runs that do not touch are bridged by extruding the earlier
outline to the next run's first plane. A dome end (a run whose end ring is
much smaller than the ring below it) is finished with a short ruled cone
to the mesh's apex instead of a flat cap.

Phase 0 measurements (docs/SLICED-LOFT.md) behind the recipe:
  * Every ring of a run is fitted by least squares onto one shared
    uniform periodic cubic B-spline basis (`fit_ring_poles`: K poles,
    doubling until the samples sit within 0.05 mm), and
    `BRepOffsetAPI_ThruSections` lofts those. The surface then has K x M
    poles and ThruSections has no knot vectors to unify. The alternatives
    measured: chord-length interpolating splines (the prototype) took
    minutes and `makeSplineApprox` rings 13 s per cornered run (knot
    unification), uniform interpolating splines built in 0.7 s but gave
    an 11k-pole surface that could not be tessellated in minutes;
    `SetSmoothing` and `SetMaxDegree` make ThruSections fail outright.
  * A slice within a hair of a dome tip (a 0.2 mm ring) makes the smooth
    surface flare by centimetres. End slices are therefore dropped when
    they are tiny against their neighbour, and the tip is a ruled cone.
  * A ruled loft is exact within d^2/(8R) and never flares; it is the
    per-run fallback when the smooth solid is invalid or its volume is off
    the section-area integral by more than 2 %, and the user option.

`loft_body(mesh, axis, interval, ruled)` returns (TopoDS_Shape, build_info);
build_info carries the thinned section outlines per run, which is what the
generated Fusion script draws (fusion_export.emit_fusion_loft_script).
"""
import numpy as np

from .section_fit import section_loops, section_curves, trim_slivers, to_3d, signed_area, douglas_peucker

EPS = 1e-3                 # mm: end / level slices sit this far inside the material
MAX_SECTIONS_PER_RUN = 60  # smooth loft sections per run (plan: ~60)
STEP_AREA_FRAC = 0.02      # a flat level holding this share of the bbox section is a step
END_AREA_FRAC = 0.5        # an end slice under this share of its neighbour is a dome tip
PTS_PER_MM = 2.0           # ring resampling density before the spline approximation
RING_MIN, RING_MAX = 32, 1200
SPLINE_TOL = 0.02          # mm: Douglas-Peucker thinning of the rings kept for the Fusion script
RING_FIT_TOL = 0.05        # mm: ring samples to their shared-basis B-spline
RING_POLES_MIN, RING_POLES_MAX = 16, 640   # poles per ring (doubling until RING_FIT_TOL holds)
VOL_CHECK_PCT = 1.0        # smooth run volume vs section-area integral before falling back to ruled
RULED_VOL_PCT = 5.0        # a ruled run off by more than this is refused (twisted rings)
SMOOTH_MAX_CHANGE = 0.15   # relative area jump between neighbours above which the pair is lofted ruled


def axis_index(mesh, slice_axis='auto'):
    """'x'|'y'|'z' -> 0|1|2; 'auto' -> the longest extent. The longest
    side is the rule the web app's slice slider and the partial loft use
    (it is known before anything is cut); the whole-body loft picks its
    axis by slicing structure instead (choose_axis)."""
    if slice_axis in (None, 'auto'):
        return int(np.argmax(mesh.extents))
    return 'xyz'.index(slice_axis)


def choose_axis(mesh, interval=0.2, join_mm=0.0, trim_mm=0.0, verbose=True):
    """The axis a whole-body loft should slice along: the one whose
    section stack has the fewest single-section runs, then the fewest
    runs plus steep neighbour pairs (area jumping by more than
    SMOOTH_MAX_CHANGE, which the loft can only do ruled), then the
    longest extent. Measured on a round lens cap (60 x 60 x 12 mm, X and
    Y a thousandth apart): the longest side gave 72 runs with 48 of one
    section (10 minutes, 707 loose solids), Z gave 5 runs (32 s, one
    solid). A 20 x 4 x 22 mm plate wants its 4 mm axis (one run); a disc
    is one run across its face too, but with the width racing at both
    ends, so the steep count sends it to its short axis. Scored on a
    coarse stack (about 1 mm, at least four sections along the thinnest
    side): under a second per axis. Returns (axis, scores) with
    scores['x'|'y'|'z'] = {'runs', 'single', 'steep', 'extent',
    'sections'}."""
    ext = np.asarray(mesh.extents, float)
    coarse = min(max(1.0, float(interval)), max(float(interval), float(ext.min()) / 4.0))
    scores, best, best_key = {}, None, None
    for ax in range(3):
        try:
            slices, _, _ = slice_mesh(mesh, ax, coarse, verbose=False, join_mm=join_mm, trim_mm=trim_mm)
            slices, _, _ = drop_dome_ends(slices)
            runs = build_runs(slices) if slices else []
        except Exception:
            slices, runs = [], []
        single = sum(1 for a, b, _ in runs if b == a)
        steep = 0
        for a, b, _ in runs:
            ar = [_total_area(slices[k][2]) for k in range(a, b + 1)]
            steep += sum(1 for i0, i1, smooth in _stretches(ar) if not smooth)
        sc = {'runs': len(runs), 'single': int(single), 'steep': int(steep),
              'extent': float(ext[ax]), 'sections': len(slices)}
        scores['xyz'[ax]] = sc
        if len(slices) < 2:
            continue
        key = (single, len(runs) + steep, -float(ext[ax]))
        if best_key is None or key < best_key:
            best, best_key = ax, key
    if best is None:
        best = int(np.argmax(ext))
    if verbose:
        print("[loft] axis auto -> " + 'xyz'[best].upper() + " by slicing structure: "
              + ", ".join(f"{k.upper()} {v['runs']} run(s)"
                          + (f" ({v['single']} of one section)" if v['single'] else "")
                          + (f" + {v['steep']} steep" if v['steep'] else "")
                          for k, v in scores.items()))
    return best, scores


def _frame(axis):
    n = np.zeros(3)
    n[axis] = 1.0
    from .section_fit import plane_basis
    u, v, n = plane_basis(n)
    return u, v, n


# ---------------------------------------------------------------------------
# where to slice

def step_levels(mesh, axis, frac=STEP_AREA_FRAC, gap=0.05):
    """Heights of flat faces perpendicular to the axis that hold more than
    `frac` of the bounding cross-section area: the shoulders a loft must
    not smooth across. Faces within `gap` mm of each other are one level
    (a parting-line rim exported at three heights 0.01 mm apart is one
    step, not three 0.01 mm runs)."""
    n = mesh.face_normals[:, axis]
    perp = np.abs(n) > 0.999
    if not perp.any():
        return []
    zc = mesh.triangles_center[perp][:, axis]
    ar = mesh.area_faces[perp]
    o = np.argsort(zc)
    zc, ar = zc[o], ar[o]
    u, v = [i for i in range(3) if i != axis]
    ref = float(mesh.extents[u] * mesh.extents[v])
    out, i = [], 0
    while i < len(zc):
        j = i
        while j + 1 < len(zc) and zc[j + 1] - zc[j] < gap:
            j += 1
        if ar[i:j + 1].sum() > frac * ref:
            out.append(float(np.average(zc[i:j + 1], weights=ar[i:j + 1])))
        i = j + 1
    return out


def slice_plan(mesh, axis, interval, levels, z_range=None):
    """[(z_query, z_target, kind)] sorted by z_query. Regular slices every
    `interval` from the low end; an 'end' slice EPS inside each end
    (snapped onto the end); a 'level' pair EPS either side of each step
    level (both snapped onto it, so the run above and the run below meet
    exactly). Regular slices within 3*EPS of a snapped one are dropped.
    `z_range` = (lo, hi) plans only that stretch of the body (clipped to
    the body), with 'end' slices at the stretch's ends."""
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    if z_range is not None:
        lo, hi = max(lo, float(min(z_range))), min(hi, float(max(z_range)))
        if hi - lo < 2 * EPS:
            return []
    reqs = [(lo + EPS, lo, 'end'), (hi - EPS, hi, 'end')]
    reqs += [(float(z), float(z), 'reg') for z in np.arange(lo + interval, hi - interval * 0.5, interval)]
    for L in levels:
        if L - lo < 2 * EPS or hi - L < 2 * EPS:
            continue
        reqs.append((L - EPS, L, 'level'))
        reqs.append((L + EPS, L, 'level'))
    reqs.sort()
    keep = []
    for r in reqs:
        if r[2] == 'reg' and any(abs(r[0] - k[0]) < 3 * EPS for k in keep[-2:]):
            continue
        if keep and r[2] == 'reg' and keep[-1][2] != 'reg' and abs(r[0] - keep[-1][0]) < 3 * EPS:
            continue
        keep.append(r)
    # a regular slice just before a snapped one
    out = []
    for i, r in enumerate(keep):
        if r[2] == 'reg' and i + 1 < len(keep) and keep[i + 1][2] != 'reg' \
                and abs(keep[i + 1][0] - r[0]) < 3 * EPS:
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# cutting

class Cutter:
    """Plane cuts of one mesh perpendicular to one axis, with the faces
    pre-filtered by their extent along the axis (one boolean mask per cut
    instead of the whole mesh through the chainer).

    With `join_mm` or `trim_mm` set, a cut goes through the sketch path
    (`section_curves`: free ends within join_mm joined, open chains left
    out, slivers thinner than trim_mm cut) instead of `section_loops`,
    which closes every open chain by its chord. On a watertight body both
    give the same loops; on a leaky one only the sketch path gives the
    outline the single-slice view shows."""

    def __init__(self, mesh, axis, join_mm=0.0, trim_mm=0.0):
        self.join_mm = float(join_mm or 0.0)
        self.trim_mm = float(trim_mm or 0.0)
        self.V = np.asarray(mesh.vertices, float)
        self.F = np.asarray(mesh.faces, np.int64)
        self.axis = axis
        z = self.V[:, axis][self.F]
        self.zmin, self.zmax = z.min(1), z.max(1)
        self.u, self.v, self.n = _frame(axis)

    def cut(self, z):
        """[(outer_xy, [hole_xy...])] in the plane frame, largest outer first."""
        m = (self.zmin <= z) & (self.zmax >= z)
        if not m.any():
            return []
        o = self.n * z
        if self.join_mm <= 0 and self.trim_mm <= 0:
            loops, _ = section_loops(self.V, self.F[m], o, self.n, min_area=1e-4)
            return loops
        loops, _, _, _ = section_curves(self.V, self.F[m], o, self.n, min_area=1e-4,
                                        join_mm=self.join_mm)
        if self.trim_mm > 0:
            loops = [(trim_slivers(outer, self.trim_mm)[0],
                      [trim_slivers(h, self.trim_mm)[0] for h in holes]) for outer, holes in loops]
            loops = [(o_, h_) for o_, h_ in loops if len(o_) >= 3]
        return loops

    def to_3d(self, xy, z):
        return to_3d(xy, (self.n * z, self.u, self.v, self.n))


def slice_mesh(mesh, axis, interval, verbose=True, z_range=None, join_mm=0.0, trim_mm=0.0):
    """All sections: [(z_target, kind, loops)] with empty cuts dropped,
    plus the step levels used. `z_range` limits the stack to one stretch
    of the body; `join_mm` / `trim_mm` pick the sketch-path cutter."""
    levels = step_levels(mesh, axis, gap=max(0.05, interval / 2))
    if z_range is not None:
        lo, hi = min(z_range), max(z_range)
        levels = [L for L in levels if lo < L < hi]
    plan = slice_plan(mesh, axis, interval, levels, z_range)
    cutter = Cutter(mesh, axis, join_mm, trim_mm)
    out = []
    for zq, zt, kind in plan:
        loops = cutter.cut(zq)
        if loops:
            out.append((zt, kind, loops))
    if verbose:
        print(f"[loft] axis {'xyz'[axis]}, {interval} mm: {len(out)} sections, "
              f"{len(levels)} step level(s)"
              + (f" at {[round(l, 2) for l in levels]}" if levels else ""))
    return out, levels, cutter


# ---------------------------------------------------------------------------
# rings

def resample(P, N):
    """N points at equal arc length along the closed 2-D polyline P."""
    P = np.asarray(P, float)
    if len(P) > 1 and np.allclose(P[0], P[-1]):
        P = P[:-1]
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    L = seg.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, L, N, endpoint=False)
    idx = np.clip(np.searchsorted(cum, t, side='right') - 1, 0, len(P) - 1)
    frac = (t - cum[idx]) / np.maximum(seg[idx], 1e-12)
    return P[idx] + (P[(idx + 1) % len(P)] - P[idx]) * frac[:, None]


def ccw(Q):
    return Q if signed_area(Q) > 0 else Q[::-1].copy()


def align(Q, prev):
    """Roll the ring so its start is nearest the previous ring's start
    (least total squared distance), so the loft does not twist. The first
    ring starts at its max-x point."""
    if prev is None:
        k = int(np.argmax(Q[:, 0] + 1e-3 * Q[:, 1]))
        return np.roll(Q, -k, 0)
    # sum |roll(Q, -k) - prev|^2 = |Q|^2 + |prev|^2 - 2 c[k], with c the
    # circular cross-correlation over both coordinates: all n rolls in
    # one FFT instead of an n^2 Python loop (n up to 1200, 60 rings a run)
    Q = np.asarray(Q, float)
    prev = np.asarray(prev, float)
    c = np.fft.ifft(np.fft.fft(Q, axis=0) * np.conj(np.fft.fft(prev, axis=0)), axis=0).real.sum(1)
    bk = int(np.argmax(c))
    return np.roll(Q, -bk, 0)


def _perimeter(xy):
    return float(np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1).sum())


def _ring_n(xy):
    return int(np.clip(round(_perimeter(xy) * PTS_PER_MM), RING_MIN, RING_MAX))


def _overlap(a, b, shrink=0.02):
    """Overlap test of two 2-D loops without shapely: bounding boxes and
    the fraction of a's vertices inside b (and b's inside a). The sample
    points are pulled `shrink` of the way towards their loop's centroid
    first: two identical sections (any prism) put every vertex exactly on
    the other's edges, where a ray cast is a coin toss, and an axis-aligned
    rectangle then scored 0 against its own twin."""
    from .section_fit import point_in_loop
    lo = np.maximum(a.min(0), b.min(0))
    hi = np.minimum(a.max(0), b.max(0))
    if (hi < lo).any():
        return 0.0
    ca, cb = a.mean(0), b.mean(0)
    sa = a[::max(1, len(a) // 40)]
    sb = b[::max(1, len(b) // 40)]
    sa = ca + (sa - ca) * (1.0 - shrink)
    sb = cb + (sb - cb) * (1.0 - shrink)
    fa = np.mean([point_in_loop(p, b) for p in sa])
    fb = np.mean([point_in_loop(p, a) for p in sb])
    return max(fa, fb)


def match(loops_a, loops_b):
    """Outer i in a -> outer j in b, when the counts agree, every pair
    overlaps and the hole counts agree; else None (a topology change)."""
    if len(loops_a) != len(loops_b) or not loops_a:
        return None
    used, m = set(), []
    for i, (ea, ha) in enumerate(loops_a):
        best, bj = 0.0, -1
        for j, (eb, hb) in enumerate(loops_b):
            if j in used:
                continue
            ov = _overlap(ea, eb)
            if ov > best:
                best, bj = ov, j
        if bj < 0 or best < 0.2 or len(ha) != len(loops_b[bj][1]):
            return None
        used.add(bj)
        m.append(bj)
    return m


def match_holes(ha, hb):
    """Hole i in ha -> nearest hole (by centroid) in hb."""
    used, m = set(), []
    ca = [h.mean(0) for h in ha]
    cb = [h.mean(0) for h in hb]
    for i in range(len(ha)):
        best, bj = None, -1
        for j in range(len(hb)):
            if j in used:
                continue
            d = float(np.linalg.norm(ca[i] - cb[j]))
            if best is None or d < best:
                best, bj = d, j
        used.add(bj)
        m.append(bj)
    return m


# ---------------------------------------------------------------------------
# runs

def _total_area(loops):
    return sum(abs(signed_area(o)) - sum(abs(signed_area(h)) for h in hs) for o, hs in loops)


def drop_dome_ends(slices):
    """Remove an end slice whose material area is under END_AREA_FRAC of
    its neighbour's: that ring is the last sliver of a dome, and lofting
    through it makes the smooth surface flare. Returns (slices, apex_lo,
    apex_hi): the flags say which ends want a cone to the apex."""
    apex_lo = apex_hi = False
    if len(slices) >= 2 and slices[0][1] == 'end' and \
            _total_area(slices[0][2]) < END_AREA_FRAC * _total_area(slices[1][2]):
        slices = slices[1:]
        apex_lo = True
    if len(slices) >= 2 and slices[-1][1] == 'end' and \
            _total_area(slices[-1][2]) < END_AREA_FRAC * _total_area(slices[-2][2]):
        slices = slices[:-1]
        apex_hi = True
    return slices, apex_lo, apex_hi


def build_runs(slices):
    """Group consecutive sections into runs of constant topology.

    Returns [(i, j, chains)]: slices i..j inclusive belong to the run, and
    chains[c] is the list of (outer_xy, holes_xy) of outer c through the
    run. A run ends where the next slice does not match (loop or hole
    count, overlap) and at a snapped level pair (two slices at the same
    height), so the flat face between them becomes a real planar face."""
    runs, i = [], 0
    while i < len(slices):
        j = i
        chains = [[lp] for lp in slices[i][2]]
        while j + 1 < len(slices):
            if abs(slices[j + 1][0] - slices[j][0]) < 1e-9:
                break                      # snapped pair: the step is the boundary
            m = match(slices[j][2], slices[j + 1][2])
            if m is None:
                break
            nxt = slices[j + 1][2]
            for ci, mj in enumerate(m):
                chains[ci].append(nxt[mj])
            j += 1
        runs.append((i, j, chains))
        i = j + 1
    return runs


def thin_indices(n, max_n=MAX_SECTIONS_PER_RUN, areas=None, change=0.02):
    """Indices to keep out of n sections, at most max_n: first and last
    always; with `areas` (one per section) a section is kept where the
    outline has changed by more than `change` (relative area) since the
    last kept one, plus an even backbone so no gap exceeds n/max_n
    sections. Where the outline changes fast (a dome tip, a fillet) the
    kept sections crowd; along a straight stretch they thin out. Without
    areas, an even spread."""
    if n <= max_n:
        return list(range(n))
    even = set(int(round(x)) for x in np.linspace(0, n - 1, max_n // 2))
    if areas is None:
        return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, max_n)))
    keep = [0]
    for i in range(1, n - 1):
        a0, a1 = areas[keep[-1]], areas[i]
        if i in even or abs(a1 - a0) > change * max(a0, 1e-12):
            keep.append(i)
    keep.append(n - 1)
    if len(keep) > max_n:
        # too many changes to keep them all: an even pick among them
        keep = [keep[int(round(x))] for x in np.linspace(0, len(keep) - 1, max_n)]
    return sorted(set(keep))


# ---------------------------------------------------------------------------
# OCC

# --- ring curves: one shared uniform periodic cubic B-spline basis per run

def _ubs_design(N, K):
    """Design matrix (N x K) of a uniform periodic cubic B-spline with K
    poles, knots at the integers 0..K, sampled at t_i = i K / N. Pole
    indexing follows OCC's Geom_BSplineCurve(periodic=True) convention:
    the span [i, i+1) blends poles i .. i+3 (mod K)."""
    t = np.arange(N) * (K / N)
    seg = np.floor(t).astype(int)
    u = t - seg
    b = ((1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
         (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6)
    A = np.zeros((N, K))
    rows = np.arange(N)
    for k in range(4):
        np.add.at(A, (rows, (seg + k) % K), b[k])
    return A


def fit_ring_poles(rings, tol=RING_FIT_TOL, kmin=None, kmax=None):
    """Least-squares poles of every ring on one shared basis.

    `rings`: (N, d) arrays with the same N (equal-arc-length samples,
    aligned). The pole count K doubles from kmin until every ring's
    samples sit within `tol` of its curve, or K reaches kmax or N/2.
    Sharing the basis is what keeps the loft light and robust: the
    surface has K x M poles (a sphere: 16 x 60) and ThruSections has no
    knot vectors to unify. Returns (poles [(K, d)], K, worst_dev)."""
    kmin = RING_POLES_MIN if kmin is None else kmin
    kmax = RING_POLES_MAX if kmax is None else kmax
    N = len(rings[0])
    kmax = max(kmin, min(kmax, N // 2))
    K = min(kmin, kmax)
    while True:
        A = _ubs_design(N, K)
        pinv = np.linalg.pinv(A)
        poles = [pinv @ np.asarray(r, float) for r in rings]
        dev = max(float(np.linalg.norm(A @ q - np.asarray(r, float), axis=1).max())
                  for q, r in zip(poles, rings))
        if dev <= tol or K >= kmax:
            return poles, K, dev
        K = min(kmax, K * 2)


def _wire_from_poles(poles3d):
    """Closed periodic cubic B-spline wire from (K, 3) poles on the
    uniform knot vector 0..K (see _ubs_design)."""
    import cadquery as cq
    from OCP.Geom import Geom_BSplineCurve
    from OCP.TColgp import TColgp_Array1OfPnt
    from OCP.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    K = len(poles3d)
    arr = TColgp_Array1OfPnt(1, K)
    for i, q in enumerate(poles3d):
        arr.SetValue(i + 1, gp_Pnt(float(q[0]), float(q[1]), float(q[2])))
    knots = TColStd_Array1OfReal(1, K + 1)
    mults = TColStd_Array1OfInteger(1, K + 1)
    for i in range(K + 1):
        knots.SetValue(i + 1, float(i))
        mults.SetValue(i + 1, 1)
    c = Geom_BSplineCurve(arr, knots, mults, 3, True)
    return cq.Wire.assembleEdges([cq.Edge(BRepBuilderAPI_MakeEdge(c).Edge())])


def _ring_wires(rings3d, tol=RING_FIT_TOL):
    """Wires for a run's rings on one shared basis; (wires, K, dev)."""
    poles, K, dev = fit_ring_poles(rings3d, tol)
    return [_wire_from_poles(q) for q in poles], K, dev


def _ring_wire(pts3d):
    """One ring on its own basis."""
    return _ring_wires([pts3d])[0][0]


def _thru(wires, ruled, apex=None, apex_first=False):
    """ThruSections solid through wires (cadquery's defaults: no smoothing
    flags, those make OCC fail). `apex` is an optional 3-D point added as
    a vertex section at the start (apex_first) or the end."""
    import cadquery as cq
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.gp import gp_Pnt
    ts = BRepOffsetAPI_ThruSections(True, bool(ruled), 1e-6)
    if apex is not None and apex_first:
        ts.AddVertex(BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, apex))).Vertex())
    for w in wires:
        ts.AddWire(w.wrapped)
    if apex is not None and not apex_first:
        ts.AddVertex(BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, apex))).Vertex())
    ts.Build()
    if not ts.IsDone():
        raise RuntimeError('ThruSections failed')
    s = cq.Shape.cast(ts.Shape())
    if s.wrapped.IsNull():
        raise RuntimeError('ThruSections gave a null shape')
    return s


def _valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return BRepCheck_Analyzer(shape.wrapped).IsValid()


def _volume(shape):
    """Adaptive volume: cq's Volume() is 1-2 % off on B-spline faces."""
    from .pipeline import accurate_volume
    return accurate_volume(shape)


def _polyline_gap(P, Q, chunk=256):
    """Largest distance from a point of closed polyline P to closed
    polyline Q (point-to-segment, vectorised in chunks)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    A = Q
    B = np.roll(Q, -1, 0)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-18)
    worst = 0.0
    for c in range(0, len(P), chunk):
        p = P[c:c + chunk]
        AP = p[:, None, :] - A[None, :, :]
        t = np.clip((AP * AB[None, :, :]).sum(2) / L2[None, :], 0.0, 1.0)
        D = AP - t[:, :, None] * AB[None, :, :]
        worst = max(worst, float(np.sqrt((D * D).sum(2)).min(1).max()))
    return worst


def _merge_identical(raw2d, rings2d, tol=RING_FIT_TOL):
    """Collapse runs of sections that are the same outline. A group of
    sections whose raw cut loops all lie within `tol` of the group's
    *first* loop (not of the previous one, so a shallow draft still steps
    to a new group every `tol` of drift) keeps only its first ring and
    repeats that ring at the last section's height, so the pair is an
    exact extrusion: a stack of 60 equal rectangles becomes one prism
    instead of 59 bands, with no twist from the sample phase drifting
    slice to slice. The raw loops are compared (as polylines, both ways),
    not the resampled rings: resampling rounds the corners by up to half
    a sample spacing, which is more than `tol`. Returns (rings2d, keep,
    forced): the new ring list, the source index of each ring, and, in
    new-index space, the pairs (i, i+1) that were a group."""
    n = len(rings2d)
    out, keep, forced, i = [np.asarray(rings2d[0], float)], [0], set(), 0
    while i < n - 1:
        j = i
        while j + 1 < n and _polyline_gap(raw2d[j + 1], raw2d[i]) <= tol \
                and _polyline_gap(raw2d[i], raw2d[j + 1]) <= tol:
            j += 1
        if j > i:
            forced.add(len(out) - 1)
            out.append(np.asarray(rings2d[i], float).copy())
            keep.append(j)
            i = j
        else:
            out.append(np.asarray(rings2d[i + 1], float))
            keep.append(i + 1)
            i += 1
    return out, keep, forced


def _stretches(areas, max_change=None, forced=()):
    """Split a run's ring indices into stretches: consecutive rings whose
    material area changes by at most `max_change` (relative) stay in one
    smooth stretch; a jump bigger than that is lofted ruled between the
    two rings. A smooth B-spline through sections that shrink by half per
    step (a dome tip) overshoots by tenths of a millimetre; a ruled pair
    there is exact within d^2/8R. Pairs in `forced` (the collapsed
    identical groups of _merge_identical) are ruled as well and never
    join a smooth stretch: a smooth surface through a repeated ring is
    the flare case. Returns [(i0, i1, smooth)] covering 0..n-1 with
    shared ends."""
    if max_change is None:
        max_change = SMOOTH_MAX_CHANGE
    n = len(areas)
    forced = set(forced)
    steep = [i in forced or
             abs(areas[i + 1] - areas[i]) > max_change * max(min(areas[i], areas[i + 1]), 1e-12)
             for i in range(n - 1)]
    out, i = [], 0
    while i < n - 1:
        j = i
        while j < n - 1 and steep[j] == steep[i]:
            j += 1
        out.append((i, j, not steep[i]))
        i = j
    return out


def _loft_rings(rings3d, areas, zs, ruled, tag, verbose, merge=None, ax_vec=None, forced=()):
    """Solid through 3-D rings (each (N, 3), same N, aligned) on one shared
    spline basis. Smooth unless `ruled`, in stretches: where the section
    area jumps by more than SMOOTH_MAX_CHANGE between neighbours the pair
    is lofted ruled (_stretches). A smooth stretch that comes out invalid
    or off the trapezoid integral of its areas by more than VOL_CHECK_PCT
    is rebuilt ruled. The stretch solids are glued into one. A result off
    the whole run's integral by more than RULED_VOL_PCT is refused
    (raises): the rings do not correspond.

    Two short cuts, both measured. Rings the shared basis cannot fit
    within RING_FIT_TOL (cornered outlines at the pole cap) go ruled
    straight away, because every smooth loft through such rings came out
    wrong by orders of magnitude before falling back anyway. And a ruled
    run then takes `merge` = (rings3d, areas, zs, forced) from
    _merge_identical: identical sections collapsed to one straight
    extrusion each (the `forced` pairs). A smooth run keeps every ring:
    near a sphere's equator neighbours are also "identical" within tol,
    and a straight band there would cut the one smooth face in three.
    `forced` pairs of the full ring list are ruled in a smooth run as
    well (a bridge: the last ring repeated at the next run's plane).
    `ax_vec` is the slicing axis, along which an extruded pair's top is
    moved. Returns (solid, used_ruled, stats) with stats =
    {'ruled_direct': bool, 'merged': sections dropped, 'extruded':
    [[z0, z1], ...]}."""
    stats = {'ruled_direct': False, 'merged': 0, 'extruded': []}
    forced = set(forced)
    if ax_vec is None:
        ax_vec = _ring_axis(rings3d)
    _, K, dev = fit_ring_poles(rings3d)
    if dev > RING_FIT_TOL and not ruled:
        ruled = True
        stats['ruled_direct'] = True
        if verbose:
            print(f"[loft] {tag}: rings fit to {dev:.3f} mm with {K} poles (cap); ruled directly")
    if ruled and merge is not None and len(merge[0]) < len(rings3d):
        stats['merged'] = len(rings3d) - len(merge[0])
        rings3d, areas, zs, forced = merge
        if verbose:
            print(f"[loft] {tag}: {stats['merged']} repeated section(s) merged into "
                  f"{len(forced)} straight stretch(es)")
    wires, K, dev = _ring_wires(rings3d)
    est = float(np.trapezoid(areas, zs)) if len(zs) > 1 else 0.0
    try:
        solid, used_ruled = _loft_stretches(wires, areas, zs, ruled, forced, est, tag, verbose)
        return solid, used_ruled, stats
    except Exception as ex:
        # the run must not vanish: pair by pair, a bad pair extruded
        if verbose:
            print(f"[loft] {tag}: {ex}; lofting pair by pair instead")
    solid, ext = _loft_pairs(wires, ax_vec, areas, zs, tag, verbose)
    stats['extruded'] = ext
    return solid, True, stats


def _ring_axis(rings3d):
    """Unit normal of the ring planes (the slicing axis) by least squares
    on the first ring. Only a fallback: the centroid-to-centroid
    direction is tilted wherever the outline drifts sideways, and an
    extrusion along it once stopped 0.02 mm short of the next piece."""
    P = np.asarray(rings3d[0], float)
    P = P - P.mean(0)
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    n = vt[-1]
    if len(rings3d) > 1:
        d = np.asarray(rings3d[-1], float).mean(0) - np.asarray(rings3d[0], float).mean(0)
        if d @ n < 0:
            n = -n
    return n / max(np.linalg.norm(n), 1e-12)


def _loft_pairs(wires, ax, areas, zs, tag, verbose):
    """Fallback for a run whose rings do not correspond: one ruled loft
    per neighbouring pair, each checked against its own trapezoid volume;
    a pair that fails becomes an extrusion of its lower ring (the wire
    translated, so the bottom of every pair is exactly the top of the one
    before). Nothing is dropped: at worst the outline steps once per
    interval where the real one changed. Returns (solid, extruded) with
    extruded = [[z0, z1], ...] of the pairs that were extruded."""
    import cadquery as cq
    ax = np.asarray(ax, float)
    pieces, ext = [], []
    for i in range(len(wires) - 1):
        dz = float(zs[i + 1] - zs[i])
        e = 0.5 * (areas[i] + areas[i + 1]) * dz
        piece = None
        try:
            cand = _thru([wires[i], wires[i + 1]], True)
            off = abs(_volume(cand) - e) / e * 100 if e > 0 else 0.0
            if _valid(cand) and off <= RULED_VOL_PCT:
                piece = cand
        except Exception:
            piece = None
        if piece is None:
            top = wires[i].translate(cq.Vector(*(ax * dz)))
            piece = _thru([wires[i], top], True)
            ext.append([float(zs[i]), float(zs[i + 1])])
        pieces.append(piece)
    if verbose and ext:
        print(f"[loft] {tag}: {len(ext)} of {len(pieces)} pair(s) extruded straight "
              f"(their rings do not correspond)")
    solid, how = _fuse_all(pieces, verbose)
    return solid, ext


def _loft_stretches(wires, areas, zs, ruled, forced, est, tag, verbose):
    """The smooth / ruled stretches of _loft_rings; raises when the result
    is off the run's section integral (rings that do not correspond)."""
    used_ruled = bool(ruled)
    pieces = []
    for i0, i1, smooth in _stretches(areas, forced=forced):
        w = wires[i0:i1 + 1]
        e = float(np.trapezoid(areas[i0:i1 + 1], zs[i0:i1 + 1]))
        if smooth and not ruled and len(w) >= 2:
            try:
                piece = _thru(w, False)
                off = abs(_volume(piece) - e) / e * 100 if e > 0 else 0.0
                if _valid(piece) and off <= VOL_CHECK_PCT:
                    pieces.append(piece)
                    continue
                if verbose:
                    print(f"[loft] {tag} {zs[i0]:.2f}..{zs[i1]:.2f}: smooth loft "
                          f"{'invalid' if not _valid(piece) else f'volume off by {off:.1f}%'}; ruled instead")
            except Exception as ex:
                if verbose:
                    print(f"[loft] {tag} {zs[i0]:.2f}..{zs[i1]:.2f}: smooth loft failed "
                          f"({type(ex).__name__}: {ex}); ruled instead")
        pieces.append(_thru(w, True))
        used_ruled = True
    if len(pieces) == 1:
        solid = pieces[0]
    else:
        solid, how = _fuse_all(pieces, False)
        if how == 'compound':
            solid = _thru(wires, True)
            used_ruled = True
    vol = _volume(solid)
    off = abs(vol - est) / est * 100 if est > 0 else 0.0
    if off > RULED_VOL_PCT:
        raise RuntimeError(f'loft volume off the section integral by {off:.0f}%; '
                           f'the rings do not correspond')
    return solid, used_ruled


def _extrude(outer3d, holes3d, d):
    """Prism of an outline (minus its holes) along d: a ruled loft between
    the ring and its translated copy. extrudeLinear refuses B-spline
    wires as 'not planar' at its tolerance, so it is not used."""
    d = np.asarray(d, float)
    body = _thru(_ring_wires([outer3d, outer3d + d])[0], True)
    for h in holes3d:
        tool = _thru(_ring_wires([h - d * 0.01, h + d * 1.01])[0], True)
        body = body.cut(tool)
    return body


def _fuse_two(a, b, want, tol_pct):
    """One valid solid from two touching pieces, or None: glued fuse (the
    pieces share planar caps), then plain, then fuzzy; a result that is
    one solid holding both volumes (within tol_pct) but reads invalid
    gets one ShapeFix pass (two caps whose B-spline rings overlap along a
    straight stretch come out with inflated tolerances). Returns (shape,
    volume, how) or None."""
    for how, kw in (('glue', dict(glue=True)), ('plain', {}), ('fuzzy', dict(tol=1e-3))):
        try:
            f = a.fuse(b, **kw)
        except Exception:
            continue
        try:
            if len(f.Solids()) != 1:
                continue
            v = _volume(f)
            if abs(v - want) / want * 100 > tol_pct if want > 0 else False:
                continue
            if _valid(f):
                return f, v, how
            g = f.fix()
            if len(g.Solids()) == 1 and _valid(g) and abs(_volume(g) - want) / want * 100 <= tol_pct:
                return g, _volume(g), how + '+fix'
        except Exception:
            continue
    return None


def _fuse_all(solids, verbose, tol_pct=2.0):
    """One shape from the run solids, fused one at a time in the order
    given (which is along the axis). Each piece joins the chain it
    touches: the latest chain first, then any earlier one, else it starts
    a chain of its own; the chains are joined at the end. A fuse that
    looks valid but lost volume is wrong however valid it looks (a glued
    fuse of 19 pieces of a block once returned 533 mm^3 out of 24000, and
    of the bracket's 14 pieces 2153 out of 3675), so every step checks
    the volume (_fuse_two). Chains that will not join stay loose in a
    compound, with a warning. Returns (shape, how) with how in 'single'
    | 'glue' | 'fused' | 'compound'."""
    import cadquery as cq
    if len(solids) == 1:
        return solids[0], 'single'
    chains, hows = [], set()
    for piece in solids:
        pv = _volume(piece)
        placed = False
        for ci in range(len(chains) - 1, -1, -1):
            acc, av = chains[ci]
            got = _fuse_two(acc, piece, av + pv, tol_pct)
            if got is not None:
                chains[ci] = (got[0], got[1])
                hows.add(got[2])
                placed = True
                break
        if not placed:
            chains.append((piece, pv))
    # join the chains
    changed = True
    while changed and len(chains) > 1:
        changed = False
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                got = _fuse_two(chains[i][0], chains[j][0], chains[i][1] + chains[j][1], tol_pct)
                if got is not None:
                    chains[i] = (got[0], got[1])
                    hows.add(got[2])
                    del chains[j]
                    changed = True
                    break
            if changed:
                break
    if len(chains) > 1:
        if verbose:
            print(f"[loft] {len(solids)} piece(s) fused into {len(chains)} solids that would not "
                  f"join; left as separate solids in one compound")
        return cq.Compound.makeCompound([c for c, _ in chains]), 'compound'
    acc, acc_vol = chains[0]
    try:
        c = acc.clean()
        if _valid(c) and len(c.Solids()) == 1 and abs(_volume(c) - acc_vol) / acc_vol * 100 <= tol_pct:
            acc = c
    except Exception:
        pass
    return acc, 'glue' if hows <= {'glue'} else 'fused'


# ---------------------------------------------------------------------------
# the body

def loft_body(mesh, axis, interval=0.2, ruled=False, verbose=True, z_range=None,
              join_mm=0.0, trim_mm=0.0):
    """Sliced-loft solid of one closed body. Returns (TopoDS_Shape, info).

    `z_range` = (z0, z1), absolute along the axis, lofts only that stretch
    of the body with flat ends (no dome tips at a cut end); info gains
    'range'. `join_mm` / `trim_mm` are the sketch-path cutter's settings
    (see Cutter).

    info: {'mode': 'loft', 'axis': [..], 'axis_name', 'interval', 'ruled'
    (requested), 'n_sections', 'n_runs', 'n_levels', 'n_holes',
    'n_ruled_runs', 'fuse', 'runs': [{'z0', 'z1', 'n', 'ruled', 'sections':
    [{'z', 'outer': [[x, y, z], ...], 'holes': [[...], ...]}]}]} where the
    section outlines are the thinned rings the loft went through, thinned
    again by Douglas-Peucker at SPLINE_TOL for the Fusion script.
    """
    import cadquery as cq
    if isinstance(axis, str):
        axis = axis_index(mesh, axis)
    slices, levels, cutter = slice_mesh(mesh, axis, interval, verbose, z_range, join_mm, trim_mm)
    if len(slices) < 2:
        raise RuntimeError(f'only {len(slices)} section(s) along {"xyz"[axis]}; nothing to loft')
    if z_range is None:
        slices, apex_lo, apex_hi = drop_dome_ends(slices)
    else:
        apex_lo = apex_hi = False          # a cut end is a flat face, never a dome tip
    runs = build_runs(slices)
    if verbose:
        print(f"[loft] {len(runs)} run(s): "
              + ", ".join(f"{slices[a][0]:.2f}..{slices[b][0]:.2f} ({b - a + 1} sections, {len(c)} outline(s))"
                          for a, b, c in runs))
    lo, hi = float(mesh.bounds[0][axis]), float(mesh.bounds[1][axis])
    if z_range is not None:
        lo, hi = max(lo, float(min(z_range))), min(hi, float(max(z_range)))
    ax_vec = np.zeros(3)
    ax_vec[axis] = 1.0

    solids, run_infos, n_holes, n_ruled, n_merged, n_extruded, skipped = [], [], 0, 0, 0, 0, []
    for ri, (a, b, chains) in enumerate(runs):
        zs_all = [slices[k][0] for k in range(a, b + 1)]
        keep = thin_indices(len(zs_all), areas=[_total_area(slices[k][2]) for k in range(a, b + 1)])
        zs = [zs_all[k] for k in keep]
        rinfo = {'z0': zs[0], 'z1': zs[-1], 'n': len(zs), 'ruled': bool(ruled), 'sections': []}
        secs = [{'z': z, 'outer': None, 'holes': []} for z in zs]
        # a gap to the next run (a topology change between two slices, not
        # a flat step) is bridged inside this run: its last section is
        # repeated at the next run's first plane, one straight stretch,
        # so the two runs share that plane exactly and no sliver solid
        # is needed
        bridge_to = None
        if ri + 1 < len(runs):
            zb = slices[runs[ri + 1][0]][0]
            if zb - zs[-1] > 1e-6:
                bridge_to = float(zb)
                zs = zs + [bridge_to]
                rinfo['bridge_to'] = bridge_to
        distinct = abs(zs[-1] - zs[0]) > 1e-6 and len(zs) >= 2
        for ci, chain in enumerate(chains):
            chain = [chain[k] for k in keep]
            if bridge_to is not None:
                chain = chain + [chain[-1]]
            N = max(_ring_n(e) for e, _ in chain)
            rings, prev = [], None
            for (e, _) in chain:
                Q = align(ccw(resample(e, N)), prev)
                rings.append(Q)
                prev = Q
            if bridge_to is not None:
                rings[-1] = rings[-2].copy()
            rings3d = [cutter.to_3d(Q, z) for Q, z in zip(rings, zs)]
            for si, r3 in enumerate(rings3d[:len(secs)]):
                pts = r3[douglas_peucker(np.vstack([r3, r3[:1]]), SPLINE_TOL)[:-1]]
                secs[si]['outer'] = pts.tolist() if ci == 0 else secs[si].get('outer')
                if ci > 0:
                    secs[si].setdefault('more_outers', []).append(pts.tolist())
            if not distinct:
                continue
            tag = f"run {ri + 1} outline {ci + 1}"
            areas = [abs(signed_area(Q)) for Q in rings]
            bridge_pairs = {len(rings) - 2} if bridge_to is not None else set()
            # identical sections (a prism) collapse to one straight
            # stretch each, if the run comes out ruled
            mrings, mkeep, forced = _merge_identical([e for e, _ in chain], rings)
            mzs = [zs[k] for k in mkeep]
            merge = ([cutter.to_3d(Q, z) for Q, z in zip(mrings, mzs)],
                     [abs(signed_area(Q)) for Q in mrings], mzs, forced)
            try:
                body, used_ruled, lstats = _loft_rings(rings3d, areas, zs, ruled, tag, verbose, merge,
                                                       ax_vec, bridge_pairs)
            except Exception as e:
                # even the pair chain failed: the first outline extruded
                # over the whole run keeps the material; only if that
                # fails too is the outline skipped (a hole in the part,
                # reported)
                try:
                    body = _extrude(rings3d[0], [], ax_vec * (zs[-1] - zs[0]))
                    used_ruled, lstats = True, {'merged': 0, 'extruded': [[zs[0], zs[-1]]]}
                    if verbose:
                        print(f"[loft] {tag}: {type(e).__name__}: {e}; first section extruded over the run")
                except Exception as e2:
                    skipped.append({'what': 'run', 'z0': float(zs[0]), 'z1': float(zs[-1]),
                                    'text': f"{tag}: {type(e).__name__}: {e}; extrusion failed too: {e2}"})
                    if verbose:
                        print(f"[loft] {tag} skipped ({type(e).__name__}: {e})")
                    continue
            n_ruled += used_ruled
            n_merged += lstats['merged']
            rinfo['ruled'] = rinfo['ruled'] or used_ruled
            rinfo['merged'] = rinfo.get('merged', 0) + lstats['merged']
            if lstats.get('extruded'):
                rinfo.setdefault('extruded', []).extend(lstats['extruded'])
                n_extruded += len(lstats['extruded'])
            # holes: chained by centroid, lofted the same way, cut
            nh = len(chain[0][1])
            for hi_ in range(nh):
                hraw, hidx = [], hi_
                for k, (e, holes) in enumerate(chain):
                    if k > 0:
                        hidx = match_holes(chain[k - 1][1], holes)[hidx]
                    hraw.append(holes[hidx])
                Nh = max(_ring_n(hp) for hp in hraw)
                hrings, prev = [], None
                for hp in hraw:
                    Q = align(ccw(resample(hp, Nh)), prev)
                    hrings.append(Q)
                    prev = Q
                if bridge_to is not None:
                    hrings[-1] = hrings[-2].copy()
                h3d = [cutter.to_3d(Q, z) for Q, z in zip(hrings, zs)]
                for si, r3 in enumerate(h3d[:len(secs)]):
                    pts = r3[douglas_peucker(np.vstack([r3, r3[:1]]), SPLINE_TOL)[:-1]]
                    secs[si]['holes'].append(pts.tolist())
                hareas = [abs(signed_area(Q)) for Q in hrings]
                mh, mk, mf = _merge_identical(hraw, hrings)
                mhz = [zs[k] for k in mk]
                mh3 = [cutter.to_3d(Q, z) for Q, z in zip(mh, mhz)]
                htag = tag + f" hole {hi_ + 1}"
                d = ax_vec * EPS
                try:
                    cut, tool = None, None
                    # the tool's caps on the run's caps first (its cap edges
                    # are then the rings, which the next run shares); a
                    # hair of overshoot only if that cut comes out wrong
                    for over in (0.0, 1.0):
                        th = [r.copy() for r in h3d]
                        tm = [r.copy() for r in mh3]
                        for lst in (th, tm):
                            lst[0] = lst[0] - d * over
                            lst[-1] = lst[-1] + d * over
                        tool, _, hstats = _loft_rings(th, hareas, zs, ruled or used_ruled, htag, verbose,
                                                      (tm, [abs(signed_area(Q)) for Q in mh], mhz, mf),
                                                      ax_vec, bridge_pairs)
                        c = body.cut(tool)
                        if not c.wrapped.IsNull() and _valid(c) and len(c.Solids()) >= 1:
                            cut = c
                            break
                    if cut is None:
                        raise RuntimeError('cut gave an invalid solid')
                    n_merged += hstats['merged']
                    body = cut
                    n_holes += 1
                except Exception as e:
                    skipped.append({'what': 'hole', 'z0': float(zs[0]), 'z1': float(zs[-1]),
                                    'text': f"{htag}: {type(e).__name__}: {e}"})
                    if verbose:
                        print(f"[loft] {htag} left filled ({type(e).__name__}: {e})")
            # dome tips: a ruled cone from the end ring to the mesh apex
            for at_end, want, ring, r3, zend in ((ri == 0, apex_lo, rings[0], rings3d[0], lo),
                                                 (ri == len(runs) - 1, apex_hi, rings[-1], rings3d[-1], hi)):
                if not (at_end and want and nh == 0):
                    continue
                touching = abs((zs[0] if zend == lo else zs[-1]) - (slices[0][0] if zend == lo else slices[-1][0])) < 1e-9
                if not touching:
                    continue
                try:
                    apex = cutter.to_3d(ring.mean(0)[None, :], zend)[0]
                    solids.append(_thru([_ring_wire(r3)], True, apex, apex_first=(zend == lo)))
                except Exception as e:
                    skipped.append({'what': 'cone', 'z0': float(zend), 'z1': float(zend),
                                    'text': f"{tag} apex cone: {type(e).__name__}: {e}"})
                    if verbose:
                        print(f"[loft] {tag} apex cone skipped ({type(e).__name__}: {e})")
            solids.append(body)
        rinfo['sections'] = secs
        run_infos.append(rinfo)

    if not solids:
        raise RuntimeError('no run could be lofted')
    shape, how = _fuse_all(solids, verbose)
    if verbose:
        print(f"[loft] {len(solids)} solid(s) -> {len(shape.Faces())} faces "
              f"({how} fuse), volume {_volume(shape):.0f} mm^3")
    info = {'mode': 'loft', 'axis': ax_vec.tolist(), 'axis_name': 'xyz'[axis],
            'interval': float(interval), 'ruled': bool(ruled),
            'range': [lo, hi] if z_range is not None else None,
            'join_mm': float(join_mm or 0.0), 'trim_mm': float(trim_mm or 0.0),
            'n_sections': len(slices), 'n_runs': len(runs), 'n_levels': len(levels),
            'n_holes': n_holes, 'n_ruled_runs': int(n_ruled), 'n_merged': int(n_merged),
            'n_extruded_pairs': int(n_extruded), 'fuse': how, 'n_solids': len(shape.Solids()),
            'faces': len(shape.Faces()), 'skipped': skipped, 'runs': run_infos}
    return shape.wrapped, info
