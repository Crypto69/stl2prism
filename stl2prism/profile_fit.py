"""Fit clean line/arc primitives to section polylines.

The cross-section of a faceted mesh is a dense polyline: true arcs appear
as chords, true lines as one or two long segments (a planar wall is two
triangles, so the section crosses it at its ends and at the diagonal). We
recover the design intent with:

  * Taubin algebraic circle fitting (bias-corrected, noise-robust)
  * deviation measured against the whole polyline, not just its vertices —
    the sagitta of a chord against a fitted circle is checked at the foot
    of the perpendicular, so a big circle through the ends of a straight
    wall cannot masquerade as an arc
  * recursive split (at real corners first, else at the max chord
    deviation) followed by a cyclic merge pass, so an arc split in two or a
    line split at a diagonal crossing become one primitive again
  * frame snapping (lines within a small angle of the dominant in-plane
    directions are squared up), radius/centre unification
  * junction solving: consecutive primitives meet exactly — line/line at
    their intersection, line/arc tangent where the data says tangent
    (fillets), else at the line-circle intersection

Primitives are dicts: {'type': 'line', 'p0', 'p1'} or
{'type': 'arc', 'center', 'r', 'p0', 'p1', 'ccw'}; a full circle is
{'type': 'arc', 'center', 'r', 'full': True}.
"""
import numpy as np


# ---------- fitting primitives ----------

def fit_line(pts):
    """Total-least-squares line through pts. Returns dict with 'dir',
    'point' (centroid) and 'dev' (max distance of pts to the line)."""
    c = pts.mean(axis=0)
    d = pts - c
    if len(pts) == 2:
        v = pts[1] - pts[0]
        L = np.linalg.norm(v)
        direction = v / L if L > 1e-12 else np.array([1.0, 0.0])
        return {'type': 'line', 'p0': pts[0], 'p1': pts[-1], 'dir': direction,
                'point': c, 'dev': 0.0}
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    direction = vt[0]
    dev = np.abs(d @ np.array([-direction[1], direction[0]]))
    return {'type': 'line', 'p0': pts[0], 'p1': pts[-1],
            'dir': direction, 'point': c, 'dev': float(dev.max())}


def fit_circle_taubin(pts, polyline=True):
    """Taubin algebraic circle fit (SVD form). Returns center, radius and
    'dev' = max deviation from the circle — of the polyline (vertices AND
    chord interiors) when `polyline`, else of the points only — or None if
    the points are collinear."""
    pts = np.asarray(pts, float)
    x, y = pts[:, 0], pts[:, 1]
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    z = u * u + v * v
    zm = z.mean()
    if zm <= 0:
        return None
    Z = (z - zm) / (2.0 * np.sqrt(zm))
    A = np.column_stack([Z, u, v])
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    a = vt[-1]
    a0 = a[0] / (2.0 * np.sqrt(zm))
    if abs(a0) < 1e-12:
        return None
    cx = -a[1] / (2 * a0) + xm
    cy = -a[2] / (2 * a0) + ym
    c = np.array([cx, cy])
    r = float(np.hypot(x - cx, y - cy).mean())
    c, r = _refine_circle_geometric(pts, c, r)
    dev = (polyline_circle_dev(pts, c, r) if polyline
           else float(np.abs(np.hypot(x - cx, y - cy) - r).max()))
    return {'type': 'arc', 'center': c, 'r': r, 'dev': dev}


def _refine_circle_geometric(pts, c, r, iters=8):
    """A few Gauss-Newton steps on the geometric residuals |p - c| - r,
    starting from the algebraic (Taubin) solution. The algebraic fit is
    biased on short arcs with uneven chords; the geometric optimum can
    sit inside a tolerance the algebraic one just misses."""
    c = np.asarray(c, float).copy()
    r = float(r)
    for _ in range(iters):
        d = pts - c
        dist = np.hypot(d[:, 0], d[:, 1])
        if np.any(dist < 1e-12):
            break
        res = dist - r
        J = np.column_stack([-d[:, 0] / dist, -d[:, 1] / dist,
                             -np.ones(len(pts))])
        try:
            step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        c = c + step[:2]
        r = r + step[2]
        if r <= 0:
            return np.asarray(c, float), float(abs(r))
        if np.abs(step).max() < 1e-9:
            break
    return c, r


def polyline_circle_dev(pts, c, r):
    """Max deviation of the polyline (vertices + chord interiors) from the
    circle (c, r). The extreme along a chord is at the foot of the
    perpendicular from the centre when that foot lies within the chord."""
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    dev = float(np.abs(d - r).max())
    if len(pts) > 1:
        a = pts[:-1]
        b = pts[1:]
        ab = b - a
        L2 = (ab * ab).sum(axis=1)
        ok = L2 > 1e-18
        t = np.zeros(len(a))
        t[ok] = ((c - a[ok]) * ab[ok]).sum(axis=1) / L2[ok]
        inside = (t > 0) & (t < 1)
        if inside.any():
            foot = a[inside] + t[inside, None] * ab[inside]
            df = np.hypot(foot[:, 0] - c[0], foot[:, 1] - c[1])
            dev = max(dev, float(np.abs(df - r).max()))
    return dev


def _arc_params(pts, fit):
    """Start/end angles + sweep direction for an arc through pts (uses the
    signed area sweep of the polyline about the centre, robust to noise)."""
    c = fit['center']
    rel = pts - c
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    dang = np.diff(ang)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    total = float(dang.sum())
    return {'a0': float(ang[0]), 'a1': float(ang[0] + total),
            'ccw': bool(total > 0), 'sweep': abs(total)}


# ---------- polyline segmentation ----------

def turning_angles(pts, closed=True):
    """Absolute turning angle (rad) at each vertex of the polyline."""
    n = len(pts)
    if closed:
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
    else:
        prev = np.vstack([pts[:1], pts[:-1]])
        nxt = np.vstack([pts[1:], pts[-1:]])
    a = pts - prev
    b = nxt - pts
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    ok = (la > 1e-12) & (lb > 1e-12)
    cosang = np.ones(n)
    cosang[ok] = (a[ok] * b[ok]).sum(axis=1) / (la[ok] * lb[ok])
    ang = np.arccos(np.clip(cosang, -1, 1))
    if not closed:
        ang[0] = ang[-1] = 0.0
    return ang


def _dedupe(pts, eps=1e-9):
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > eps:
            keep.append(i)
    pts = pts[keep]
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) <= eps:
        pts = pts[:-1]
    return pts


CORNER_RAD = np.radians(30.0)     # a turn this sharp is a real corner


def segment_polyline(pts, tol=0.08, closed=True, min_arc_pts=4):
    """Split a (closed) polyline into line/arc primitives within tol.

    Returns a list of primitives in order; for a closed ring the chain is
    cyclic (last primitive ends where the first starts).
    """
    pts = _dedupe(np.asarray(pts, float))
    n = len(pts)
    if n < 2:
        return []
    if closed:
        # start at the sharpest real corner, if any: splits then land on
        # corners rather than mid-arc
        ta = turning_angles(pts, closed=True)
        k = int(np.argmax(ta))
        if ta[k] > CORNER_RAD:
            pts = np.roll(pts, -k, axis=0)
        pts = np.vstack([pts, pts[:1]])
    prims = _segment_span(pts, tol, min_arc_pts, closed=closed)
    # merge and boundary refinement feed each other: handing a stray
    # vertex back to its line can bring two halves of a fillet within tol
    # of one circle, so alternate until nothing changes
    for _ in range(3):
        prims = merge_pass(prims, pts_of(prims), tol, min_arc_pts, closed=closed)
        n = len(prims)
        prims = refine_boundaries(prims, tol, min_arc_pts, closed=closed)
        merged = merge_pass(prims, pts_of(prims), tol, min_arc_pts, closed=closed)
        if len(merged) == n:
            prims = merged
            break
        prims = merged
    return prims


def _refit_same_type(p, pts, tol, min_arc_pts):
    """Refit primitive `p`'s type to `pts`; None if it no longer fits."""
    if len(pts) < 2:
        return None
    if p['type'] == 'line':
        f = fit_line(pts)
        if f['dev'] <= tol and not _line_hides_bend(pts, f['dev'], tol):
            return _mk_line(pts)
        return None
    if len(pts) < min_arc_pts:
        return None
    circ = fit_circle_taubin(pts)
    if circ and circ['dev'] <= tol and _arc_sane(pts, circ) \
            and not _arc_hides_straight_run(pts, circ, tol):
        circ.update(_arc_params(pts, circ))
        circ['p0'], circ['p1'] = pts[0].copy(), pts[-1].copy()
        circ['_pts'] = pts
        return circ
    return None


def _residual(p, pts):
    if p['type'] == 'line':
        return fit_line(pts)['dev']
    return polyline_circle_dev(pts, p['center'], p['r'])


def refine_boundaries(prims, tol, min_arc_pts, closed=True, max_shift=3):
    """Move the shared vertex between adjacent primitives so that each
    vertex belongs to the primitive that explains it best. A recursive
    split lands within `tol` of the true junction, which lets a line
    swallow the first vertex of a neighbouring arc (tilting the line by a
    fraction of a degree); handing that vertex back makes CAD lines exact."""
    n = len(prims)
    if n < 2:
        return prims
    for _ in range(2):
        moved = False
        for i in range(n if closed else n - 1):
            j = (i + 1) % n
            if i == j:
                continue
            a, b = prims[i], prims[j]
            if a['type'] == b['type'] == 'line':
                continue                       # a corner: leave it
            pa, pb = a['_pts'], b['_pts']
            best = None
            base = _residual(a, pa) + _residual(b, pb)
            for k in range(-max_shift, max_shift + 1):
                if k == 0:
                    continue
                if k > 0:                      # give k vertices from a to b
                    if len(pa) - k < 2:
                        continue
                    na, nb = pa[:len(pa) - k], np.vstack([pa[len(pa) - k - 1:-1], pb])
                else:                          # take -k vertices from b to a
                    if len(pb) + k < 2:
                        continue
                    na, nb = np.vstack([pa, pb[1:1 - k]]), pb[-k:]
                fa = _refit_same_type(a, na, tol, min_arc_pts)
                fb = _refit_same_type(b, nb, tol, min_arc_pts)
                if fa is None or fb is None:
                    continue
                cost = _residual(fa, na) + _residual(fb, nb)
                if cost < base - 1e-9 and (best is None or cost < best[0]):
                    best = (cost, fa, fb)
            if best is not None:
                _, fa, fb = best
                for key in ('snapped',):
                    fa.pop(key, None)
                    fb.pop(key, None)
                prims[i], prims[j] = fa, fb
                moved = True
        if not moved:
            break
    return prims


def pts_of(prims):
    """Reconstruct the polyline vertex arrays each primitive was fitted to
    (kept on the primitive as '_pts')."""
    return [p['_pts'] for p in prims]


# Section points that come from one planar facet are collinear to float
# precision; anything tighter than this is "exactly straight".
EXACT_EPS = 1e-3
# A facet of a tessellated arc has a chord sagitta set by the exporter's
# chordal tolerance, so sagittas along one arc are alike. A collinear run
# whose sagitta against the fitted circle is this many times the others'
# is a straight wall, not a facet of the arc.
STRAIGHT_SAG_RATIO = 20.0
# A line within this fraction of tol is taken as is (wobble, not shape).
# A sloppier line within tol still wins over an arc unless the arc's
# deviation is smaller by ARC_OVER_LINE: a 3 mm stretch of an r=12 arc
# stays within 0.08 of a chord but is not what the mesh says.
LINE_CLEAN_FRAC = 0.25
ARC_OVER_LINE = 1.5


def _collinear_runs(pts, eps=EXACT_EPS):
    """Maximal runs of consecutive points collinear within eps, as
    (i0, i1) index pairs (inclusive, i1 > i0) covering the polyline."""
    n = len(pts)
    runs = []
    i = 0
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            seg = pts[i:j + 2]
            v = seg[-1] - seg[0]
            L = np.linalg.norm(v)
            if L < 1e-12:
                break
            nrm = np.array([-v[1], v[0]]) / L
            if np.abs((seg - seg[0]) @ nrm).max() > eps:
                break
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _arc_hides_straight_run(pts, fit, tol):
    """True when the circle `fit` over `pts` absorbs a real straight
    segment: an exactly collinear run, long relative to the span, whose
    sagitta against the circle dwarfs that of the other chords. One circle
    through a straight wall and a gentle arc can stay within tol; the
    wall must stay a line.

    A section crossing one planar facet yields up to three collinear
    points (it crosses the facet's diagonal), so a 3-point run may be a
    single coarse facet at the tangent end of the arc; it counts as a wall
    only when it is most of the span or its sagitta is plain (> tol/3).
    Four or more collinear points are two facets in a row: a wall."""
    runs = _collinear_runs(pts)
    if len(runs) < 2:
        return False
    L = np.array([np.linalg.norm(pts[j] - pts[i]) for i, j in runs])
    r = float(fit['r'])
    sag = r - np.sqrt(np.maximum(r * r - (L / 2) ** 2, 0.0))
    total = float(L.sum())
    order = np.argsort(sag)
    others = sag[order[:-1]]
    # the tessellation's own sagitta: the median of the other chords, or
    # the smallest when there are too few for a median to mean anything
    med_sag = (float(np.min(others)) if len(others) <= 2
               else float(np.median(others))) if len(others) else 0.0
    floor = max(4 * EXACT_EPS, STRAIGHT_SAG_RATIO * med_sag)
    for k, (i, j) in enumerate(runs):
        nseg = j - i
        if sag[k] <= floor:
            continue
        med_len = float(np.median(np.delete(L, k)))
        if L[k] < max(0.3 * total, 3 * med_len):
            continue
        if nseg >= 3:
            return True                       # two facets in a row: a wall
        if nseg == 2 and L[k] >= 0.5 * total:
            return True                       # one facet, but most of the span
        if sag[k] > tol / 3 and L[k] >= 0.4 * total:
            return True                       # plain sagitta, whatever it is
    return False


def _line_hides_bend(pts, line_dev, tol):
    """True when a line over `pts` is mostly an exact straight run with a
    curve peeling away at one end: the line would come out tilted by a
    fraction of a degree and the wall would move. A wobble (facets a few
    microns apart) is not a bend, and neither is a transition facet or a
    slight kink that leaves the wall within tol/3 (the line is anchored on
    the exact run, see `_mk_line`): the deviation must grow monotonically
    away from the run and exceed tol/3."""
    if line_dev <= 4 * EXACT_EPS:
        return False
    runs = _collinear_runs(pts)
    if len(runs) < 2:
        return False
    L = np.array([np.linalg.norm(pts[j] - pts[i]) for i, j in runs])
    k = int(np.argmax(L))
    if L[k] < 0.6 * L.sum():
        return False
    i, j = runs[k]
    if j - i < 2 and L[k] < 0.8 * L.sum():
        return False
    v = pts[j] - pts[i]
    nrm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    d = (pts - pts[i]) @ nrm
    lim = max(4 * EXACT_EPS, tol / 3)
    for tail in (d[j + 1:], d[:i][::-1]):
        if len(tail) == 0:
            continue
        a = np.abs(tail)
        same_side = (np.sign(tail) == np.sign(tail[-1])) | (a <= EXACT_EPS)
        if a.max() > lim and np.all(same_side) \
                and np.all(np.diff(a) >= -EXACT_EPS):
            return True
    return False


def _try_fit(pts, tol, min_arc_pts):
    """Best single primitive for a span, or None. Lines are preferred:
    an arc is only accepted when no line fits, its radius is sane, and it
    bends by more than the tolerance (otherwise a line would do). Neither
    may absorb an exactly straight run that belongs to a wall (see
    `_arc_hides_straight_run` / `_line_hides_bend`)."""
    if len(pts) < 2:
        return None
    line = fit_line(pts)
    line_ok = line['dev'] <= tol and not _line_hides_bend(pts, line['dev'], tol)
    if line_ok and line['dev'] <= max(4 * EXACT_EPS, tol * LINE_CLEAN_FRAC):
        return _mk_line(pts)
    if len(pts) >= min_arc_pts:
        circ = fit_circle_taubin(pts)
        if circ and circ['dev'] <= tol and _arc_sane(pts, circ) \
                and not _arc_hides_straight_run(pts, circ, tol):
            if not line_ok or circ['dev'] < line['dev'] / ARC_OVER_LINE:
                circ.update(_arc_params(pts, circ))
                circ['p0'], circ['p1'] = pts[0].copy(), pts[-1].copy()
                circ['_pts'] = pts
                return circ
    if line_ok:
        return _mk_line(pts)
    return None


def _segment_span(pts, tol, min_arc_pts, closed, depth=0):
    n = len(pts)
    if n < 2:
        return []
    fit = _try_fit(pts, tol, min_arc_pts)
    if fit is not None:
        return [fit]
    if depth > 40 or n <= 2:
        return [_mk_line(pts)]
    # split at the sharpest interior corner if there is one, else at the
    # point of max deviation from the chord (Douglas-Peucker)
    ta = turning_angles(pts, closed=False)
    ta[0] = ta[-1] = 0.0
    k = int(np.argmax(ta))
    if ta[k] <= CORNER_RAD or n <= 3:
        v = pts[-1] - pts[0]
        L = np.linalg.norm(v)
        if L < 1e-9:
            k = n // 2
        else:
            vn = v / L
            perp = np.abs((pts - pts[0]) @ np.array([-vn[1], vn[0]]))
            k = int(np.clip(np.argmax(perp), 1, n - 2))
    return (_segment_span(pts[:k + 1], tol, min_arc_pts, closed, depth + 1) +
            _segment_span(pts[k:], tol, min_arc_pts, closed, depth + 1))


def _mk_line(pts):
    """Line primitive over `pts`. The line is the chord p0-p1, except when
    most of the span is one exactly collinear run (a wall) with a short
    transition facet at an end: then the wall's own line is used and the
    endpoints are projected onto it, so the wall stays where the mesh has
    it instead of tilting towards the facet."""
    p0, p1 = pts[0].copy(), pts[-1].copy()
    if len(pts) >= 3:
        runs = _collinear_runs(pts)
        if len(runs) > 1:
            L = np.array([np.linalg.norm(pts[j] - pts[i]) for i, j in runs])
            k = int(np.argmax(L))
            i, j = runs[k]
            if L[k] >= 0.5 * L.sum() and j - i >= 2:
                f = fit_line(pts[i:j + 1])
                d, c = f['dir'], f['point']
                p0 = c + d * ((p0 - c) @ d)
                p1 = c + d * ((p1 - c) @ d)
    return {'type': 'line', 'p0': p0, 'p1': p1, '_pts': pts}


def _arc_sane(pts, fit):
    """Reject arc fits with absurd radius vs span, or that barely bend."""
    chord = np.linalg.norm(pts[-1] - pts[0])
    arc_span = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
    if fit['r'] > 50 * max(chord, arc_span):
        return False
    # every vertex should sit near the circle *and* the polyline should
    # actually curve: max distance of vertices from the chord > 0
    v = pts[-1] - pts[0]
    L = np.linalg.norm(v)
    if L > 1e-9:
        vn = v / L
        sag = np.abs((pts - pts[0]) @ np.array([-vn[1], vn[0]])).max()
        if sag < 1e-9:
            return False
    return True


def merge_pass(prims, pts_list, tol, min_arc_pts, closed=True, max_rounds=50):
    """Greedily merge adjacent primitives when one primitive fits their
    combined polyline within tol (cyclic for closed rings). Lines are
    preferred over arcs; the merge that lowers the count most first."""
    prims = list(prims)
    pts_list = list(pts_list)
    for _ in range(max_rounds):
        n = len(prims)
        if n < 2:
            break
        best = None
        pairs = range(n) if closed else range(n - 1)
        for i in pairs:
            j = (i + 1) % n
            if j == i:
                continue
            a, b = pts_list[i], pts_list[j]
            if not np.allclose(a[-1], b[0]):
                continue
            span = np.vstack([a, b[1:]])
            fit = _try_fit(span, tol, min_arc_pts)
            if fit is None:
                continue
            # score: prefer merges that produce a line, then larger spans
            score = (fit['type'] == 'line', len(span))
            if best is None or score > best[0]:
                best = (score, i, j, fit, span)
        if best is None and n >= 3:
            # pairwise merging can stall one step short: a fillet whose two
            # tangent-end facets each push a pair over tol while all three
            # together fit. Try triples before giving up.
            for i in (range(n) if closed else range(n - 2)):
                j, k = (i + 1) % n, (i + 2) % n
                if len({i, j, k}) < 3:
                    continue
                a, b, c = pts_list[i], pts_list[j], pts_list[k]
                if not (np.allclose(a[-1], b[0]) and np.allclose(b[-1], c[0])):
                    continue
                span = np.vstack([a, b[1:], c[1:]])
                fit = _try_fit(span, tol, min_arc_pts)
                if fit is None:
                    continue
                score = (fit['type'] == 'line', len(span))
                if best is None or score > best[0]:
                    best = (score, i, k, fit, span)
        if best is None:
            break
        _, i, j, fit, span = best
        fit['_pts'] = span
        if closed and j < i:            # wrap-around merge: (last, first)
            prims = [fit] + prims[j + 1:i]
            pts_list = [span] + pts_list[j + 1:i]
        else:
            prims = prims[:i] + [fit] + prims[j + 1:]
            pts_list = pts_list[:i] + [span] + pts_list[j + 1:]
    return prims


# ---------- full-circle detection ----------

def try_full_circle(pts, tol=0.08):
    """If a closed ring of points is one circle, return it."""
    pts = _dedupe(np.asarray(pts, float))
    if len(pts) < 8:
        return None
    ring = np.vstack([pts, pts[:1]])
    fit = fit_circle_taubin(ring)
    if fit and fit['dev'] <= tol:
        fit['full'] = True
        fit['_pts'] = pts
        return fit
    return None


# ---------- snapping ----------

def dominant_frame(prims, bin_deg=1.0):
    """Angle (rad, in [0, pi/2)) of the dominant orthogonal frame of the
    lines. Lines are weighted by length; among lines within the winning
    1-degree bin, the frame is refined from those whose points are exactly
    collinear (CAD walls) when any exist, so a line that swallowed one arc
    vertex cannot tilt the whole frame. None if there are no lines."""
    angs, ws, exact = [], [], []
    for p in prims:
        if p['type'] != 'line':
            continue
        v = np.asarray(p['p1'], float) - np.asarray(p['p0'], float)
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        angs.append(np.degrees(np.arctan2(v[1], v[0])) % 90.0)
        ws.append(L)
        pts = p.get('_pts')
        exact.append(pts is not None and len(pts) >= 2
                     and fit_line(np.asarray(pts, float))['dev'] < 1e-6 * max(1.0, L))
    if not angs:
        return None
    angs = np.array(angs)
    ws = np.array(ws)
    exact = np.array(exact, bool)
    nb = int(round(90 / bin_deg))
    hist = np.zeros(nb)
    for a, w in zip(angs, ws):
        hist[int(a // bin_deg) % nb] += w
    k = int(np.argmax(hist))
    centre = (k + 0.5) * bin_deg
    d = (angs - centre + 45) % 90 - 45
    sel = np.abs(d) < 1.5
    if (sel & exact).any():
        sel = sel & exact
    if sel.any():
        centre = centre + float((d[sel] * ws[sel]).sum() / ws[sel].sum())
    return np.radians(centre % 90.0)


def snap_lines_to_frame(prims, frame=None, angle_snap_deg=1.5):
    """Rotate lines within angle_snap_deg of the frame's axes onto them
    (about their midpoints). Endpoints are re-solved later by
    solve_junctions, so this may open tiny gaps for now."""
    if frame is None:
        frame = dominant_frame(prims)
    if frame is None:
        return prims
    targets = [frame + k * np.pi / 2 for k in range(4)]
    for p in prims:
        if p['type'] != 'line':
            continue
        v = p['p1'] - p['p0']
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        ang = np.arctan2(v[1], v[0])
        for t in targets:
            d = (ang - t + np.pi) % (2 * np.pi) - np.pi
            if abs(d) < np.radians(angle_snap_deg):
                mid = 0.5 * (p['p0'] + p['p1'])
                u = np.array([np.cos(t), np.sin(t)])
                p['p0'] = mid - u * L / 2
                p['p1'] = mid + u * L / 2
                p['snapped'] = True
                break
    return prims


def snap_profile(prims, angle_snap_deg=1.5, radius_merge_tol=0.05,
                 center_merge_tol=0.15):
    """Local snapping inside one ring: frame-align lines, unify near-equal
    radii and near-identical centres. Junctions are NOT re-closed here;
    call solve_junctions after any global snapping."""
    snap_lines_to_frame(prims, angle_snap_deg=angle_snap_deg)
    arcs = [p for p in prims if p['type'] == 'arc']
    for i, a in enumerate(arcs):
        for b in arcs[i + 1:]:
            if abs(a['r'] - b['r']) < radius_merge_tol:
                r = round((a['r'] + b['r']) / 2, 4)
                a['r'] = b['r'] = r
            if np.linalg.norm(a['center'] - b['center']) < center_merge_tol:
                c = (a['center'] + b['center']) / 2
                a['center'] = b['center'] = c
    return prims


# ---------- junction solving ----------

TANGENT_DEG = 3.0


def _line_dir(p):
    v = np.asarray(p['p1'], float) - np.asarray(p['p0'], float)
    L = np.linalg.norm(v)
    return v / L if L > 1e-12 else np.array([1.0, 0.0])


def _line_line(a, b):
    """Intersection of the infinite lines of a and b (None if parallel)."""
    p, r = np.asarray(a['p0'], float), _line_dir(a)
    q, s = np.asarray(b['p0'], float), _line_dir(b)
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    return p + t * r


def _line_circle_nearest(line, arc, near):
    """Point on the line closest to being on the circle, nearest `near`:
    the intersection if the line cuts the circle, else the projection of
    the circle's nearest point."""
    p, u = np.asarray(line['p0'], float), _line_dir(line)
    c, r = np.asarray(arc['center'], float), float(arc['r'])
    w = c - p
    t0 = w @ u
    foot = p + t0 * u
    h2 = r * r - float(np.sum((foot - c) ** 2))
    if h2 >= 0:
        h = np.sqrt(h2)
        cands = [foot + h * u, foot - h * u]
        return min(cands, key=lambda q: np.linalg.norm(q - near))
    return foot


def _arc_tangent_at(arc, pt):
    c = np.asarray(arc['center'], float)
    rad = np.asarray(pt, float) - c
    L = np.linalg.norm(rad)
    if L < 1e-12:
        return np.array([1.0, 0.0])
    rad /= L
    t = np.array([-rad[1], rad[0]])
    return t if arc.get('ccw', True) else -t


def _make_tangent(line, arc):
    """Move the arc's centre perpendicular to the line so the circle is
    exactly tangent to it (the line is the longer, more reliable fit)."""
    p, u = np.asarray(line['p0'], float), _line_dir(line)
    nrm = np.array([-u[1], u[0]])
    c = np.asarray(arc['center'], float)
    d = (c - p) @ nrm                    # signed distance centre->line
    target = np.sign(d) * arc['r'] if abs(d) > 1e-12 else arc['r']
    arc['center'] = c + (target - d) * nrm
    arc['tangent'] = True                # centre now set by a line
    return p + ((arc['center'] - p) @ u) * u    # tangent point


def _solve_fillet(prev, arc, nxt):
    """Arc tangent to two lines: centre = intersection of the two lines
    offset by r towards the arc; endpoints = tangent points. Returns True
    if solved (lines are left where they are — they are the reliable
    fits; only the arc moves, by less than the fit tolerance)."""
    r = float(arc['r'])
    c0 = np.asarray(arc['center'], float)
    p1, u1 = np.asarray(prev['p0'], float), _line_dir(prev)
    p2, u2 = np.asarray(nxt['p0'], float), _line_dir(nxt)
    n1 = np.array([-u1[1], u1[0]])
    n2 = np.array([-u2[1], u2[0]])
    s1 = np.sign((c0 - p1) @ n1) or 1.0
    s2 = np.sign((c0 - p2) @ n2) or 1.0
    # (c - p1).n1 = s1 r ; (c - p2).n2 = s2 r
    A = np.array([n1, n2])
    b = np.array([s1 * r + p1 @ n1, s2 * r + p2 @ n2])
    if abs(np.linalg.det(A)) < 1e-9:
        return False
    c = np.linalg.solve(A, b)
    if np.linalg.norm(c - c0) > max(0.5, 0.2 * r):
        return False                       # not the fillet the data shows
    t1 = c - s1 * r * n1
    t2 = c - s2 * r * n2
    arc['center'] = c
    arc['p0'] = t1
    arc['p1'] = t2
    prev['p1'] = t1.copy()
    nxt['p0'] = t2.copy()
    arc['fillet'] = True
    return True


def _is_tangent(line, arc, at):
    tan = _arc_tangent_at(arc, at)
    u = _line_dir(line)
    ang = np.degrees(np.arccos(np.clip(abs(tan @ u), -1, 1)))
    return ang < TANGENT_DEG


def solve_junctions(prims, closed=True, freeze_arcs=False):
    """Make consecutive primitives meet exactly, in place.

    With `freeze_arcs` no arc centre moves (no fillet/tangent solving):
    junctions are intersections, or pinned to the line when the circle
    misses it by a hair. Used after arcs have been unified across slabs.

    Fillets first: an arc whose two neighbours are lines it is tangent to
    is re-solved as a true fillet (centre from the offset lines, tangent
    points as endpoints; the lines do not move).
    Then pairwise —
    line/line: intersection (or the shared endpoint if parallel).
    line/arc: tangent if the fitted arc's tangent at the junction is
      within TANGENT_DEG of the line — the arc centre is nudged so the
      circle touches the line; else the line-circle intersection.
    arc/arc: nearest circle-circle intersection to the current joint, or
      the shared point projected onto both circles' mean.
    Finally arc endpoints are put exactly on their circles and the
    adjoining lines follow.
    """
    n = len(prims)
    if n == 0:
        return prims
    if n == 1:
        p = prims[0]
        if p['type'] == 'arc' and not p.get('full'):
            _project_arc_ends(p)
        return prims
    pairs = [(i, (i + 1) % n) for i in range(n if closed else n - 1)]
    solved = set()
    if n >= 3 and not freeze_arcs:
        for i, arc in enumerate(prims):
            if arc['type'] != 'arc':
                continue
            ip, inx = (i - 1) % n, (i + 1) % n
            if not closed and (i == 0 or i == n - 1):
                continue
            prev, nxt = prims[ip], prims[inx]
            if prev['type'] == 'line' and nxt['type'] == 'line' and prev is not nxt:
                if (_is_tangent(prev, arc, arc['p0']) and _is_tangent(nxt, arc, arc['p1'])
                        and _solve_fillet(prev, arc, nxt)):
                    solved.add((ip, i))
                    solved.add((i, inx))
    for i, j in pairs:
        if (i, j) in solved:
            continue
        a, b = prims[i], prims[j]
        joint = 0.5 * (np.asarray(a['p1'], float) + np.asarray(b['p0'], float))
        if a['type'] == 'line' and b['type'] == 'line':
            x = _line_line(a, b)
            if x is None or np.linalg.norm(x - joint) > 5.0:
                x = joint
        elif a['type'] == 'line' and b['type'] == 'arc':
            x = _joint_line_arc(a, b, joint, arc_first=False, freeze=freeze_arcs)
        elif a['type'] == 'arc' and b['type'] == 'line':
            x = _joint_line_arc(b, a, joint, arc_first=True, freeze=freeze_arcs)
        else:
            x = _joint_arc_arc(a, b, joint)
        a['p1'] = np.asarray(x, float).copy()
        b['p0'] = np.asarray(x, float).copy()
    for p in prims:
        if p['type'] == 'arc' and not p.get('fillet'):
            _project_arc_ends(p)
    # arcs own their endpoints (they were projected onto the circle);
    # lines follow
    for i, j in pairs:
        a, b = prims[i], prims[j]
        if a['type'] == 'arc' and b['type'] == 'line':
            if a.pop('_pin_p1', False):
                a['p1'] = np.asarray(b['p0'], float).copy()
            else:
                b['p0'] = np.asarray(a['p1'], float).copy()
        elif a['type'] == 'line' and b['type'] == 'arc':
            if b.pop('_pin_p0', False):
                b['p0'] = np.asarray(a['p1'], float).copy()
            else:
                a['p1'] = np.asarray(b['p0'], float).copy()
    return prims


# A circle that misses its neighbouring line by up to this is pinned to
# the line at the junction (the arc end sits that far off its circle); a
# bigger miss is a real gap and the old nearest-point rule applies.
PIN_GAP = 0.2


def _joint_line_arc(line, arc, joint, arc_first, freeze=False):
    tan = _arc_tangent_at(arc, joint)
    u = _line_dir(line)
    ang = np.degrees(np.arccos(np.clip(abs(tan @ u), -1, 1)))
    if ang < TANGENT_DEG and not freeze:
        return _make_tangent(line, arc)
    # the circle misses the line by a hair: there is no intersection, so
    # the nearest point is off the circle, and "lines follow arcs" below
    # would drag the line's end onto the circle and tilt the wall. Pin the
    # junction to the line instead (the line is the trusted fit) and keep
    # the arc end there: the drawn three-point arc passes within the gap
    # of the fitted circle, which is inside the fit tolerance.
    p = np.asarray(line['p0'], float)
    c, r = np.asarray(arc['center'], float), float(arc['r'])
    nrm = np.array([-u[1], u[0]])
    gap = abs((c - p) @ nrm) - r
    if 0 < gap <= PIN_GAP:
        x = p + ((np.asarray(joint, float) - p) @ u) * u
        arc['_pin_p1' if arc_first else '_pin_p0'] = True
        return x
    return _line_circle_nearest(line, arc, joint)


def _joint_arc_arc(a, b, joint):
    c0, r0 = np.asarray(a['center'], float), float(a['r'])
    c1, r1 = np.asarray(b['center'], float), float(b['r'])
    d = np.linalg.norm(c1 - c0)
    if d < 1e-9:
        return joint
    if abs(r0 - r1) - 1e-9 <= d <= r0 + r1 + 1e-9:
        x = (d * d + r0 * r0 - r1 * r1) / (2 * d)
        h2 = max(r0 * r0 - x * x, 0.0)
        h = np.sqrt(h2)
        e = (c1 - c0) / d
        nrm = np.array([-e[1], e[0]])
        base = c0 + x * e
        cands = [base + h * nrm, base - h * nrm]
        return min(cands, key=lambda q: np.linalg.norm(q - joint))
    # tangent-ish or disjoint circles: point on the line of centres
    if d > r0 + r1:
        return c0 + (c1 - c0) / d * r0
    return joint


def _project_arc_ends(p):
    c = np.asarray(p['center'], float)
    for key in ('p0', 'p1'):
        if p.get('_pin_' + key):
            continue
        v = np.asarray(p[key], float) - c
        L = np.linalg.norm(v)
        if L > 1e-12:
            p[key] = c + v / L * p['r']


# ---------- vertex-based refinement ----------

def refine_arcs_with_points(prims, pts2d, tol, band=None):
    """Re-fit each arc's circle to `pts2d` (e.g. mesh vertices projected
    into the section plane, which lie exactly on the CAD surface) that lie
    within `band` of the current circle and inside the arc's angular span.
    Section-polyline vertices sit on chords, so a fit through them under-
    estimates the radius by up to the facet sagitta; the mesh vertices do
    not."""
    if pts2d is None or len(pts2d) < 4:
        return prims
    band = 2 * tol if band is None else band
    pts2d = np.asarray(pts2d, float)
    for p in prims:
        if p['type'] != 'arc':
            continue
        c, r = np.asarray(p['center'], float), float(p['r'])
        d = np.hypot(pts2d[:, 0] - c[0], pts2d[:, 1] - c[1])
        sel = np.abs(d - r) < band
        if p.get('full'):
            cand = pts2d[sel]
        else:
            ang = np.arctan2(pts2d[:, 1] - c[1], pts2d[:, 0] - c[0])
            a0 = np.arctan2(p['p0'][1] - c[1], p['p0'][0] - c[0])
            sweep = p.get('sweep', abs(p.get('a1', 0) - p.get('a0', 0)))
            rel = (ang - a0) % (2 * np.pi)
            if not p.get('ccw', True):
                rel = (2 * np.pi - rel) % (2 * np.pi)
            margin = 0.15
            sel &= (rel <= sweep + margin) | (rel >= 2 * np.pi - margin)
            cand = pts2d[sel]
        if len(cand) < 4:
            continue
        # unique points only (vertices are shared by many triangles)
        cand = np.unique(np.round(cand, 6), axis=0)
        if len(cand) < 4:
            continue
        fit = fit_circle_taubin(cand, polyline=False)
        if fit is None:
            continue
        dv = np.abs(np.hypot(cand[:, 0] - fit['center'][0],
                             cand[:, 1] - fit['center'][1]) - fit['r'])
        # the refined circle must still explain the section polyline: a
        # neighbouring feature's vertices inside the band would otherwise
        # drag the fit away from the arc it is meant to sharpen
        own = p.get('_pts')
        still_ok = True
        if own is not None and len(own) >= 2:
            still_ok = polyline_circle_dev(np.asarray(own, float),
                                           fit['center'], fit['r']) <= tol * 1.5
        # how far the refined circle may differ from the chord fit: a short
        # arc (20 degrees of r=12) pins its radius only to a few tenths, so
        # allow a share of the radius; `still_ok` keeps a wrong circle out
        slack = max(band, 0.1 * r)
        if still_ok and dv.max() <= max(tol, 0.5 * band) \
                and abs(fit['r'] - r) < slack \
                and np.linalg.norm(fit['center'] - c) < slack:
            p['center'] = fit['center']
            p['r'] = float(fit['r'])
            p['refined'] = True
    return prims
