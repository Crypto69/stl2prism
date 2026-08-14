"""Fit clean line/arc primitives to noisy section polylines.

The section of a faceted mesh gives dense polylines where true arcs
appear as many short chords. We recover intent with:
  * Taubin algebraic circle fitting (robust, bias-corrected)
  * greedy split: try one primitive over a span; if max deviation
    exceeds tol, split at the worst point and recurse
  * constraint snapping (GlobFit-lite): axis-align near-horizontal /
    vertical lines, unify near-equal radii and near-identical centers
"""
import numpy as np


# ---------- fitting primitives ----------

def fit_line(pts):
    c = pts.mean(axis=0)
    d = pts - c
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    direction = vt[0]
    dev = np.abs(d @ np.array([-direction[1], direction[0]]))
    return {'type': 'line', 'p0': pts[0], 'p1': pts[-1],
            'dir': direction, 'dev': float(dev.max())}


def fit_circle_taubin(pts):
    """Taubin algebraic circle fit. Returns center, radius, max deviation."""
    x, y = pts[:, 0], pts[:, 1]
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    z = u * u + v * v
    zm = z.mean()
    Z = (z - zm) / (2.0 * np.sqrt(zm)) if zm > 0 else z
    A = np.column_stack([Z, u, v])
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    a = vt[-1]
    a0 = a[0] / (2.0 * np.sqrt(zm)) if zm > 0 else a[0]
    if abs(a0) < 1e-12:
        return None
    cx = -a[1] / (2 * a0) + xm
    cy = -a[2] / (2 * a0) + ym
    r = np.sqrt(a[1] ** 2 + a[2] ** 2) / (4 * a0 ** 2) + zm - \
        (a[1] ** 2 + a[2] ** 2) / (4 * a0 ** 2) * 0  # radius via distances:
    d = np.hypot(x - cx, y - cy)
    r = d.mean()
    return {'type': 'arc', 'center': np.array([cx, cy]), 'r': float(r),
            'dev': float(np.abs(d - r).max())}


def _arc_params(pts, fit):
    """Start/end angles + sweep direction for an arc through pts."""
    c = fit['center']
    ang = np.unwrap(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return {'a0': float(ang[0]), 'a1': float(ang[-1]),
            'ccw': bool(ang[-1] > ang[0])}


# ---------- polyline segmentation ----------

def segment_polyline(pts, tol=0.08, closed=True, min_arc_pts=5):
    """Greedy split of a polyline into line/arc primitives within tol."""
    if closed and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    prims = _segment_span(pts, tol, min_arc_pts)
    return prims


def _segment_span(pts, tol, min_arc_pts, depth=0):
    n = len(pts)
    if n < 2:
        return []
    line = fit_line(pts)
    if line['dev'] <= tol:
        return [_finalize_line(pts)]
    if n >= min_arc_pts:
        circ = fit_circle_taubin(pts)
        if circ and circ['dev'] <= tol and _arc_sane(pts, circ):
            circ.update(_arc_params(pts, circ))
            circ['p0'], circ['p1'] = pts[0], pts[-1]
            return [circ]
    if depth > 24 or n <= 3:
        return [_finalize_line(pts)]
    # split at point of max deviation from chord
    chord = fit_line(np.vstack([pts[0], pts[-1]]))
    d = pts - pts[0]
    v = pts[-1] - pts[0]
    L = np.linalg.norm(v)
    if L < 1e-9:
        k = n // 2
    else:
        vn = v / L
        perp = np.abs(d @ np.array([-vn[1], vn[0]]))
        k = int(np.clip(np.argmax(perp), 1, n - 2))
    return (_segment_span(pts[:k + 1], tol, min_arc_pts, depth + 1) +
            _segment_span(pts[k:], tol, min_arc_pts, depth + 1))


def _finalize_line(pts):
    return {'type': 'line', 'p0': pts[0], 'p1': pts[-1]}


def _arc_sane(pts, fit):
    """Reject arc fits with absurd radius vs chord (i.e. straight lines)."""
    chord = np.linalg.norm(pts[-1] - pts[0])
    arc_span = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
    if fit['r'] > 50 * max(chord, arc_span):
        return False
    return True


# ---------- full-circle detection ----------

def try_full_circle(pts, tol=0.08):
    """If a closed ring of points is one circle, return it."""
    if len(pts) < 8:
        return None
    fit = fit_circle_taubin(pts)
    if fit and fit['dev'] <= tol:
        return fit
    return None


# ---------- constraint snapping (GlobFit-lite) ----------

def snap_profile(prims, angle_snap_deg=1.5, radius_merge_tol=0.05,
                 center_merge_tol=0.15):
    """Snap near-axis-aligned lines and merge near-equal arc params."""
    # snap line directions to 0/90 degrees
    for p in prims:
        if p['type'] != 'line':
            continue
        v = p['p1'] - p['p0']
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        ang = np.degrees(np.arctan2(v[1], v[0])) % 180
        for target in (0.0, 90.0, 180.0):
            if abs(ang - target) < angle_snap_deg:
                mid = (p['p0'] + p['p1']) / 2
                if target in (0.0, 180.0):
                    p['p0'] = np.array([p['p0'][0], mid[1]])
                    p['p1'] = np.array([p['p1'][0], mid[1]])
                else:
                    p['p0'] = np.array([mid[0], p['p0'][1]])
                    p['p1'] = np.array([mid[0], p['p1'][1]])
                break
    # merge near-equal radii across arcs
    arcs = [p for p in prims if p['type'] == 'arc']
    for i, a in enumerate(arcs):
        for b in arcs[i + 1:]:
            if abs(a['r'] - b['r']) < radius_merge_tol:
                r = round((a['r'] + b['r']) / 2, 3)
                a['r'] = b['r'] = r
            if np.linalg.norm(a['center'] - b['center']) < center_merge_tol:
                c = (a['center'] + b['center']) / 2
                a['center'] = b['center'] = c
    # re-close chain: force each primitive to start where previous ended
    for prev, cur in zip(prims, prims[1:] + prims[:1]):
        cur['p0'] = prev['p1'].copy() if isinstance(prev['p1'], np.ndarray) \
            else np.array(prev['p1'])
    return prims
