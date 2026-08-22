"""Face-group engine (C5): one analytic B-rep face per surface region.

Fusion 360's "Prismatic" mesh conversion, done in the open: segment the mesh
into face groups, fit a plane / cylinder / cone / sphere to each group's
VERTICES, build one trimmed face per group on the fitted surface, sew the
faces into a solid, heal and check it. Regions that no primitive explains
stay as their triangles, so a body never fails as a whole because of one
blend.

Pipeline position (see pipeline._convert_body): strict prismatic first (it is
the only route that yields sketch+extrude structure for the scripts), then
this engine, then the hybrid patch, then faceted.

Segmentation is a greedy, fit-driven region growing over the mesh's coplanar
components ("seeds", rebuild.planar_groups). Two residual levels keep it
honest on CAD exports, where every vertex sits exactly on the true surface:
vertices must lie within `merge_tol` (a few microns; noise-adaptive) and the
facet interiors (centroids and edge midpoints) within `fit_tol`. The second
test is what stops a wide plane and its first fillet strip from being fitted
by an exact, absurdly large cylinder — any two facets fit a cylinder.

v1 builds face boundaries from the mesh boundary polylines projected onto the
fitted surfaces (chord edges); shared edges from surface–surface intersections
are the planned upgrade (face-group engine v2).
"""
from dataclasses import dataclass
import time
import numpy as np

from .rebuild import planar_groups, _group_loops

# --- tunables -----------------------------------------------------------------
SHARP_DEG = 25.0            # never grow a region across a sharper dihedral
MERGE_TOL_MIN = 2e-3        # vertex residual floor (float32 STL noise ~1e-3)
MAX_SEEDS = 30000           # above this the engine steps aside (scan-like meshes)
MAX_FACES = 300000
MAX_FITS = 20000            # least-squares fit budget per body (~20 s worst case)
SEW_TOL = 1e-3              # sewing tolerance — deliberately NOT the fit tolerance
ORDER = ('cylinder', 'sphere', 'cone')                 # simplest first (LMM)
MINSUP = {'cylinder': 3, 'sphere': 8, 'cone': 6}       # seeds AND distinct normals
MIN_SPREAD_DEG = {'cylinder': 6.0, 'cone': 6.0, 'sphere': 20.0}
NORMAL_AGREE_DEG = 8.0      # facet normal vs fitted-surface normal at the centroid
FIT_SAMPLE = 2000           # fit on at most this many vertices (deterministic stride)


class FaceGroupError(RuntimeError):
    """The face-group engine could not produce a closed, valid solid."""


@dataclass
class Region:
    id: int
    kind: str                       # 'plane' | 'cylinder' | 'cone' | 'sphere'
    faces: np.ndarray               # triangle indices
    seeds: list
    params: dict                    # see fit_* below
    resid: float = 0.0              # max |d| over vertices + facet interiors
    resid_v: float = 0.0            # vertices only
    area: float = 0.0
    concave: bool = False           # mesh normals point against the surface normal
    built: str = ''                 # 'analytic' | reason it fell back to triangles
    n_faces_out: int = 0


# --- signed distance / surface normal per kind ---------------------------------

def _d_plane(p, pr):
    return (p - pr['point']) @ pr['normal']


def _d_cyl(p, pr):
    rel = p - pr['point']
    h = rel @ pr['axis']
    return np.linalg.norm(rel - np.outer(h, pr['axis']), axis=1) - pr['r']


def _d_sph(p, pr):
    return np.linalg.norm(p - pr['center'], axis=1) - pr['r']


def _d_cone(p, pr):
    rel = p - pr['apex']
    h = rel @ pr['axis']
    rr = np.linalg.norm(rel - np.outer(h, pr['axis']), axis=1)
    return rr * np.cos(pr['half_angle']) - h * np.sin(pr['half_angle'])


def _n_plane(p, pr):
    return np.tile(pr['normal'], (len(p), 1))


def _n_cyl(p, pr):
    rel = p - pr['point']
    h = rel @ pr['axis']
    r = rel - np.outer(h, pr['axis'])
    return r / np.maximum(np.linalg.norm(r, axis=1)[:, None], 1e-12)


def _n_sph(p, pr):
    r = p - pr['center']
    return r / np.maximum(np.linalg.norm(r, axis=1)[:, None], 1e-12)


def _n_cone(p, pr):
    rel = p - pr['apex']
    h = rel @ pr['axis']
    r = rel - np.outer(h, pr['axis'])
    rn = r / np.maximum(np.linalg.norm(r, axis=1)[:, None], 1e-12)
    a = pr['half_angle']
    return rn * np.cos(a) - pr['axis'] * np.sin(a)


DIST = {'plane': _d_plane, 'cylinder': _d_cyl, 'sphere': _d_sph, 'cone': _d_cone}
NORM = {'plane': _n_plane, 'cylinder': _n_cyl, 'sphere': _n_sph, 'cone': _n_cone}


def _project(p, kind, pr):
    """Closest point on the surface for one 3-D point (closed forms)."""
    if kind == 'plane':
        return p - ((p - pr['point']) @ pr['normal']) * pr['normal']
    if kind == 'sphere':
        d = p - pr['center']
        n = np.linalg.norm(d)
        return p if n < 1e-12 else pr['center'] + d / n * pr['r']
    if kind == 'cylinder':
        rel = p - pr['point']
        h = rel @ pr['axis']
        rad = rel - h * pr['axis']
        n = np.linalg.norm(rad)
        return p if n < 1e-12 else pr['point'] + h * pr['axis'] + rad / n * pr['r']
    if kind == 'cone':
        rel = p - pr['apex']
        h = rel @ pr['axis']
        rad = rel - h * pr['axis']
        n = np.linalg.norm(rad)
        if n < 1e-12:
            return p
        a = pr['half_angle']
        g = np.cos(a) * pr['axis'] + np.sin(a) * (rad / n)   # generatrix direction
        return pr['apex'] + max(float(rel @ g), 0.0) * g
    return p


# --- primitive fits (on vertices) ---------------------------------------------

def fit_plane(P, weights=None):
    """PCA plane through points: {'normal', 'point'}."""
    P = np.asarray(P, float)
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return {'normal': vt[2], 'point': c}


def fit_sphere(P):
    """Least-squares sphere: algebraic (linear) solve, then a geometric
    Gauss-Newton refine. Returns {'center', 'r', 'resid'} or None."""
    P = np.asarray(P, float)
    if len(P) < 4:
        return None
    A = np.column_stack([2 * P, np.ones(len(P))])
    b = (P ** 2).sum(axis=1)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    c = x[:3]
    k = x[3] + c @ c
    if not np.isfinite(k) or k <= 0:
        return None
    r = float(np.sqrt(k))

    def resid(y):
        return np.linalg.norm(P - y[:3], axis=1) - y[3]
    try:
        from scipy.optimize import least_squares
        sol = least_squares(resid, np.r_[c, r], method='lm', max_nfev=100)
        c, r = sol.x[:3], float(sol.x[3])
    except Exception:
        pass
    if not np.isfinite(r) or r <= 0 or not np.all(np.isfinite(c)):
        return None
    return {'center': c, 'r': r, 'resid': float(np.abs(resid(np.r_[c, r])).max())}


def _fit(kind, P, N, W, pre_tol=None):
    """Fit `kind` to vertices P; N/W are the facet normals/areas of the region
    (used to seed the axis, as features._split_region does).

    `pre_tol`: skip the (expensive) least-squares refinement when a closed-form
    fit with the seeded axis is already worse than this — most candidates in
    region growing are hopeless, and the LM cone fit costs ~20 ms each."""
    from .features import fit_cylinder, fit_cone, _plane_basis
    from .profile_fit import fit_circle_taubin
    if kind == 'plane':
        return fit_plane(P)
    if kind == 'sphere':
        if pre_tol is not None and len(P) >= 5:
            A = np.column_stack([2 * P, np.ones(len(P))])
            b = (P ** 2).sum(axis=1)
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            k = x[3] + x[:3] @ x[:3]
            if not np.isfinite(k) or k <= 0:
                return None
            if np.abs(np.linalg.norm(P - x[:3], axis=1) - np.sqrt(k)).max() > pre_tol:
                return None
        return fit_sphere(P)
    if len(P) < 6:
        return None
    if kind == 'cylinder':
        C0 = (N * W[:, None]).T @ N
        _, v0 = np.linalg.eigh(C0)
        ax = v0[:, 0]
        if pre_tol is not None:
            b0, b1 = _plane_basis(ax)
            circ = fit_circle_taubin(np.column_stack([P @ b0, P @ b1]), polyline=False)
            if circ is None or circ['dev'] > pre_tol:
                return None
        return fit_cylinder(P, ax)
    if kind == 'cone':
        mean_n = (N * W[:, None]).sum(axis=0) / W.sum()
        cn = N - mean_n
        _, v = np.linalg.eigh((cn * W[:, None]).T @ cn)
        ax = v[:, 0]
        s = float((N @ ax * W).sum() / W.sum())
        half = np.arcsin(np.clip(abs(s), 0.0, 1.0))
        if np.degrees(half) < 2.0:
            return None
        if s < 0:
            ax = -ax
        n_distinct = len(np.unique(np.round(N, 3), axis=0))
        if n_distinct < 3:
            # two facets leave the axis undetermined: the seed is arbitrary,
            # so keep the search short — a real cone converges in < 20 evals,
            # garbage would grind to the cap
            return fit_cone(P, ax, float(half), max_nfev=40)
        if pre_tol is not None:
            # fixed seeded axis: r = (h - h_apex) tan(alpha) is linear in h
            c0 = P.mean(axis=0)
            rel = P - c0
            h = rel @ ax
            rad = np.linalg.norm(rel - np.outer(h, ax), axis=1)
            (a, b), *_ = np.linalg.lstsq(np.column_stack([h, np.ones(len(h))]), rad, rcond=None)
            if not (a > 1e-6):
                return None
            al = np.arctan(a)
            if np.abs((rad - (a * h + b)) * np.cos(al)).max() > pre_tol:
                return None
        return fit_cone(P, ax, float(half))
    return None


def _subsample(P, n=FIT_SAMPLE):
    if len(P) <= n:
        return P
    idx = np.linspace(0, len(P) - 1, n).astype(np.int64)
    return P[idx]


# --- segmentation --------------------------------------------------------------

def segment(mesh, fit_tol=0.08, merge_tol=None, sharp_deg=SHARP_DEG,
            verbose=False, max_fits=MAX_FITS):
    """Segment a mesh into fitted regions. Returns list[Region], or None when
    the mesh is outside the engine's envelope (too many seeds/faces)."""
    t0 = time.time()
    seeds = planar_groups(mesh)
    ns = len(seeds)
    nf = len(mesh.faces)
    if ns > MAX_SEEDS or nf > MAX_FACES:
        if verbose:
            print(f"[fgroup] {ns} seeds / {nf} faces exceed the engine envelope; skipped")
        return None
    V = mesh.vertices
    F = mesh.faces
    C = mesh.triangles_center
    FN = mesh.face_normals
    FA = mesh.area_faces
    T = mesh.triangles
    MID = np.concatenate([(T[:, 0] + T[:, 1]) / 2, (T[:, 1] + T[:, 2]) / 2,
                          (T[:, 2] + T[:, 0]) / 2])
    diag = float(np.linalg.norm(mesh.extents))

    sid = np.empty(nf, np.int64)
    for i, s in enumerate(seeds):
        sid[s] = i
    adj = mesh.face_adjacency
    ang = np.degrees(np.abs(mesh.face_adjacency_angles))
    a0, a1 = sid[adj[:, 0]], sid[adj[:, 1]]
    ok = (a0 != a1) & (ang < sharp_deg)
    nbrs = [set() for _ in range(ns)]
    for x, y in zip(a0[ok].tolist(), a1[ok].tolist()):
        nbrs[x].add(y)
        nbrs[y].add(x)

    seed_area = np.array([FA[s].sum() for s in seeds])
    seed_normal = np.empty((ns, 3))
    for i, s in enumerate(seeds):
        n = (FN[s] * FA[s][:, None]).sum(axis=0)
        nn = np.linalg.norm(n)
        seed_normal[i] = n / nn if nn > 0 else FN[s[0]]

    # noise-adaptive vertex tolerance: how far off their own plane the
    # vertices of coplanar seeds sit (float32 exports: ~1e-3)
    if merge_tol is None:
        res = []
        for s in seeds:
            if len(s) >= 3:
                P = V[np.unique(F[s])]
                if len(P) >= 4:
                    pl = fit_plane(P)
                    res.append(np.abs(_d_plane(P, pl)).max())
        noise = float(np.percentile(res, 99)) if res else 0.0
        merge_tol = float(np.clip(3 * noise, MERGE_TOL_MIN, min(1e-2, fit_tol / 4)))
    else:
        noise = float('nan')
    merge_relaxed = float(min(4 * merge_tol, fit_tol / 2))

    nfits = [0]

    def data(members):
        faces = np.concatenate([seeds[s] for s in members])
        P = V[np.unique(F[faces])]
        return faces, P

    def evaluate(kind, pr, faces, P, final):
        """max residual if the region is explained by (kind, pr), else None."""
        if pr is None:
            return None
        d = DIST[kind]
        rv = float(np.abs(d(P, pr)).max())
        if rv > cur_tol[0]:
            return None
        inner = np.concatenate([C[faces], MID[faces], MID[faces + nf], MID[faces + 2 * nf]])
        rc = float(np.abs(d(inner, pr)).max())
        if rc > fit_tol:
            return None
        if kind != 'plane':
            nf_ = NORM[kind]
            dots = (nf_(C[faces], pr) * FN[faces]).sum(axis=1)
            # A facet's normal only equals the surface normal at one point
            # of the facet; on a coarse tessellation (long triangles at a
            # sphere pole) it can differ from the centroid normal by more
            # than the fixed allowance. Allow the spread of the surface
            # normal across the facet's own vertices on top of it.
            n0 = nf_(V[F[faces, 0]], pr)
            n1 = nf_(V[F[faces, 1]], pr)
            n2 = nf_(V[F[faces, 2]], pr)
            span = np.arccos(np.clip(np.minimum.reduce([
                (n0 * n1).sum(axis=1), (n1 * n2).sum(axis=1), (n0 * n2).sum(axis=1)]), -1, 1))
            allow = np.cos(np.minimum(np.radians(NORMAL_AGREE_DEG) + span, np.radians(60)))
            if np.any(np.abs(dots) < allow):
                return None
            if not (np.all(dots > 0) or np.all(dots < 0)):
                return None
            r = pr.get('r')
            if final and r is not None and (r > 2 * diag or r < 1e-3):
                return None
            if final and kind == 'cone':
                h = (P - pr['apex']) @ pr['axis']
                if h.min() < max(5 * merge_tol, 0.05 * (h.max() - h.min())):
                    return None            # loop through the apex: ShapeFix dies there
        return max(rv, rc)

    def support_ok(kind, members):
        if kind == 'plane':
            return True
        N = seed_normal[members]
        nn = len(np.unique(np.round(N, 3), axis=0))
        if nn < MINSUP[kind] or len(members) < MINSUP[kind]:
            return False
        spread = np.degrees(np.arccos(np.clip((N @ N.T).min(), -1.0, 1.0)))
        return spread >= MIN_SPREAD_DEG[kind]

    def try_fit(kind, P, N, W):
        nfits[0] += 1
        return _fit(kind, _subsample(P), N, W, pre_tol=max(10 * cur_tol[0], fit_tol / 2))

    assigned = -np.ones(ns, np.int64)      # seed -> region index, -1 = free
    regions = []
    cur_tol = [merge_tol]

    def grow(candidates):
        """Greedy region growing from each free candidate seed, largest first."""
        for s0 in candidates:
            if assigned[s0] >= 0:
                continue
            if nfits[0] > max_fits:
                break
            rid = len(regions)
            members = [s0]
            assigned[s0] = rid
            kind = 'plane'
            pr = None
            frontier = [n for n in nbrs[s0] if assigned[n] < 0]
            tried = set()
            while True:
                while frontier:
                    cand = frontier.pop(0)
                    if assigned[cand] >= 0 or cand in tried:
                        continue
                    faces, P = data(members + [cand])
                    N, W = FN[faces], FA[faces]
                    accepted = None
                    for k in ORDER:
                        if kind != 'plane' and ORDER.index(k) < ORDER.index(kind):
                            continue          # a region's type may only upgrade
                        if k == kind and len(members) >= 6:
                            r = evaluate(k, pr, faces, P, True)
                            if r is not None:
                                accepted = (k, pr, r)
                                break
                            if len(members) >= 30:
                                continue      # well determined: no refit of this kind
                        prk = try_fit(k, P, N, W)
                        r = evaluate(k, prk, faces, P, len(members) + 1 >= MINSUP[k])
                        if r is not None:
                            accepted = (k, prk, r)
                            break
                    if accepted is None:
                        tried.add(cand)
                        continue
                    k2, pr, _ = accepted
                    if k2 != kind:
                        tried = set()         # rejected under the old type: retry
                        kind = k2
                    members.append(cand)
                    assigned[cand] = rid
                    for n in nbrs[cand]:
                        if assigned[n] < 0 and n not in tried:
                            frontier.append(n)
                if kind == 'plane':
                    break
                # refit on everything, then sweep the rejected neighbours once
                # more against the final parameters (no fitting)
                faces, P = data(members)
                prf = try_fit(kind, P, FN[faces], FA[faces])
                if evaluate(kind, prf, faces, P, True) is not None:
                    pr = prf
                retry = [c for c in tried if assigned[c] < 0]
                tried = set()
                added = False
                for c in retry:
                    faces2, P2 = data(members + [c])
                    if evaluate(kind, pr, faces2, P2, True) is not None:
                        members.append(c)
                        assigned[c] = rid
                        added = True
                        frontier.extend(n for n in nbrs[c] if assigned[n] < 0)
                    else:
                        tried.add(c)
                if not added and not frontier:
                    break
            if kind == 'plane':
                assigned[s0] = -1             # stays a free single seed for now
                continue
            faces, P = data(members)
            r = evaluate(kind, pr, faces, P, True)
            if r is None or not support_ok(kind, members):
                for s in members:
                    assigned[s] = -1          # dissolve; the seeds stay free
                continue
            regions.append(Region(rid, kind, faces, list(members), pr, resid=r))
        # keep region ids == list index even after dissolves
        for i, rg in enumerate(regions):
            if rg.id != i:
                for s in rg.seeds:
                    assigned[s] = i
                rg.id = i

    order = list(np.argsort(-seed_area))
    grow(order)                                # strict pass
    cur_tol[0] = merge_relaxed
    grow([s for s in order if assigned[s] < 0])   # relaxed pass on leftovers only
    cur_tol[0] = merge_relaxed

    # leftover single seeds next to a curved region: absorb if the region's
    # own surface explains them (no refit)
    for _ in range(5):
        changed = False
        for s in range(ns):
            if assigned[s] >= 0:
                continue
            for n in nbrs[s]:
                j = assigned[n]
                if j < 0:
                    continue
                rg = regions[j]
                faces2, P2 = data(rg.seeds + [s])
                if evaluate(rg.kind, rg.params, faces2, P2, True) is not None:
                    rg.seeds.append(s)
                    rg.faces = faces2
                    assigned[s] = j
                    changed = True
                    break
        if not changed:
            break

    # everything still free is a plane region of its own
    for s in range(ns):
        if assigned[s] >= 0:
            continue
        faces = seeds[s]
        P = V[np.unique(F[faces])]
        pr = fit_plane(P) if len(faces) >= 3 and len(P) >= 4 else \
            {'normal': seed_normal[s], 'point': P.mean(axis=0)}
        if pr['normal'] @ seed_normal[s] < 0:
            pr['normal'] = -pr['normal']
        rid = len(regions)
        assigned[s] = rid
        regions.append(Region(rid, 'plane', faces, [s], pr))

    # final numbers per region
    for rg in regions:
        faces = rg.faces
        P = V[np.unique(F[faces])]
        d = DIST[rg.kind]
        rg.resid_v = float(np.abs(d(P, rg.params)).max())
        inner = np.concatenate([C[faces], MID[faces], MID[faces + nf], MID[faces + 2 * nf]])
        rg.resid = max(rg.resid_v, float(np.abs(d(inner, rg.params)).max()))
        rg.area = float(FA[faces].sum())
        if rg.kind != 'plane':
            dots = (NORM[rg.kind](C[faces], rg.params) * FN[faces]).sum(axis=1)
            rg.concave = bool(dots.mean() < 0)
    if verbose:
        from collections import Counter
        cnt = Counter(r.kind for r in regions)
        print(f"[fgroup] {ns} seeds -> {len(regions)} regions "
              f"({', '.join(f'{k} {v}' for k, v in sorted(cnt.items()))}); "
              f"merge tol {merge_tol:.4f}mm, {nfits[0]} fits, {time.time() - t0:.1f}s")
    segment.last_stats = {'seeds': ns, 'merge_tol': merge_tol, 'noise': noise,
                          'fits': nfits[0], 'fit_budget_hit': nfits[0] > max_fits,
                          't_segment': time.time() - t0}
    return regions


# --- regularise (GlobFit-lite) -------------------------------------------------

def regularise(regions, mesh, fit_tol=0.08, merge_tol=None, verbose=False):
    """Snap near-parallel/orthogonal directions, coaxial axes, coplanar
    planes and equal radii — within measurement uncertainty only. Every
    touched region is re-checked and reverted if the snap moved its surface
    off the vertices. Returns counts."""
    from .profile_fit import fit_circle_taubin
    from .features import _plane_basis
    V, F = mesh.vertices, mesh.faces
    C = mesh.triangles_center
    T = mesh.triangles
    nf = len(F)
    MID = np.concatenate([(T[:, 0] + T[:, 1]) / 2, (T[:, 1] + T[:, 2]) / 2,
                          (T[:, 2] + T[:, 0]) / 2])
    if merge_tol is None:
        merge_tol = max(MERGE_TOL_MIN, float(np.median([r.resid_v for r in regions]) if regions else MERGE_TOL_MIN))
    stats = {'dirs': 0, 'axes': 0, 'planes': 0, 'radii': 0, 'reverted': 0}
    if not regions:
        return stats
    diag = float(np.linalg.norm(mesh.extents))

    def region_pts(rg):
        return V[np.unique(F[rg.faces])]

    def extent(rg):
        P = region_pts(rg)
        return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0

    # 1. directions -----------------------------------------------------------
    items = []      # (weight, direction, region, key)
    for rg in regions:
        if rg.kind == 'plane':
            items.append((rg.area, rg.params['normal'].copy(), rg, 'normal'))
        elif rg.kind in ('cylinder', 'cone'):
            items.append((rg.area, rg.params['axis'].copy(), rg, 'axis'))
    items.sort(key=lambda t: -t[0])
    reps = []       # [weight, dir]
    memb = []       # list of (item index, sign)
    ang_tol = np.radians(0.5)
    for i, (w, d, rg, key) in enumerate(items):
        tol_i = max(ang_tol, np.arctan(2 * merge_tol / extent(rg)))
        for j, (rw, rd) in enumerate(reps):
            c = float(d @ rd)
            if abs(c) > np.cos(tol_i):
                sgn = 1.0 if c > 0 else -1.0
                nd = rd * rw + sgn * d * w
                reps[j] = [rw + w, nd / np.linalg.norm(nd)]
                memb[j].append((i, sgn))
                break
        else:
            reps.append([w, d / np.linalg.norm(d)])
            memb.append([(i, 1.0)])
    # orthogonalise lighter reps against heavier ones, snap to XYZ
    for j in range(len(reps)):
        d = reps[j][1]
        for k in range(j):
            e = reps[k][1]
            c = float(d @ e)
            if 0 < abs(c) < np.sin(ang_tol):
                d = d - c * e
                d /= np.linalg.norm(d)
        for ax in np.eye(3):
            c = float(d @ ax)
            if abs(c) > np.cos(ang_tol) and abs(c) < 1.0:
                d = ax * (1.0 if c > 0 else -1.0)
        reps[j][1] = d
    snapped = {}
    dir_cluster = {}                                  # region id -> cluster j
    for j, mem in enumerate(memb):
        for i, sgn in mem:
            w, d, rg, key = items[i]
            nd = reps[j][1] * sgn
            if np.linalg.norm(nd - d) > 1e-12:
                snapped[rg.id] = dict(rg.params)      # remember for revert
                dir_cluster[rg.id] = j
                _set_direction(rg, key, nd, region_pts(rg), fit_circle_taubin, _plane_basis)
                stats['dirs'] += 1

    # 2. placement: coaxial cylinders/cones, coplanar planes -----------------------
    pos_tol = fit_tol / 2
    axes = [rg for rg in regions if rg.kind in ('cylinder', 'cone')]
    used = set()
    for a in axes:
        if a.id in used:
            continue
        group = [a]
        da = a.params['axis']
        pa = a.params['point'] if a.kind == 'cylinder' else a.params['apex']
        for b in axes:
            if b.id == a.id or b.id in used:
                continue
            db = b.params['axis']
            if abs(float(da @ db)) < np.cos(np.radians(0.5)):
                continue
            pb = b.params['point'] if b.kind == 'cylinder' else b.params['apex']
            rel = pb - pa
            if np.linalg.norm(rel - (rel @ da) * da) <= pos_tol:
                group.append(b)
        if len(group) > 1:
            wsum = sum(g.area for g in group)
            # weighted mean of the axis lines' foot points on a common line
            foot = np.zeros(3)
            for g in group:
                p = g.params['point'] if g.kind == 'cylinder' else g.params['apex']
                foot += g.area * (p - ((p - pa) @ da) * da)
            foot /= wsum
            for g in group:
                used.add(g.id)
                snapped.setdefault(g.id, dict(g.params))
                p = g.params['point'] if g.kind == 'cylinder' else g.params['apex']
                shift = foot - (p - ((p - pa) @ da) * da)
                if np.linalg.norm(shift) > 1e-12:
                    if g.kind == 'cylinder':
                        g.params['point'] = p + shift
                    else:
                        g.params['apex'] = p + shift
                    stats['axes'] += 1
    planes = [rg for rg in regions if rg.kind == 'plane']
    used = set()
    # two parallel planes are one plane only within the uncertainty the
    # revert below would accept anyway (4 * merge_tol, at least 0.02 mm):
    # a shallower offset than fit_tol/2 can be a real step (a ledge), and
    # proposing it just to revert it leaves the region's resid untouched
    plane_tol = min(pos_tol, max(4 * merge_tol, 0.02))
    for a in planes:
        if a.id in used:
            continue
        na = a.params['normal']
        oa = float(na @ a.params['point'])
        group = [a]
        for b in planes:
            if b.id == a.id or b.id in used:
                continue
            nb = b.params['normal']
            if float(na @ nb) < np.cos(np.radians(0.5)):
                continue
            if abs(float(na @ b.params['point']) - oa) <= plane_tol:
                group.append(b)
        if len(group) > 1:
            wsum = sum(g.area for g in group)
            off = sum(g.area * float(na @ g.params['point']) for g in group) / wsum
            for g in group:
                used.add(g.id)
                cur = float(na @ g.params['point'])
                if abs(cur - off) > 1e-12:
                    snapped.setdefault(g.id, dict(g.params))
                    g.params['point'] = g.params['point'] + (off - cur) * na
                    stats['planes'] += 1

    # 3. equality: radii (double-cap clustering as rebuild._global_snap) -------
    r_tol = fit_tol / 2
    rr = sorted([rg for rg in regions if rg.kind in ('cylinder', 'sphere')],
                key=lambda g: g.params['r'])
    cluster = []

    def flush(cl):
        if len(cl) < 2:
            return
        wsum = sum(g.area for g in cl)
        rmean = sum(g.area * g.params['r'] for g in cl) / wsum
        grid = round(rmean, 1)
        if abs(grid - rmean) < max(2 * merge_tol, 1e-3):
            rmean = grid
        for g in cl:
            if abs(g.params['r'] - rmean) > 1e-12:
                snapped.setdefault(g.id, dict(g.params))
                g.params['r'] = float(rmean)
                stats['radii'] += 1
    for g in rr:
        if cluster and (g.params['r'] - cluster[-1].params['r'] > r_tol or
                        g.params['r'] - cluster[0].params['r'] > r_tol):
            flush(cluster)
            cluster = []
        cluster.append(g)
    flush(cluster)

    # 4. verify every touched region; revert what moved off the vertices -----
    def measure(rg):
        P = region_pts(rg)
        d = DIST[rg.kind]
        rv = float(np.abs(d(P, rg.params)).max())
        inner = np.concatenate([C[rg.faces], MID[rg.faces], MID[rg.faces + nf], MID[rg.faces + 2 * nf]])
        rc = float(np.abs(d(inner, rg.params)).max())
        return rv, rc, (rv > max(4 * merge_tol, 1.5 * rg.resid_v) or rc > fit_tol)

    verdict = {}
    for rg in regions:
        if rg.id in snapped:
            verdict[rg.id] = measure(rg)
    # a direction cluster snaps or reverts as a whole: if the big face
    # cannot take the snap (33 mm x 0.1 deg is 0.03 mm off its vertices)
    # while the ledge next to it can, snapping only the ledge would put
    # two faces that are parallel in the mesh 0.1 deg apart in the solid
    bad_clusters = {dir_cluster[i] for i, (rv, rc, bad) in verdict.items()
                    if bad and i in dir_cluster}
    for rg in regions:
        if rg.id not in snapped:
            continue
        rv, rc, bad = verdict[rg.id]
        if bad or dir_cluster.get(rg.id) in bad_clusters:
            rg.params = snapped[rg.id]
            stats['reverted'] += 1
        else:
            rg.resid_v = rv
            rg.resid = max(rv, rc)
    if verbose and any(stats.values()):
        print(f"[fgroup] regularise: {stats['dirs']} direction(s), {stats['axes']} axis, "
              f"{stats['planes']} plane offset(s), {stats['radii']} radius snap(s), "
              f"{stats['reverted']} reverted")
    return stats


def _set_direction(rg, key, nd, P, fit_circle_taubin, _plane_basis):
    """Give region rg a new normal/axis and re-solve its linear parameters."""
    if rg.kind == 'plane':
        rg.params['normal'] = nd
        rg.params['point'] = P.mean(axis=0)
        return
    b0, b1 = _plane_basis(nd)
    if rg.kind == 'cylinder':
        uv = np.column_stack([P @ b0, P @ b1])
        circ = fit_circle_taubin(uv, polyline=False)
        if circ is None:
            return
        cu, cv = circ['center']
        rg.params['axis'] = nd
        rg.params['point'] = cu * b0 + cv * b1
        rg.params['r'] = float(circ['r'])
        return
    if rg.kind == 'cone':
        # fixed axis: r = (h - h_apex) tan(alpha) is linear in h
        rel = P - rg.params['apex']
        h = rel @ nd
        rad = np.linalg.norm(rel - np.outer(h, nd), axis=1)
        A = np.column_stack([h, np.ones(len(h))])
        (a, b), *_ = np.linalg.lstsq(A, rad, rcond=None)
        if not (a > 1e-6):
            return
        h_apex = -b / a
        rg.params['axis'] = nd
        rg.params['apex'] = rg.params['apex'] + h_apex * nd
        rg.params['half_angle'] = float(np.arctan(a))


# --- face construction (B1: projected boundary polylines) ----------------------

def _vertex_positions(mesh, regions):
    """One position per mesh vertex, on every fitted surface it touches:
    both sides of every region boundary must use identical chords for
    sewing to pair them.

    Alternating projection onto the incident surfaces, smallest region
    first and the largest last, so the last (exact) projection lands on
    the face that matters most and the few microns of closure error go to
    the fillet strips. (A least-squares junction point is not better here:
    fillets meet their neighbours tangentially, where the intersection is
    ill-conditioned and the "exact" junction can sit tens of microns from
    the mesh vertex.)"""
    V, F = mesh.vertices, mesh.faces
    inc = [[] for _ in range(len(V))]
    for rg in regions:
        for vi in np.unique(F[rg.faces]).tolist():
            inc[vi].append(rg)
    vpos = V.copy()
    for vi in range(len(V)):
        rs = inc[vi]
        if not rs or (len(rs) == 1 and rs[0].kind == 'plane'):
            continue
        rs = sorted(rs, key=lambda rg: rg.area)
        p = V[vi].copy()
        for _ in range(4):
            for rg in rs:
                p = _project(p, rg.kind, rg.params)
        if np.all(np.isfinite(p)):
            vpos[vi] = p
    return vpos, inc


def _seam_xdir(org, ax, pts, bloops):
    """Direction of the parametric seam (U = 0) for a surface of revolution
    about (org, ax): in the largest angular gap of the region's vertices when
    the region does not wrap all the way round (the seam then never touches
    the face), else through a boundary vertex — preferably an angle shared by
    several boundary vertices (both rings of a cylinder) — so ShapeFix's
    seam splits the boundary at an existing vertex and never creates a new
    one on a chord (which does not sew against the neighbour's chord)."""
    b0 = np.cross([0, 1, 0], ax)
    if np.linalg.norm(b0) < 1e-6:
        b0 = np.cross([1, 0, 0], ax)
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(ax, b0)

    def angles(P):
        rel = P - org
        rad = rel - np.outer(rel @ ax, ax)
        ok = np.linalg.norm(rad, axis=1) > 1e-9
        return np.mod(np.arctan2(rad[ok] @ b1, rad[ok] @ b0), 2 * np.pi)
    a_all = np.sort(angles(pts))
    if len(a_all) == 0:
        return b0
    # does any boundary loop wind around the axis? (a cap around a pole, a
    # full ring) — then the seam has to cross the boundary and must do so at
    # a vertex; otherwise it can sit in the largest angular gap, untouched
    winds = False
    for L in bloops:
        a = angles(L)
        if len(a) < 3:
            continue
        du = np.diff(np.r_[a, a[0]])
        du = (du + np.pi) % (2 * np.pi) - np.pi
        if abs(du.sum()) > np.pi:
            winds = True
            break
    gaps = np.diff(np.r_[a_all, a_all[0] + 2 * np.pi])
    i = int(np.argmax(gaps))
    if not winds and gaps[i] >= np.radians(5):
        seam = a_all[i] + gaps[i] / 2
    else:
        a_b = np.sort(angles(np.vstack(bloops))) if bloops else a_all
        if len(a_b) == 0:
            seam = a_all[0]
        else:
            # the boundary angle with the most companions within 1e-4 rad
            q = np.round(a_b / 1e-4).astype(np.int64)
            vals, counts = np.unique(q, return_counts=True)
            seam = float(a_b[np.searchsorted(q, vals[int(np.argmax(counts))])])
    return np.cos(seam) * b0 + np.sin(seam) * b1


def _make_surface(kind, pr, pts, bloops=None, sphere_axis=None):
    """Geom surface for a fitted region. `pts` = all region vertices,
    `bloops` = list of boundary-loop vertex arrays (seam placement, see
    _seam_xdir). A sphere's
    pole goes to the region's centroid direction unless `sphere_axis` says
    otherwise (holes around the pole break the trimmed face)."""
    from OCP.gp import gp_Ax3, gp_Pnt, gp_Dir, gp_Pln
    from OCP.Geom import (Geom_Plane, Geom_CylindricalSurface,
                          Geom_ConicalSurface, Geom_SphericalSurface)
    if bloops is None:
        bloops = []
    if kind == 'plane':
        return gp_Pln(gp_Pnt(*map(float, pr['point'])), gp_Dir(*map(float, pr['normal'])))
    if kind == 'sphere':
        c = pr['center']
        if sphere_axis is None:
            d = pts.mean(axis=0) - c
            n = np.linalg.norm(d)
            d = d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        else:
            d = np.asarray(sphere_axis, float)
        xd = _seam_xdir(c, d, pts, bloops)
        return Geom_SphericalSurface(gp_Ax3(gp_Pnt(*map(float, c)), gp_Dir(*map(float, d)),
                                            gp_Dir(*map(float, xd))), float(pr['r']))
    ax = pr['axis']
    org = pr['point'] if kind == 'cylinder' else pr['apex']
    xd = _seam_xdir(org, ax, pts, bloops)
    ax3 = gp_Ax3(gp_Pnt(*map(float, org)), gp_Dir(*map(float, ax)), gp_Dir(*map(float, xd)))
    if kind == 'cylinder':
        return Geom_CylindricalSurface(ax3, float(pr['r']))
    return Geom_ConicalSurface(ax3, float(pr['half_angle']), 0.0)


def _loops_param_ok(kind, loops, vpos, org, ax, r_ref):
    """The boundary polylines must be well parametrised on a surface of
    revolution about (org, ax): no vertex within a sliver of the axis (its
    angle is meaningless there), no jump of more than 120 deg between
    consecutive vertices (a chord passing the axis/pole), each loop either
    closed in UV or winding exactly once, the winding count consistent with
    the surface (a cone region must not contain the apex; a cylinder region
    is a patch or a band), and no self-intersection in UV. ShapeFix_Face
    crashes the process on wires that break these, so this runs BEFORE any
    OCC call."""
    from shapely.geometry import Polygon, LineString
    b0 = np.cross([0, 1, 0], ax)
    if np.linalg.norm(b0) < 1e-6:
        b0 = np.cross([1, 0, 0], ax)
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(ax, b0)
    rmin = max(5 * MERGE_TOL_MIN, 0.02 * r_ref)
    n_wind = 0
    for l in loops:
        rel = vpos[l] - org
        h = rel @ ax
        rad = rel - np.outer(h, ax)
        rn = np.linalg.norm(rad, axis=1)
        if rn.min() < rmin:
            return False
        u = np.arctan2(rad @ b1, rad @ b0)
        du = np.diff(np.r_[u, u[0]])
        du = (du + np.pi) % (2 * np.pi) - np.pi          # shortest signed step
        if np.abs(du).max() > np.radians(120):
            return False
        wind = float(du.sum())
        uu = np.r_[u[0], u[0] + np.cumsum(du)]           # unwrapped, closes at u0+wind
        v = h if kind != 'sphere' else np.arcsin(np.clip(h / max(r_ref, 1e-12), -1, 1))
        vv = np.r_[v, v[0]]
        if abs(wind) < 0.5:
            if not Polygon(np.column_stack([uu[:-1], vv[:-1]])).is_valid:
                return False
        elif abs(abs(wind) - 2 * np.pi) < 0.5:
            n_wind += 1
            if not LineString(np.column_stack([uu, vv])).is_simple:
                return False
        else:
            return False
    if kind == 'sphere':
        return n_wind <= 2
    return n_wind in (0, 2)


def _polygon(vpos, loop):
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    mp = BRepBuilderAPI_MakePolygon()
    last = None
    for vi in loop:
        p = vpos[vi]
        if last is not None and np.linalg.norm(p - last) < 1e-9:
            continue
        mp.Add(gp_Pnt(*map(float, p)))
        last = p
    mp.Close()
    return mp.Wire() if mp.IsDone() else None


def _face_vertices_near(face, vpos, loops, tol):
    """Every vertex of the built face lies on (or very near) the polyline;
    catches surfaces whose parametrisation moved the boundary."""
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_VERTEX, TopAbs_EDGE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    segs_a, segs_b = [], []
    for l in loops:
        P = vpos[l]
        segs_a.append(P)
        segs_b.append(np.roll(P, -1, axis=0))
    A = np.vstack(segs_a)
    B = np.vstack(segs_b)
    AB = B - A
    L2 = np.maximum((AB * AB).sum(axis=1), 1e-18)
    # a sphere cap's pole comes in as a degenerate edge (and the end of the
    # seam) whose vertex sits far from every chord — legitimate, skip it
    poles = []
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        if BRep_Tool.Degenerated_s(e):
            vx = TopExp_Explorer(e, TopAbs_VERTEX)
            while vx.More():
                p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(vx.Current()))
                poles.append(np.array([p.X(), p.Y(), p.Z()]))
                vx.Next()
        ex.Next()
    ex = TopExp_Explorer(face, TopAbs_VERTEX)
    while ex.More():
        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(ex.Current()))
        q = np.array([p.X(), p.Y(), p.Z()])
        ex.Next()
        if any(np.linalg.norm(q - pp) < 1e-6 for pp in poles):
            continue
        t = np.clip(((q - A) * AB).sum(axis=1) / L2, 0.0, 1.0)
        d = np.linalg.norm(A + t[:, None] * AB - q, axis=1).min()
        if d > tol:
            return False
    return True


def _region_face(mesh, rg, vpos, loops, fit_tol):
    """One trimmed OCC face for a region, or (None, reason)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    F = mesh.faces
    pts = vpos[np.unique(F[rg.faces])]
    if rg.kind == 'cone':
        h = (pts - rg.params['apex']) @ rg.params['axis']
        if h.min() <= max(5 * MERGE_TOL_MIN, 0.05 * (h.max() - h.min())):
            return None, 'apex'
        rad = np.linalg.norm((pts - rg.params['apex']) - np.outer(h, rg.params['axis']), axis=1)
        if not _loops_param_ok('cone', loops, vpos, rg.params['apex'], rg.params['axis'], float(rad.max())):
            return None, 'axis'
    if rg.kind == 'cylinder':
        if not _loops_param_ok('cylinder', loops, vpos, rg.params['point'], rg.params['axis'], rg.params['r']):
            return None, 'axis'
    # sphere: pole placement candidates (centroid direction first; then away
    # from it, the largest facet, the world axes) — a hole around the pole
    # yields a broken face, and only building tells
    axes = [None]
    if rg.kind == 'sphere':
        c = rg.params['center']
        d = pts.mean(axis=0) - c
        d = d / (np.linalg.norm(d) or 1.0)
        big = int(np.argmax(mesh.area_faces[rg.faces]))
        e = mesh.triangles_center[rg.faces[big]] - c
        e = e / (np.linalg.norm(e) or 1.0)
        axes = [ax for ax in (d, -d, e, np.array([0., 0., 1.]), np.array([1., 0., 0.]),
                              np.array([0., 1., 0.]))
                if _loops_param_ok('sphere', loops, vpos, c, ax, rg.params['r'])]
        if not axes:
            return None, 'pole'
    bloops = [vpos[np.asarray(l)] for l in loops]
    surfs = []
    for ax in axes:
        try:
            surfs.append(_make_surface(rg.kind, rg.params, pts, bloops, sphere_axis=ax))
        except Exception as e:                   # noqa: BLE001
            return None, f'surface {type(e).__name__}'
    if rg.kind == 'plane':
        n = rg.params['normal']
        b0 = np.cross([0, 1, 0], n)
        if np.linalg.norm(b0) < 1e-6:
            b0 = np.cross([1, 0, 0], n)
        b0 /= np.linalg.norm(b0)
        b1 = np.cross(n, b0)

        def area2d(loop):
            P = vpos[loop]
            u, v = P @ b0, P @ b1
            return 0.5 * np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v)
        loops = sorted(loops, key=lambda l: -abs(area2d(l)))
        flips = (False,)
        inside = True
        # the face is built from the (possibly moved) vpos polygons, so that
        # is the area to expect — not the original triangles'
        area_mesh = abs(area2d(loops[0])) - sum(abs(area2d(l)) for l in loops[1:])
    else:
        loops = sorted(loops, key=lambda l: -len(l))
        flips = (rg.concave, not rg.concave)
        inside = False
        area_mesh = float(mesh.area_faces[rg.faces].sum())
    why = 'notdone'
    for surf, flip in [(s, f) for s in surfs for f in flips]:
        wires = []
        for l in loops:
            w = _polygon(vpos, l[::-1] if flip else l)
            if w is None:
                return None, 'polygon'
            wires.append(w)
        mk = BRepBuilderAPI_MakeFace(surf, wires[0], inside)
        if not mk.IsDone():
            why = 'notdone'
            continue
        for w in wires[1:]:
            mk.Add(w)
        if not mk.IsDone():
            why = 'notdone'
            continue
        face = mk.Face()
        fx = ShapeFix_Face(face)
        fx.Perform()
        face = fx.Face()
        if not BRepCheck_Analyzer(face).IsValid():
            why = 'invalid'
            continue
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g)
        a = g.Mass()
        if not (a > 0 and abs(a - area_mesh) <= 0.05 * area_mesh + 1e-3):
            why = f'area {a:.3f} vs {area_mesh:.3f}'
            continue
        if not _face_vertices_near(face, vpos, loops, fit_tol):
            why = 'vertices moved'
            continue
        return face, 'analytic'
    return None, why


def _full_sphere_solid(rg):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt
    c = rg.params['center']
    return BRepPrimAPI_MakeSphere(gp_Pnt(*map(float, c)), float(rg.params['r'])).Solid()


def build_solid(mesh, regions, fit_tol=0.08, sew_tol=SEW_TOL, verbose=False):
    """Faces for every region → sew → largest solid. Returns (TopoDS_Solid,
    stats); raises FaceGroupError when the result is not one closed shell."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
    from OCP.BRep import BRep_Tool
    from OCP.TopoDS import TopoDS
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.BRepCheck import BRepCheck_Analyzer
    from .rebuild import _largest_solid, _naked_edges
    t0 = time.time()
    F = mesh.faces
    vpos, inc = _vertex_positions(mesh, regions)
    fallbacks = []
    faces_by_region = {}
    if len(regions) == 1 and regions[0].kind == 'sphere':
        # the whole body is one sphere: no boundary at all, nothing to sew —
        # the solid is the closed surface itself
        rg = regions[0]
        solid = _full_sphere_solid(rg)
        rg.built = 'analytic'
        rg.n_faces_out = 1
        vol = float(4 / 3 * np.pi * rg.params['r'] ** 3)
        return solid, {'faces_added': 1, 'faces_kept': 1, 'shells': 1, 'free_edges': 0,
                       'triangle_faces': 0, 'naked_edges': 0, 'volume': vol,
                       'valid': True, 'fallbacks': [], 't_build': time.time() - t0}
    for rg in regions:
        loops = _group_loops(mesh, rg.faces)
        if not loops:
            rg.built = 'pinched'
            fallbacks.append((rg.id, rg.kind, int(len(rg.faces)), 'pinched'))
            continue
        face, why = _region_face(mesh, rg, vpos, loops, fit_tol)
        rg.built = why
        if face is None:
            fallbacks.append((rg.id, rg.kind, int(len(rg.faces)), why))
        else:
            faces_by_region[rg.id] = face

    def sew_all(triangle_regions):
        sew = BRepBuilderAPI_Sewing(sew_tol)
        n_added = n_tri = 0
        for rg in regions:
            f = faces_by_region.get(rg.id)
            if f is not None and rg.id not in triangle_regions:
                sew.Add(f)
                n_added += 1
                rg.n_faces_out = 1
                continue
            rg.n_faces_out = 0
            for tri in F[rg.faces]:
                w = _polygon(vpos, tri)
                if w is None:
                    continue
                mk = BRepBuilderAPI_MakeFace(w, True)
                if mk.IsDone():
                    sew.Add(mk.Face())
                    n_added += 1
                    n_tri += 1
                    rg.n_faces_out += 1
        sew.Perform()
        return sew, n_added, n_tri

    def owners_of_free_edges(sew, triangle_regions):
        """Regions whose analytic face carries a free edge after sewing."""
        from OCP.TopAbs import TopAbs_EDGE
        out = set()
        free_edges = [sew.FreeEdge(i) for i in range(1, sew.NbFreeEdges() + 1)]
        if not free_edges:
            return out
        for rg in regions:
            f = faces_by_region.get(rg.id)
            if f is None or rg.id in triangle_regions:
                continue
            fs = sew.Modified(f) if sew.IsModified(f) else f
            ex = TopExp_Explorer(fs, TopAbs_EDGE)
            edges = []
            while ex.More():
                edges.append(ex.Current())
                ex.Next()
            if any(e.IsSame(fe) for fe in free_edges for e in edges):
                out.add(rg.id)
        return out

    def neighbours_of_free_edges(sew, triangle_regions):
        """Every analytic region touching a free-edge vertex (coarse)."""
        key = {tuple(np.round(vpos[i], 5)): i for i in range(len(vpos))}
        out = set()
        for i in range(1, sew.NbFreeEdges() + 1):
            e = sew.FreeEdge(i)
            ex = TopExp_Explorer(e, TopAbs_VERTEX)
            while ex.More():
                p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(ex.Current()))
                vi = key.get((round(p.X(), 5), round(p.Y(), 5), round(p.Z(), 5)))
                ex.Next()
                if vi is None:
                    continue
                for rg in inc[vi]:
                    if rg.id in faces_by_region and rg.id not in triangle_regions:
                        out.add(rg.id)
        return out

    triangle_regions = set()
    sew, n_added, n_tri = sew_all(triangle_regions)
    free = sew.NbFreeEdges()
    # free edges: rebuild the region whose face owns the edge from
    # triangles first (the neighbour across the edge is usually fine —
    # demoting everything that touches the edge's vertices threw a 500 mm2
    # plane away for a fillet strip that did not sew); widen to all
    # touching regions only if that was not enough
    def curved_owners(sew, triangle_regions):
        # an unsewn edge has two owners; a plane's straight chords are
        # what the neighbour was given, a curved face may have had its
        # chords split by the seam ShapeFix inserted — try those first
        own = owners_of_free_edges(sew, triangle_regions)
        curved = {rid for rid in own if regions[rid].kind != 'plane'}
        return curved if curved else own

    for pick in (curved_owners, owners_of_free_edges, neighbours_of_free_edges):
        if free <= 0:
            break
        culprits = pick(sew, triangle_regions)
        if not culprits:
            continue
        for rid in culprits:
            regions[rid].built = 'free edges'
            fallbacks.append((rid, regions[rid].kind, int(len(regions[rid].faces)), 'free edges'))
        triangle_regions = triangle_regions | culprits
        sew, n_added, n_tri = sew_all(triangle_regions)
        free = sew.NbFreeEdges()
    solid, vol, kept, n_shells, total = _largest_solid(sew.SewedShape(), 'facegroup')
    frac = kept / n_added if n_added else 0.0
    naked = _naked_edges(solid)
    valid = BRepCheck_Analyzer(solid).IsValid()
    stats = {'faces_added': n_added, 'faces_kept': kept, 'shells': n_shells,
             'free_edges': int(free), 'triangle_faces': n_tri,
             'naked_edges': naked, 'volume': float(vol), 'valid': bool(valid),
             'fallbacks': fallbacks, 't_build': time.time() - t0}
    if verbose:
        n_an = sum(1 for rg in regions if rg.built == 'analytic')
        print(f"[fgroup] {n_an} analytic face(s), {n_tri} triangle face(s) from "
              f"{len(fallbacks)} unfitted region(s); sewn: {kept}/{n_added} faces kept, "
              f"{free} free edge(s), {naked} naked, {'valid' if valid else 'INVALID'}")
    if free > 0 or frac < 0.9 or naked > 0:
        raise FaceGroupError(
            f"sewn shell not closed ({free} free edges, {naked} naked, "
            f"{kept}/{n_added} faces kept)")
    return solid, stats


def convert(mesh, tol=0.08, verbose=False):
    """segment → regularise → build → finish. Returns (TopoDS_Shape, stats);
    (None, stats) when the mesh is outside the engine's envelope; raises
    FaceGroupError when no closed valid solid comes out."""
    import cadquery as cq
    from .rebuild import finish_solid
    from OCP.BRepCheck import BRepCheck_Analyzer
    from collections import Counter
    regions = segment(mesh, fit_tol=tol, verbose=verbose)
    stats = dict(getattr(segment, 'last_stats', {}))
    if regions is None:
        stats['skipped'] = 'envelope'
        return None, stats
    stats['snaps'] = regularise(regions, mesh, fit_tol=tol,
                                merge_tol=stats.get('merge_tol'), verbose=verbose)
    solid, bstats = build_solid(mesh, regions, fit_tol=tol, verbose=verbose)
    stats.update(bstats)
    wp = finish_solid(cq.Workplane('XY').newObject([cq.Shape.cast(solid)]), verbose=False)
    shape = wp.val()
    if not BRepCheck_Analyzer(shape.wrapped).IsValid():
        raise FaceGroupError('solid failed BRepCheck after healing')
    from .rebuild import _surface_kind
    kinds = Counter(_surface_kind(f.wrapped) for f in shape.Faces())
    by_type = Counter(r.kind for r in regions)
    stats.update({
        'regions': len(regions),
        'by_type': dict(by_type),
        'unfitted_regions': sum(1 for r in regions if r.built != 'analytic'),
        'unfitted_faces': int(sum(len(r.faces) for r in regions if r.built != 'analytic')),
        'faces_out': len(shape.Faces()),
        'faces_by_kind': dict(kinds),
        'resid_max': float(max(r.resid for r in regions)) if regions else 0.0,
        'volume': float(shape.Volume()),
    })
    if verbose:
        print(f"[fgroup] solid: {stats['faces_out']} faces "
              f"({', '.join(f'{k} {v}' for k, v in sorted(kinds.items()))})")
    return shape.wrapped, stats
