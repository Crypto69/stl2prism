"""PROTOTYPE (not wired into the pipeline): slice a mesh along an axis, loft the section loops, measure. See docs/SLICED-LOFT.md."""
import sys, time, os
import numpy as np, trimesh, cadquery as cq
from shapely.geometry import Polygon
from OCP.BRepCheck import BRepCheck_Analyzer
sys.path.insert(0, os.getcwd())
from stl_to_solid.pipeline import validate

EPS = 1e-3

def load(path):
    m = trimesh.load(path, force='mesh', process=True)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        parts = sorted(parts, key=lambda p: -(p.volume if p.is_watertight else p.area))
        m = parts[0]
    return m

def in_plane(axis):
    return [i for i in range(3) if i != axis]

def sections(mesh, axis, zs):
    n = np.zeros(3); n[axis] = 1.0
    paths = mesh.section_multiplane(plane_origin=np.zeros(3), plane_normal=n, heights=zs)
    u, v = in_plane(axis)
    out = []
    for z, p in zip(zs, paths):
        loops = []
        if p is not None:
            T = p.metadata['to_3D']
            for poly in p.polygons_full:
                def to3(coords):
                    c = np.asarray(coords, float)[:, :2]
                    h = np.hstack([c, np.zeros((len(c), 1)), np.ones((len(c), 1))])
                    return (h @ T.T)[:, :3]
                ext = to3(poly.exterior.coords)
                holes = [to3(r.coords) for r in poly.interiors]
                loops.append((ext, holes))
        out.append(loops)
    return out

def resample(P, N):
    P = np.asarray(P, float)
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    L = seg.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, L, N, endpoint=False)
    idx = np.clip(np.searchsorted(cum, t, side='right') - 1, 0, len(P) - 1)
    frac = (t - cum[idx]) / np.maximum(seg[idx], 1e-12)
    return P[idx] + (P[(idx + 1) % len(P)] - P[idx]) * frac[:, None]

def ccw(Q, axis):
    u, v = in_plane(axis)
    x, y = Q[:, u], Q[:, v]
    a = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return Q if a > 0 else Q[::-1].copy()

def align(Q, prev):
    if prev is None:
        # start at max-u point for determinism
        k = int(np.argmax(Q[:, 0] + 1e-3 * Q[:, 1]))
        return np.roll(Q, -k, 0)
    best, bk = None, 0
    for k in range(len(Q)):
        d = np.sum((np.roll(Q, -k, 0) - prev) ** 2)
        if best is None or d < best:
            best, bk = d, k
    return np.roll(Q, -bk, 0)

def poly2(P, axis):
    u, v = in_plane(axis)
    return Polygon(np.asarray(P)[:, [u, v]])

def match(loops_a, loops_b, axis):
    """map outer index in a -> outer index in b when counts equal and each
    pair overlaps and has the same hole count; else None."""
    if len(loops_a) != len(loops_b) or not loops_a:
        return None
    pa = [poly2(e, axis).buffer(0) for e, _ in loops_a]
    pb = [poly2(e, axis).buffer(0) for e, _ in loops_b]
    used, m = set(), []
    for i, A in enumerate(pa):
        best, bj = 0.0, -1
        for j, B in enumerate(pb):
            if j in used: continue
            ov = A.intersection(B).area
            if ov > best: best, bj = ov, j
        if bj < 0 or best < 0.2 * min(A.area, pb[bj].area):
            return None
        if len(loops_a[i][1]) != len(loops_b[bj][1]):
            return None
        used.add(bj); m.append(bj)
    return m

def match_holes(ha, hb, axis):
    if len(ha) != len(hb): return None
    if not ha: return []
    pa = [poly2(h, axis).buffer(0) for h in ha]
    pb = [poly2(h, axis).buffer(0) for h in hb]
    used, m = set(), []
    for i, A in enumerate(pa):
        best, bj = -1.0, -1
        for j, B in enumerate(pb):
            if j in used: continue
            d = A.centroid.distance(B.centroid)
            if bj < 0 or d < best: best, bj = d, j
        used.add(bj); m.append(bj)
    return m

def step_levels(mesh, axis, frac=0.02):
    n = mesh.face_normals[:, axis]
    perp = np.abs(n) > 0.999
    if not perp.any(): return []
    zc = mesh.triangles_center[perp][:, axis]
    ar = mesh.area_faces[perp]
    o = np.argsort(zc); zc, ar = zc[o], ar[o]
    u, v = in_plane(axis)
    ext = mesh.extents
    ref = ext[u] * ext[v]
    out, i = [], 0
    while i < len(zc):
        j = i
        while j + 1 < len(zc) and zc[j + 1] - zc[i] < 1e-3: j += 1
        a = ar[i:j + 1].sum()
        if a > frac * ref: out.append(float(zc[i:j + 1].mean()))
        i = j + 1
    return out

def wire(Q):
    e = cq.Edge.makeSpline([cq.Vector(*p) for p in Q], periodic=True)
    return cq.Wire.assembleEdges([e])

def loft_chain(rings, ruled):
    if len(rings) == 1:
        raise ValueError('single section')
    return cq.Solid.makeLoft([wire(Q) for Q in rings], ruled=ruled)

def build(mesh, axis, interval, ruled=False, N_per_mm=1.0, verbose=True):
    lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
    zs = list(np.arange(lo + interval, hi - interval * 0.5, interval))
    reqs = [(lo + EPS, lo), (hi - EPS, hi)] + [(z, z) for z in zs]
    for L in step_levels(mesh, axis):
        if L - lo < 2 * EPS or hi - L < 2 * EPS: continue
        reqs.append((L - EPS, L)); reqs.append((L + EPS, L))
    reqs.sort()
    # drop regular slices that sit within EPS*3 of a snapped one
    keep = []
    for r in reqs:
        if keep and abs(r[0] - keep[-1][0]) < 3 * EPS: continue
        keep.append(r)
    reqs = keep
    secs = sections(mesh, axis, [r[0] for r in reqs])
    # snap
    for (zq, zt), loops in zip(reqs, secs):
        for ext, holes in loops:
            ext[:, axis] = zt
            for h in holes: h[:, axis] = zt
    # build runs: list of (start idx, end idx, chains) — chains: list of list of (ext, holes) per slice
    slices = [(zt, loops) for (zq, zt), loops in zip(reqs, secs) if loops]
    if verbose: print(f"  {len(slices)} sections, step levels {len(step_levels(mesh, axis))}")
    runs = []
    i = 0
    while i < len(slices):
        j = i
        chains = [[lp] for lp in slices[i][1]]     # one chain per outer
        while j + 1 < len(slices):
            m = match(slices[j][1], slices[j + 1][1], axis)
            if m is None: break
            nxt = slices[j + 1][1]
            for ci, mj in enumerate(m):
                chains[ci].append(nxt[mj])
            j += 1
            # a snapped pair at the same z means a step level: end the run here
            if abs(slices[j][0] - slices[j - 1][0]) < 1e-9:
                break
        runs.append((i, j, chains))
        i = j + 1
    if verbose: print(f"  {len(runs)} runs: {[(round(slices[a][0],2), round(slices[b][0],2), len(c)) for a,b,c in runs]}")
    solids = []
    for (a, b, chains) in runs:
        for chain in chains:
            perim = max(poly2(e, axis).length for e, _ in chain)
            N = int(np.clip(round(perim * N_per_mm), 16, 240))
            rings, prev = [], None
            for e, _ in chain:
                Q = align(ccw(resample(e, N), axis), prev); rings.append(Q); prev = Q
            zs_chain = [r[0][axis] for r in rings]
            if len(rings) == 1 or abs(zs_chain[-1] - zs_chain[0]) < 1e-6:
                continue
            body = loft_chain(rings, ruled)
            # holes
            nh = len(chain[0][1])
            for hi_ in range(nh):
                hrings, prev, hidx = [], None, hi_
                for k, (e, holes) in enumerate(chain):
                    if k > 0:
                        mm = match_holes(chain[k-1][1], holes, axis)
                        hidx = mm[hidx]
                    hp = holes[hidx]
                    Nh = int(np.clip(round(poly2(hp, axis).length * N_per_mm), 16, 240)) if k == 0 else len(hrings[0])
                    Q = align(ccw(resample(hp, Nh), axis), prev); hrings.append(Q); prev = Q
                # extend hole solid a bit past both ends so the cut is clean
                hs = loft_chain(hrings, ruled)
                body = body.cut(hs)
            solids.append(body)
    # bridges between runs (gap along axis)
    for r0, r1 in zip(runs, runs[1:]):
        za, zb = slices[r0[1]][0], slices[r1[0]][0]
        gap = zb - za
        if gap < 1e-6: continue
        for chain in r0[2]:
            e, holes = chain[-1]
            ow = wire(align(ccw(resample(e, int(np.clip(round(poly2(e, axis).length*N_per_mm),16,240))), axis), None))
            hw = [wire(ccw(resample(h, int(np.clip(round(poly2(h, axis).length*N_per_mm),16,240))), axis)) for h in holes]
            d = np.zeros(3); d[axis] = gap
            solids.append(cq.Solid.extrudeLinear(ow, hw, cq.Vector(*d)))
    if verbose: print(f"  {len(solids)} solids before union")
    shape = solids[0]
    if len(solids) > 1:
        shape = shape.fuse(*solids[1:], tol=1e-3).clean()
    return shape

if __name__ == '__main__':
    path = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else None
    axis = int(sys.argv[3]) if len(sys.argv) > 3 else None
    ruled = '--ruled' in sys.argv
    mesh = load(path)
    if axis is None: axis = int(np.argmax(mesh.extents))
    if interval is None: interval = float(mesh.extents[axis]) / 40
    print(f"{os.path.basename(path)}: {len(mesh.faces)} tris, extents {np.round(mesh.extents,1)}, axis {axis}, interval {interval:.2f}, ruled={ruled}")
    t0 = time.time()
    shape = build(mesh, axis, interval, ruled=ruled)
    t1 = time.time()
    ok = BRepCheck_Analyzer(shape.wrapped).IsValid()
    print(f"  built in {t1-t0:.1f}s: {len(shape.Faces())} faces, {len(shape.Solids())} solids, valid={ok}, vol {shape.Volume():.0f} vs mesh {mesh.volume:.0f}")
    m = validate(cq.Workplane('XY').add(shape), mesh)
    print(f"  dev p95 {m['dev_p95']:.3f} max {m['dev_max']:.3f} rev p95 {m['rev_dev_p95']:.3f} rev max {m['rev_dev_max']:.3f} vol err {m['vol_err_pct']:.2f}%  ({time.time()-t1:.1f}s check)")
    out = f"/private/tmp/claude-501/-Volumes-ExternalHD-code-STLTOSOLIDKEEP/20dedaa7-5571-4a38-bf8c-ea28d88801c1/scratchpad/{os.path.splitext(os.path.basename(path))[0]}_loft.step"
    cq.exporters.export(cq.Workplane('XY').add(shape), out)
    print(f"  -> {out}")
