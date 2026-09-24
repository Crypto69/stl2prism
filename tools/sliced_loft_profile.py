"""PROTOTYPE profiler for tools/sliced_loft_proto.py: times each stage. See docs/SLICED-LOFT.md."""
import sys, time, os
import numpy as np, trimesh, cadquery as cq
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import sliced_loft_proto as P
from OCP.BRepCheck import BRepCheck_Analyzer

def T(): return time.time()
def build_dbg(mesh, axis, interval, ruled=False, N_per_mm=1.0):
    lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
    zs = list(np.arange(lo + interval, hi - interval * 0.5, interval))
    reqs = [(lo + P.EPS, lo), (hi - P.EPS, hi)] + [(z, z) for z in zs]
    for L in P.step_levels(mesh, axis):
        if L - lo < 2 * P.EPS or hi - L < 2 * P.EPS: continue
        reqs.append((L - P.EPS, L)); reqs.append((L + P.EPS, L))
    reqs.sort()
    keep = []
    for r in reqs:
        if keep and abs(r[0] - keep[-1][0]) < 3 * P.EPS: continue
        keep.append(r)
    reqs = keep
    t = T(); secs = P.sections(mesh, axis, [r[0] for r in reqs]); print(f"  sections {len(reqs)}: {T()-t:.2f}s")
    for (zq, zt), loops in zip(reqs, secs):
        for ext, holes in loops:
            ext[:, axis] = zt
            for h in holes: h[:, axis] = zt
    slices = [(zt, loops) for (zq, zt), loops in zip(reqs, secs) if loops]
    runs, i = [], 0
    while i < len(slices):
        j = i; chains = [[lp] for lp in slices[i][1]]
        while j + 1 < len(slices):
            m = P.match(slices[j][1], slices[j + 1][1], axis)
            if m is None: break
            nxt = slices[j + 1][1]
            for ci, mj in enumerate(m): chains[ci].append(nxt[mj])
            j += 1
            if abs(slices[j][0] - slices[j - 1][0]) < 1e-9: break
        runs.append((i, j, chains)); i = j + 1
    print(f"  {len(runs)} runs")
    solids = []
    for (a, b, chains) in runs:
        for chain in chains:
            perim = max(P.poly2(e, axis).length for e, _ in chain)
            N = int(np.clip(round(perim * N_per_mm), 16, 240))
            rings, prev = [], None
            for e, _ in chain:
                Q = P.align(P.ccw(P.resample(e, N), axis), prev); rings.append(Q); prev = Q
            zc = [r[0][axis] for r in rings]
            if len(rings) == 1 or abs(zc[-1] - zc[0]) < 1e-6:
                print(f"   run {zc[0]:.2f}: skipped (single/zero-height)"); continue
            t = T(); body = P.loft_chain(rings, ruled); dt = T()-t
            print(f"   run {zc[0]:.2f}..{zc[-1]:.2f} n={len(rings)} N={N}: loft {dt:.2f}s vol {body.Volume():.1f} faces {len(body.Faces())} valid {BRepCheck_Analyzer(body.wrapped).IsValid()}")
            nh = len(chain[0][1])
            for hi_ in range(nh):
                hrings, prev, hidx = [], None, hi_
                for k, (e, holes) in enumerate(chain):
                    if k > 0:
                        mm = P.match_holes(chain[k-1][1], holes, axis); hidx = mm[hidx]
                    hp = holes[hidx]
                    Nh = int(np.clip(round(P.poly2(hp, axis).length * N_per_mm), 16, 240)) if k == 0 else len(hrings[0])
                    Q = P.align(P.ccw(P.resample(hp, Nh), axis), prev); hrings.append(Q); prev = Q
                t = T(); hs = P.loft_chain(hrings, ruled); body2 = body.cut(hs); dt = T()-t
                print(f"     hole {hi_}: loft+cut {dt:.2f}s hole vol {hs.Volume():.1f} -> body vol {body2.Volume():.1f} valid {BRepCheck_Analyzer(body2.wrapped).IsValid()}")
                body = body2
            solids.append(body)
    for r0, r1 in zip(runs, runs[1:]):
        za, zb = slices[r0[1]][0], slices[r1[0]][0]; gap = zb - za
        if gap < 1e-6: continue
        for chain in r0[2]:
            e, holes = chain[-1]
            ow = P.wire(P.align(P.ccw(P.resample(e, int(np.clip(round(P.poly2(e, axis).length*N_per_mm),16,240))), axis), None))
            hw = [P.wire(P.ccw(P.resample(h, int(np.clip(round(P.poly2(h, axis).length*N_per_mm),16,240))), axis)) for h in holes]
            d = np.zeros(3); d[axis] = gap
            s = cq.Solid.extrudeLinear(ow, hw, cq.Vector(*d))
            print(f"   bridge {za:.2f}..{zb:.2f}: vol {s.Volume():.1f} valid {BRepCheck_Analyzer(s.wrapped).IsValid()}")
            solids.append(s)
    print(f"  {len(solids)} solids, sum vol {sum(s.Volume() for s in solids):.1f}")
    t = T()
    shape = solids[0]
    if len(solids) > 1:
        shape = shape.fuse(*solids[1:], glue=True).clean()
    print(f"  fuse(glue) {T()-t:.2f}s -> vol {shape.Volume():.1f} solids {len(shape.Solids())} faces {len(shape.Faces())} valid {BRepCheck_Analyzer(shape.wrapped).IsValid()}")
    return shape

if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'sphere':
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=15.0)
    elif what == 'bottle':
        wp = (cq.Workplane('XZ').spline([(0,0),(14,0),(16,10),(13,30),(6,42),(5,50),(5,60),(0,60)], includeCurrent=False).close().revolve(360, (0,0,0), (0,1,0)))
        p = os.path.join(os.path.dirname(__file__), 'bottle.stl')
        cq.exporters.export(wp, p, tolerance=0.02, angularTolerance=0.2)
        mesh = trimesh.load(p, force='mesh')
    else:
        mesh = P.load(what)
    axis = int(np.argmax(mesh.extents)) if len(sys.argv) < 3 else int(sys.argv[2])
    interval = float(mesh.extents[axis]) / 40 if len(sys.argv) < 4 else float(sys.argv[3])
    print(f"{what}: {len(mesh.faces)} tris extents {np.round(mesh.extents,1)} axis {axis} interval {interval:.2f} watertight {mesh.is_watertight}")
    t0 = T(); shape = build_dbg(mesh, axis, interval); print(f"  total {T()-t0:.1f}s")
    from stl_to_solid.pipeline import validate
    m = validate(cq.Workplane('XY').add(shape), mesh)
    print(f"  dev p95 {m['dev_p95']:.3f} max {m['dev_max']:.3f} rev max {m['rev_dev_max']:.3f} vol err {m['vol_err_pct']:.2f}%")
