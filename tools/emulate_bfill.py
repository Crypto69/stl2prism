"""Emulate a generated Fusion Boundary Fill script in OpenCascade.

Builds the same oversized tool surfaces from the script's BODIES table,
runs BOPAlgo_MakerVolume (faces -> cells), keeps the cells that contain the
inside points or whose centre of mass is inside the mesh, and reports how
many probe points are enclosed. A development check only — Fusion's kernel
is the real judge. Usage:

    python tools/emulate_bfill.py <stem>_fusion_bfill.py [fused.step]
"""
import sys
import math
import time
import numpy as np
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeSphere,
                             BRepPrimAPI_MakeTorus)
from OCP.BOPAlgo import BOPAlgo_MakerVolume
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_IN
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.TopoDS import TopoDS
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.BRepCheck import BRepCheck_Analyzer

text = open(sys.argv[1]).read()
head, rt = text.split('\nimport adsk.core')
ns = {}
exec(head, ns)
# pure-python helpers from the script's runtime
for a, b in (('def _offset_convex', 'def _plane_body'), ('class _Mesh', 'def _select_cells'),
             ('def _expand', 'def _plane_body')):
    exec('import math\n' + rt[rt.index(a):rt.index(b)], ns)
_Mesh, _offset_convex, _expand = ns['_Mesh'], ns['_offset_convex'], ns['_expand']


def faces_of(shape, nonplanar_only=False):
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    out = []
    while ex.More():
        f = TopoDS.Face_s(ex.Current())
        if not nonplanar_only or BRepAdaptor_Surface(f).GetType() != GeomAbs_Plane:
            out.append(f)
        ex.Next()
    return out


def _ax2(p1, ax, xref, r, a0, a1, E):
    """gp_Ax2 whose X direction is the arc start, plus the angular span."""
    ax = np.asarray(ax, float)
    xref = np.asarray(xref, float)
    if a0 is None or r <= 0:
        return gp_Ax2(gp_Pnt(*p1), gp_Dir(*ax)), 2 * math.pi
    da = min(math.pi, E / r)
    if (a1 - a0) + 2 * da >= 2 * math.pi - 0.02:
        return gp_Ax2(gp_Pnt(*p1), gp_Dir(*ax)), 2 * math.pi
    phi = a0 - da
    xd = xref * math.cos(phi) + np.cross(ax, xref) * math.sin(phi)
    return gp_Ax2(gp_Pnt(*p1), gp_Dir(*ax), gp_Dir(*xd)), (a1 - a0) + 2 * da


def make(s):
    k = s[0]
    if k == 'plane':
        _, o, u, v, xy, emin = s
        n = list(np.cross(u, v))
        hull = [[o[i] + x * u[i] + y * v[i] for i in range(3)] for x, y in xy]
        area2 = sum(xy[i][0] * xy[(i + 1) % len(xy)][1] - xy[i][1] * xy[(i + 1) % len(xy)][0]
                    for i in range(len(xy)))
        pts = _offset_convex(hull, n, _expand(math.sqrt(abs(area2) / 2.0), emin))
        pg = BRepBuilderAPI_MakePolygon()
        for p in pts:
            pg.Add(gp_Pnt(*p))
        pg.Close()
        return [BRepBuilderAPI_MakeFace(pg.Wire(), True).Face()]
    if k == 'cyl':
        _, o, ax, r, t0, t1, emin, xref, a0, a1 = s
        E = _expand(t1 - t0, emin)
        p1 = [o[i] + (t0 - E) * ax[i] for i in range(3)]
        ax2, span = _ax2(p1, ax, xref, r, a0, a1, E)
        return faces_of(BRepPrimAPI_MakeCylinder(ax2, r, (t1 - t0) + 2 * E, span).Shape(), True)
    if k == 'cone':
        _, apex, ax, half, t0, t1, emin, xref, a0, a1 = s
        sl = math.tan(half)
        slant = _expand((t1 - t0) / max(math.cos(half), 1e-6), emin)
        E = slant * math.cos(half)
        ta, tb = max(0.0, t0 - E), t1 + E
        r1, r2 = max(1e-4, sl * ta), sl * tb
        p1 = [apex[i] + ta * ax[i] for i in range(3)]
        if sl * ta < 0.005:
            # reaches the apex: the Fusion runtime builds a SOLID cone there
            # (createCylinderOrCone), so keep the base disc too
            ax2, span = _ax2(p1, ax, xref, r1, None, None, slant)
            return faces_of(BRepPrimAPI_MakeCone(ax2, r1, r2, tb - ta, span).Shape())
        ax2, span = _ax2(p1, ax, xref, r1, a0, a1, slant)
        return faces_of(BRepPrimAPI_MakeCone(ax2, r1, r2, tb - ta, span).Shape(), True)
    if k == 'sphere':
        return faces_of(BRepPrimAPI_MakeSphere(gp_Pnt(*s[1]), s[2]).Shape())
    if k == 'torus':
        _, c, ax, R, r = s
        return faces_of(BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(*c), gp_Dir(*ax)), R, r).Shape())
    raise ValueError(k)


def vol(shape):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return g.Mass()


for bd in ns['BODIES']:
    t0 = time.time()
    args = TopTools_ListOfShape()
    n = 0
    for s in bd['surfaces']:
        for f in make(s):
            args.Append(f)
            n += 1
    mv = BOPAlgo_MakerVolume()
    mv.SetArguments(args)
    mv.SetRunParallel(True)
    mv.Perform()
    if mv.HasErrors():
        print('MakerVolume reports errors')
    ex = TopExp_Explorer(mv.Shape(), TopAbs_SOLID)
    cells = []
    while ex.More():
        cells.append(TopoDS.Solid_s(ex.Current()))
        ex.Next()
    print(f"{bd['name']}: {n} tool faces -> {len(cells)} cells in {time.time() - t0:.1f}s")
    pts = [gp_Pnt(*p[:3]) for p in bd['inside']]
    mesh = _Mesh(bd['mesh_v'], bd['mesh_f'])
    kept, by_centre = [], 0
    for c in cells:
        hit = any(BRepClass3d_SolidClassifier(c, p, 1e-7).State() == TopAbs_IN for p in pts)
        if not hit:
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(c, g)
            cm = g.CentreOfMass()
            if (BRepClass3d_SolidClassifier(c, cm, 1e-7).State() == TopAbs_IN
                    and mesh.contains((cm.X(), cm.Y(), cm.Z()))):
                hit = True
                by_centre += 1
        if hit:
            kept.append(c)
    missed = [bd['inside'][k][3] if len(bd['inside'][k]) > 3 else '?' for k, p in enumerate(pts)
              if not any(BRepClass3d_SolidClassifier(c, p, 1e-7).State() == TopAbs_IN for c in kept)]
    miss = len(missed)
    if missed:
        from collections import Counter
        print('  unenclosed probe points by region:', dict(Counter(missed)))
    vols = sorted([abs(vol(c)) for c in cells], reverse=True)
    print(f"  kept {len(kept)} cells ({by_centre} by centre of mass); largest cells {[round(v, 4) for v in vols[:5]]}")
    print(f"  probe points outside every kept cell: {miss}/{len(pts)}  (mesh {bd['volume']:.4f} cm^3)")
    if kept and len(kept) < 80:
        fused = kept[0]
        for c in kept[1:]:
            fused = BRepAlgoAPI_Fuse(fused, c).Shape()
        u = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
        u.Build()
        fused = u.Shape()
        print(f"  union: {len(faces_of(fused))} faces, {vol(fused):.4f} cm^3 "
              f"(err {100 * (vol(fused) - bd['volume']) / bd['volume']:+.2f}%), "
              f"valid={BRepCheck_Analyzer(fused).IsValid()}")
        if len(sys.argv) > 2:
            import cadquery as cq
            cq.exporters.export(cq.Workplane().newObject([cq.Shape.cast(fused)]), sys.argv[2])
            print('  wrote', sys.argv[2])
