"""OCC dry run of an emitted Fusion Boundary Fill script.

Rebuilds every tool surface from the script's BODIES table with the OCC
kernel, runs BOPAlgo_MakerVolume (faces -> cells), classifies the cells the
way the Fusion runtime will (inside probe points, then centre-of-mass in
the mesh) and reports how much of the mesh volume the kept cells enclose.

The pipeline uses this for the Boundary Fill outlook: no heuristic on the
region list predicts kernel behaviour — a chain of tapered blend bands can
cut perfectly well while a gently curved plate segmented into near-parallel
plane strips never closes, and only the arrangement itself tells the two
apart. Fusion's kernel is still the final judge (it has its own quirks the
emulation cannot see), but a closure failure here is a reliable FAIL.
`tools/emulate_bfill.py` is the CLI over the same code."""
import math
import numpy as np
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone,
                             BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus)
from OCP.BOPAlgo import BOPAlgo_MakerVolume
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_IN
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane
from OCP.TopoDS import TopoDS


def parse_script(text):
    """Execute the data head of a generated script plus the pure-python
    helpers of its runtime; returns the namespace (BODIES, EXPAND, _Mesh,
    _expand, _offset_convex...)."""
    head, rt = text.split('\nimport adsk.core')
    ns = {}
    exec(head, ns)
    for a, b in (('def _offset_convex', 'def _plane_body'),
                 ('class _Mesh', 'def _select_cells'),
                 ('def _expand', 'def _plane_body')):
        exec('import math\n' + rt[rt.index(a):rt.index(b)], ns)
    return ns


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
    if a0 is None or a1 is None:
        return gp_Ax2(gp_Pnt(*p1), gp_Dir(*ax)), 2 * math.pi
    grow = min(E / max(r, 1e-6), math.pi / 6)
    lo, hi = a0 - grow, a1 + grow
    span = min(hi - lo, 2 * math.pi)
    x = np.asarray(xref, float)
    a = np.asarray(ax, float)
    y = np.cross(a, x)
    d = math.cos(lo) * x + math.sin(lo) * y
    return gp_Ax2(gp_Pnt(*p1), gp_Dir(*a), gp_Dir(*d)), span


def make_tool(s, ns):
    """OCC faces for one SURFACES record (same growth as the Fusion runtime)."""
    _expand = ns['_expand']
    _offset_convex = ns['_offset_convex']
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


def volume_of(shape):
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return g.Mass()


def check_body(bd, ns):
    """Cells + classification for one BODIES entry. Returns a dict with the
    cell/probe counts and the enclosed volume fraction — the outlook signal."""
    args = TopTools_ListOfShape()
    n = 0
    for s in bd['surfaces']:
        for f in make_tool(s, ns):
            args.Append(f)
            n += 1
    mv = BOPAlgo_MakerVolume()
    mv.SetArguments(args)
    mv.SetRunParallel(True)
    mv.Perform()
    cells = []
    ex = TopExp_Explorer(mv.Shape(), TopAbs_SOLID)
    while ex.More():
        cells.append(TopoDS.Solid_s(ex.Current()))
        ex.Next()
    pts = [gp_Pnt(*p[:3]) for p in bd['inside']]
    mesh = ns['_Mesh'](bd['mesh_v'], bd['mesh_f'])
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
    missed = [bd['inside'][k][3] if len(bd['inside'][k]) > 3 else '?'
              for k, p in enumerate(pts)
              if not any(BRepClass3d_SolidClassifier(c, p, 1e-7).State() == TopAbs_IN
                         for c in kept)]
    kept_vol = sum(abs(volume_of(c)) for c in kept)
    mesh_vol = float(bd.get('volume', 0.0)) or 1e-12
    return {'name': bd.get('name', '?'), 'tool_faces': n, 'cells': len(cells),
            'kept_cells': len(kept), 'by_centre': by_centre,
            'kept_volume': kept_vol, 'mesh_volume': mesh_vol,
            'enclosed_pct': 100.0 * kept_vol / mesh_vol,
            'unenclosed': missed, 'n_probes': len(pts),
            'kept': kept, 'cell_volumes': sorted((abs(volume_of(c)) for c in cells),
                                                 reverse=True)}


def check_script(text):
    """Run check_body over every body of an emitted script's text."""
    ns = parse_script(text)
    return [check_body(bd, ns) for bd in ns['BODIES']], ns
