"""OCC dry run of an emitted Fusion Boundary Fill script.

Rebuilds every tool surface from the script's BODIES table with the OCC
kernel, grown by the script's own arithmetic (its _expand / _arc_span are
executed from the script text, so the tools are the ones Fusion will get),
runs BOPAlgo_MakerVolume (faces -> cells), classifies the cells the way the
runtime's _select_cells does — a probe point inside, else a point of the
cell's own interior inside the mesh, else a face probe for slivers, else the
cell closest to the mesh volume — and reports how much of the mesh volume
the kept cells enclose and which probe points no cell holds.

The pipeline uses this for the Boundary Fill outlook (`outlook`): no
heuristic on the region list predicts kernel behaviour — a chain of tapered
blend bands can cut perfectly well while a gently curved plate segmented
into near-parallel plane strips never closes, and only the arrangement
itself tells the two apart. Fusion's kernel is still the final judge (it has
its own quirks the emulation cannot see), but a closure failure here is a
reliable FAIL. `tools/emulate_bfill.py` is the CLI over the same code."""
import math
import time
import numpy as np
from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Pnt2d
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone,
                             BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeTorus)
from OCP.BOPAlgo import BOPAlgo_MakerVolume
from OCP.TopTools import TopTools_ListOfShape
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID, TopAbs_IN
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepTools import BRepTools
from OCP.BRep import BRep_Tool
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS

from .rebuild import _faces, _volume

PROBE_TOL = 1e-7                  # point-in-solid tolerance (cm)
NUDGE_CM = 0.005                  # face probes are pushed this far into the cell (0.05 mm)
ENCLOSED_PCT_OK = (99.0, 103.0)   # kept cells must enclose this much of the mesh volume


def parse_script(text):
    """Execute the data head of a generated script plus the pure-python
    helpers of its runtime; returns the namespace (BODIES, EXPAND, SKIP,
    _Mesh, SLIVER_CM, _expand, _arc_span, _offset_convex...)."""
    head, rt = text.split('\nimport adsk.core')
    ns = {}
    exec(head, ns)
    for a, b in (('def _offset_convex', 'def _plane_body'),
                 ('class _Mesh', 'def _select_cells'),
                 ('def _expand', 'def _plane_body')):
        exec('import math\n' + rt[rt.index(a):rt.index(b)], ns)
    if '_arc_span' not in ns or 'SLIVER_CM' not in ns:
        raise ValueError('this script predates v0.3.5 (no _arc_span / SLIVER_CM in its '
                         'runtime): regenerate it from the STL')
    return ns


def _ax2(p1, ax, xref, r, span):
    """gp_Ax2 whose X direction is the arc start, plus the angular length
    (2*pi for a whole circle); `span` is the runtime's _arc_span window."""
    if span is None:
        return gp_Ax2(gp_Pnt(*p1), gp_Dir(*ax)), 2 * math.pi
    lo, hi = span
    x = np.asarray(xref, float)
    a = np.asarray(ax, float)
    y = np.cross(a, x)
    d = math.cos(lo) * x + math.sin(lo) * y
    return gp_Ax2(gp_Pnt(*p1), gp_Dir(*a), gp_Dir(*d)), hi - lo


def make_tool(s, ns):
    """OCC faces for one SURFACES record, grown exactly as the Fusion
    runtime grows it: _expand, _offset_convex and _arc_span are the
    script's own functions, executed from its text."""
    _expand = ns['_expand']
    _offset_convex = ns['_offset_convex']
    _arc_span = ns['_arc_span']
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
        ax2, span = _ax2(p1, ax, xref, r, _arc_span(r, a0, a1, E))
        return _faces(BRepPrimAPI_MakeCylinder(ax2, r, (t1 - t0) + 2 * E, span).Shape(), True)
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
            ax2, span = _ax2(p1, ax, xref, r1, None)
            return _faces(BRepPrimAPI_MakeCone(ax2, r1, r2, tb - ta, span).Shape())
        # one window for both ends, grown at the small end (as the runtime)
        ax2, span = _ax2(p1, ax, xref, r1, _arc_span(r1, a0, a1, slant))
        return _faces(BRepPrimAPI_MakeCone(ax2, r1, r2, tb - ta, span).Shape(), True)
    if k == 'sphere':
        return _faces(BRepPrimAPI_MakeSphere(gp_Pnt(*s[1]), s[2]).Shape())
    if k == 'torus':
        _, c, ax, R, r = s
        return _faces(BRepPrimAPI_MakeTorus(gp_Ax2(gp_Pnt(*c), gp_Dir(*ax)), R, r).Shape())
    raise ValueError(k)


def _point_on_face(face):
    """A point on the face (Fusion's BRepFace.pointOnFace): the first of a
    few parametric samples that classifies as on the face, or None."""
    umin, umax, vmin, vmax = BRepTools.UVBounds_s(face)
    surf = BRep_Tool.Surface_s(face)
    for fu, fv in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.75), (0.25, 0.75), (0.75, 0.25)):
        u = umin + fu * (umax - umin)
        v = vmin + fv * (vmax - vmin)
        if BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), PROBE_TOL).State() == TopAbs_IN:
            return surf.Value(u, v)
    return None


def _cell_is_material(cell, gprops, pts, pt_hit, mesh, sliver_cm):
    """The runtime's _cell_is_material on an OCC solid: (hit, how) — a probe
    point in the cell ('probe'), else a point of its own interior (box
    centre, else centre of mass) inside the mesh ('centre'), else — a
    crescent whose interior points are not its own, or a sliver straddling
    the part's skin — a point on one of its faces nudged inward ('face').
    One classifier per cell and a bounding-box test before every point
    query: nearly every (cell, probe) pair is a miss the box rejects for
    free."""
    cl = BRepClass3d_SolidClassifier(cell)

    def inside(p):
        cl.Perform(p, PROBE_TOL)
        return cl.State() == TopAbs_IN

    bb = Bnd_Box()
    BRepBndLib.Add_s(cell, bb)
    hit = False
    for k, p in enumerate(pts):
        if not bb.IsOut(p) and inside(p):
            hit = True
            pt_hit[k] = True
    if hit:
        return True, 'probe'
    if mesh is None:
        return False, None
    xmin, ymin, zmin, xmax, ymax, zmax = bb.Get()
    if mesh.box_outside((xmin, ymin, zmin), (xmax, ymax, zmax)):
        return False, None
    centre = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    diag = math.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    inner = None
    c = gp_Pnt(*centre)
    if inside(c):
        inner = c
    else:
        cm = gprops.CentreOfMass()
        if inside(cm):
            inner = cm
    if inner is not None:
        if mesh.contains((inner.X(), inner.Y(), inner.Z())):
            return True, 'centre'
        if diag > sliver_cm:
            return False, None
    for f in _faces(cell)[:6]:
        p0 = _point_on_face(f)
        if p0 is None:
            continue
        d = np.array(centre) - np.array([p0.X(), p0.Y(), p0.Z()])
        nn = float(np.linalg.norm(d)) or 1.0
        p1 = gp_Pnt(p0.X() + NUDGE_CM * d[0] / nn, p0.Y() + NUDGE_CM * d[1] / nn,
                    p0.Z() + NUDGE_CM * d[2] / nn)
        if inside(p1) and mesh.contains((p1.X(), p1.Y(), p1.Z())):
            return True, 'face'
    return False, None


def check_body(bd, ns):
    """Cells + classification for one BODIES entry, the way the Fusion
    runtime does it: tools that cannot be built are skipped and counted,
    the script's SKIP list is honoured (indices into the tools that were
    built, as in the runtime), cells are classified by _cell_is_material,
    and with nothing kept the cell closest to the mesh volume is. Returns
    a dict with the counts and the enclosed volume fraction — the outlook
    signal — or with 'error' set when the kernel could not compute the
    cells."""
    from collections import Counter
    mesh_vol = float(bd.get('volume', 0.0)) or 1e-12
    labels = [p[3] if len(p) > 3 else '?' for p in bd['inside']]
    res = {'name': bd.get('name', '?'), 'tool_faces': 0, 'tools_failed': 0,
           'tools_skipped': 0, 'cells': 0, 'kept_cells': 0, 'by_centre': 0,
           'by_face': 0, 'fallback': False, 'kept_volume': 0.0,
           'mesh_volume': mesh_vol, 'enclosed_pct': 0.0, 'unenclosed': list(labels),
           'n_probes': len(labels), 'probes_by_region': dict(Counter(labels)),
           'kept': [], 'cell_volumes': [], 'error': None}
    skip = {int(i) for i in (ns.get('SKIP') or [])}
    args = TopTools_ListOfShape()
    built = 0
    for s in bd['surfaces']:
        try:
            # a degenerate record (Fusion refuses the body; OCC may hand back
            # an invalid face) counts as a tool that could not be built
            fs = [f for f in make_tool(s, ns) if BRepCheck_Analyzer(f).IsValid()]
            if not fs:
                raise ValueError('no valid face')
        except Exception:
            res['tools_failed'] += 1
            continue
        k, built = built, built + 1
        if k in skip:
            res['tools_skipped'] += 1
            continue
        for f in fs:
            args.Append(f)
            res['tool_faces'] += 1
    if res['tool_faces'] == 0:
        res['error'] = 'no tool could be built'
        return res
    mv = BOPAlgo_MakerVolume()
    mv.SetArguments(args)
    mv.SetRunParallel(True)
    mv.Perform()
    if mv.HasErrors():
        res['error'] = 'MakerVolume reports errors (the cell computation failed)'
        return res
    cells = []
    ex = TopExp_Explorer(mv.Shape(), TopAbs_SOLID)
    while ex.More():
        cells.append(TopoDS.Solid_s(ex.Current()))
        ex.Next()
    res['cells'] = len(cells)
    pts = [gp_Pnt(*p[:3]) for p in bd['inside']]
    mesh = ns['_Mesh'](bd['mesh_v'], bd['mesh_f']) if bd.get('mesh_f') else None
    sliver_cm = float(ns.get('SLIVER_CM', 0.1))
    pt_hit = [False] * len(pts)
    kept_idx, vols = [], []
    for i, c in enumerate(cells):
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(c, g)
        vols.append(abs(g.Mass()))
        hit, how = _cell_is_material(c, g, pts, pt_hit, mesh, sliver_cm)
        if how == 'centre':
            res['by_centre'] += 1
        elif how == 'face':
            res['by_face'] += 1
        if hit:
            kept_idx.append(i)
    if not kept_idx and cells:
        kept_idx = [min(range(len(cells)), key=lambda i: abs(vols[i] - mesh_vol))]
        res['fallback'] = True
    kept_vol = sum(vols[i] for i in kept_idx)
    res.update(kept_cells=len(kept_idx), kept_volume=kept_vol,
               enclosed_pct=100.0 * kept_vol / mesh_vol,
               unenclosed=[labels[k] for k, h in enumerate(pt_hit) if not h],
               kept=[cells[i] for i in kept_idx],
               cell_volumes=sorted(vols, reverse=True))
    return res


def check_script(text, budget_s=None):
    """check_body over every body of an emitted script's text. With a time
    budget (seconds) the bodies past it are not checked (the first always
    is): their result has 'error' set and no cells."""
    ns = parse_script(text)
    t0 = time.time()
    out = []
    for bd in ns['BODIES']:
        if budget_s is not None and out and time.time() - t0 > budget_s:
            out.append({'name': bd.get('name', '?'), 'cells': 0, 'kept_cells': 0,
                        'enclosed_pct': 0.0, 'unenclosed': [], 'n_probes': len(bd['inside']),
                        'error': f'not checked: past the {budget_s:.0f} s budget'})
            continue
        out.append(check_body(bd, ns))
    return out


def outlook(results, dropped=()):
    """The Boundary Fill outlook from the dry-run results and the bodies
    the script left out (fusion_boundary_fill.skipped_bodies): {'ok': True
    | False | None (could not be checked), 'reason', 'enclosed_pct' (the
    worst body's)}. Every body is judged on its own. It fails when a
    region has none of its probe points in any cell (a feature the cells
    do not enclose at all) or when the kept cells enclose too little or
    too much of the mesh volume. A stray probe of a region that has
    others in cells is only mentioned: next to an approximate (fit-tol)
    tool a point 0.03 mm under the skin can lie outside the tool, within
    the tolerance the tool was accepted at."""
    from collections import Counter
    bad, notes, pcts, unchecked = [], [], [], []
    for r in results:
        name = r.get('name', '?')
        if r.get('error'):
            unchecked.append(f"{name}: {r['error']}")
            continue
        pct = float(r['enclosed_pct'])
        pcts.append(pct)
        missed = Counter(r['unenclosed'])
        totals = r.get('probes_by_region') or {}
        lost = sorted(lb for lb, n in missed.items() if n >= totals.get(lb, n))
        if lost:
            bad.append(f"{name}: no cell holds region{'s' if len(lost) > 1 else ''} "
                       f"{', '.join(lost)} (all their probe points lie outside every cell)")
        elif not (ENCLOSED_PCT_OK[0] <= pct <= ENCLOSED_PCT_OK[1]):
            bad.append(f"{name}: cells enclose {pct:.1f}% of the mesh volume")
        elif missed:
            notes.append(f"{name}: {sum(missed.values())} of {r['n_probes']} probe points lie "
                         f"in no cell, next to an approximate tool most likely")
        if r.get('tools_failed'):
            notes.append(f"{name}: {r['tools_failed']} tool(s) could not be built")
    for bi, n in dropped:
        bad.append(f"body {bi + 1}: {n} regions is above the script's limit, left out")
    worst = round(min(pcts), 2) if pcts else None
    if bad:
        return {'ok': False, 'enclosed_pct': worst,
                'reason': 'OCC dry run: ' + '; '.join(bad + notes)}
    if unchecked:
        return {'ok': None, 'enclosed_pct': worst,
                'reason': 'not checked: ' + '; '.join(unchecked + notes)}
    return {'ok': True, 'enclosed_pct': worst,
            'reason': f'OCC dry run: cells enclose {worst:.1f}% of the mesh volume'
                      + (' (' + '; '.join(notes) + ')' if notes else '')}
