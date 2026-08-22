"""Synthetic CAD parts for tests, generated with CadQuery on the fly.

Every part is a clean CAD export (the tool's home turf) with a known face
count, so tests can assert on the artefact: mode, ADVANCED_FACE count,
surface types, deviation. Nothing here depends on the git-ignored samples.
"""
import math
import numpy as np
import cadquery as cq
import trimesh


def export(wp, path, tol=0.02, ang=0.2):
    cq.exporters.export(wp, str(path), tolerance=tol, angularTolerance=ang)
    return str(path)


def plate_holes():
    """60x40x6 plate, 4 x D5 through holes -> 10 faces."""
    return (cq.Workplane('XY').box(60, 40, 6).faces('>Z').workplane()
            .rect(40, 25, forConstruction=True).vertices().hole(5))


def plate_holes_fillets():
    """plate_holes with r5 fillets on the vertical edges -> 14 faces."""
    return plate_holes().edges('|Z').fillet(5)


def rounded_rect():
    """40x30x10 box, r4 fillets on vertical edges -> 12 faces (8 sides)."""
    return cq.Workplane('XY').box(40, 30, 10).edges('|Z').fillet(4)


def obround():
    """slot2D 40x12 extruded 8, minus a 30x4 slot -> 12 faces."""
    return (cq.Workplane('XY').slot2D(40, 12).extrude(8)
            .faces('>Z').workplane().slot2D(30, 4).cutThruAll())


def hollow_cube():
    """40 cube with a centred 20 cube void -> 12 faces, one solid, 56000 mm3."""
    return cq.Workplane('XY').box(40, 40, 40).cut(cq.Workplane('XY').box(20, 20, 20))


def cross_blind():
    """40x30x20 block; D6 through hole along Y at x=5; D4 blind hole along X
    from x=12 to the +X face -> prismatic, 9 faces."""
    blk = cq.Workplane('XY').box(40, 30, 20)
    blk = blk.cut(cq.Workplane('XZ').center(5, 0).circle(3).extrude(40, both=True))
    blk = blk.cut(cq.Workplane('YZ', origin=(12, 0, 0)).center(-8, 3).circle(2).extrude(20))
    return blk


def stepped_shaft():
    return (cq.Workplane('XY').circle(10).extrude(20).faces('>Z').workplane()
            .circle(7).extrude(15).faces('>Z').workplane().circle(4).extrude(10))


def slot_hex():
    return (cq.Workplane('XY').box(60, 30, 6).faces('>Z').workplane().slot2D(30, 8).cutThruAll()
            .faces('>Z').workplane().center(0, 10).polygon(6, 8).cutBlind(-3))


def cross3():
    return (cq.Workplane('XY').box(60, 10, 10).union(cq.Workplane('XY').box(10, 60, 10))
            .union(cq.Workplane('XY').box(10, 10, 60)))


def drafted_block(deg=2.0, h=20.0):
    d = 2 * h * math.tan(math.radians(deg))
    return cq.Workplane('XY').rect(40, 30).workplane(offset=h).rect(40 - d, 30 - d).loft()


def chamfer_top():
    return cq.Workplane('XY').box(30, 20, 10).edges('>Z').chamfer(1)


def csk_plate():
    return (cq.Workplane('XY').box(60, 60, 8).faces('>Z').workplane()
            .rect(40, 40, forConstruction=True).vertices().cskHole(5, 10, 90))


def sphere_boss():
    return cq.Workplane('XY').box(30, 30, 10).union(cq.Workplane('XY').sphere(8).translate((0, 0, 5)))


def fillet_top():
    return cq.Workplane('XY').box(40, 30, 10).edges('>Z').fillet(2)


def small_step_inside():
    """Plate 100x100x8 with a 6 mm square, 0.4 mm-high pad in the middle of
    the top: too small (0.036 % of area) for the old level filter."""
    return (cq.Workplane('XY').box(100, 100, 8).faces('>Z').workplane()
            .rect(6, 6).extrude(0.4))


def flat_gentle_arc(ledge=0.0):
    """Profile in YZ: a 5 mm flat top, then 20 degrees of an r=12 arc, then
    a tangent slope, walls and floor; extruded 8 along X -> 8 faces. One
    circle through the flat AND the arc stays within 0.08 mm, which is how
    a flat mesh face used to come out as an r=50..70 cylinder. With
    `ledge` a 6x5 pocket that deep is cut into the +X face (-> 13 faces);
    a 0.15 mm ledge used to be merged into the neighbouring level."""
    a = math.radians(20)
    r = 12.0
    p1 = (5.0, 0.0)
    p2 = (5 + r * math.sin(a), -r * (1 - math.cos(a)))
    t = (math.cos(a), -math.sin(a))
    p3 = (p2[0] + 6 * t[0], p2[1] + 6 * t[1])
    wp = (cq.Workplane('YZ').moveTo(0, 0).lineTo(*p1).radiusArc(p2, r).lineTo(*p3)
          .lineTo(p3[0], -10).lineTo(0, -10).close().extrude(8))
    if ledge:
        wp = wp.faces('>X').workplane().center(4, -6).rect(6, 5).cutBlind(-ledge)
    return wp


def rotate(wp, ax, deg):
    return wp.rotate((0, 0, 0), ax, deg)


def crack_plate(path):
    """plate_holes exported, then the top-face vertices offset by 1e-5 so
    exact merging cannot close the crack."""
    m = trimesh.load(export(plate_holes(), path), force='mesh')
    v = m.vertices.copy()
    f = m.faces.copy()
    top = m.triangles_center[:, 2] > 2.9
    newv = []
    for i in np.where(top)[0]:
        for k in range(3):
            newv.append(v[f[i, k]] + np.array([0, 0, 1e-5]))
            f[i, k] = len(v) + len(newv) - 1
    v = np.vstack([v, np.array(newv)])
    trimesh.Trimesh(v, f, process=False).export(path)
    return path


def jitter(path_in, path_out, sigma=0.01, max_edge=0.5, seed=0):
    """A noisy 'scan' of a CAD part: uniform remesh (scanner-like triangle
    size) + gaussian jitter."""
    m = trimesh.load(path_in, force='mesh')
    m = m.subdivide_to_size(max_edge=max_edge, max_iter=12)
    v = m.vertices + np.random.default_rng(seed).normal(0, sigma, m.vertices.shape)
    trimesh.Trimesh(v, m.faces).export(path_out)
    return path_out


def scan_like_cube(path, size=40.0, subdivisions=6, sigma=0.02, seed=0):
    """Dense, uniformly triangulated cube with scanner-like noise: ~49k
    faces, ~0.6 mm edges, mean dihedral a few degrees, no coplanar pairs."""
    m = trimesh.creation.box((size, size, size))
    for _ in range(subdivisions):
        m = m.subdivide()
    v = m.vertices + np.random.default_rng(seed).normal(0, sigma, m.vertices.shape)
    trimesh.Trimesh(v, m.faces).export(path)
    return path


def step_faces(path):
    """ADVANCED_FACE count and surface-type histogram from the STEP text."""
    import re
    kinds = {}
    faces = 0
    pat = re.compile(r'=\s*([A-Z_0-9]+)\s*\(')
    with open(path, errors='replace') as fh:
        for line in fh:
            mm = pat.search(line)
            if not mm:
                continue
            k = mm.group(1)
            if k == 'ADVANCED_FACE':
                faces += 1
            elif k in ('PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE',
                       'SPHERICAL_SURFACE', 'TOROIDAL_SURFACE') or k.startswith('B_SPLINE_SURFACE'):
                kinds[k] = kinds.get(k, 0) + 1
    return faces, kinds


def reimport(path):
    """Round-trip through OCC: solids, faces, naked edges, validity, volume."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from stl2prism.rebuild import _naked_edges
    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    s = r.OneShape()

    def n(kind):
        e = TopExp_Explorer(s, kind)
        c = 0
        while e.More():
            c += 1
            e.Next()
        return c
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(s, g)
    return {'solids': n(TopAbs_SOLID), 'faces': n(TopAbs_FACE),
            'naked_edges': _naked_edges(s), 'valid': BRepCheck_Analyzer(s).IsValid(),
            'volume': abs(g.Mass())}


def face_areas(path):
    """Areas of every face in the STEP (for sliver checks)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    s = r.OneShape()
    out = []
    e = TopExp_Explorer(s, TopAbs_FACE)
    while e.More():
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(e.Current(), g)
        out.append(g.Mass())
        e.Next()
    return out
