"""C5 — face-group engine: segmentation, fits, face construction, pipeline
route. Every part here is a CAD export whose true face count and surface
types are known, so the assertions are on the artefact (as test_matrix)."""
import math
import numpy as np
import pytest

from . import synth


def _mesh(wp, tmp_path, name):
    from stl2prism.mesh_prep import load_and_prep_bodies
    p = synth.export(wp, tmp_path / f'{name}.stl')
    bodies, _, _ = load_and_prep_bodies(p, verbose=False)
    return bodies[0].mesh


# --- fits ---------------------------------------------------------------------

def test_fit_sphere_exact_and_noisy():
    from stl2prism.facegroups import fit_sphere
    rng = np.random.default_rng(0)
    d = rng.normal(size=(200, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    c, r = np.array([1.5, -2.0, 7.0]), 4.25
    f = fit_sphere(c + r * d)
    assert np.allclose(f['center'], c, atol=1e-9) and abs(f['r'] - r) < 1e-9
    assert f['resid'] < 1e-9
    f2 = fit_sphere(c + r * d + rng.normal(0, 0.01, (200, 3)))
    assert np.allclose(f2['center'], c, atol=0.01) and abs(f2['r'] - r) < 0.01
    # a hemisphere's worth of facets (the real case) is enough
    top = d[d[:, 2] > 0]
    f3 = fit_sphere(c + r * top)
    assert abs(f3['r'] - r) < 1e-9
    # degenerate input: fewer than 4 points
    assert fit_sphere((c + r * d)[:3]) is None


def test_fit_torus_exact_and_noisy():
    from stl2prism.features import fit_torus
    rng = np.random.default_rng(1)
    c, a, R, r = np.array([2.0, -1.0, 3.0]), np.array([0.0, 0.6, 0.8]), 6.5, 1.5
    b0 = np.cross([1.0, 0.0, 0.0], a)
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(a, b0)
    # a quarter of the tube (a fillet's worth) all the way round the axis
    u = rng.uniform(0, 2 * np.pi, 400)
    v = rng.uniform(np.pi, 1.5 * np.pi, 400)
    P = (c + np.outer((R + r * np.cos(v)) * np.cos(u), b0) + np.outer((R + r * np.cos(v)) * np.sin(u), b1)
         + np.outer(r * np.sin(v), a))
    # seed off by a few percent / degrees, as the band-chain seed is
    a0 = a + np.array([0.03, -0.02, 0.0])
    f = fit_torus(P, c + 0.2, a0, R * 1.05, r * 0.97)
    assert np.allclose(f['center'], c, atol=1e-7) and abs(abs(f['axis'] @ a) - 1) < 1e-9
    assert abs(f['R'] - R) < 1e-7 and abs(f['r'] - r) < 1e-7 and f['resid'] < 1e-7
    f2 = fit_torus(P + rng.normal(0, 0.01, P.shape), c + 0.2, a0, R * 1.05, r * 0.97)
    assert np.allclose(f2['center'], c, atol=0.01) and abs(f2['R'] - R) < 0.01 and abs(f2['r'] - r) < 0.01
    assert fit_torus(P[:5], c, a, R, r) is None


def test_fit_cone_exact_and_noisy():
    from stl2prism.features import fit_cone
    rng = np.random.default_rng(2)
    apex, a, al = np.array([1.0, 2.0, -3.0]), np.array([0.6, 0.0, 0.8]), np.radians(20.0)
    b0 = np.cross([0.0, 1.0, 0.0], a)
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(a, b0)
    # a 140-degree sector of a frustum (the real case: a partial taper patch)
    u = rng.uniform(0.3, 0.3 + np.radians(140), 300)
    h = rng.uniform(2.0, 8.0, 300)
    rad = h * np.tan(al)
    P = apex + np.outer(h, a) + np.outer(rad * np.cos(u), b0) + np.outer(rad * np.sin(u), b1)
    # seed off by a few degrees, as the chain seed is
    a0 = a + np.array([0.05, 0.03, 0.0])
    f = fit_cone(P, a0 / np.linalg.norm(a0), al * 1.15)
    assert abs(f['axis'] @ a - 1) < 1e-9 and abs(f['half_angle'] - al) < 1e-9
    assert np.allclose(f['apex'], apex, atol=1e-7) and f['resid'] < 1e-7
    f2 = fit_cone(P + rng.normal(0, 0.01, P.shape), a0 / np.linalg.norm(a0), al * 1.15)
    assert abs(f2['half_angle'] - al) < 0.01 and np.allclose(f2['apex'], apex, atol=0.05)
    assert fit_cone(P[:5], a, al) is None


# --- segmentation -------------------------------------------------------------

SEG_CASES = [
    ('drafted_block', synth.drafted_block, {'plane': 6}),
    ('cross3', synth.cross3, {'plane': 30}),
    ('csk_plate', synth.csk_plate, {'plane': 6, 'cylinder': 4, 'cone': 4}),
    ('sphere_boss', synth.sphere_boss, {'plane': 6, 'sphere': 1}),
    ('fillet_top', synth.fillet_top, {'plane': 6, 'cylinder': 4}),
    ('plate_holes_fillets', synth.plate_holes_fillets, {'plane': 6, 'cylinder': 8}),
    ('stepped_shaft', synth.stepped_shaft, {'plane': 4, 'cylinder': 3}),
    # rolling-ball blends: one torus each (concave boss base, convex hole mouth)
    ('boss_fillet', synth.boss_fillet, {'plane': 7, 'cylinder': 1, 'torus': 1}),
    ('filleted_hole', synth.filleted_hole, {'plane': 6, 'cylinder': 1, 'torus': 1}),
    ('boss_fillet_two', synth.boss_fillet_two, {'plane': 8, 'cylinder': 2, 'torus': 2}),
    # a pointed tip: one apex cone (closed at the tip by a degenerate edge)
    ('pencil', synth.pencil, {'plane': 1, 'cylinder': 1, 'cone': 1}),
]


@pytest.mark.parametrize('name,builder,expected', SEG_CASES, ids=[c[0] for c in SEG_CASES])
def test_segment_finds_the_cad_face_groups(tmp_path, name, builder, expected):
    from collections import Counter
    from stl2prism.facegroups import segment
    m = _mesh(builder(), tmp_path, name)
    regions = segment(m, fit_tol=0.08)
    got = Counter(r.kind for r in regions)
    assert dict(got) == expected, (name, dict(got))
    # every triangle in exactly one region
    covered = np.concatenate([r.faces for r in regions])
    assert len(covered) == len(m.faces) and len(np.unique(covered)) == len(m.faces)
    for r in regions:
        assert r.resid <= 0.08, (name, r.kind, r.resid)


def test_segment_does_not_let_a_plane_eat_fillet_strips(tmp_path):
    """The failure mode of vertex-only fitting: a wide plane plus its first
    fillet strip fit an exact, huge cylinder. The interior-residual test must
    keep the top plane a plane and the fillet a full quarter cylinder."""
    from stl2prism.facegroups import segment
    m = _mesh(synth.fillet_top(), tmp_path, 'fillet_top')
    regions = segment(m, fit_tol=0.08)
    top = [r for r in regions if r.kind == 'plane' and abs(r.params['normal'][2] - 1) < 1e-6]
    assert len(top) == 1 and abs(top[0].area - (40 - 4) * (30 - 4)) < 1e-6
    for r in regions:
        if r.kind == 'cylinder':
            assert abs(r.params['r'] - 2.0) < 1e-6


# --- face construction + full engine ------------------------------------------

ENGINE_CASES = [
    # name, builder, faces, STEP surface histogram
    ('drafted_block', synth.drafted_block, 6, {'PLANE': 6}),
    ('cross3', synth.cross3, 30, {'PLANE': 30}),
    ('csk_plate', synth.csk_plate, 14, {'PLANE': 6, 'CYLINDRICAL_SURFACE': 4, 'CONICAL_SURFACE': 4}),
    ('sphere_boss', synth.sphere_boss, 7, {'PLANE': 6, 'SPHERICAL_SURFACE': 1}),
    ('fillet_top', synth.fillet_top, 10, {'PLANE': 6, 'CYLINDRICAL_SURFACE': 4}),
    ('plate_holes_fillets', synth.plate_holes_fillets, 14, {'PLANE': 6, 'CYLINDRICAL_SURFACE': 8}),
    ('boss_fillet', synth.boss_fillet, 9, {'PLANE': 7, 'CYLINDRICAL_SURFACE': 1, 'TOROIDAL_SURFACE': 1}),
    ('filleted_hole', synth.filleted_hole, 8, {'PLANE': 6, 'CYLINDRICAL_SURFACE': 1, 'TOROIDAL_SURFACE': 1}),
    ('boss_fillet_two', synth.boss_fillet_two, 12, {'PLANE': 8, 'CYLINDRICAL_SURFACE': 2, 'TOROIDAL_SURFACE': 2}),
    ('pencil', synth.pencil, 3, {'PLANE': 1, 'CYLINDRICAL_SURFACE': 1, 'CONICAL_SURFACE': 1}),
]


@pytest.mark.parametrize('name,builder,faces,kinds', ENGINE_CASES, ids=[c[0] for c in ENGINE_CASES])
def test_engine_builds_the_cad_solid(tmp_path, name, builder, faces, kinds):
    from stl2prism import facegroups
    from stl2prism.pipeline import validate, gate_values
    from stl2prism.rebuild import write_step
    m = _mesh(builder(), tmp_path, name)
    shape, stats = facegroups.convert(m, tol=0.08)
    assert stats['unfitted_regions'] == 0, stats['fallbacks']
    assert stats['faces_out'] == faces, (name, stats['faces_out'], stats['faces_by_kind'])
    out = str(tmp_path / f'{name}.step')
    write_step([shape], out, names=[name])
    n, got = synth.step_faces(out)
    assert n == faces and got == kinds, (name, n, got)
    ri = synth.reimport(out)
    assert ri['solids'] == 1 and ri['valid'] and ri['naked_edges'] == 0, ri
    met = validate(shape, m)
    p95, mx = gate_values(met)
    assert p95 <= 0.25 and mx <= 0.26 and met['vol_err_pct'] < 0.5, met


def test_engine_reports_fitted_radius_exactly(tmp_path):
    """Fits are on vertices, which lie on the true surface — the radius must
    come out exact, not sagitta-biased low."""
    from OCP.BRep import BRep_Tool
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Sphere
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from stl2prism import facegroups
    m = _mesh(synth.sphere_boss(), tmp_path, 'sphere_boss')
    shape, _ = facegroups.convert(m, tol=0.08)
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    radii = []
    while ex.More():
        ad = GeomAdaptor_Surface(BRep_Tool.Surface_s(TopoDS.Face_s(ex.Current())))
        if ad.GetType() == GeomAbs_Sphere:
            radii.append(ad.Sphere().Radius())
        ex.Next()
    assert len(radii) == 1 and abs(radii[0] - 8.0) < 1e-6, radii


def test_engine_reports_torus_radii_exactly(tmp_path):
    from OCP.BRep import BRep_Tool
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Torus
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from stl2prism import facegroups
    m = _mesh(synth.boss_fillet(), tmp_path, 'boss_fillet')
    shape, stats = facegroups.convert(m, tol=0.08)
    assert stats['torus_merges'] == 1
    tori = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        ad = GeomAdaptor_Surface(BRep_Tool.Surface_s(TopoDS.Face_s(ex.Current())))
        if ad.GetType() == GeomAbs_Torus:
            t = ad.Torus()
            tori.append((t.MajorRadius(), t.MinorRadius(), t.Location().Z()))
        ex.Next()
    assert len(tori) == 1, tori
    R, r, z = tori[0]
    assert abs(R - 6.5) < 1e-6 and abs(r - 1.5) < 1e-6 and abs(z - 4.5) < 1e-6, tori
    # the blend sits on the material side: concave, and exported as such
    t = [g for g in stats['export']['regions'] if g['kind'] == 'torus'][0]
    assert t['concave'] and t['R'] == 6.5 and t['r'] == 1.5 and t['a0'] is None
    assert abs(t['v0'] - np.pi) < 0.02 and abs(t['v1'] - 1.5 * np.pi) < 0.02, t


def test_engine_reports_cone_angle_exactly(tmp_path):
    """The pencil's tip: one conical face closed at the apex by a degenerate
    edge, with the exact half-angle and apex of the CAD part."""
    from OCP.BRep import BRep_Tool
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cone
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from stl2prism import facegroups
    m = _mesh(synth.pencil(), tmp_path, 'pencil')
    shape, stats = facegroups.convert(m, tol=0.08)
    cones = []
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    while ex.More():
        ad = GeomAdaptor_Surface(BRep_Tool.Surface_s(TopoDS.Face_s(ex.Current())))
        if ad.GetType() == GeomAbs_Cone:
            c = ad.Cone()
            a = c.Apex()
            cones.append((c.SemiAngle(), a.X(), a.Y(), a.Z()))
        ex.Next()
    assert len(cones) == 1, cones
    semi, ax_, ay_, az_ = cones[0]
    assert abs(semi - math.atan2(4.0, 6.0)) < 1e-6
    assert abs(ax_) < 1e-4 and abs(ay_) < 1e-4 and abs(az_ - 16.0) < 1e-4, cones
    # the export record reaches the apex (t0 = 0) and runs all the way round
    c = [g for g in stats['export']['regions'] if g['kind'] == 'cone'][0]
    assert c['t0'] == 0.0 and c['a0'] is None and not c['concave'], c


def test_taper_pin_grows_two_exact_cones():
    """Two straight taper slopes: with Taubin-centred cone seeding, growth
    alone fits exactly two cones — no bands, no merge needed."""
    from collections import Counter
    from stl2prism import facegroups
    m = synth.taper_pin_mesh()
    regs = facegroups.segment(m, fit_tol=0.08)
    assert Counter(r.kind for r in regs) == {'plane': 2, 'cone': 2}
    halves = sorted(np.degrees(r.params['half_angle']) for r in regs if r.kind == 'cone')
    assert abs(halves[0] - np.degrees(np.arctan(0.05))) < 0.05
    assert abs(halves[1] - np.degrees(np.arctan(0.22))) < 0.05
    assert all(r.resid <= 0.08 for r in regs)


def test_spiral_taper_sphere_bands_merge_to_cones():
    """A curved taper on ring-aligned tessellation grows as thin 'sphere'
    bands (any two vertex rings lie exactly on some sphere): the cone-chain
    merge must chain those ring-like spheres and refit them as cones."""
    from collections import Counter
    from stl2prism import facegroups
    m = synth.spiral_taper_mesh()
    regs = facegroups.segment(m, fit_tol=0.08)
    kinds = Counter(r.kind for r in regs)
    assert kinds.get('sphere', 0) == 0 and kinds.get('cone', 0) >= 2, kinds
    assert len(regs) <= 10 and facegroups.segment.last_stats['cone_merges'] >= 2
    assert all(r.resid <= 0.08 for r in regs)


def test_pinched_region_is_peeled_into_clean_pieces():
    """An annulus whose hole touches the rim at one vertex has a pinched
    boundary (one vertex with two outgoing boundary edges): _group_loops
    refuses it, and before v0.3.4 the whole region fell to triangles.
    _split_pinched must peel the faces at the pinch and return pieces whose
    loops all group cleanly."""
    import trimesh
    from stl2prism.facegroups import _split_pinched
    from stl2prism.rebuild import _group_loops
    # a proper annulus strip whose inner ring reuses rim vertex 0: the hole
    # touches the rim there, degenerate triangles at the weld are dropped
    n = 12
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([4 * np.cos(a), 4 * np.sin(a), np.zeros(n)])
    inner = np.column_stack([1.5 * np.cos(a), 1.5 * np.sin(a), np.zeros(n)])
    V = np.vstack([outer, inner])
    idx_in = [0] + list(range(n + 1, 2 * n))          # inner ring, v0 welded
    tri = []
    for i in range(n):
        j = (i + 1) % n
        for t in ([i, j, idx_in[i]], [j, idx_in[j], idx_in[i]]):
            if len(set(t)) == 3:
                tri.append(t)
    m = trimesh.Trimesh(V, np.asarray(tri), process=False)
    faces = np.arange(len(m.faces))
    # deterministic construction: if _group_loops ever accepts this boundary,
    # build_solid's repair gate (the same refusal) stops firing — fail loudly
    assert _group_loops(m, faces) is None, 'construction must pinch'
    pieces = _split_pinched(m, faces)
    assert pieces is not None and len(pieces) >= 2
    assert sum(len(p) for p in pieces) == len(faces)
    for p in pieces:
        assert _group_loops(m, p) is not None


def test_two_lobes_touching_at_a_vertex_are_split_into_both():
    """A bow-tie region (two planar patches sharing one vertex) pinches at
    that vertex; the repair must hand back one clean piece per lobe plus
    the peeled fans, not a two-component 'core' that _region_face then
    builds as a plate with a hole."""
    import trimesh
    from stl2prism.facegroups import _split_pinched
    from stl2prism.rebuild import _group_loops

    def grid(x0, y0):
        V = [(x0 + i, y0 + j, 0.0) for j in range(3) for i in range(3)]
        T = []
        for j in range(2):
            for i in range(2):
                a = j * 3 + i
                T += [[a, a + 1, a + 4], [a, a + 4, a + 3]]
        return np.asarray(V, float), np.asarray(T)
    VA, TA = grid(0, 0)
    VB, TB = grid(2, 2)                       # B's corner 0 is A's corner 8
    V = np.vstack([VA, VB[1:]])
    remap = np.array([8] + list(range(9, 17)))
    T = np.vstack([TA, remap[TB]])
    m = trimesh.Trimesh(V, T, process=False)
    faces = np.arange(len(m.faces))
    assert _group_loops(m, faces) is None, 'construction must pinch'
    pieces = _split_pinched(m, faces)
    assert pieces is not None and len(pieces) >= 4, pieces
    assert sum(len(p) for p in pieces) == len(faces)
    for p in pieces:
        assert _group_loops(m, p) is not None
    # every piece lies in one lobe
    for p in pieces:
        xs = m.triangles_center[p][:, 0]
        assert (xs <= 2.0).all() or (xs >= 2.0).all()


def test_blend_chain_is_walked_from_an_end():
    """consolidate_blends tries runs of consecutive bands: the component
    must be in chain order (from an end band), not BFS order from the
    smallest id, or a mid-chain start hides the real sub-chains."""
    from stl2prism.facegroups import _chain_order, _connected
    # a-b-c-d-e with ids c=0, b=1, d=2, a=3, e=4
    adj = {3: {1}, 1: {3, 0}, 0: {1, 2}, 2: {0, 4}, 4: {2}}
    order = _chain_order([0, 1, 2, 3, 4], adj)
    assert order in ([3, 1, 0, 2, 4], [4, 2, 0, 1, 3]), order
    assert _connected([3, 1, 0], adj) and _connected([0], adj)
    assert not _connected([1, 2], adj) and not _connected([3, 4], adj)
    ring = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {2, 0}}
    assert _chain_order([0, 1, 2, 3], ring) == [0, 1, 2, 3]       # no end: as is
    branch = {0: {1, 2, 3}, 1: {0}, 2: {0}, 3: {0}}
    assert _chain_order([0, 1, 2, 3], branch) == [0, 1, 2, 3]


def test_rounded_box_corners_stay_spheres(tmp_path):
    """Three fillets meeting at a box corner blend as a sphere, not a torus:
    the band-chain merge must leave them alone (straight fillets are not
    bands, and the corner has no band chain)."""
    from stl2prism import facegroups
    m = _mesh(synth.rounded_box(), tmp_path, 'rounded_box')
    shape, stats = facegroups.convert(m, tol=0.08)
    assert stats['torus_merges'] == 0 and stats['by_type'].get('torus', 0) == 0
    assert stats['by_type']['sphere'] == 8 and stats['by_type']['plane'] == 6
    assert stats['faces_by_kind']['sphere'] == 8 and stats['unfitted_regions'] == 0


# --- guards -------------------------------------------------------------------

def test_engine_steps_aside_on_scan_like_meshes(tmp_path):
    """No coplanar structure = tens of thousands of seeds: the engine must
    return quickly and let the faceted route handle it."""
    import time
    from stl2prism.mesh_prep import load_and_prep_bodies
    from stl2prism import facegroups
    p = synth.scan_like_cube(str(tmp_path / 'scan.stl'))
    bodies, _, _ = load_and_prep_bodies(p, verbose=False)
    t = time.time()
    shape, stats = facegroups.convert(bodies[0].mesh, tol=0.08)
    assert shape is None and stats.get('skipped') == 'envelope'
    assert time.time() - t < 20


def test_cone_tip_loops_are_refused_before_occ():
    """A boundary loop passing next to a cone's axis (a drill tip) is not a
    usable trimmed face — ShapeFix_Face crashes the process on it, so it
    has to be caught beforehand and the region emitted as triangles."""
    from stl2prism.facegroups import _loops_param_ok
    ax = np.array([0.0, 0.0, 1.0])
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    ring = np.column_stack([np.cos(ang), np.sin(ang), np.ones_like(ang)])
    # a good frustum: two rings -> ok
    ring2 = np.column_stack([2 * np.cos(ang), 2 * np.sin(ang), 2 * np.ones_like(ang)])
    vpos = np.vstack([ring, ring2])
    loops = [list(range(12)), list(range(12, 24))[::-1]]
    assert _loops_param_ok('cone', loops, vpos, np.zeros(3), ax, 2.0)
    # a single ring around the axis: the region contains the apex -> refused
    # unless the caller says the apex cap is wanted (one ring + degenerate tip)
    assert not _loops_param_ok('cone', [list(range(12))], vpos, np.zeros(3), ax, 2.0)
    assert _loops_param_ok('cone', [list(range(12))], vpos, np.zeros(3), ax, 2.0,
                           apex_ok=True)
    # a vertex on the axis -> refuse
    vpos2 = np.vstack([ring, [[0.0, 0.0, 1.0]]])
    assert not _loops_param_ok('cylinder', [list(range(6)) + [12]], vpos2, np.zeros(3), ax, 1.0)
    # a sphere cap around one pole is fine
    assert _loops_param_ok('sphere', [list(range(12))], vpos, np.zeros(3), ax, math.sqrt(2))


def test_torus_loops_are_checked_in_both_periodic_directions():
    """A torus is periodic round the axis (u) and round the tube (v): a
    blend is a ring about the axis (two loops winding in u), a pipe elbow a
    ring about the tube (two loops winding in v), or a patch; a loop that
    winds both ways, three rings, or a chord jumping across the tube are
    refused before OCC sees them."""
    from stl2prism.facegroups import _loops_param_ok
    R, r = 5.0, 1.5
    c, ax = np.zeros(3), np.array([0.0, 0.0, 1.0])

    def tp(u, v):
        return np.array([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])

    us = np.linspace(0, 2 * np.pi, 25)[:-1]
    vs = np.linspace(0, 2 * np.pi, 13)[:-1]
    ring_u = lambda v: np.array([tp(u, v) for u in us])          # noqa: E731
    ring_v = lambda u: np.array([tp(u, v) for v in vs])          # noqa: E731
    # fillet band: inner-lower quadrant, two rings about the axis
    vpos = np.vstack([ring_u(np.pi), ring_u(1.5 * np.pi)])
    assert _loops_param_ok('torus', [list(range(24)), list(range(24, 48))[::-1]], vpos, c, ax, R, r_minor=r)
    # the same on the outer equator (v = 0 seam) is fine too
    vpos = np.vstack([ring_u(0.0), ring_u(0.5 * np.pi)])
    assert _loops_param_ok('torus', [list(range(24)), list(range(24, 48))[::-1]], vpos, c, ax, R, r_minor=r)
    # one ring alone: a ring about the axis needs its partner
    assert not _loops_param_ok('torus', [list(range(24))], vpos, c, ax, R, r_minor=r)
    # pipe elbow: two rings about the tube, 90 deg apart in u
    vpos = np.vstack([ring_v(0.0), ring_v(0.5 * np.pi)])
    assert _loops_param_ok('torus', [list(range(12)), list(range(12, 24))[::-1]], vpos, c, ax, R, r_minor=r)
    # three rings about the axis -> refuse
    vpos = np.vstack([ring_u(np.pi), ring_u(1.25 * np.pi), ring_u(1.5 * np.pi)])
    assert not _loops_param_ok('torus', [list(range(24)), list(range(24, 48)), list(range(48, 72))],
                               vpos, c, ax, R, r_minor=r)
    # a patch (60 deg of u, a quarter of v): one closed loop
    uu = np.linspace(0, np.radians(60), 7)
    vv = np.linspace(np.pi, 1.5 * np.pi, 5)
    patch = np.vstack([[tp(u, vv[0]) for u in uu], [tp(uu[-1], v) for v in vv[1:]],
                       [tp(u, vv[-1]) for u in uu[-2::-1]], [tp(uu[0], v) for v in vv[-2:0:-1]]])
    assert _loops_param_ok('torus', [list(range(len(patch)))], patch, c, ax, R, r_minor=r)
    # a loop that winds round the axis AND round the tube (a Villarceau-like
    # helix closing on itself) is not a trimming loop
    t = np.linspace(0, 2 * np.pi, 49)[:-1]
    helix = np.array([tp(x, x) for x in t])
    assert not _loops_param_ok('torus', [list(range(48))], helix, c, ax, R, r_minor=r)
    # a chord jumping 180 deg across the tube -> refuse
    bad = np.vstack([patch[:5], [tp(np.radians(40), 0.0)], patch[5:]])
    assert not _loops_param_ok('torus', [list(range(len(bad)))], bad, c, ax, R, r_minor=r)


def test_regularise_snaps_within_uncertainty_and_reverts_the_rest(tmp_path):
    """Two parallel top faces 0.03 mm apart: a real step, not one plane.
    The coplanar snap must leave the tiny pad where its vertices are —
    either by never proposing the merge (offset beyond the uncertainty the
    revert would accept, the current behaviour) or by reverting it."""
    import cadquery as cq
    from stl2prism.facegroups import segment, regularise
    wp = (cq.Workplane('XY').box(100, 100, 8).faces('>Z').workplane()
          .rect(6, 6).extrude(0.03))
    m = _mesh(wp, tmp_path, 'thin_pad')
    regions = segment(m, fit_tol=0.08)
    pad = [r for r in regions if r.kind == 'plane' and abs(r.area - 36.0) < 1e-6
           and abs(r.params['normal'][2] - 1) < 1e-6]
    assert len(pad) == 1
    before = pad[0].params['point'].copy()
    stats = regularise(regions, m, fit_tol=0.08)
    assert np.allclose(pad[0].params['point'], before)
    assert abs(float(pad[0].params['normal'] @ pad[0].params['point']) - 4.03) < 1e-3
    for r in regions:
        assert r.resid_v <= 0.01, (r.kind, r.resid_v)


# --- pipeline route -----------------------------------------------------------

def test_pipeline_routes_to_facegroup_before_hybrid(tmp_path):
    from stl2prism.pipeline import run
    p = synth.export(synth.sphere_boss(), tmp_path / 'sb.stl')
    out = str(tmp_path / 'sb.step')
    r = run(p, out, verbose=False)
    assert r['mode'] == 'facegroup', r['metrics']
    assert r['metrics']['fgroup']['by_type'] == {'plane': 6, 'sphere': 1}
    assert r['metrics']['faces_out'] == 7
    assert r['script'] is None                 # no extrusion structure to script
    n, kinds = synth.step_faces(out)
    assert kinds.get('SPHERICAL_SURFACE') == 1


def test_pipeline_flag_disables_the_engine(tmp_path):
    from stl2prism.pipeline import run
    p = synth.export(synth.fillet_top(), tmp_path / 'ft.stl')
    r = run(p, str(tmp_path / 'ft.step'), verbose=False, face_groups=False)
    assert r['mode'] == 'faceted'
    r2 = run(p, str(tmp_path / 'ft2.step'), verbose=False)
    assert r2['mode'] == 'facegroup' and r2['metrics']['faces_out'] == 10


def test_cli_accepts_no_face_groups(tmp_path):
    import subprocess, sys
    p = synth.export(synth.fillet_top(), tmp_path / 'ft.stl')
    out = tmp_path / 'ft.step'
    res = subprocess.run([sys.executable, '-m', 'stl2prism.pipeline', p, str(out),
                          '--no-face-groups', '--quiet'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    n, kinds = synth.step_faces(str(out))
    assert 'CYLINDRICAL_SURFACE' not in kinds        # faceted: triangles only
