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


# --- segmentation -------------------------------------------------------------

SEG_CASES = [
    ('drafted_block', synth.drafted_block, {'plane': 6}),
    ('cross3', synth.cross3, {'plane': 30}),
    ('csk_plate', synth.csk_plate, {'plane': 6, 'cylinder': 4, 'cone': 4}),
    ('sphere_boss', synth.sphere_boss, {'plane': 6, 'sphere': 1}),
    ('fillet_top', synth.fillet_top, {'plane': 6, 'cylinder': 4}),
    ('plate_holes_fillets', synth.plate_holes_fillets, {'plane': 6, 'cylinder': 8}),
    ('stepped_shaft', synth.stepped_shaft, {'plane': 4, 'cylinder': 3}),
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
    # a single ring around the axis: the region contains the apex -> refuse
    assert not _loops_param_ok('cone', [list(range(12))], vpos, np.zeros(3), ax, 2.0)
    # a vertex on the axis -> refuse
    vpos2 = np.vstack([ring, [[0.0, 0.0, 1.0]]])
    assert not _loops_param_ok('cylinder', [list(range(6)) + [12]], vpos2, np.zeros(3), ax, 1.0)
    # a sphere cap around one pole is fine
    assert _loops_param_ok('sphere', [list(range(12))], vpos, np.zeros(3), ax, math.sqrt(2))


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
