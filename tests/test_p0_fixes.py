"""P0 correctness fixes from the 2026-08 review (F1, F3, F4, F7, F8, A6-A8).

Every test asserts on the produced artefact (STEP round-trip), not just on
the reported mode.
"""
import os
import pytest
import numpy as np
import trimesh

from . import synth


def _run(wp_or_path, tmp_path, name, **kw):
    from stl2prism.pipeline import run
    src = wp_or_path if isinstance(wp_or_path, str) else synth.export(wp_or_path, tmp_path / f'{name}.stl')
    out = str(tmp_path / f'{name}.step')
    r = run(src, out, verbose=False, **kw)
    return r, out


# --- F1: voids ---------------------------------------------------------------

def test_hollow_cube_is_one_solid_with_void(tmp_path):
    r, out = _run(synth.hollow_cube(), tmp_path, 'hollow')
    assert r['mode'] == 'prismatic'
    assert r['n_bodies'] == 1                          # void attached, not a body
    assert r['metrics']['voids'] == 1
    got = synth.reimport(out)
    assert got['solids'] == 1
    assert got['volume'] == pytest.approx(40 ** 3 - 20 ** 3, rel=1e-6)
    assert r['metrics']['vol_err_pct'] < 0.01


def test_nest_shells_depth_parity():
    from stl2prism.mesh_prep import nest_shells
    outer = trimesh.creation.box((40, 40, 40))
    void = trimesh.creation.box((20, 20, 20))
    island = trimesh.creation.box((5, 5, 5))          # part inside the cavity
    other = trimesh.creation.box((10, 10, 10)); other.apply_translation((100, 0, 0))
    bodies = nest_shells(sorted([outer, void, island, other],
                                key=lambda p: len(p.faces), reverse=True))
    # box meshes all have 12 faces; sort is stable so order is as given
    vols = sorted(round(b.mesh.volume) for b in bodies)
    assert vols == [125, 1000, 64000]                  # island, other, outer
    hollow = [b for b in bodies if round(b.mesh.volume) == 64000][0]
    assert len(hollow.voids) == 1 and round(hollow.voids[0].volume) == 8000
    assert round(hollow.volume()) == 56000


# --- F3: blind holes -----------------------------------------------------------

def test_blind_cross_hole_not_overcut(tmp_path):
    r, out = _run(synth.cross_blind(), tmp_path, 'blind')
    assert r['mode'] == 'prismatic', r['metrics']
    faces, kinds = synth.step_faces(out)
    assert kinds.get('CYLINDRICAL_SURFACE', 0) == 2
    assert faces == 9
    assert r['metrics']['dev_max'] < 0.05
    got = synth.reimport(out)
    exp = 40 * 30 * 20 - np.pi * 9 * 30 - np.pi * 4 * 8
    assert got['volume'] == pytest.approx(exp, rel=2e-3)


def test_hole_end_exits_probe():
    from stl2prism.features import hole_end_exits
    m = trimesh.creation.box((40, 30, 20))
    c = {'axis': np.array([1.0, 0, 0]), 'basis': (np.array([0, 0, -1.0]), np.array([0, 1.0, 0])),
         'center2': np.array([-3.0, 0.0]), 'r': 2.0, 'h0': 12.0, 'h1': 20.0}
    assert hole_end_exits(m, c, 'h1') is True          # breaks out at x=20
    assert hole_end_exits(m, c, 'h0') is False         # floor at x=12 is inside material


# --- F4: axis snap ------------------------------------------------------------

@pytest.mark.parametrize('ax,deg', [((1, 0, 0), 3.0), ((0, 1, 0), 3.0), ((0, 0, 1), 3.0)])
def test_slightly_tilted_plate_keeps_true_axis(tmp_path, ax, deg):
    r, out = _run(synth.rotate(synth.plate_holes(), ax, deg), tmp_path, f'tilt{ax}{deg}')
    assert r['mode'] == 'prismatic', r['metrics']
    assert synth.step_faces(out)[0] == 10
    assert r['metrics']['dev_max'] < 0.05


def test_dominant_axis_offers_snapped_and_raw():
    from stl2prism.extrusion import dominant_axis
    m = trimesh.load(synth.export(synth.rotate(synth.plate_holes(), (1, 0, 0), 3.0),
                                  '/tmp/_tilt_probe.stl'), force='mesh')
    axes = [a for _, a in dominant_axis(m)]
    has_snapped = any(abs(abs(a[2]) - 1) < 1e-9 for a in axes)
    has_raw = any(abs(abs(a[2]) - np.cos(np.radians(3))) < 1e-3 for a in axes)
    assert has_snapped and has_raw


# --- F7: deterministic, symmetric validation ------------------------------------

def test_validation_is_deterministic(tmp_path):
    p = synth.export(synth.plate_holes_fillets(), tmp_path / 'p.stl')
    from stl2prism.pipeline import run
    r1 = run(p, str(tmp_path / 'a.step'), verbose=False)
    r2 = run(p, str(tmp_path / 'b.step'), verbose=False)
    for k in ('dev_p95', 'dev_max', 'rev_dev_p95', 'rev_dev_max', 'vol_err_pct'):
        assert r1['metrics'][k] == r2['metrics'][k]


def test_reverse_deviation_catches_added_material():
    """A solid that is the mesh plus an extra lump: forward deviation is ~0
    (every mesh point lies on the solid) but the reverse direction sees it."""
    import cadquery as cq
    from stl2prism.pipeline import validate, gate_values
    mesh = trimesh.creation.box((20, 10, 5))
    solid = cq.Workplane('XY').box(20, 10, 5).union(
        cq.Workplane('XY').box(4, 4, 4).translate((0, 0, 4.5)))
    m = validate(solid, mesh)
    # forward only sees the covered patch obliquely (~2 mm to the lump's
    # sides); reverse measures the lump's top face 4 mm off the mesh
    assert m['symmetric'] and m['rev_dev_max'] > 3.5
    assert m['rev_dev_max'] > m['dev_max']
    p95, mx = gate_values(m)
    assert mx == m['rev_dev_max']


def test_open_mesh_reports_volume_unverified(tmp_path):
    m = trimesh.creation.box((20, 10, 5)).subdivide().subdivide()
    c = m.triangles_center
    keep = ~((c[:, 2] > 2.4) & (c[:, 0] > 2))          # rip a big patch off the top
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    assert not m.is_watertight
    p = str(tmp_path / 'leaky.stl'); m.export(p)
    from stl2prism.pipeline import run
    r = run(p, str(tmp_path / 'leaky.step'), verbose=False)
    assert r['metrics']['vol_verified'] is False
    assert r['metrics']['symmetric'] is False


# --- F8: welding -----------------------------------------------------------------

def test_microcrack_is_welded(tmp_path):
    p = synth.crack_plate(str(tmp_path / 'crack.stl'))
    from stl2prism.mesh_prep import load_mesh
    m = load_mesh(p)
    assert m.is_watertight and m.body_count == 1
    r, out = _run(p, tmp_path, 'crack')
    assert r['mode'] == 'prismatic' and r['n_bodies'] == 1
    assert synth.step_faces(out)[0] == 10


def test_weld_does_not_merge_distinct_vertices():
    from stl2prism.mesh_prep import weld_vertices
    m = trimesh.creation.box((1, 1, 1))
    n = len(m.vertices)
    weld_vertices(m)                       # tol ~1e-4 << 1 mm edges
    assert len(m.vertices) == n and m.is_watertight


# --- A6: CLI flag ---------------------------------------------------------------

def test_cli_accepts_vol_pct(tmp_path):
    import subprocess, sys
    p = synth.export(synth.plate_holes(), tmp_path / 'p.stl')
    out = str(tmp_path / 'p.step')
    res = subprocess.run([sys.executable, '-m', 'stl2prism.pipeline', p, out,
                          '--accept-vol-pct', '3', '--quiet'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert os.path.getsize(out) > 0


# --- A7: levels / constancy --------------------------------------------------------

def test_small_interior_step_is_recovered(tmp_path):
    r, out = _run(synth.small_step_inside(), tmp_path, 'step')
    assert r['mode'] == 'prismatic', r['metrics']
    assert synth.step_faces(out)[0] == 6 + 5             # box + pad (4 sides + top)
    assert r['metrics']['dev_max'] < 0.05


def test_shape_constancy_rejects_drift():
    from stl2prism.extrusion import score_axis
    m = trimesh.load(synth.export(synth.rotate(synth.plate_holes(), (1, 0, 0), 3.0),
                                  '/tmp/_tilt_probe2.stl'), force='mesh')
    sc_wrong, _, slabs = score_axis(m, np.array([0, 0, 1.0]))
    sc_true, _, _ = score_axis(m, np.array([0, -np.sin(np.radians(3)), np.cos(np.radians(3))]))
    assert sc_true > 0.99 and sc_wrong < 0.5


# --- A8: classifier -----------------------------------------------------------------

def test_dense_cad_export_is_not_a_scan(tmp_path):
    from stl2prism.mesh_prep import load_mesh, classify, SCAN_MIN_FACES
    p = synth.export(synth.plate_holes_fillets(), tmp_path / 'dense.stl', tol=0.0002, ang=0.01)
    m = load_mesh(p)
    assert len(m.faces) > SCAN_MIN_FACES // 4
    assert classify(m, verbose=False, scan_min_faces=1000) is False


def test_jittered_dense_mesh_is_a_scan(tmp_path):
    from stl2prism.mesh_prep import load_mesh, classify, mean_dihedral_deg, coplanar_fraction
    p = synth.scan_like_cube(str(tmp_path / 'noisy.stl'))
    m = load_mesh(p)
    assert mean_dihedral_deg(m) < 8 and coplanar_fraction(m) < 0.02
    assert classify(m, verbose=False, scan_min_faces=1000) is True


def test_real_scans_with_some_coplanar_pairs_are_scans():
    """Regression: the two real scans measure 3.6% and 4.3% coplanar pairs
    (mean dihedral 3.4/4.1 deg). A 2% cutoff sent Mesh_90p (2.17M faces)
    down the CAD path — no repair/reduce, a 40+ minute run. The cutoff must
    sit between the scans (~4%) and the CAD exports (30-66%)."""
    from stl2prism.mesh_prep import (SCAN_MAX_COPLANAR_FRAC, SCAN_DIHEDRAL_DEG,
                                     SCAN_MIN_FACES)
    measured_scans = [(2166597, 4.1, 0.0434), (709546, 3.4, 0.0358)]
    measured_cad = [(4200, 11.2, 0.664), (1644, 25.1, 0.468), (301220, 6.5, 0.303),
                    (25180, 30.0, 0.500)]
    for n, dih, cop in measured_scans:
        assert n > SCAN_MIN_FACES and dih < SCAN_DIHEDRAL_DEG
        assert cop < SCAN_MAX_COPLANAR_FRAC, 'real scan would be treated as CAD'
    for n, dih, cop in measured_cad:
        assert cop >= SCAN_MAX_COPLANAR_FRAC or dih >= SCAN_DIHEDRAL_DEG \
            or n <= SCAN_MIN_FACES
    assert 2 * 0.0434 < SCAN_MAX_COPLANAR_FRAC < 0.303 / 2, 'keep margin both ways'


# --- scan repair ladder: pre-reduce once, not per rung ------------------------------

def _fake_scan_worker(src, dst, target_faces, method, arg):
    """Stand-in for `_pymeshlab_worker` (runs in a spawned child, so it must
    be importable by name). Logs what it was asked to do and the input size,
    then does a cheap version of the job with trimesh/fast_simplification."""
    import os
    import trimesh as tm
    import fast_simplification as fs
    m = tm.load(src, force='mesh')
    with open(os.environ['STL2PRISM_FAKE_LOG'], 'a') as fh:
        fh.write(f'{method} {arg} {len(m.faces)}\n')
    if method == 'decimate':
        v, f = fs.simplify(m.vertices, m.faces, target_count=int(arg))
        tm.Trimesh(v, f).export(dst)
        return
    tm.repair.fill_holes(m)
    v, f = fs.simplify(m.vertices, m.faces, target_count=int(target_faces))
    tm.Trimesh(v, f).export(dst)


def test_scan_ladder_pre_reduces_once(tmp_path, monkeypatch):
    from stl2prism.mesh_prep import _poisson_rebuild
    log = tmp_path / 'log.txt'
    monkeypatch.setenv('STL2PRISM_FAKE_LOG', str(log))
    m = trimesh.load(synth.scan_like_cube(str(tmp_path / 'scan.stl')), force='mesh')
    n_in = len(m.faces)                                   # ~49k
    out = _poisson_rebuild(m, target_faces=2000, verbose=False,
                           attempts=(('close', 3000),), pre_faces=10000,
                           worker=_fake_scan_worker)
    calls = [line.split() for line in log.read_text().splitlines()]
    assert calls[0][0] == 'decimate' and int(calls[0][2]) == n_in
    assert calls[1][0] == 'close' and int(calls[1][2]) <= 10000, \
        'repair rung ran on the full-size mesh'
    assert out.is_watertight and len(out.faces) <= 2000


def test_scan_ladder_skips_pre_reduce_for_small_scans(tmp_path, monkeypatch):
    from stl2prism.mesh_prep import _poisson_rebuild
    log = tmp_path / 'log.txt'
    monkeypatch.setenv('STL2PRISM_FAKE_LOG', str(log))
    m = trimesh.load(synth.scan_like_cube(str(tmp_path / 'scan.stl')), force='mesh')
    _poisson_rebuild(m, target_faces=2000, verbose=False,
                     attempts=(('close', 3000),), pre_faces=300000,
                     worker=_fake_scan_worker)
    calls = [line.split() for line in log.read_text().splitlines()]
    assert [c[0] for c in calls] == ['close']
