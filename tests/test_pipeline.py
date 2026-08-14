"""Regression tests for classification, faceted export sanity, and modes.

The failure these guard against: a run that prints `mode: faceted` and exits 0
while the STEP file holds two triangles. Asserting on the exit mode alone is
what let that ship, so every test here also asserts on the artifact.
"""
import os
import glob
import pytest
import trimesh

SAMPLES = os.path.join(os.path.dirname(__file__), '..', 'samples')
SCANS = ['Mesh_90p.stl', 'rc2-clean-controller-v2.stl']


def _sample(name):
    p = os.path.join(SAMPLES, name)
    if not os.path.exists(p):
        pytest.skip(f'{name} not present (samples/ is gitignored)')
    return p


def _cad_samples():
    out = []
    for p in sorted(glob.glob(os.path.join(SAMPLES, '*.stl'))):
        if os.path.basename(p) not in SCANS:
            out.append(p)
    return out


def _step_faces(path):
    """Count ADVANCED_FACE entities without a full OCC import."""
    with open(path, 'r', errors='ignore') as fh:
        return sum(line.count('ADVANCED_FACE') for line in fh)


def _reimport(path):
    """Re-read a written STEP and report what OCC actually makes of it.

    Declaring CLOSED_SHELL is not the same as being closed: a shell with
    naked edges re-reads as a shell, not a solid, so the only trustworthy
    check is a round-trip.
    """
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE
    r = STEPControl_Reader()
    r.ReadFile(path)
    r.TransferRoots()
    s = r.OneShape()

    def n(kind):
        e = TopExp_Explorer(s, kind)
        c = 0
        while e.More():
            c += 1
            e.Next()
        return c

    from stl2prism.rebuild import _naked_edges
    return {'solids': n(TopAbs_SOLID), 'shells': n(TopAbs_SHELL),
            'faces': n(TopAbs_FACE), 'naked_edges': _naked_edges(s)}


# --- classification -------------------------------------------------------

@pytest.mark.parametrize('name', SCANS)
def test_scans_classified_as_scan(name):
    from stl2prism.mesh_prep import mean_dihedral_deg, SCAN_DIHEDRAL_DEG
    m = trimesh.load(_sample(name), force='mesh')
    assert mean_dihedral_deg(m) < SCAN_DIHEDRAL_DEG


@pytest.mark.parametrize('path', _cad_samples() or [pytest.param(
    None, marks=pytest.mark.skip(reason='no CAD samples present'))])
def test_cad_not_classified_as_scan(path):
    """A CAD export must never be routed away from the prismatic path."""
    from stl2prism.mesh_prep import mean_dihedral_deg, SCAN_DIHEDRAL_DEG
    m = trimesh.load(path, force='mesh')
    assert mean_dihedral_deg(m) > SCAN_DIHEDRAL_DEG


def _leaky_plate(path):
    """A prismatic plate with one facet removed: not watertight, obviously
    not a scan. The old `is_scan = big or not watertight` misrouted it."""
    m = trimesh.creation.box(extents=(20, 10, 5))
    m.update_faces([i for i in range(len(m.faces)) if i != 0])
    m.remove_unreferenced_vertices()
    assert not m.is_watertight
    m.export(path)
    return path


def test_non_watertight_cad_not_classified_as_scan(tmp_path):
    from stl2prism.mesh_prep import load_and_prep
    _, is_scan = load_and_prep(_leaky_plate(str(tmp_path / 'leaky.stl')),
                               verbose=False)
    assert is_scan is False


@pytest.mark.slow
def test_leaky_cad_part_still_reaches_prismatic(tmp_path):
    """End-to-end: repair must not cost a CAD part the prismatic path."""
    from stl2prism.pipeline import run
    out = str(tmp_path / 'leaky.step')
    r = run(_leaky_plate(str(tmp_path / 'leaky.stl')), out, verbose=False)
    assert r['mode'] == 'prismatic', 'leaky CAD export was misrouted'
    assert os.path.getsize(out) > 0


# --- faceted export sanity ------------------------------------------------

def test_faceted_export_keeps_the_geometry(tmp_path):
    """The bug that produced a 6KB two-face STEP from a 40k-face mesh."""
    from stl2prism.rebuild import faceted_fallback
    m = trimesh.creation.icosphere(subdivisions=3)
    m.apply_scale(10.0)
    out = str(tmp_path / 'sphere.step')
    stats = faceted_fallback(m, out, verbose=False)
    assert stats['faces_out'] >= 0.5 * stats['faces_in']
    assert _step_faces(out) == stats['faces_out']
    assert stats['volume'] == pytest.approx(m.volume, rel=0.02)
    assert stats['naked_edges'] == 0
    # Round-trip: a watertight input must come back as a SOLID, not a shell.
    got = _reimport(out)
    assert got['solids'] == 1, f'exported a shell, not a solid: {got}'
    assert got['naked_edges'] == 0


def test_faceted_export_refuses_fragments(tmp_path):
    """Two loose triangles must raise, not silently export as a solid."""
    from stl2prism.rebuild import faceted_fallback, FacetedError
    m = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9], [10, 9, 9], [9, 10, 9]],
        faces=[[0, 1, 2], [3, 4, 5]], process=False)
    with pytest.raises(FacetedError):
        faceted_fallback(m, str(tmp_path / 'frag.step'), verbose=False)


# --- end-to-end modes -----------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('name', SCANS)
def test_scan_endtoend_produces_real_solid(tmp_path, name):
    from stl2prism.pipeline import run
    out = str(tmp_path / 'scan.step')
    r = run(_sample(name), out, verbose=False)
    assert r['mode'] == 'faceted'
    assert r['metrics']['faces_out'] > 1000, 'exported a fragment'
    assert os.path.getsize(out) > 1_000_000
    assert _step_faces(out) == r['metrics']['faces_out']


@pytest.mark.slow
def test_cad_endtoend_hits_prismatic_gate(tmp_path):
    """CAD parts either pass the prismatic gate or fall back honestly."""
    from stl2prism.pipeline import run
    paths = _cad_samples()
    if not paths:
        pytest.skip('no CAD samples present')
    out = str(tmp_path / 'cad.step')
    r = run(paths[0], out, verbose=False)
    assert r['mode'] in ('prismatic', 'faceted')
    if r['mode'] == 'prismatic':
        assert r['metrics']['dev_p95'] <= 0.25
    assert os.path.getsize(out) > 0
