"""Regression tests for classification, faceted export sanity, and modes.

The failure these guard against: a run that prints `mode: faceted` and exits 0
while the STEP file holds two triangles. Asserting on the exit mode alone is
what let that ship, so every test here also asserts on the artifact.
"""
import os
import glob
import numpy as np
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


# --- OBJ input ------------------------------------------------------------

def _hard_obj(path, mesh=None):
    """Write `mesh` (default: a 20x10x5 box) as the kind of OBJ real exporters
    produce: per-corner normals (`vn`), UVs (`vt`), several `g` groups, and a
    `mtllib` pointing at a file that is not there. Every one of those makes
    trimesh split vertices, so the mesh only reads as closed if the loader
    merges on position alone."""
    m = mesh if mesh is not None else trimesh.creation.box(extents=(20, 10, 5))
    lines = ['mtllib missing.mtl', 'o part', 'vt 0 0', 'vt 1 0', 'vt 0 1']
    lines += ['v %.6f %.6f %.6f' % tuple(v) for v in m.vertices]
    lines += ['vn %.6f %.6f %.6f' % tuple(n) for n in m.face_normals]
    half = len(m.faces) // 2
    for i, f in enumerate(m.faces):
        if i == 0:
            lines += ['g A', 'usemtl matA']
        if i == half:
            lines += ['g B', 'usemtl matB']
        lines.append('f %d/1/%d %d/2/%d %d/3/%d' % (
            f[0] + 1, i + 1, f[1] + 1, i + 1, f[2] + 1, i + 1))
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return path


def test_obj_with_split_normals_loads_watertight(tmp_path):
    """OBJ per-corner vn/vt must not turn a closed part into an open one."""
    from stl2prism.mesh_prep import load_and_prep
    src = trimesh.creation.box(extents=(20, 10, 5))
    m, is_scan = load_and_prep(_hard_obj(str(tmp_path / 'hard.obj'), src),
                               verbose=False)
    assert m.is_watertight
    assert len(m.faces) == len(src.faces)
    assert len(m.vertices) == len(src.vertices)
    assert m.volume == pytest.approx(src.volume, rel=1e-6)
    assert is_scan is False


def test_obj_quads_are_triangulated(tmp_path):
    from stl2prism.mesh_prep import load_mesh
    cube = """v 0 0 0
v 10 0 0
v 10 20 0
v 0 20 0
v 0 0 5
v 10 0 5
v 10 20 5
v 0 20 5
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
"""
    p = tmp_path / 'quads.OBJ'   # mixed-case extension must work too
    p.write_text(cube)
    m = load_mesh(str(p))
    assert len(m.faces) == 12
    assert m.is_watertight
    assert m.volume == pytest.approx(1000.0)


def test_units_scale_on_load(tmp_path):
    """A 20x10x5 box in cm must arrive as 200x100x50 mm; unknown units
    are refused before anything is loaded."""
    from stl2prism.mesh_prep import load_and_prep, PrepError
    p = str(tmp_path / 'box.obj')
    trimesh.creation.box(extents=(20, 10, 5)).export(p)
    m, _ = load_and_prep(p, verbose=False, units='cm')
    assert m.bounding_box.primitive.extents == pytest.approx([200, 100, 50])
    assert m.is_watertight
    m_in, _ = load_and_prep(p, verbose=False, units='in')
    assert m_in.bounding_box.primitive.extents == pytest.approx(
        [508, 254, 127])
    with pytest.raises(PrepError, match='unknown units'):
        load_and_prep(p, verbose=False, units='furlongs')


def test_unsupported_extension_rejected(tmp_path):
    from stl2prism.mesh_prep import load_mesh, PrepError
    p = tmp_path / 'part.xyz'
    p.write_text('0 0 0\n1 0 0\n0 1 0\n')
    with pytest.raises(PrepError, match='unsupported'):
        load_mesh(str(p))


@pytest.mark.parametrize('ext', ['ply', 'off', 'glb', '3mf'])
def test_other_mesh_formats_load(tmp_path, ext):
    from stl2prism.mesh_prep import load_mesh
    p = tmp_path / f'box.{ext}'
    trimesh.creation.box((20, 10, 5)).export(str(p))
    m = load_mesh(str(p))
    assert m.is_watertight and m.volume == pytest.approx(1000.0)


def test_dji_obj_sample_loads_and_is_scan():
    """The real-world sample: 21 objects, no .mtl, open, dense -> scan path.
    Only the load + classification is checked; the repair ladder needs
    pymeshlab, which is not available everywhere the tests run."""
    from stl2prism.mesh_prep import (load_mesh, mean_dihedral_deg,
                                     SCAN_DIHEDRAL_DEG, SCAN_MIN_FACES)
    m = load_mesh(_sample('DJI_RC-N1_controller.obj'))
    assert len(m.faces) == 301220
    assert m.body_count > 1
    assert not m.is_watertight
    assert len(m.faces) > SCAN_MIN_FACES
    assert mean_dihedral_deg(m) < SCAN_DIHEDRAL_DEG


@pytest.mark.slow
def test_obj_matches_stl_endtoend(tmp_path):
    """End-to-end: a CAD part must convert identically whether it arrives as
    STL or as a hard OBJ of the same triangles (same mode, same fidelity)."""
    from stl2prism.pipeline import run
    stl = _sample('servo_bracket_1.stl')
    r_stl = run(stl, str(tmp_path / 'stl.step'), verbose=False)
    obj = _hard_obj(str(tmp_path / 'cad.obj'), trimesh.load(stl, force='mesh'))
    out = str(tmp_path / 'obj.step')
    r_obj = run(obj, out, verbose=False)
    assert r_obj['mode'] == r_stl['mode'] == 'prismatic', 'OBJ was misrouted'
    assert r_obj['metrics']['dev_p95'] == pytest.approx(
        r_stl['metrics']['dev_p95'], abs=0.01)
    assert os.path.getsize(out) > 0


# --- multi-body -----------------------------------------------------------

def _assembly(path):
    """Two boxes, a sphere and a 2-triangle sliver in one file: three real
    bodies (two prismatic, one not) plus the kind of stray fragment exporters
    leave behind."""
    a = trimesh.creation.box((20, 10, 5))
    b = trimesh.creation.box((8, 8, 8))
    b.apply_translation((40, 0, 0))
    c = trimesh.creation.icosphere(subdivisions=3, radius=5)
    c.apply_translation((0, 40, 0))
    sliver = trimesh.Trimesh(
        vertices=[[80, 0, 0], [81, 0, 0], [80, 1, 0], [80, 0, 1]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False)
    trimesh.util.concatenate([a, b, c, sliver]).export(path)
    return path


def test_split_bodies_drops_only_slivers(tmp_path):
    """CAD input keeps every closable body, however small (a 12-face box is
    a part); only fragments that cannot close are dropped."""
    from stl2prism.mesh_prep import load_mesh, split_bodies
    m = load_mesh(_assembly(str(tmp_path / 'asm.stl')))
    parts, dropped = split_bodies(m, is_scan=False, verbose=False)
    assert dropped == 1
    assert [len(p.faces) for p in parts] == [1280, 12, 12]   # largest first


def test_zero_volume_bodies_are_slivers():
    """A 'closed' body with no volume — two triangles back to back, or a
    flattened fan — passed the old face-count rule and then failed the
    faceted volume gate 58 times over on a real assembly export."""
    from stl2prism.mesh_prep import is_sliver
    # two coincident triangles, opposite winding: closed, zero volume
    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                           faces=[[0, 1, 2], [0, 2, 1]], process=False)
    assert flat.is_watertight and is_sliver(flat) is True
    # The pattern from the export: a triangular bipyramid (5 verts, 9 edges,
    # 6 faces, every edge on exactly two faces) with both apexes squashed
    # into the base plane — trimesh calls it watertight, volume is zero.
    fan = trimesh.Trimesh(
        vertices=[[0, 0, 0], [2, 0, 0], [1, 1, 0], [1, 0.4, 0], [1, 0.3, 0]],
        faces=[[3, 0, 1], [3, 1, 2], [3, 2, 0],
               [4, 1, 0], [4, 2, 1], [4, 0, 2]], process=False)
    assert fan.is_watertight and is_sliver(fan) is True
    # a thin but real washer-like plate is a part
    plate = trimesh.creation.box((10, 10, 0.2))
    assert is_sliver(plate) is False
    assert is_sliver(trimesh.creation.box()) is False


def test_faceted_merge_never_changes_volume():
    """Coplanar merging is cosmetic; if it moves the volume by more than 1%
    the unmerged (exact) solid must be kept. Regression for micron-thick
    decal bodies where unify collapsed the two skins into each other."""
    from stl2prism.rebuild import faceted_solid
    # A thin, slightly bent sheet: many near-coplanar facets, 20um thick.
    g = trimesh.creation.box((4, 3, 0.02))
    g = g.subdivide().subdivide()
    v = g.vertices.copy()
    v[:, 2] += 0.002 * np.sin(v[:, 0])      # bend so faces are near-coplanar
    g = trimesh.Trimesh(v, g.faces, process=True)
    assert g.is_watertight
    _, st = faceted_solid(g, verbose=False)
    assert st['volume'] == pytest.approx(g.volume, rel=0.01)


def test_multi_body_writes_every_body_into_one_step(tmp_path):
    """The bug that shipped one knurled dial out of an 85-body controller:
    every body must be converted on its own and land in the same STEP."""
    from stl2prism.pipeline import run
    out = str(tmp_path / 'asm.step')
    r = run(_assembly(str(tmp_path / 'asm.stl')), out, verbose=False)
    assert r['mode'] == 'mixed'
    assert (r['n_bodies'], r['n_written'], r['n_dropped']) == (3, 3, 1)
    modes = sorted(b['mode'] for b in r['bodies'])
    assert modes == ['facegroup', 'prismatic', 'prismatic']   # the sphere: 1 analytic face
    assert all(b['error'] is None for b in r['bodies'])
    assert r['metrics']['n_prismatic'] == 2 and r['metrics']['n_facegroup'] == 1
    got = _reimport(out)
    assert got['solids'] == 3, f'expected 3 solids in one STEP: {got}'
    assert got['naked_edges'] == 0


def test_single_body_result_shape_unchanged(tmp_path):
    """Callers of run() on ordinary parts must see the old keys and values."""
    from stl2prism.pipeline import run
    p = str(tmp_path / 'box.stl')
    trimesh.creation.box((20, 10, 5)).export(p)
    r = run(p, str(tmp_path / 'box.step'), verbose=False)
    assert r['mode'] == 'prismatic'
    assert 'bodies' not in r
    assert r['metrics']['dev_p95'] == pytest.approx(0.0, abs=1e-6)
    assert (r['n_bodies'], r['n_written'], r['n_dropped']) == (1, 1, 0)


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
    pytest.importorskip('pymeshlab')          # scan repair ladder: Linux x86_64 only
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
