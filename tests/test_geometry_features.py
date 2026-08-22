"""B2 (clean solids), B3/C2 (cylinder & cone fitting), B4 (hybrid patch),
C1 (lofted tapers/chamfers/countersinks)."""
import numpy as np
import pytest
import trimesh
import cadquery as cq

from . import synth


def _run(wp, tmp_path, name, **kw):
    from stl2prism.pipeline import run
    p = synth.export(wp, tmp_path / f'{name}.stl')
    out = str(tmp_path / f'{name}.step')
    r = run(p, out, verbose=False, **kw)
    return r, out


# --- B2: one valid solid, no slivers, unified faces -----------------------------

@pytest.mark.parametrize('name,wp,faces', [
    ('rounded_rect', synth.rounded_rect(), 10),
    ('stepped_shaft', synth.stepped_shaft(), 7),
    ('cross3', synth.cross3(), 30),
    ('slot_hex', synth.slot_hex(), 17),
])
def test_clean_single_solid(tmp_path, name, wp, faces):
    r, out = _run(wp, tmp_path, name)
    assert r['mode'] == 'prismatic', r['metrics']
    got = synth.reimport(out)
    assert got['solids'] == 1 and got['valid'] and got['naked_edges'] == 0
    assert got['faces'] == faces
    assert min(synth.face_areas(out)) > 1e-3          # no sliver faces


def test_stacked_slabs_share_one_cylinder(tmp_path):
    """A through hole spanning three slabs must be ONE cylindrical face."""
    part = (cq.Workplane('XY').box(30, 30, 4)
            .union(cq.Workplane('XY').box(20, 20, 4).translate((0, 0, 4)))
            .union(cq.Workplane('XY').box(10, 10, 4).translate((0, 0, 8)))
            .faces('>Z').workplane().hole(3))
    r, out = _run(part, tmp_path, 'stack')
    assert r['mode'] == 'prismatic'
    faces, kinds = synth.step_faces(out)
    assert kinds.get('CYLINDRICAL_SURFACE', 0) == 1
    assert faces == 17                                  # 16 planes + 1 cylinder


# --- B3 / C2: primitive fitting on vertices ---------------------------------------

def test_cylinder_fit_exact_on_coarse_mesh(tmp_path):
    from stl2prism.features import find_cross_cylinders
    m = trimesh.load(synth.export(synth.cross_blind(), tmp_path / 'cb.stl', tol=0.3, ang=0.5),
                     force='mesh')
    cyls = find_cross_cylinders(m, np.array([0, 0, 1.0]), exclude_parallel=False)
    radii = sorted(round(c['r'], 3) for c in cyls)
    assert radii == [2.0, 3.0]
    assert all(c['resid'] < 0.01 for c in cyls)


def test_cross_axis_countersink_is_a_cone(tmp_path):
    """Countersink along X in a block; primary axis given as Z, so the cone
    is a cross-axis feature."""
    from stl2prism.features import find_cross_cones
    blk = (cq.Workplane('XY').box(30, 30, 30).faces('>X').workplane().cskHole(6, 12, 90))
    m = trimesh.load(synth.export(blk, tmp_path / 'csk.stl'), force='mesh')
    cones = find_cross_cones(m, np.array([0, 0, 1.0]))
    assert len(cones) == 1
    c = cones[0]
    assert c['concave']
    assert abs(np.degrees(c['half_angle']) - 45) < 1.0
    assert abs(abs(c['axis'][0]) - 1) < 1e-3
    assert c['resid'] < 0.02


def test_partial_cylinder_is_not_a_cone(tmp_path):
    from stl2prism.features import find_cross_cones
    m = trimesh.load(synth.export(synth.rounded_rect(), tmp_path / 'rr.stl'), force='mesh')
    assert find_cross_cones(m, np.array([1.0, 0, 0])) == []


# --- C1: lofts --------------------------------------------------------------------

def test_drafted_block_is_six_planes(tmp_path):
    r, out = _run(synth.drafted_block(), tmp_path, 'draft')
    assert r['mode'] == 'prismatic'
    faces, kinds = synth.step_faces(out)
    assert faces == 6 and kinds == {'PLANE': 6}
    assert r['metrics']['dev_max'] < 0.02


def test_chamfered_box_is_ten_planes(tmp_path):
    r, out = _run(synth.chamfer_top(), tmp_path, 'chamfer')
    assert r['mode'] == 'prismatic'
    faces, kinds = synth.step_faces(out)
    assert faces == 10 and kinds == {'PLANE': 10}
    assert r['metrics']['dev_max'] < 0.02


def test_countersunk_plate_has_cones(tmp_path):
    r, out = _run(synth.csk_plate(), tmp_path, 'csk')
    assert r['mode'] == 'prismatic', r['metrics']
    faces, kinds = synth.step_faces(out)
    assert kinds.get('CONICAL_SURFACE') == 4 and kinds.get('CYLINDRICAL_SURFACE') == 4
    assert faces == 14
    assert r['metrics']['dev_max'] < 0.1


def test_boss_with_chamfered_top(tmp_path):
    part = (cq.Workplane('XY').box(30, 30, 5).faces('>Z').workplane().circle(5).extrude(6)
            .faces('>Z').edges().chamfer(1))
    r, out = _run(part, tmp_path, 'bosschamfer')
    assert r['mode'] == 'prismatic', r['metrics']
    faces, kinds = synth.step_faces(out)
    assert kinds.get('CONICAL_SURFACE') == 1
    assert r['metrics']['dev_max'] < 0.1


# --- B4: hybrid patch --------------------------------------------------------------

def test_hybrid_patch_keeps_analytic_faces(tmp_path):
    """A plate with holes plus one spherical dimple: the dimple cannot be
    expressed, but the rest must stay analytic (planes + cylinders), the
    dimple region patched with facets, and the result accepted."""
    part = (synth.plate_holes()
            .cut(cq.Workplane('XY').sphere(4).translate((15, 5, 3))))
    # hybrid rung (face-group engine off): patched prismatic
    r, out = _run(part, tmp_path, 'dimple', face_groups=False)
    assert r['mode'] == 'prismatic', r['metrics']
    assert r['metrics'].get('patched') is True
    faces, kinds = synth.step_faces(out)
    assert kinds.get('CYLINDRICAL_SURFACE', 0) >= 4
    assert r['metrics']['dev_max'] < 0.26
    got = synth.reimport(out)
    assert got['solids'] == 1 and got['valid']
    # with the face-group engine (default) the dimple is a real sphere face
    r2, out2 = _run(part, tmp_path, 'dimple_fg')
    assert r2['mode'] == 'facegroup', r2['metrics']
    faces2, kinds2 = synth.step_faces(out2)
    assert kinds2.get('SPHERICAL_SURFACE') == 1 and kinds2.get('CYLINDRICAL_SURFACE') == 4
    assert faces2 == 11
