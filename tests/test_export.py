"""D1 (CadQuery script export), D4 (named/coloured STEP), B5 (faceted route:
planar merge + tolerance-driven reduce)."""
import os
import subprocess
import sys
import numpy as np
import pytest
import trimesh

from . import synth


def test_script_rebuilds_the_part(tmp_path):
    from stl2prism.pipeline import run
    p = synth.export(synth.slot_hex(), tmp_path / 'slot.stl')
    out = str(tmp_path / 'slot.step')
    r = run(p, out, verbose=False)
    assert r['mode'] == 'prismatic'
    script = r['script']
    assert script and os.path.exists(script)
    txt = open(script).read()
    assert 'cq.Workplane' in txt and '.extrude(' in txt and 'H_1' in txt and 'threePointArc' in txt
    # run it in a clean directory: it must regenerate a STEP of the same volume
    res = subprocess.run([sys.executable, script], cwd=str(tmp_path), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    regenerated = str(tmp_path / 'slot.step')      # script writes next to itself (cwd)
    got = synth.reimport(regenerated)
    src = trimesh.load(p, force='mesh')
    assert got['solids'] == 1
    assert got['volume'] == pytest.approx(src.volume, rel=2e-3)


def test_script_for_multi_body_and_hole_cut(tmp_path):
    from stl2prism.pipeline import run
    p = synth.export(synth.cross_blind(), tmp_path / 'cb.stl')
    r = run(p, str(tmp_path / 'cb.step'), verbose=False)
    txt = open(r['script']).read()
    assert 'cross-axis hole' in txt
    res = subprocess.run([sys.executable, r['script']], cwd=str(tmp_path), capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_step_has_body_names_and_colours(tmp_path):
    from stl2prism.pipeline import run
    p = synth.export(synth.csk_plate(), tmp_path / 'named_part.stl')
    out = str(tmp_path / 'named_part.step')
    r = run(p, out, verbose=False)
    txt = open(out).read()
    assert 'named_part' in txt
    assert 'COLOUR_RGB' in txt
    # names round-trip through XDE
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDataStd import TDataStd_Name
    doc = TDocStd_Document(TCollection_ExtendedString('x'))
    rd = STEPCAFControl_Reader()
    rd.SetNameMode(True)
    assert rd.ReadFile(out) == 1
    rd.Transfer(doc)
    st = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    st.GetFreeShapes(labels)
    names = []
    for i in range(1, labels.Length() + 1):
        n = TDataStd_Name()
        if labels.Value(i).FindAttribute(TDataStd_Name.GetID_s(), n):
            names.append(n.Get().ToExtString())
    assert 'named_part' in names


def test_multi_body_step_names(tmp_path):
    from stl2prism.pipeline import run
    a = trimesh.creation.box((20, 10, 5))
    b = trimesh.creation.box((8, 8, 8)); b.apply_translation((40, 0, 0))
    p = str(tmp_path / 'asm.stl'); trimesh.util.concatenate([a, b]).export(p)
    out = str(tmp_path / 'asm.step')
    r = run(p, out, verbose=False)
    txt = open(out).read()
    assert 'asm_body1' in txt and 'asm_body2' in txt
    assert synth.reimport(out)['solids'] == 2


# --- B5 ---------------------------------------------------------------------

def test_planar_merge_before_sewing_gives_few_faces():
    from stl2prism.rebuild import faceted_solid
    m = trimesh.creation.box((20, 10, 5)).subdivide().subdivide()   # 12*16 = 192 tris
    shape, st = faceted_solid(m, verbose=False, merge_planar=True)
    assert st['faces_out'] == 6 and st['is_solid'] and st['planar_faces_merged'] == 6


def test_reduce_mesh_respects_tolerance():
    from stl2prism.rebuild import reduce_mesh
    m = trimesh.creation.icosphere(subdivisions=5, radius=20)
    r, info = reduce_mesh(m, 0.05, verbose=False)
    assert info['reduced'] and info['faces_after'] < 0.25 * info['faces_before']
    assert info['max_dev'] <= 0.05 and r.is_watertight
    r2, info2 = reduce_mesh(m, 0.0, verbose=False)
    assert not info2['reduced']


def test_faceted_fallback_uses_reduce(tmp_path):
    from stl2prism.pipeline import run
    m = trimesh.creation.icosphere(subdivisions=4, radius=10)
    p = str(tmp_path / 'sphere.stl'); m.export(p)
    # (the face-group engine would make this one analytic sphere; this test
    # is about the faceted route, so switch it off)
    r = run(p, str(tmp_path / 'sphere.step'), verbose=False, reduce_tol=0.05,
            face_groups=False)
    assert r['mode'] == 'faceted'
    assert r['metrics']['faces_out'] < 0.6 * len(m.faces)
    assert r['metrics']['reduce']['reduced'] is True
    r0 = run(p, str(tmp_path / 'sphere0.step'), verbose=False, reduce_tol=0.0,
             face_groups=False)
    assert r0['metrics']['faces_out'] == len(m.faces)


def test_no_script_when_nothing_is_prismatic(tmp_path):
    """A faceted-only result must not offer a script: it would be an empty
    program (no recognised bodies) that crashes on `bodies[0]`."""
    from stl2prism.pipeline import run
    m = trimesh.creation.icosphere(subdivisions=3, radius=10)
    p = str(tmp_path / 'sphere.stl'); m.export(p)
    r = run(p, str(tmp_path / 'sphere.step'), verbose=False, face_groups=False)
    assert r['mode'] == 'faceted' and r['is_scan'] is False
    assert r['script'] is None
    # the face-group engine makes it one analytic sphere — still no script
    r2 = run(p, str(tmp_path / 'sphere2.step'), verbose=False)
    assert r2['mode'] == 'facegroup' and r2['script'] is None
    assert not os.path.exists(tmp_path / 'sphere.py')
    assert not os.path.exists(tmp_path / 'sphere_fusion.py')


def test_fusion_script_is_emitted_and_parses(tmp_path):
    """Cannot run Fusion here: check the script exists, parses, and carries
    the expected structure (planes, sketches, extrude/loft/cut calls)."""
    import ast
    from stl2prism.pipeline import run
    p = synth.export(synth.csk_plate(), tmp_path / 'csk.stl')
    r = run(p, str(tmp_path / 'csk.step'), verbose=False)
    f = tmp_path / 'csk_fusion.py'
    assert f.exists()
    src = f.read_text()
    ast.parse(src)
    assert 'lofts.createInput' in src and 'extrudes.createInput' in src
    assert 'modelToSketchSpace' in src and 'def run(context)' in src
