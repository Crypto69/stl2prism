"""Sliced loft: plane sections -> fitted outlines -> lofted B-spline solid.

Synthetic parts only (tests/synth.py style, built here), each asserted on
the artefact: mode 'loft', one valid solid, few faces, deviation and
volume against the mesh, and a STEP that re-reads as one solid.
"""
import os
import numpy as np
import pytest
import trimesh
import cadquery as cq

from . import synth
from .test_pipeline import _reimport


def _sphere():
    return trimesh.creation.icosphere(subdivisions=4, radius=15.0)


def _ellipsoid():
    e = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    e.apply_scale([20.0, 12.0, 8.0])
    return e


def _bottle_mesh(tmp_path):
    wp = (cq.Workplane('XY').spline([(0, 0), (14, 0), (16, 10), (13, 30), (6, 42), (5, 50), (5, 60), (0, 60)],
                                    includeCurrent=False).close()
          .revolve(360, (0, 0, 0), (0, 1, 0)))
    p = synth.export(wp, tmp_path / 'bottle.stl', tol=0.01)
    return trimesh.load(p, force='mesh')


def _valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return BRepCheck_Analyzer(shape).IsValid()


def _check(shape, mesh, faces_max, dev_p95=0.1, vol_pct=1.0):
    from stl2prism.pipeline import validate
    assert _valid(shape)
    s = cq.Shape.cast(shape)
    assert len(s.Solids()) == 1
    assert len(s.Faces()) <= faces_max, len(s.Faces())
    m = validate(shape, mesh)
    assert m['dev_p95'] <= dev_p95, m
    assert m['rev_dev_p95'] <= dev_p95, m
    assert m['vol_err_pct'] <= vol_pct, m
    return m, s


# --- the section cutter ---------------------------------------------------

def test_section_loops_box_and_hole():
    from stl2prism.section_fit import section_loops, signed_area
    m = trimesh.creation.box([40, 30, 10])
    loops, frame = section_loops(m.vertices, m.faces, [0, 0, 1.0], [0, 0, 1])
    assert len(loops) == 1 and not loops[0][1]
    outer = loops[0][0]
    assert signed_area(outer) == pytest.approx(1200.0, rel=1e-6)   # CCW
    # a plate with a hole: one outer, one CW hole
    p = synth.export(synth.plate_holes(), '/tmp/_s2p_plate.stl')
    m = trimesh.load(p, force='mesh')
    loops, _ = section_loops(m.vertices, m.faces, [0, 0, 0.5], [0, 0, 1])
    assert len(loops) == 1 and len(loops[0][1]) == 4
    assert all(signed_area(h) < 0 for h in loops[0][1])
    assert sum(abs(signed_area(h)) for h in loops[0][1]) == pytest.approx(4 * np.pi * 2.5 ** 2, rel=0.02)


def test_section_matches_trimesh():
    from stl2prism.section_fit import section_loops, signed_area
    m = _ellipsoid()
    z = 2.0
    loops, _ = section_loops(m.vertices, m.faces, [0, 0, z], [0, 0, 1])
    s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    p2, _ = s.to_2D()
    ref = sum(abs(p.area) for p in p2.polygons_closed if p is not None)
    assert len(loops) == 1
    assert abs(signed_area(loops[0][0])) == pytest.approx(ref, rel=1e-6)


def test_fit_loop_rounded_rect_is_lines_and_arcs():
    from stl2prism.section_fit import section_loops, fit_loop
    p = synth.export(synth.rounded_rect(), '/tmp/_s2p_rr.stl')
    m = trimesh.load(p, force='mesh')
    loops, _ = section_loops(m.vertices, m.faces, [0, 0, 0.0], [0, 0, 1])
    prims = fit_loop(loops[0][0], tol=0.08)
    kinds = sorted(p['type'] for p in prims)
    assert kinds.count('line') == 4 and kinds.count('arc') == 4 and 'spline' not in kinds
    assert all(abs(p['r'] - 4.0) < 0.05 for p in prims if p['type'] == 'arc')


def test_fit_loop_spline_fallback_on_ripple():
    """A ring with ripples too short for any line or arc to hold at the
    tolerance comes back as a spline, within tolerance of the raw points."""
    from stl2prism.section_fit import fit_loop, prim_points, polyline_deviation
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    r = 10 + 0.25 * np.sin(80 * t)
    xy = np.c_[r * np.cos(t), r * np.sin(t)]
    prims = fit_loop(xy, tol=0.05)
    assert any(p['type'] == 'spline' for p in prims)
    assert polyline_deviation(xy[::5], prim_points(prims)) < 0.1


def test_fit_loop_holds_tolerance_on_wave():
    """Gentle waves are arcs; whatever the mix, the fit stays within tol."""
    from stl2prism.section_fit import fit_loop, prim_points, polyline_deviation
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    r = 10 + 1.5 * np.sin(5 * t)
    xy = np.c_[r * np.cos(t), r * np.sin(t)]
    prims = fit_loop(xy, tol=0.05)
    assert len(prims) > 4
    assert polyline_deviation(xy[::4], prim_points(prims)) < 0.1


def test_trace_rc_n2_section_like_fusion():
    """The section a user cuts in Fusion (plane normal to the depth axis,
    26 mm from the box centre) traced from the mesh: a few hundred lines
    and arcs plus splines on the domes, all within 0.15 mm of the raw cut."""
    from .test_pipeline import _sample
    from stl2prism.section_fit import fit_section
    p = _sample('fixed-rc-n2-360.stl')
    m = trimesh.load(p, force='mesh')
    m.apply_scale(0.1)                      # the file is written 10x too big
    z = m.bounding_box.centroid[2] - 26.0
    sec = fit_section(m.vertices, m.faces, [0, 0, z], [0, 0, 1], tol=0.08)
    st = sec['stats']
    assert st['loops'] >= 5 and st['holes'] >= 3
    assert st['lines'] > 50 and st['arcs'] > 50 and st['splines'] > 0
    assert st['dev_max'] < 0.15, st


# --- the loft -------------------------------------------------------------

def test_step_levels_on_shaft(tmp_path):
    from stl2prism.sliced_loft import step_levels
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    m = trimesh.load(p, force='mesh')
    lv = step_levels(m, 2)
    assert [round(l, 3) for l in lv] == [0.0, 20.0, 35.0, 45.0]


def test_loft_sphere_one_face_per_run():
    from stl2prism.sliced_loft import loft_body
    m = _sphere()
    shape, info = loft_body(m, 2, 0.2, verbose=False)
    assert info['mode'] == 'loft' and info['n_runs'] == 1
    # one smooth face for the body, a few ruled bands where the rings
    # shrink fast near the poles, two apex cones: tens of faces, not the
    # mesh's 5120
    _check(shape, m, faces_max=24, dev_p95=0.08, vol_pct=0.5)


def test_loft_ellipsoid_along_x():
    from stl2prism.sliced_loft import loft_body
    m = _ellipsoid()
    shape, info = loft_body(m, 'auto', 0.2, verbose=False)
    assert info['axis_name'] == 'x'
    _check(shape, m, faces_max=30, dev_p95=0.08, vol_pct=0.5)


def test_loft_stepped_shaft_breaks_at_levels(tmp_path):
    from stl2prism.sliced_loft import loft_body
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    m = trimesh.load(p, force='mesh')
    shape, info = loft_body(m, 2, 0.2, verbose=False)
    assert info['n_runs'] == 3 and info['n_levels'] == 4
    m2, s = _check(shape, m, faces_max=3 * 3, dev_p95=0.05, vol_pct=0.5)
    # the shoulders are real planes
    assert sum(1 for f in s.Faces() if f.geomType() == 'PLANE') >= 4


def test_loft_plate_holes_are_cut(tmp_path):
    from stl2prism.sliced_loft import loft_body
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    m = trimesh.load(p, force='mesh')
    shape, info = loft_body(m, 2, 0.5, verbose=False)
    assert info['n_holes'] == 4
    _check(shape, m, faces_max=3 + 4 + 4, dev_p95=0.1, vol_pct=1.0)


def test_loft_ruled_option(tmp_path):
    from stl2prism.sliced_loft import loft_body
    m = _sphere()
    shape, info = loft_body(m, 2, 0.5, ruled=True, verbose=False)
    assert info['ruled'] and info['n_ruled_runs'] == 1
    s = cq.Shape.cast(shape)
    assert len(s.Faces()) > 20            # one face per section pair
    _check(shape, m, faces_max=200, dev_p95=0.1, vol_pct=1.0)


def test_pipeline_loft_mode_writes_step_and_fusion_script(tmp_path):
    from stl2prism.pipeline import run
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    out = str(tmp_path / 'shaft.step')
    r = run(p, out, method='loft', slice_mm=0.5, verbose=False)
    assert r['mode'] == 'loft'
    assert r['metrics']['loft']['n_runs'] == 3
    assert r['metrics']['gate_ok'] is True
    got = _reimport(out)
    assert got['solids'] == 1 and got['naked_edges'] == 0
    assert r['script'] is None
    fs = r['fusion_script']
    assert fs and os.path.exists(fs) and fs.endswith('_fusion.py')
    txt = open(fs).read()
    assert 'loftFeatures' in txt and 'sketchFittedSplines' in txt
    assert txt.count('lofts.add(li)') == 3
    compile(txt, fs, 'exec')              # at least valid Python


def test_pipeline_loft_gate_is_reported_not_enforced(tmp_path):
    """A sideways hole smears under a loft; the loft is still written."""
    from stl2prism.pipeline import run
    p = synth.export(synth.cross_blind(), tmp_path / 'cb.stl')
    out = str(tmp_path / 'cb.step')
    r = run(p, out, method='loft', slice_mm=0.5, slice_axis='z', verbose=False)
    assert r['mode'] == 'loft'
    assert r['metrics']['gate_ok'] is False          # the smeared holes
    # the run pieces do not always fuse across a sideways hole; a compound
    # of a few solids is the honest result, never nothing
    assert 1 <= _reimport(out)['solids'] <= 8
    assert r['metrics']['vol_err_pct'] < 1.0


def test_cli_flags_parse(tmp_path):
    import subprocess, sys
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    out = str(tmp_path / 'shaft.step')
    res = subprocess.run([sys.executable, '-m', 'stl2prism.pipeline', p, out,
                          '--method', 'loft', '--slice-mm', '1.0', '--slice-axis', 'z',
                          '--workers', '0', '--quiet'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert os.path.exists(out)


# --- single slice: API + Fusion sketch script ------------------------------

def test_section_api_and_sketch_script(tmp_path, monkeypatch):
    """Upload a stepped shaft, trace one slice through its middle step,
    get the curves back as JSON and the same slice as a Fusion sketch."""
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STL2PRISM_DATA', str(tmp_path / 'data'))
    from backend import jobs, main
    importlib.reload(jobs)
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    with TestClient(main.app) as c:
        with open(p, 'rb') as fh:
            r = c.post('/api/jobs', files={'file': ('shaft.stl', fh, 'application/octet-stream')})
        assert r.status_code == 200, r.text
        job = r.json()['id']
        # the middle step is r=7 between z=20 and 35; the box centre is z=22.5
        r = c.get(f'/api/jobs/{job}/section', params={'axis': 'z', 'offset': 5.0, 'tol': 0.08})
        assert r.status_code == 200, r.text
        sec = r.json()
        assert sec['axis'] == 'z' and sec['at'] == pytest.approx(27.5, abs=0.01)
        assert sec['stats']['loops'] == 1 and sec['stats']['circles'] == 1
        assert len(sec['polylines']) == 1
        r = c.get(f'/api/jobs/{job}/section-script', params={'axis': 'z', 'offset': 5.0})
        assert r.status_code == 200
        txt = r.text
        assert 'sketchCircles.addByCenterRadius' in txt and 'root.sketches.add' in txt
        compile(txt, 'section.py', 'exec')
        assert 'attachment' in r.headers['content-disposition']
        r = c.get(f'/api/jobs/{job}/section', params={'axis': 'q'})
        assert r.status_code == 400


def test_sections_script_draws_lines_arcs_and_splines():
    from stl2prism.section_fit import section_preview
    from stl2prism.fusion_export import emit_fusion_sections_script
    p = synth.export(synth.rounded_rect(), '/tmp/_s2p_rr2.stl')
    m = trimesh.load(p, force='mesh')
    sec = section_preview(m.vertices, m.faces, [0, 0, 0], [0, 0, 1], tol=0.08)
    txt = emit_fusion_sections_script([{'origin': sec['origin'], 'normal': sec['normal'],
                                        'name': 'rr', 'loops': sec['loops']}])
    assert txt.count('sketchLines.addByTwoPoints') == 4
    assert txt.count('sketchArcs.addByThreePoints') == 4
    compile(txt, 'rr.py', 'exec')
