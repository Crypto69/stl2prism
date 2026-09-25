"""X-Ray: a stack of section sketches between two planes (v0.4.5).

The stack rule (`section_fit.stack_offsets`), the backend's stack tracer
(one mesh load, each plane cut with only the faces that span it), the
/xray and /xray-script routes, and the embedded-data Fusion script."""
import ast

import numpy as np
import pytest
import trimesh

from stl_to_solid.section_fit import stack_offsets, section_preview

from . import synth


# --- where the planes go ------------------------------------------------------

def test_stack_offsets_regular_and_end_included():
    offs = stack_offsets(-10, 10, 1, 50)
    assert len(offs) == 21
    assert offs == pytest.approx(list(range(-10, 11)))
    # the end plane the user chose is always the last slice
    assert stack_offsets(0, 1, 0.3, 50) == pytest.approx([0, 0.3, 0.6, 0.9, 1.0])
    # ...but not twice when a multiple already lands on it
    assert stack_offsets(0, 1, 0.25, 50) == pytest.approx([0, 0.25, 0.5, 0.75, 1.0])


def test_stack_offsets_clamp_swap_single():
    # swapped, both ends clamped just inside +-8, end included
    offs = stack_offsets(10, -10, 5, 8)
    assert offs == pytest.approx([-7.999, -2.999, 2.001, 7.001, 7.999])
    assert stack_offsets(3, 3, 1, 10) == [3]
    assert stack_offsets(0, 5, 0, 10) == [0]
    # k*step, never accumulated: 100 planes end exactly on 20
    offs = stack_offsets(0.2, 20, 0.2, 100)
    assert len(offs) == 100
    assert offs[-1] == pytest.approx(20.0, abs=1e-12)


# --- the stack tracer ----------------------------------------------------------

def test_trace_stack_matches_unmasked_preview(tmp_path):
    from backend.sections import trace_stack, trace
    from stl_to_solid.mesh_prep import load_mesh
    p = str(synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl'))
    m = load_mesh(p)
    c = m.bounding_box.centroid
    offs = [-15.0, -5.0, 5.0, 15.0]
    stack = trace_stack(p, 'z', offs, tol=0.08)
    assert [s['offset'] for s in stack] == offs
    for off, sec in zip(offs, stack):
        origin = np.array([0, 0, c[2] + off])
        ref = section_preview(m.vertices, m.faces, origin, [0, 0, 1], tol=0.08, join_mm=2.5)
        assert sec['stats'] == ref['stats']
        assert len(sec['polylines']) == len(ref['polylines'])
        for a, b in zip(sec['polylines'], ref['polylines']):
            assert np.allclose(a, b)
        assert not sec.get('empty')
    one = trace(p, 'z', 5.0)
    assert one['stats'] == stack[2]['stats'] and one['at'] == stack[2]['at']


def test_trace_stack_marks_empty_planes_and_budget(tmp_path):
    from backend.sections import trace_stack
    p = str(synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl'))
    far = trace_stack(p, 'z', [1000.0])[0]
    assert far['empty'] and far['polylines'] == [] and far['stats']['loops'] == 0
    with pytest.raises(TimeoutError) as e:
        trace_stack(p, 'z', [-5.0, 0.0, 5.0], budget_s=0)
    assert '1 of 3' in str(e.value)


# --- the routes + the script ------------------------------------------------------

def _data(txt):
    """TITLE and SECTIONS as the script embeds them (skipping its adsk import)."""
    ns = {}
    exec(txt[txt.index('TITLE ='):txt.index('def _plane(')], ns)
    return ns


def _client(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STLTOSOLID_DATA', str(tmp_path / 'data'))
    from backend import jobs, main
    importlib.reload(jobs)
    return TestClient(main.app)


def _upload(c, path, name='shaft.stl'):
    with open(path, 'rb') as fh:
        r = c.post('/api/jobs', files={'file': (name, fh, 'application/octet-stream')})
    assert r.status_code == 200, r.text
    return r.json()['id']


def test_xray_api_count_script_and_cap(tmp_path, monkeypatch):
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, p)
        r = c.get(f'/api/jobs/{job}/xray', params={'axis': 'z', 'from': -10, 'to': 10, 'step': 1})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d['count'] == 21 and not d['over_cap'] and d['max'] == 500
        assert d['offsets'][0] == pytest.approx(-10) and d['offsets'][-1] == pytest.approx(10)

        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -10, 'to': 10, 'step': 1})
        assert r.status_code == 200, r.text
        txt = r.text
        ast.parse(txt)
        assert txt.count("'name': 'xray Z=") == 21
        assert '(21/21)' in txt and '(1/21)' in txt
        assert 'isComputeDeferred' in txt and 'createProgressDialog' in txt
        assert 'def _extrude_slice' in txt and txt.count("'extrude_mm': None") == 21
        assert 'sketchCircles.addByCenterRadius' in txt      # the drawing loop
        assert '_xray_z-10.0_+10.0_s1.py' in r.headers['content-disposition']

        # the sections embedded in the script hold the cut: a circle per plane
        secs = _data(txt)['SECTIONS']
        assert len(secs) == 21
        assert all(isinstance(s['loops'][0][0], dict) for s in secs)   # full circles
        assert secs[0]['origin'][2] == pytest.approx(12.5, abs=0.01)

        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -20, 'to': 20, 'step': 0.01})
        assert r.status_code == 400 and 'over the limit' in r.json()['detail']
        r = c.get(f'/api/jobs/{job}/xray-script', params={'axis': 'z', 'step': 0})
        assert r.status_code == 400
        r = c.get(f'/api/jobs/{job}/xray', params={'axis': 'q'})
        assert r.status_code == 400


def test_xray_script_single_plane_matches_section(tmp_path, monkeypatch):
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, p)
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': 5, 'to': 5, 'step': 1})
        assert r.status_code == 200, r.text
        secs = _data(r.text)['SECTIONS']
        assert len(secs) == 1 and secs[0]['name'] == 'xray Z=27.50 mm (1/1)'
        one = c.get(f'/api/jobs/{job}/section', params={'axis': 'z', 'offset': 5}).json()
        circle = secs[0]['loops'][0][0]
        assert circle['r'] == pytest.approx(one['loops'][0][0]['r'], abs=1e-3)
        assert '_xray_z+5.0_+5.0_s1.py' in r.headers['content-disposition']


def test_xray_script_extrudes_each_slab(tmp_path, monkeypatch):
    """extrude=true: every slice carries the signed distance to the next
    plane, the last one a full spacing; the script holds the join/new-body
    extrude code. Without it no slice extrudes."""
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, p, 'plate.stl')
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -2, 'to': 2, 'step': 2, 'extrude': 'true'})
        assert r.status_code == 200, r.text
        secs = _data(r.text)['SECTIONS']
        assert [s['extrude_mm'] for s in secs] == pytest.approx([2.0, 2.0, 2.0])
        assert 'JoinFeatureOperation' in r.text and 'NewBodyFeatureOperation' in r.text
        assert "sgn * sec['extrude_mm'] / 10.0" in r.text
        assert r.headers['content-disposition'].endswith('_s2_solid.py"')
        # a plate with holes: outer + holes in every sketch
        assert all(len(s['loops'][0][1]) >= 1 for s in secs)
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -2, 'to': 2, 'step': 2})
        assert r.text.count("'extrude_mm': None") == 3
        assert not r.headers['content-disposition'].endswith('_solid.py"')


def test_emit_xray_script_draws_lines_arcs_and_splines():
    from stl_to_solid.fusion_export import emit_fusion_xray_script
    p = synth.export(synth.rounded_rect(), '/tmp/_s2s_xray_rr.stl')
    m = trimesh.load(p, force='mesh')
    sec = section_preview(m.vertices, m.faces, [0, 0, 0], [0, 0, 1], tol=0.08)
    txt = emit_fusion_xray_script([{'origin': sec['origin'], 'normal': sec['normal'],
                                    'name': 'rr', 'loops': sec['loops'], 'open': []}], title='t')
    ast.parse(txt)
    ns = _data(txt)
    prims = ns['SECTIONS'][0]['loops'][0][0]
    kinds = [q['type'] for q in prims]
    assert kinds.count('line') == 4 and kinds.count('arc') == 4
    assert ns['TITLE'] == 't'
