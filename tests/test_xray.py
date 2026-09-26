"""X-Ray: a stack of section sketches between two planes (v0.4.5, solid
slabs reworked in v0.4.7).

The stack rule (`section_fit.stack_offsets`), the backend's stack tracer
(one mesh load, each plane cut with only the faces that span it), the
/xray and /xray-script routes, the embedded-data Fusion script, and its
extrude runtime run against a stand-in `adsk` that behaves like Fusion
(a Join that touches nothing quietly makes a new body; a hole's interior
is a profile too)."""
import ast
import sys
import types
from types import SimpleNamespace

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
    plane, the last one a full spacing but never past the part's far face
    (the 6 mm plate ends 1 mm above the last plane); each section carries
    its loops' material areas so the script can leave the hole interiors
    alone; the script holds the join/new-body extrude code. Without
    extrude no slice extrudes."""
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, p, 'plate.stl')
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -2, 'to': 2, 'step': 2, 'extrude': 'true'})
        assert r.status_code == 200, r.text
        secs = _data(r.text)['SECTIONS']
        assert [s['extrude_mm'] for s in secs] == pytest.approx([2.0, 2.0, 0.999], abs=1e-3)
        assert 'JoinFeatureOperation' in r.text and 'NewBodyFeatureOperation' in r.text
        assert "sgn * sec['extrude_mm'] / 10.0" in r.text
        assert 'def _material_profiles' in r.text and 'def _track' in r.text
        assert r.headers['content-disposition'].endswith('_s2_solid.py"')
        # a plate with holes: outer + holes in every sketch, and the material
        # area is the plate minus its four D5 holes
        assert all(len(s['loops'][0][1]) >= 1 for s in secs)
        want = 60 * 40 - 4 * np.pi * 2.5 ** 2
        for s in secs:
            assert len(s['areas_mm2']) == 1
            assert s['areas_mm2'][0] == pytest.approx(want, rel=0.01)
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -2, 'to': 2, 'step': 2})
        assert r.text.count("'extrude_mm': None") == 3
        assert not r.headers['content-disposition'].endswith('_solid.py"')


def test_xray_script_end_plane_leftover_runs_through(tmp_path, monkeypatch):
    """The end plane is always cut, so the slice before it can be a hair
    away. Then that slice's slab runs through to where the end slab would
    have finished and the end plane is drawn but not extruded: no 0.05 mm
    slab, and the stack still ends on the part's face."""
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, p, 'plate.stl')
        # z in -3..3, clamped to +-2.999: planes every 0.8 from -2.999 end at
        # 2.601, then 2.999 (0.398 away, under half a spacing)
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -3, 'to': 3, 'step': 0.8, 'extrude': 'true'})
        assert r.status_code == 200, r.text
        secs = _data(r.text)['SECTIONS']
        exts = [s['extrude_mm'] for s in secs]
        assert len(secs) == 9 and exts[-1] is None
        assert exts[-2] == pytest.approx(0.398, abs=1e-3)
        assert exts[:-2] == pytest.approx([0.8] * 7)
        assert sum(e for e in exts if e) == pytest.approx(5.998, abs=1e-3)
        # a stack that stops well inside the part keeps a full last slab
        r = c.get(f'/api/jobs/{job}/xray-script',
                  params={'axis': 'z', 'from': -1, 'to': 1, 'step': 0.5, 'extrude': 'true'})
        exts = [s['extrude_mm'] for s in _data(r.text)['SECTIONS']]
        assert exts == pytest.approx([0.5] * 5)
        # slivers trimmed anywhere in the stack are said in the closing message
        assert _data(r.text)['NOTES'] == []


def test_xray_servo_bracket_hole_strips_and_face(tmp_path, monkeypatch):
    """The part that showed the bugs (2026-09-25): a 5 x 5 mm arm with a 1.6 mm
    cross hole, sliced along X at 0.2 mm with Trim slivers 1.5. The 1.3 mm
    strip beside the hole used to come back as a stub (the notch with a
    pillar in it in Fusion); now the slices through the hole are the same
    with trim 1.5 as with trim 0, the stack ends on the +X face and not a
    spacing past it, and the wall slices carry their bore as a hole with
    the material area to match."""
    import os
    p = os.path.join(os.path.dirname(__file__), '..', 'samples', 'servo_bracket_1.stl')
    if not os.path.exists(p):
        pytest.skip('sample not present')
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, p, 'servo_bracket_1.stl')
        q = {'axis': 'x', 'from': 9.62, 'to': -9.63, 'step': 0.2, 'tol': 0.08, 'join': 4,
             'extrude': 'true'}
        a = _data(c.get(f'/api/jobs/{job}/xray-script', params={**q, 'trim': 1.5}).text)
        b = _data(c.get(f'/api/jobs/{job}/xray-script', params={**q, 'trim': 0}).text)
        sa, sb = a['SECTIONS'], b['SECTIONS']
        assert len(sa) == len(sb) == 98 and a['NOTES'] == []
        for x, y in zip(sa, sb):
            assert sorted(x['areas_mm2']) == pytest.approx(sorted(y['areas_mm2']), abs=1e-3)
        # through the hole: four strips, two of them 5 x 1.3..1.5, all full width
        assert len(sa[29]['loops']) == 4 and min(sa[29]['areas_mm2']) > 6.0
        # the stack: first plane just inside the -X face, slabs end on the +X face
        x0 = sa[0]['origin'][0]
        end = x0 + sum(s['extrude_mm'] or 0 for s in sa)
        assert x0 == pytest.approx(-79.0943, abs=1e-3) and end == pytest.approx(-59.8395, abs=2e-3)
        assert sa[-1]['extrude_mm'] is None and 0.04 < sa[-2]['extrude_mm'] < 0.06
        # the wall: one outline with the bore as a hole, area = outline minus bore
        assert len(sa[80]['loops']) == 1 and len(sa[80]['loops'][0][1]) == 1
        assert 600 < sa[80]['areas_mm2'][0] < 640


# --- the extrude runtime, against a stand-in Fusion ------------------------------

class _Body:
    _n = 0

    def __init__(self, col):
        _Body._n += 1
        self.entityToken = 'body%d' % _Body._n
        self.col = col                      # which column of the part it belongs to
        self.isValid = True


class _Profile:
    def __init__(self, area_cm2, col):
        self.area = area_cm2
        self.col = col

    def areaProperties(self):
        return SimpleNamespace(area=self.area)


class _Coll(list):
    @property
    def count(self):
        return len(self)

    def item(self, i):
        return self[i]

    def add(self, x):
        self.append(x)


class _Extrudes:
    """Behaves like Fusion: a Join whose profile touches a participant (same
    column) merges into it and reports that body; a Join that touches none
    does not fail, it quietly makes a new body; NewBody makes a new body."""
    def __init__(self):
        self.log = []

    def createInput(self, prof, op):
        return SimpleNamespace(prof=prof, op=op, dist=None, participantBodies=None,
                               setDistanceExtent=lambda *a: None)

    def add(self, inp):
        op = 'join' if inp.op == 1 else 'new'
        if op == 'join':
            hit = [b for b in (inp.participantBodies or []) if b.isValid and b.col == inp.prof.col]
            if hit:
                self.log.append(('join', inp.prof.col, hit[0].entityToken))
                return SimpleNamespace(bodies=_Coll([hit[0]]))
            b = _Body(inp.prof.col)
            self.log.append(('join-made-new', inp.prof.col, b.entityToken))
            return SimpleNamespace(bodies=_Coll([b]))
        b = _Body(inp.prof.col)
        self.log.append(('new', inp.prof.col, b.entityToken))
        return SimpleNamespace(bodies=_Coll([b]))


def _fake_adsk(monkeypatch):
    adsk = types.ModuleType('adsk')
    core, fusion = types.ModuleType('adsk.core'), types.ModuleType('adsk.fusion')
    core.ValueInput = SimpleNamespace(createByReal=staticmethod(lambda v: v))
    core.ObjectCollection = SimpleNamespace(create=staticmethod(_Coll))
    core.Point3D = SimpleNamespace(create=staticmethod(lambda *a: a))
    fusion.FeatureOperations = SimpleNamespace(NewBodyFeatureOperation=0, JoinFeatureOperation=1)
    adsk.core, adsk.fusion = core, fusion
    for name, mod in (('adsk', adsk), ('adsk.core', core), ('adsk.fusion', fusion)):
        monkeypatch.setitem(sys.modules, name, mod)
    return adsk


def _runtime(monkeypatch):
    from stl_to_solid.fusion_export import emit_fusion_xray_script
    _fake_adsk(monkeypatch)
    txt = emit_fusion_xray_script([], title='t')
    ns = {}
    exec(txt, ns)
    return ns


def test_xray_runtime_tracks_the_body_a_silent_join_makes(monkeypatch):
    """Two columns (the bracket's two arms). Slice 1: arm A is a new body,
    arm B's Join touches nothing and Fusion quietly makes a body for it.
    That body must be a participant of slice 2, or every later slab of arm
    B is a loose body of its own (73 of them on the servo bracket)."""
    ns = _runtime(monkeypatch)
    ext = _Extrudes()
    root = SimpleNamespace(features=SimpleNamespace(extrudeFeatures=ext))
    bodies, seen = [], set()
    for _ in range(4):
        sk = SimpleNamespace(profiles=_Coll([_Profile(0.25, 'A'), _Profile(0.25, 'B')]))
        made, ok = ns['_extrude_slice'](root, sk, 0.02, bodies, seen, [0.25])
        assert made == 2 and ok
    live = ns['_live'](bodies)
    assert len(live) == 2 and sorted(b.col for b in live) == ['A', 'B']
    kinds = [e[0] for e in ext.log]
    assert kinds[:2] == ['new', 'join-made-new']
    assert kinds[2:] == ['join'] * 6                   # every later slab joined its column


def test_xray_runtime_leaves_hole_interiors_alone(monkeypatch):
    """A plate slice: Fusion lists the ring (plate minus hole) and the hole's
    disc as two profiles. Only the one whose area matches the section's
    material area is extruded; with no match every profile is taken and
    the run is told (matched False); with no areas at all every profile is
    taken quietly (a script from before the areas existed)."""
    ns = _runtime(monkeypatch)
    ext = _Extrudes()
    root = SimpleNamespace(features=SimpleNamespace(extrudeFeatures=ext))
    ring, disc = _Profile(23.2146, 'A'), _Profile(0.19635, 'A')
    sk = SimpleNamespace(profiles=_Coll([ring, disc]))
    made, ok = ns['_extrude_slice'](root, sk, 0.02, [], set(), [23.2146])
    assert (made, ok) == (1, True) and ext.log[-1][1] == 'A' and len(ext.log) == 1
    made, ok = ns['_extrude_slice'](root, sk, 0.02, [], set(), [99.0])
    assert (made, ok) == (2, False)
    made, ok = ns['_extrude_slice'](root, sk, 0.02, [], set(), [])
    assert (made, ok) == (2, True)
    with pytest.raises(RuntimeError):
        ns['_extrude_slice'](root, SimpleNamespace(profiles=_Coll()), 0.02, [], set(), [])


def test_emit_xray_script_draws_lines_arcs_and_splines():
    from stl_to_solid.fusion_export import emit_fusion_xray_script
    p = synth.export(synth.rounded_rect(), '/tmp/_s2s_xray_rr.stl')
    m = trimesh.load(p, force='mesh')
    sec = section_preview(m.vertices, m.faces, [0, 0, 0], [0, 0, 1], tol=0.08)
    txt = emit_fusion_xray_script([{'origin': sec['origin'], 'normal': sec['normal'],
                                    'name': 'rr', 'loops': sec['loops'], 'open': [],
                                    'areas_mm2': sec['areas_mm2']}], title='t',
                                  notes=['3 slivers trimmed'])
    ast.parse(txt)
    ns = _data(txt)
    prims = ns['SECTIONS'][0]['loops'][0][0]
    kinds = [q['type'] for q in prims]
    assert kinds.count('line') == 4 and kinds.count('arc') == 4
    assert ns['TITLE'] == 't' and ns['NOTES'] == ['3 slivers trimmed']
    # 40 x 30 with r4 corners
    assert ns['SECTIONS'][0]['areas_mm2'][0] == pytest.approx(40 * 30 - (4 - np.pi) * 16, rel=0.01)
