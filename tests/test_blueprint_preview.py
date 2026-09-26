"""The live preview: which feature owns each triangle, the warm helper
process (answers, refusals, a crash it survives), and the /preview route."""
import json
import os
import time

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
pytest.importorskip('PIL')

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'sg90_recipe.json')


@pytest.fixture
def sg90():
    with open(FIX) as f:
        return json.load(f)


def test_feature_map_names_the_maker_of_each_face(sg90):
    from collections import Counter
    from stl_to_solid.blueprint.validate import validate
    from stl_to_solid.blueprint.build_cq import build, preview_mesh, feature_map
    rep = validate(sg90)
    b = build(rep.resolved)
    m = preview_mesh(b)
    fm = feature_map(b, m)
    assert len(fm) == len(m.faces)
    by = Counter(fm)
    ids = [f['id'] for f in b.features]
    # the body owns most of the surface, the tabs and boss a good share,
    # the tab holes their walls; nothing is left unassigned to a feature
    assert set(by) <= set(range(len(ids)))
    import numpy as np
    owner = np.asarray(fm)
    body_area = m.area_faces[owner == ids.index('body')].sum()
    assert body_area > 0.4 * m.area          # a box: few triangles, most of the skin
    for fid in ('tabs', 'boss', 'tab_holes', 'slot', 'hub'):
        assert by[ids.index(fid)] > 0, fid
    # the tab-hole triangles sit at the holes' centres
    hole_tris = [i for i, o in enumerate(fm) if ids[o] == 'tab_holes']
    xs = m.triangles_center[hole_tris][:, 0]
    assert all((abs(x - (-4.85 + 2.35)) < 1.2) or (abs(x - (22.5 + 4.85 - 2.35)) < 1.2) for x in xs)


def test_quick_preview_writes_only_the_live_files(sg90, tmp_path):
    from stl_to_solid.blueprint.compile import quick_preview
    r = quick_preview(sg90, str(tmp_path))
    assert r['ok'] and r['solids'] == 1
    assert (tmp_path / 'live.stl').exists() and (tmp_path / 'live_features.json').exists()
    assert not (tmp_path / 'output.step').exists()
    assert len(r['triangle_feature']) > 100 and [f['id'] for f in r['features']][0] == 'body'


def test_helper_process_builds_refuses_and_survives_a_crash(sg90, tmp_path):
    from backend.preview import Helper, PreviewError
    h = Helper()
    try:
        t0 = time.time()
        ans = h.build(sg90, str(tmp_path / 'a'))
        first = time.time() - t0
        assert ans['ok'] and ans['bbox']['size'] == pytest.approx([32.2, 11.8, 29.9], abs=1e-3)
        t0 = time.time()
        ans = h.build(sg90, str(tmp_path / 'b'))
        second = time.time() - t0
        assert (tmp_path / 'b' / 'live.stl').exists()
        assert second < max(2.0, first)          # warm: no import cost the second time
        bad = json.loads(json.dumps(sg90))
        bad['features'][0]['shapes'][0]['w'] = '0'
        with pytest.raises(PreviewError) as e:
            h.build(bad, str(tmp_path / 'c'))
        assert e.value.kind == 'recipe' and 'w and h > 0' in str(e.value)
        assert h.alive()
        # the helper dies mid-request: the caller gets a sentence, the next call restarts it
        h._proc.kill()
        with pytest.raises(PreviewError):
            h.build(sg90, str(tmp_path / 'd'))
        ans = h.build(sg90, str(tmp_path / 'e'))
        assert ans['ok']
    finally:
        h.close()
    assert not h.alive()


def _client(tmp_path, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STLTOSOLID_DATA', str(tmp_path / 'data'))
    from backend import jobs, main
    importlib.reload(jobs)
    jobs._jobs.clear()
    return TestClient(main.app)


def test_preview_route(sg90, tmp_path, monkeypatch):
    from PIL import Image
    png = tmp_path / 'd.png'
    Image.new('RGB', (320, 240), 'white').save(png)
    with _client(tmp_path, monkeypatch) as c:
        with open(png, 'rb') as fh:
            job = c.post('/api/blueprints', files={'file': ('d.png', fh, 'image/png')}).json()['id']
        assert c.get(f'/api/blueprints/{job}/live.stl').status_code == 404
        assert c.get(f'/api/blueprints/{job}/preview-map').status_code == 404
        r = c.post(f'/api/blueprints/{job}/preview', json={'recipe': sg90})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d['solids'] == 1 and len(d['triangle_feature']) > 100
        import base64
        stl = base64.b64decode(d['stl_b64'])
        assert len(stl) > 1000 and 'stl_bytes' not in d
        # binary STL: 80-byte header then the triangle count, matching the map
        import struct
        assert struct.unpack('<I', stl[80:84])[0] == len(d['triangle_feature'])
        r2 = c.get(f'/api/blueprints/{job}/live.stl')
        assert r2.status_code == 200 and r2.headers['cache-control'] == 'no-store'
        bad = json.loads(json.dumps(sg90))
        bad['overall']['w'] = 50
        r = c.post(f'/api/blueprints/{job}/preview', json={'recipe': bad})
        assert r.status_code == 400 and 'along X' in r.json()['detail']
        # the job itself is untouched by previews
        assert c.get(f'/api/jobs/{job}').json()['status'] == 'uploaded'
    from backend.preview import helper
    helper.close()
