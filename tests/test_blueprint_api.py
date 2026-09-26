"""Blueprint's routes and job plumbing: image upload, the read phase (a
fake reader), the build path through the real worker, refusals as
readable sentences, cancel while reading, and the key never touching
disk."""
import json
import os
import threading
import time

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
pytest.importorskip('PIL')

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'sg90_recipe.json')
BOX = {'name': 'box', 'units': 'mm', 'overall': {'w': 20, 'd': 15, 'h': 10},
       'params': [{'name': 'w', 'value': 20, 'source': 'label', 'inferred': False},
                  {'name': 'h', 'value': 10, 'source': 'label', 'inferred': False}],
       'features': [{'id': 'body', 'op': 'new_body', 'plane': 'XY', 'offset': 0, 'direction': '+',
                     'distance': 'h', 'through_all': False,
                     'shapes': [{'kind': 'rect', 'center': {'u': 'w/2', 'v': 7.5}, 'w': 'w', 'h': 15,
                                 'corner_radius': None}],
                     'source': {'view': 'top', 'labels': ['20', '15', '10'], 'inferred': False},
                     'confidence': 0.9}],
       'views_found': ['top', 'front'], 'notes': []}


def _client(tmp_path, monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STLTOSOLID_DATA', str(tmp_path / 'data'))
    for k in ('STLTOSOLID_ANTHROPIC_API_KEY', 'STLTOSOLID_OPENAI_API_KEY', 'STLTOSOLID_DEEPSEEK_API_KEY'):
        monkeypatch.delenv(k, raising=False)
    from backend import jobs, main
    importlib.reload(jobs)
    jobs._jobs.clear()
    return TestClient(main.app)


def _png(tmp_path, name='drawing.png', size=(320, 240)):
    from PIL import Image
    p = tmp_path / name
    Image.new('RGB', size, 'white').save(p)
    return p


def _upload(c, path, name=None, route='/api/blueprints'):
    with open(path, 'rb') as fh:
        return c.post(route, files={'file': (name or os.path.basename(path), fh, 'application/octet-stream')})


def _fake_read(monkeypatch, recipe=BOX, raise_=None, block=None):
    import stl_to_solid.blueprint.read_drawing as rd

    def fake(image, provider, key, model=None, base_url=None, hints='', timeout=0, **kw):
        if block is not None:
            block.wait(10)
        if raise_ is not None:
            raise raise_
        return {'recipe': recipe, 'raw_text': '{}', 'usage': {'input_tokens': 11, 'output_tokens': 22, 'calls': 1},
                'model': 'fake-model', 'provider': provider, 'repaired': False,
                'validation': {'errors': [], 'warnings': []}, 'seconds': 0.1}
    monkeypatch.setattr(rd, 'read_drawing', fake)


def _wait(c, job, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = c.get(f'/api/jobs/{job}').json()
        if s['status'] not in ('reading', 'queued', 'running', 'uploaded'):
            return s
        time.sleep(0.1)
    raise AssertionError(f'job did not settle: {s}')


def test_upload_image_and_serve_it(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        r = _upload(c, _png(tmp_path))
        assert r.status_code == 200, r.text
        d = r.json()
        assert d['width'] == 320 and d['height'] == 240
        job = d['id']
        s = c.get(f'/api/jobs/{job}').json()
        assert s['status'] == 'uploaded' and s['tool'] == 'blueprint'
        r = c.get(f'/api/blueprints/{job}/drawing')
        assert r.status_code == 200 and r.headers['content-type'].startswith('image/png')
        assert c.head(f'/api/blueprints/{job}/drawing').status_code == 200
        assert c.get(f'/api/blueprints/{job}/recipe').status_code == 404


def test_upload_refusals(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        txt = tmp_path / 'notes.txt'
        txt.write_text('hello')
        r = _upload(c, txt)
        assert r.status_code == 400 and 'not a drawing image' in r.json()['detail']
        fake = tmp_path / 'fake.png'
        fake.write_bytes(b'not really a png')
        r = _upload(c, fake)
        assert r.status_code == 400 and 'Could not read fake.png' in r.json()['detail']
        from backend import jobs
        assert not any(True for _ in os.scandir(jobs.DATA_DIR)) or all(
            not os.path.exists(os.path.join(jobs.DATA_DIR, n, 'input.png')) for n in os.listdir(jobs.DATA_DIR))
        # the mesh route refuses images and the drawing routes refuse meshes
        r = _upload(c, _png(tmp_path), route='/api/jobs')
        assert r.status_code == 400 and 'not a mesh file' in r.json()['detail']
        import trimesh
        stl = tmp_path / 'box.stl'
        trimesh.creation.box((10, 20, 30)).export(str(stl))
        job = _upload(c, stl, route='/api/jobs').json()['id']
        r = c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'}, headers={'X-Api-Key': 'k'})
        assert r.status_code == 400 and 'holds a mesh' in r.json()['detail']


def test_config_and_missing_key(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        cfg = c.get('/api/blueprints/config').json()
        keys = [p['key'] for p in cfg['providers']]
        assert keys == ['anthropic', 'openai', 'deepseek', 'custom']
        assert all(p['server_key'] is False for p in cfg['providers'])
        job = _upload(c, _png(tmp_path)).json()['id']
        r = c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'})
        assert r.status_code == 400
        d = r.json()['detail']
        assert 'Blueprint panel' in d and 'STLTOSOLID_ANTHROPIC_API_KEY' in d
        assert c.get(f'/api/jobs/{job}').json()['status'] == 'uploaded'
        r = c.post(f'/api/blueprints/{job}/read', json={'provider': 'custom', 'model': 'llava'})
        assert r.status_code == 400 and 'base URL' in r.json()['detail']
        monkeypatch.setenv('STLTOSOLID_OPENAI_API_KEY', 'sk-server')
        cfg = c.get('/api/blueprints/config').json()
        assert next(p for p in cfg['providers'] if p['key'] == 'openai')['server_key'] is True
        assert 'sk-server' not in json.dumps(cfg)


def test_read_then_build_then_download(tmp_path, monkeypatch):
    _fake_read(monkeypatch)
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, _png(tmp_path, 'sg90 drawing.png')).json()['id']
        r = c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'},
                   headers={'X-Api-Key': 'sk-test-secret-key'})
        assert r.status_code == 200 and r.json()['status'] == 'reading'
        s = _wait(c, job)
        assert s['status'] == 'done', s
        res = s['result']
        assert res['ok'] and res['mode'] == 'blueprint'
        assert res['recipe']['params'][0]['name'] == 'w'
        assert res['bbox']['size'] == pytest.approx([20, 15, 10], abs=1e-6)
        assert res['params']['read']['usage']['input_tokens'] == 11
        assert res['params']['read']['model'] == 'fake-model'
        assert 'read with fake-model' in s['log']
        for route in ('fusion-script', 'download', 'preview'):
            assert c.head(f'/api/jobs/{job}/{route}').status_code == 200, route
        r = c.get(f'/api/jobs/{job}/fusion-script')
        assert 'filename="sg90_drawing_fusion.py"' in r.headers['content-disposition']
        assert b'userParameters' in r.content
        rr = c.get(f'/api/blueprints/{job}/recipe').json()
        assert rr['recipe']['name'] == 'box' and rr['read']['model'] == 'fake-model'
        # the key is nowhere on disk
        from backend import jobs
        for root, _, files in os.walk(jobs.job_dir(job)):
            for n in files:
                with open(os.path.join(root, n), 'rb') as fh:
                    assert b'sk-test-secret-key' not in fh.read(), n


def test_read_failure_is_a_sentence(tmp_path, monkeypatch):
    from stl_to_solid.blueprint.providers import ReadError
    _fake_read(monkeypatch, raise_=ReadError('Anthropic refused the API key. Check the key in the Blueprint panel.', 'key'))
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, _png(tmp_path)).json()['id']
        c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'}, headers={'X-Api-Key': 'bad'})
        s = _wait(c, job)
        assert s['status'] == 'error'
        assert s['result']['failure'] == 'read'
        assert s['result']['error'].startswith('Anthropic refused the API key')


def test_read_with_failing_checks_keeps_the_recipe(tmp_path, monkeypatch):
    import stl_to_solid.blueprint.read_drawing as rd
    bad = json.loads(json.dumps(BOX))
    bad['overall']['w'] = 50

    def fake(image, provider, key, **kw):
        return {'recipe': bad, 'raw_text': '{}', 'usage': {}, 'model': 'm', 'provider': provider,
                'repaired': True, 'validation': {'errors': ['the built part is 20.00 mm along X but the '
                                                              'drawing says 50.00 (-60.0 %)'], 'warnings': []},
                'seconds': 0.1}
    monkeypatch.setattr(rd, 'read_drawing', fake)
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, _png(tmp_path)).json()['id']
        c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'}, headers={'X-Api-Key': 'k'})
        s = _wait(c, job)
        assert s['status'] == 'error' and s['result']['failure'] == 'recipe'
        assert '1 check failed' in s['result']['error'] and 'rebuild' in s['result']['error'].lower()
        rr = c.get(f'/api/blueprints/{job}/recipe')
        assert rr.status_code == 200 and rr.json()['recipe']['overall']['w'] == 50


def test_build_refuses_bad_recipes_and_busy_jobs(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, _png(tmp_path)).json()['id']
        bad = {'params': [{'name': 'w', 'value': -1}],
               'features': [{'op': 'melt', 'plane': 'QQ', 'shapes': [], 'distance': 'nope'}]}
        r = c.post(f'/api/blueprints/{job}/build', json={'recipe': bad})
        assert r.status_code == 400
        d = r.json()['detail']
        assert d.startswith('The recipe is not valid') and 'plane must be' in d and 'op must be' in d
        from backend import jobs
        jobs._jobs[job]['status'] = 'running'
        assert c.post(f'/api/blueprints/{job}/build', json={'recipe': BOX}).status_code == 409
        assert c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'},
                      headers={'X-Api-Key': 'k'}).status_code == 409
        jobs._jobs[job]['status'] = 'uploaded'


def test_build_the_fixture_through_the_worker(tmp_path, monkeypatch):
    with open(FIX) as f:
        rec = json.load(f)
    with _client(tmp_path, monkeypatch) as c:
        job = _upload(c, _png(tmp_path)).json()['id']
        r = c.post(f'/api/blueprints/{job}/build', json={'recipe': rec})
        assert r.status_code == 200 and r.json()['status'] == 'running'
        assert any('deduced' in w for w in r.json()['warnings'])
        s = _wait(c, job)
        assert s['status'] == 'done', s.get('result')
        res = s['result']
        assert res['solids'] == 1 and res['bbox']['size'] == pytest.approx([32.2, 11.8, 29.9], abs=1e-3)
        assert res['output_stats']['solids'] == 1
        step = c.get(f'/api/jobs/{job}/download')
        assert step.status_code == 200 and step.content.startswith(b'ISO-10303-21;')
        # a refused rebuild does not touch the previous result
        rec2 = json.loads(json.dumps(rec))
        rec2['features'][0]['op'] = 'cut'
        assert c.post(f'/api/blueprints/{job}/build', json={'recipe': rec2}).status_code == 400
        assert c.get(f'/api/jobs/{job}').json()['status'] == 'done'


def test_cancel_while_reading(tmp_path, monkeypatch):
    block = threading.Event()
    _fake_read(monkeypatch, block=block)
    started = []
    with _client(tmp_path, monkeypatch) as c:
        from backend import jobs
        monkeypatch.setattr(jobs, 'start', lambda *a, **k: started.append(a))
        job = _upload(c, _png(tmp_path)).json()['id']
        c.post(f'/api/blueprints/{job}/read', json={'provider': 'anthropic'}, headers={'X-Api-Key': 'k'})
        assert c.get(f'/api/jobs/{job}').json()['status'] == 'reading'
        assert c.get('/api/jobs/running').json()['running'] == 1
        r = c.post(f'/api/jobs/{job}/cancel')
        assert r.json()['status'] == 'cancelled'
        block.set()
        s = _wait(c, job)
        assert s['status'] == 'cancelled'
        assert started == []
