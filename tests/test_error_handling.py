"""Every failure reaches the user as one readable sentence.

The API answers JSON with a 'detail' string for validation errors and for
bugs alike; an upload that cannot be read leaves no job behind; a worker
that cannot be started, that crashes, or that writes no result still ends
the job in a state the UI can show; and the worker itself writes
result.json for every failure short of being killed."""
import json
import os
import signal
import threading
import time

import pytest
import trimesh

pytest.importorskip('fastapi')
pytest.importorskip('httpx')


def _client(tmp_path, monkeypatch, raise_server_exceptions=True):
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STLTOSOLID_DATA', str(tmp_path / 'data'))
    from backend import jobs, main
    importlib.reload(jobs)
    jobs._jobs.clear()
    return TestClient(main.app, raise_server_exceptions=raise_server_exceptions)


def _box_stl(tmp_path):
    p = tmp_path / 'box.stl'
    trimesh.creation.box((10, 20, 30)).export(str(p))
    return p


def _upload(c, path, name='box.stl'):
    with open(path, 'rb') as fh:
        return c.post('/api/jobs', files={'file': (name, fh, 'application/octet-stream')})


# --- the API's error shape ---------------------------------------------------

def test_validation_error_is_one_readable_sentence(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    job = _upload(c, _box_stl(tmp_path)).json()['id']
    r = c.post(f'/api/jobs/{job}/convert', json={'tol': -1, 'units': 'furlongs'})
    assert r.status_code == 422
    detail = r.json()['detail']
    assert isinstance(detail, str)
    assert detail.startswith('Invalid settings: ')
    assert 'tol:' in detail and 'units:' in detail
    assert '[' not in detail            # not the pydantic record list


def test_unexpected_error_is_json_with_a_sentence(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch, raise_server_exceptions=False)
    from backend import jobs

    def boom(job_id):
        raise KeyError('proc')
    monkeypatch.setattr(jobs, 'public_state', boom)
    r = c.get('/api/jobs/000000000000')
    assert r.status_code == 500
    detail = r.json()['detail']
    assert detail.startswith('Something went wrong on the server')
    assert 'KeyError' in detail and 'server log' in detail


def test_unknown_job_says_what_to_do(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    for path in ('/api/jobs/000000000000', '/api/jobs/000000000000/download',
                 '/api/jobs/000000000000/bodies'):
        r = c.get(path)
        assert r.status_code == 404
        assert 'Load the file again' in r.json()['detail']
    assert c.post('/api/jobs/000000000000/cancel').status_code == 404


# --- uploads -------------------------------------------------------------------

def test_garbage_upload_is_refused_and_leaves_no_job(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    from backend import jobs
    bad = tmp_path / 'bad.stl'
    bad.write_bytes(b'this is not a mesh at all' * 100)
    r = _upload(c, bad, 'bad.stl')
    assert r.status_code == 400
    detail = r.json()['detail']
    assert detail.startswith('Could not read bad.stl') or 'holds no triangles' in detail
    assert not any(os.path.isdir(os.path.join(jobs.DATA_DIR, n))
                   for n in os.listdir(jobs.DATA_DIR)) if os.path.isdir(jobs.DATA_DIR) else True
    assert not jobs._jobs


def test_empty_upload_is_refused(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    empty = tmp_path / 'empty.stl'
    empty.write_bytes(b'')
    r = _upload(c, empty, 'empty.stl')
    assert r.status_code == 400
    assert r.json()['detail'] == 'The file is empty.'


def test_wrong_extension_names_the_formats(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = _upload(c, _box_stl(tmp_path), 'notes.txt')
    assert r.status_code == 400
    assert '.stl' in r.json()['detail'] and 'not a mesh file' in r.json()['detail']


# --- convert on a job that lost its input --------------------------------------

def test_convert_without_input_file_is_a_404_not_a_stuck_job(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    from backend import jobs
    d = jobs.job_dir('abcdefabcdef')
    os.makedirs(d)
    # a recovered job: params but no input file (the upload was cleared)
    with open(os.path.join(d, 'params.json'), 'w') as f:
        json.dump({}, f)
    r = c.post('/api/jobs/abcdefabcdef/convert', json={})
    assert r.status_code == 404
    assert 'Load it again' in r.json()['detail']
    assert jobs.get('abcdefabcdef')['status'] != 'running'


# --- job states the UI can show -------------------------------------------------

def test_corrupt_result_file_is_reported_not_a_500(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    from backend import jobs
    job_id, d = jobs.new_job()
    with open(os.path.join(d, 'result.json'), 'w') as f:
        f.write('{"ok": true, "mode": "prism')      # the worker died mid-write
    with jobs._lock:
        jobs._jobs[job_id].update(status='done', input='input.stl')
    st = jobs.public_state(job_id)
    assert st['status'] == 'error'
    assert st['result']['ok'] is False
    assert st['result']['failure'] == 'no_result'
    assert 'could not be read' in st['result']['error']


def test_launch_failure_ends_the_job_with_a_message(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    from backend import jobs
    monkeypatch.setattr(jobs, '_worker_cmd', lambda: [str(tmp_path / 'no-such-program')])
    job_id, d = jobs.new_job()
    (tmp_path / 'data' / job_id / 'input.stl').write_bytes(b'solid x\nendsolid x\n')
    with jobs._lock:
        jobs._jobs[job_id].update(status='uploaded', input='input.stl')
    jobs.start(job_id, 'x.stl', {})
    for _ in range(100):
        if jobs.get(job_id)['status'] not in ('queued', 'running'):
            break
        time.sleep(0.05)
    st = jobs.public_state(job_id)
    assert st['status'] == 'error'
    assert st['result']['failure'] == 'launch'
    assert st['result']['error'].startswith('The conversion could not be started')


def test_cancel_of_a_queued_job_never_starts_a_worker(tmp_path, monkeypatch):
    _client(tmp_path, monkeypatch)
    from backend import jobs
    job_id, d = jobs.new_job()
    with jobs._lock:
        jobs._jobs[job_id].update(status='queued', input='input.stl', proc=None)
    assert jobs.cancel(job_id) == 'cancelled'
    assert jobs._jobs[job_id]['cancelling'] is True
    # the run thread then sees the flag and ends the job as cancelled
    jobs._run_guarded(job_id)
    assert jobs._jobs[job_id]['status'] == 'cancelled'
    assert jobs.public_state(job_id)['result']['error'] == 'Conversion cancelled.'


@pytest.mark.parametrize('code, kind', [
    (0, None), (1, None), (-9, 'oom'), (-11, 'crashed'), (-6, 'crashed'), (-15, 'stopped'),
    (0xC0000005, 'crashed'), (0xC00000FD, 'crashed'), (0xC0000017, 'oom'), (0xC0000999, 'crashed'),
])
def test_killed_by_names_the_cause(code, kind):
    from backend.jobs import _killed_by
    got = _killed_by(code)
    if kind is None:
        assert got is None
    else:
        assert got['kind'] == kind
        assert got['message'][0].isupper() and got['message'].endswith('.')
        if kind == 'crashed':
            assert 'crashed' in got['message']
            assert 'Sliced Loft' in got['message']


# --- the worker always leaves a result ------------------------------------------

def _worker_files(tmp_path, params_text):
    d = tmp_path / 'job'
    d.mkdir()
    trimesh.creation.box((10, 20, 30)).export(str(d / 'input.stl'))
    (d / 'params.json').write_text(params_text)
    return [str(d / 'input.stl'), str(d / 'output.step'), str(d / 'params.json'),
            str(d / 'result.json')]


def test_worker_reports_unreadable_settings(tmp_path):
    from backend import worker
    args = _worker_files(tmp_path, '{not json')
    with pytest.raises(SystemExit) as e:
        worker.main(args)
    assert e.value.code == 1
    res = json.loads(open(args[3]).read())
    assert res['ok'] is False
    assert res['error'].startswith('The conversion failed: the conversion settings could not be read')


def test_worker_reports_pipeline_error_as_its_own_sentence(tmp_path, monkeypatch):
    import stl_to_solid
    from backend import worker

    def run(*a, **k):
        raise RuntimeError('no body crosses 1.00..2.00 mm along Z')
    monkeypatch.setattr(stl_to_solid, 'run', run)
    params = {'tol': 0.08, 'accept_p95': 0.25, 'accept_max': 0.26, 'accept_hole_max': 0.1,
              'accept_vol_pct': 2.0, 'force_prismatic': False}
    args = _worker_files(tmp_path, json.dumps(params))
    with pytest.raises(SystemExit):
        worker.main(args)
    res = json.loads(open(args[3]).read())
    assert res['ok'] is False
    assert res['error'] == 'The conversion failed: no body crosses 1.00..2.00 mm along Z'
    assert res['failure'] == 'RuntimeError'
    assert res['params'] == params


def test_worker_reports_out_of_memory_plainly(tmp_path, monkeypatch):
    import stl_to_solid
    from backend import worker

    def run(*a, **k):
        raise MemoryError()
    monkeypatch.setattr(stl_to_solid, 'run', run)
    params = {'tol': 0.08, 'accept_p95': 0.25, 'accept_max': 0.26, 'accept_hole_max': 0.1,
              'accept_vol_pct': 2.0, 'force_prismatic': False}
    args = _worker_files(tmp_path, json.dumps(params))
    with pytest.raises(SystemExit):
        worker.main(args)
    res = json.loads(open(args[3]).read())
    assert res['failure'] == 'oom'
    assert 'ran out of memory' in res['error']


def test_worker_reports_bad_body_indices_without_a_traceback_only(tmp_path, monkeypatch):
    from backend import worker, analysis

    def body_list(path):
        raise ValueError('mesh has no faces')
    monkeypatch.setattr(analysis, 'body_list', body_list)
    params = {'tol': 0.08, 'accept_p95': 0.25, 'accept_max': 0.26, 'accept_hole_max': 0.1,
              'accept_vol_pct': 2.0, 'force_prismatic': False, 'bodies': [0]}
    args = _worker_files(tmp_path, json.dumps(params))
    with pytest.raises(SystemExit):
        worker.main(args)
    res = json.loads(open(args[3]).read())
    assert res['ok'] is False
    assert 'the chosen bodies could not be found in the mesh (mesh has no faces)' in res['error']


def test_worker_refuses_wrong_argument_count(capsys):
    from backend import worker
    with pytest.raises(SystemExit) as e:
        worker.main(['only', 'two'])
    assert e.value.code == 2
    assert 'expected 4 arguments' in capsys.readouterr().err


# --- describe() -----------------------------------------------------------------

def test_describe_turns_exceptions_into_sentences():
    from backend.errors import describe
    assert 'ran out of memory' in describe(MemoryError())
    assert describe(RuntimeError('ThruSections failed')) == 'ThruSections failed'
    assert describe(KeyError('proc')) == "an internal error (KeyError: 'proc'); the log has the traceback"
    assert 'no message' in describe(ValueError())
    assert describe(OSError(28, 'No space left on device')) == 'the disk is full'
    assert describe(TimeoutError('3 of 9 slices fitted')) == '3 of 9 slices fitted'
    assert describe(FileNotFoundError(2, 'x', '/tmp/gone.stl')) == 'a file went missing (/tmp/gone.stl)'


# --- HEAD on the file routes ------------------------------------------------------

def test_file_routes_answer_head_so_the_ui_can_check_before_downloading(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    from backend import jobs
    job = _upload(c, _box_stl(tmp_path)).json()['id']
    # nothing converted yet: every file route says so, on HEAD and GET alike
    for path in ('download', 'script', 'fusion-script', 'fusion-bfill-script', 'preview'):
        assert c.head(f'/api/jobs/{job}/{path}').status_code == 404, path
        assert 'detail' in c.get(f'/api/jobs/{job}/{path}').json()
    with open(os.path.join(jobs.job_dir(job), 'output.step'), 'w') as f:
        f.write('ISO-10303-21;\n')
    h = c.head(f'/api/jobs/{job}/download')
    assert h.status_code == 200 and h.content == b''
    assert 'filename=' in h.headers['content-disposition']
    assert c.get(f'/api/jobs/{job}/download').content.startswith(b'ISO-10303-21;')
