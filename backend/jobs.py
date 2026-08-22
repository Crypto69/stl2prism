"""In-process job registry over subprocess workers.

One directory per job under DATA_DIR:
    input.<stl|obj>  params.json  output.step  result.json  log.txt
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

DATA_DIR = os.environ.get('STL2PRISM_DATA', os.path.join(os.getcwd(), 'data'))
MAX_AGE_S = int(os.environ.get('STL2PRISM_JOB_TTL', 24 * 3600))

_lock = threading.Lock()
_jobs = {}  # id -> dict

# A scan-repair run can peak at several GB; on a shared NAS conversions
# must queue up rather than run concurrently.
_run_slot = threading.BoundedSemaphore(
    max(1, int(os.environ.get('STL2PRISM_CONCURRENCY', 1))))


def job_dir(job_id):
    return os.path.join(DATA_DIR, job_id)


def new_job():
    job_id = uuid.uuid4().hex[:12]
    d = job_dir(job_id)
    os.makedirs(d, exist_ok=True)
    with _lock:
        _jobs[job_id] = {'id': job_id, 'status': 'created',
                         'created': time.time(), 'proc': None}
    return job_id, d


def get(job_id):
    with _lock:
        return _jobs.get(job_id)


def start(job_id, filename, params):
    d = job_dir(job_id)
    with open(os.path.join(d, 'params.json'), 'w') as f:
        json.dump(params, f)
    with _lock:
        _jobs[job_id].update(status='queued', filename=filename)
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()


def _run(job_id):
    d = job_dir(job_id)
    with _run_slot:
        with _lock:
            _jobs[job_id]['status'] = 'running'
            input_name = _jobs[job_id].get('input', 'input.stl')
        log = open(os.path.join(d, 'log.txt'), 'wb')
        # -u: unbuffered, so the log endpoint sees pipeline progress live.
        proc = subprocess.Popen(
            [sys.executable, '-u', '-m', 'backend.worker',
             os.path.join(d, input_name), os.path.join(d, 'output.step'),
             os.path.join(d, 'params.json'), os.path.join(d, 'result.json')],
            stdout=log, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log.close()
        with _lock:
            _jobs[job_id]['proc'] = proc
        code = proc.wait()
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job['proc'] = None
            job['status'] = 'done' if code == 0 else 'error'


def public_state(job_id):
    job = get(job_id)
    if job is None:
        return None
    d = job_dir(job_id)
    out = {'id': job_id, 'status': job['status'],
           'filename': job.get('filename')}
    try:
        with open(os.path.join(d, 'log.txt'), errors='replace') as f:
            out['log'] = f.read()
    except OSError:
        out['log'] = ''
    if job['status'] in ('done', 'error'):
        try:
            with open(os.path.join(d, 'result.json')) as f:
                out['result'] = json.load(f)
        except OSError:
            out['result'] = None
            if job['status'] == 'done':
                out['status'] = 'error'
    return out


def cleanup_old():
    """Drop job directories older than the TTL (best effort)."""
    now = time.time()
    try:
        entries = os.listdir(DATA_DIR)
    except OSError:
        return
    for name in entries:
        d = os.path.join(DATA_DIR, name)
        try:
            if os.path.isdir(d) and now - os.path.getmtime(d) > MAX_AGE_S:
                with _lock:
                    job = _jobs.get(name)
                    if job and job.get('proc'):
                        continue
                    _jobs.pop(name, None)
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
