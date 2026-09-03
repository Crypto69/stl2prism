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
            # A worker the kernel kills never gets to write result.json or
            # even a traceback: the log simply stops. Without this the UI
            # can only say "see log", and the log is empty by definition.
            job['killed'] = _killed_by(code)


# Signals that mean the worker was killed rather than failing on its own.
# SIGKILL is what the OOM killer sends (and Docker's memory limit); the
# others are a manual stop or a container shutdown.
_KILL_SIGNALS = {9: 'killed', 15: 'stopped', 2: 'stopped', 6: 'crashed'}


def _killed_by(code):
    """A note about a worker that died without reporting, or None.

    subprocess returns a negative code when a signal killed the child. -9
    is overwhelmingly an out-of-memory kill: several conversions can hold a
    few GB each, so a wide selection of bodies or too many workers for the
    container's memory limit takes the whole worker out.
    """
    if code >= 0:
        return None
    sig = -code
    kind = _KILL_SIGNALS.get(sig)
    if sig == 9:
        return {'signal': sig, 'kind': 'oom', 'message':
                'The conversion ran out of memory and was stopped by the '
                'system. Convert fewer bodies at once, or give the server '
                'more memory (STL2PRISM_WORKERS controls how many bodies '
                'are converted at the same time).'}
    return {'signal': sig, 'kind': kind or 'signal', 'message':
            f'The conversion was stopped by the system (signal {sig}) '
            f'before it could report an error.'}


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
        # No result and a killed worker: say what happened, since the log
        # stops mid-sentence and explains nothing.
        if out.get('result') is None and job.get('killed'):
            out['result'] = {'ok': False, 'error': job['killed']['message'],
                             'failure': job['killed']['kind']}
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
