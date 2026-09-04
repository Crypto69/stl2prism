"""In-process job registry over subprocess workers.

One directory per job under DATA_DIR:
    input.<stl|obj>  params.json  output.step  result.json  log.txt
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

DATA_DIR = os.environ.get('STL2PRISM_DATA', os.path.join(os.getcwd(), 'data'))
# Seconds a cancelled worker gets to stop politely before it is killed.
CANCEL_GRACE_S = 5
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
        job = _jobs.get(job_id)
    return job if job is not None else _recover(job_id)


def _recover(job_id):
    """Rebuild a job's entry from its directory, or None.

    The registry lives in memory, so a restart forgets every job while its
    files sit on disk untouched: the user is told "unknown job" about a
    conversion whose result is right there, and a restart mid-run turns a
    running job into a 404 rather than a failure they can read. Recover
    what the directory can prove and let the rest be reported honestly.
    """
    if not job_id or os.path.sep in job_id or job_id in ('.', '..'):
        return None
    d = job_dir(job_id)
    if not os.path.isdir(d):
        return None
    entry = {'id': job_id, 'proc': None, 'recovered': True,
             'created': os.path.getmtime(d), 'filename': None}
    try:
        entry['input'] = next(n for n in os.listdir(d) if n.startswith('input.'))
    except StopIteration:
        entry['input'] = None
    res = os.path.join(d, 'result.json')
    par = os.path.join(d, 'params.json')
    have_res, have_par = os.path.exists(res), os.path.exists(par)
    # A job can be converted more than once. params.json is rewritten at the
    # start of every run and result.json only at the end, so params newer
    # than result means a later run was still going when the server
    # stopped — reporting the older run's success would hide that.
    stale = (have_res and have_par
             and os.path.getmtime(par) > os.path.getmtime(res) + 1)
    if have_res and not stale:
        entry['status'] = 'done'          # public_state re-reads it
    elif have_par:
        entry['status'] = 'error'
        entry['stale_result'] = stale
        entry['killed'] = {'signal': None, 'kind': 'restart', 'message':
                           'The server restarted while this conversion was '
                           'running, so it did not finish. Convert again.'}
    else:
        entry['status'] = 'uploaded'
    with _lock:
        _jobs.setdefault(job_id, entry)
        return _jobs[job_id]


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
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            # its own process group: cancelling signals the group, so the
            # worker's shell pool goes with it instead of being orphaned
            start_new_session=True)
        log.close()
        with _lock:
            _jobs[job_id]['proc'] = proc
        code = proc.wait()
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job['proc'] = None
            if job.pop('cancelling', False):
                # The user stopped it: a state of its own, not a failure to
                # explain and not a success. Any result.json in the
                # directory belongs to an earlier run, so it is withheld.
                job['status'] = 'cancelled'
                job['killed'] = {'signal': None, 'kind': 'cancelled',
                                 'message': 'Conversion cancelled.'}
                job['stale_result'] = True
                return
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
    if job['status'] in ('done', 'error', 'cancelled'):
        try:
            if job.get('stale_result'):
                raise OSError('result belongs to an earlier run')
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


def cancel(job_id):
    """Stop a running conversion. Returns 'cancelled', 'not_running' or
    'unknown'.

    The worker is asked to stop with SIGTERM and given a moment to go, then
    killed. Its own children (the shell pool) die with it: Popen puts the
    worker in its own process group, so the signal goes to the group and no
    orphan is left converting in the background.
    """
    job = get(job_id)
    if job is None:
        return 'unknown'
    with _lock:
        proc = job.get('proc')
        if proc is None or job['status'] not in ('queued', 'running'):
            return 'not_running'
        job['cancelling'] = True
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=CANCEL_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
    return 'cancelled'


def running_summary():
    """{'running': n, 'jobs': [...]} for the jobs converting right now."""
    with _lock:
        live = [j for j in _jobs.values()
                if j.get('proc') is not None and j['status'] in ('queued', 'running')]
        return {'running': len(live),
                'jobs': [{'id': j['id'], 'filename': j.get('filename'),
                          'status': j['status']} for j in live]}


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
