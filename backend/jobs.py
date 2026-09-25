"""In-process job registry over subprocess workers.

One directory per job under DATA_DIR:
    input.<stl|obj>  params.json  output.step  result.json  log.txt
"""
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from .errors import describe

log = logging.getLogger('stltosolid.jobs')

DATA_DIR = os.environ.get('STLTOSOLID_DATA', os.path.join(os.getcwd(), 'data'))
# Seconds a cancelled worker gets to stop politely before it is killed.
CANCEL_GRACE_S = 5
MAX_AGE_S = int(os.environ.get('STLTOSOLID_JOB_TTL', 24 * 3600))

_lock = threading.Lock()
_jobs = {}  # id -> dict

# A scan-repair run can peak at several GB; on a shared NAS conversions
# must queue up rather than run concurrently.
_run_slot = threading.BoundedSemaphore(
    max(1, int(os.environ.get('STLTOSOLID_CONCURRENCY', 1))))


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


def discard(job_id):
    """Forget a job and delete its directory (an upload that failed
    half-way). Never raises: a job that cannot be removed is only a stray
    directory the TTL sweep takes later."""
    with _lock:
        job = _jobs.pop(job_id, None)
    if job is not None and job.get('proc') is not None:
        return
    shutil.rmtree(job_dir(job_id), ignore_errors=True)


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
    entry = _from_disk(job_id)
    if entry is None:
        return None
    with _lock:
        _jobs.setdefault(job_id, entry)
        return _jobs[job_id]


def _from_disk(job_id):
    """What the job's directory says about it right now, or None."""
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
        entry['stale_result'] = False
    return entry


def start(job_id, filename, params):
    """Queue a conversion. Raises OSError when params.json cannot be
    written (a full disk): the caller reports that instead of a job that
    would never start."""
    d = job_dir(job_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'params.json'), 'w') as f:
        json.dump(params, f)
    with _lock:
        if job_id not in _jobs:
            _jobs[job_id] = {'id': job_id, 'created': time.time(), 'proc': None}
        _jobs[job_id].update(status='queued', filename=filename)
    threading.Thread(target=_run, args=(job_id,), daemon=True).start()


_FROZEN = getattr(sys, 'frozen', False)
_WINDOWS = sys.platform == 'win32'


def _worker_cmd():
    """How to start backend.worker. A frozen (PyInstaller) interpreter
    cannot run `-m`, so it is started with a `--worker` flag its entry point
    dispatches. -u: unbuffered, so the log endpoint sees progress live."""
    if _FROZEN:
        return [sys.executable, '--worker']
    return [sys.executable, '-u', '-m', 'backend.worker']


def _own_group():
    """Popen kwargs that give the worker its own process group: cancelling
    stops the group, so the worker's shell pool goes with it instead of
    being orphaned."""
    if _WINDOWS:
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


def _kill_tree(proc, hard):
    """Stop the worker and its children. POSIX signals the process group
    (SIGTERM, or SIGKILL when hard); Windows has no process groups to
    signal, so taskkill /T takes the whole tree. Never raises: a worker
    that is already gone is the outcome wanted."""
    if _WINDOWS:
        try:
            r = subprocess.run(['taskkill', '/T', '/F', '/PID', str(proc.pid)],
                               capture_output=True, timeout=30)
            ok = r.returncode == 0
        except Exception:           # taskkill missing or hung
            ok = False
        if not ok:
            try:
                proc.kill()
            except Exception:
                pass
        return
    try:
        os.killpg(os.getpgid(proc.pid),
                  signal.SIGKILL if hard else signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            if hard:
                proc.kill()
            else:
                proc.terminate()
        except Exception:
            pass


def _run(job_id):
    try:
        _run_guarded(job_id)
    except Exception as e:
        # The thread is the only thing watching this job. Whatever went
        # wrong (no log file, a worker that could not be started), the job
        # must end in a state the UI can show rather than "running" forever.
        log.exception('job %s could not be run', job_id)
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            job['proc'] = None
            job.pop('cancelling', None)
            job['status'] = 'error'
            job['stale_result'] = True
            job['killed'] = {'signal': None, 'kind': 'launch', 'message':
                             'The conversion could not be started: '
                             f'{describe(e)}.'}


def _run_guarded(job_id):
    d = job_dir(job_id)
    with _run_slot:
        with _lock:
            job = _jobs.get(job_id)
            if job is None:
                return
            if job.pop('cancelling', False):
                # cancelled while it was still waiting for its turn
                job['status'] = 'cancelled'
                job['killed'] = {'signal': None, 'kind': 'cancelled',
                                 'message': 'Conversion cancelled.'}
                job['stale_result'] = True
                return
            job['status'] = 'running'
            input_name = job.get('input') or 'input.stl'
        # a stale result.json from an earlier run must never be read as
        # this run's answer if the worker dies before writing its own
        try:
            os.remove(os.path.join(d, 'result.json'))
        except OSError:
            pass
        with open(os.path.join(d, 'log.txt'), 'wb') as logf:
            proc = subprocess.Popen(
                _worker_cmd() + [
                    os.path.join(d, input_name), os.path.join(d, 'output.step'),
                    os.path.join(d, 'params.json'), os.path.join(d, 'result.json')],
                stdout=logf, stderr=subprocess.STDOUT,
                cwd=None if _FROZEN else os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))),
                **_own_group())
        with _lock:
            _jobs[job_id]['proc'] = proc
            # a cancel that landed between the slot check and Popen found
            # no process to kill and only left the flag: honour it now
            late_cancel = _jobs[job_id].get('cancelling', False)
        if late_cancel:
            _kill_tree(proc, hard=True)
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
# others are a manual stop, a container shutdown, or a crash inside the
# geometry kernel (OpenCascade is C++: a bad face can take the process
# down with SIGSEGV / SIGABRT / SIGBUS rather than raise).
_KILL_SIGNALS = {9: 'killed', 15: 'stopped', 2: 'stopped', 1: 'stopped',
                 6: 'crashed', 11: 'crashed', 7: 'crashed', 4: 'crashed', 8: 'crashed'}
# Windows has no signals: a crashed process exits with an NTSTATUS code.
_WIN_CRASH = {0xC0000005: 'an access violation', 0xC00000FD: 'a stack overflow',
              0xC0000409: 'a stack buffer overrun', 0xC0000374: 'a heap corruption',
              0xC000001D: 'an illegal instruction', 0xC0000094: 'a division by zero',
              0xC0000017: 'running out of memory', 0xC0000142: 'a DLL that failed to load',
              0xC0000135: 'a missing DLL (is the Microsoft Visual C++ runtime installed?)'}
_CRASH_HINT = (' Try converting fewer bodies at once, a slightly larger tolerance, or '
               'the Sliced Loft tool; the pipeline log above ends where it stopped.')


def _killed_by(code):
    """A note about a worker that died without reporting, or None.

    subprocess returns a negative code when a signal killed the child. -9
    is overwhelmingly an out-of-memory kill: several conversions can hold a
    few GB each, so a wide selection of bodies or too many workers for the
    container's memory limit takes the whole worker out. On Windows a
    crash shows as a large positive NTSTATUS code instead.
    """
    if code is None:
        return None
    if code >= 0:
        if code == 0xC0000017:
            return {'signal': code, 'kind': 'oom', 'message': _OOM_MESSAGE}
        if code in _WIN_CRASH or code >= 0xC0000000:
            what = _WIN_CRASH.get(code, f'code 0x{code:08X}')
            return {'signal': code, 'kind': 'crashed', 'message':
                    f'The conversion crashed ({what}) before it could report '
                    'an error.' + _CRASH_HINT}
        return None
    sig = -code
    kind = _KILL_SIGNALS.get(sig)
    if sig == 9:
        return {'signal': sig, 'kind': 'oom', 'message': _OOM_MESSAGE}
    if kind == 'crashed':
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = f'signal {sig}'
        return {'signal': sig, 'kind': 'crashed', 'message':
                f'The geometry kernel crashed ({name}) before it could report '
                'an error.' + _CRASH_HINT}
    return {'signal': sig, 'kind': kind or 'signal', 'message':
            f'The conversion was stopped by the system (signal {sig}) '
            f'before it could report an error.'}


_OOM_MESSAGE = ('The conversion ran out of memory and was stopped by the '
                'system. Convert fewer bodies at once, or give the server '
                'more memory (STLTOSOLID_WORKERS controls how many bodies '
                'are converted at the same time).')


def public_state(job_id):
    job = get(job_id)
    if job is None:
        return None
    # A recovered verdict is a snapshot of the directory at the moment it
    # was read. If the run it was waiting on has since finished, the entry
    # is stale: re-read rather than reporting a failure for a conversion
    # that succeeded afterwards.
    if job.get('recovered') and job.get('proc') is None:
        fresh = _from_disk(job_id)
        if fresh is not None and fresh['status'] != job['status']:
            with _lock:
                cur = _jobs.get(job_id)
                if cur is not None and cur.get('proc') is None:
                    cur.update(fresh)
                    job = cur
    d = job_dir(job_id)
    out = {'id': job_id, 'status': job['status'],
           'filename': job.get('filename')}
    try:
        with open(os.path.join(d, 'log.txt'), errors='replace') as f:
            out['log'] = f.read()
    except OSError:
        out['log'] = ''
    if job['status'] in ('done', 'error', 'cancelled'):
        unreadable = None
        try:
            if job.get('stale_result'):
                raise OSError('result belongs to an earlier run')
            with open(os.path.join(d, 'result.json')) as f:
                out['result'] = json.load(f)
            if not isinstance(out['result'], dict):
                raise ValueError('result is not an object')
        except OSError:
            out['result'] = None
        except ValueError as e:          # half-written or corrupt JSON
            log.warning('job %s: result.json unreadable: %s', job_id, e)
            out['result'] = None
            unreadable = e
        if out['result'] is None and job['status'] == 'done':
            out['status'] = 'error'
        # No result and a killed worker: say what happened, since the log
        # stops mid-sentence and explains nothing.
        if out.get('result') is None and job.get('killed'):
            out['result'] = {'ok': False, 'error': job['killed']['message'],
                             'failure': job['killed']['kind']}
        elif out.get('result') is None and out['status'] == 'error':
            out['result'] = {'ok': False, 'failure': 'no_result', 'error':
                             'The conversion ended without a result file'
                             + (' (it could not be read)' if unreadable else '')
                             + '. The pipeline log above shows how far it got; '
                             'convert again, and if it stops at the same place try '
                             'fewer bodies or a larger tolerance.'}
    return out


def cancel(job_id):
    """Stop a running conversion. Returns 'cancelled', 'not_running' or
    'unknown'.

    The worker is asked to stop with SIGTERM and given a moment to go, then
    killed. Its own children (the shell pool) die with it: Popen puts the
    worker in its own process group, so the signal goes to the group and no
    orphan is left converting in the background. Windows has no SIGTERM, so
    there the whole tree is killed at once.
    """
    job = get(job_id)
    if job is None:
        return 'unknown'
    with _lock:
        if job['status'] not in ('queued', 'running'):
            return 'not_running'
        proc = job.get('proc')
        job['cancelling'] = True
        if proc is None:
            # still waiting for its slot: _run_guarded sees the flag and
            # ends the job as cancelled without starting a worker
            return 'cancelled'
    _kill_tree(proc, hard=False)
    try:
        proc.wait(timeout=CANCEL_GRACE_S)
    except subprocess.TimeoutExpired:
        _kill_tree(proc, hard=True)
        try:
            proc.wait(timeout=CANCEL_GRACE_S)
        except subprocess.TimeoutExpired:
            log.warning('job %s: worker %s did not die', job_id, proc.pid)
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
