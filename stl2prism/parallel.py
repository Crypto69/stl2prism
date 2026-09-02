"""A small pool of spawned worker processes with a wall-clock limit per task.

Why not concurrent.futures: a running task there cannot be stopped without
breaking the whole pool, and one OCC crash takes every in-flight task with
it. Here each worker owns its own task queue and its own result pipe, so a
task past its limit (or a worker that died under it) costs exactly that
task: the worker is killed and replaced, the others carry on, and a
message cut short by the kill corrupts nothing shared. Workers are reused,
so the interpreter and OCP start-up is paid once per worker, not once per
task, and the clock on a task starts when the worker picks it up, not
when it is queued.

Tasks are plain picklable objects handed to a module-level function (spawn
needs an importable callable); results come back the same way. OCC shapes
do not pickle, so a task that builds one writes it to a file and returns
the path. A task's stdout goes to a file as it runs and comes back with
the result, so the parent can print one block per task, and a task that
was killed still leaves the lines it had printed.
"""
import multiprocessing as mp
import multiprocessing.connection
import os
import queue
import time
import traceback

_POLL_S = 0.25


def _worker_main(fn, task_q, conn, parent_pid):
    """The worker loop: run tasks from `task_q` until told to stop, or
    until the parent is gone (a killed parent must not leave workers
    holding hundreds of megabytes each).

    Each task's stdout is written to the file the parent named. A task's
    exception is a failed task, not a dead worker; SystemExit is reported
    apart, since the task did not finish its own ladder."""
    import contextlib
    import sys
    while True:
        try:
            item = task_q.get(timeout=1.0)
        except queue.Empty:
            if os.getppid() != parent_pid:
                return
            continue
        if item is None:
            return
        idx, task, log_path = item
        _send(conn, ('start', idx))
        with open(log_path, 'w', buffering=1) as log:
            try:
                with contextlib.redirect_stdout(log):
                    out = fn(task)
                res = {'ok': True, 'kind': 'ok', 'result': out, 'error': None}
            except SystemExit as e:
                res = {'ok': False, 'kind': 'exited',
                       'error': f'SystemExit({e.code}) in the worker', 'result': None}
            except BaseException as e:          # noqa: BLE001 - a task must not kill its worker
                log.write(traceback.format_exc())
                res = {'ok': False, 'kind': 'raised',
                       'error': f'{type(e).__name__}: {e}', 'result': None}
        sys.stdout.flush()
        _send(conn, ('done', idx, res))


def _send(conn, msg):
    """Send a message; a result that cannot be pickled becomes a failed
    task instead of a silently lost one."""
    try:
        conn.send(msg)
    except Exception as e:                       # pickling failed: nothing was written
        idx = msg[1]
        conn.send(('done', idx, {'ok': False, 'kind': 'raised', 'result': None,
                                 'error': f'result could not be sent: '
                                          f'{type(e).__name__}: {e}'}))


class _Worker:
    def __init__(self, ctx, fn):
        self.task_q = ctx.Queue()
        self.conn, child_conn = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=_worker_main,
                                args=(fn, self.task_q, child_conn, os.getpid()),
                                daemon=True)
        self.proc.start()
        child_conn.close()            # the child's death must show as EOF here
        self.current = None           # index of the task it holds
        self.started = None           # when the worker picked it up
        self.broken = False

    def send(self, idx, task, log_path):
        self.current, self.started = idx, None
        self.task_q.put((idx, task, log_path))

    def recv(self):
        """The next message, or None when the pipe is dead."""
        try:
            return self.conn.recv()
        except (EOFError, OSError):
            self.broken = True
            return None

    def stop(self):
        try:
            self.task_q.put(None)
            self.proc.join(timeout=5)
        finally:
            self.kill()

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(timeout=5)
        self.task_q.cancel_join_thread()
        self.conn.close()


def run_tasks(fn, tasks, workers, timeout=None, on_done=None, log_dir=None):
    """Run `fn(task)` for every task on `workers` spawned processes.

    Returns one entry per task, in task order: {'ok', 'kind', 'result',
    'error', 'log'} with kind 'ok', 'raised' (the task raised), 'exited'
    (SystemExit in the task), 'timeout' (killed after `timeout` seconds
    of running; None or 0 means no limit) or 'died' (the worker process
    died under it, a kernel crash). `on_done(idx, entry, n_done, n)` is
    called as each task finishes, in completion order. Tasks start in the
    order given, so put the longest first. `log_dir` holds each task's
    stdout file (a temporary directory when not given).
    """
    import tempfile
    tasks = list(tasks)
    n = len(tasks)
    results = [None] * n
    if n == 0:
        return results
    if timeout is not None and timeout <= 0:
        timeout = None
    workers = max(1, min(int(workers), n))
    ctx = mp.get_context('spawn')
    tmp = None
    if log_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix='stl2prism-pool-')
        log_dir = tmp.name

    def log_path(idx):
        return os.path.join(log_dir, f'task{idx}.log')

    def read_log(idx):
        try:
            with open(log_path(idx), errors='replace') as f:
                return f.read()
        except OSError:
            return ''

    pool = [_Worker(ctx, fn) for _ in range(workers)]
    pending = list(range(n))
    done = 0

    def finish(idx, entry):
        nonlocal done
        entry['log'] = read_log(idx)
        results[idx] = entry
        done += 1
        if on_done is not None:
            on_done(idx, entry, done, n)

    try:
        while done < n:
            for k, w in enumerate(pool):
                if w.current is None and pending:
                    if not w.proc.is_alive():        # died idle: replace, not use
                        w.kill()
                        pool[k] = w = _Worker(ctx, fn)
                    idx = pending.pop(0)
                    w.send(idx, tasks[idx], log_path(idx))
            busy = [w for w in pool if w.current is not None]
            ready = mp.connection.wait([w.conn for w in busy], timeout=_POLL_S)
            now = time.monotonic()
            for w in busy:
                if w.conn not in ready:
                    continue
                msg = w.recv()
                if msg is None:
                    continue                        # dead pipe: handled below
                if msg[0] == 'start':
                    w.started = now
                elif msg[0] == 'done' and msg[1] == w.current:
                    finish(w.current, msg[2])
                    w.current = None
            for k, w in enumerate(pool):
                if w.current is None:
                    continue
                if (timeout is not None and w.started is not None
                        and now - w.started > timeout):
                    w.kill()
                    finish(w.current, {'ok': False, 'kind': 'timeout', 'result': None,
                                       'error': f'timed out after {timeout:.0f} s'})
                    pool[k] = _Worker(ctx, fn)
                elif w.broken or not w.proc.is_alive():
                    w.kill()
                    code = w.proc.exitcode
                    finish(w.current, {'ok': False, 'kind': 'died', 'result': None,
                                       'error': f'worker process died (exit code {code})'})
                    pool[k] = _Worker(ctx, fn)
    finally:
        for w in pool:
            w.stop()
        if tmp is not None:
            tmp.cleanup()
    return results


def default_workers():
    """Half the cores this process may use, at least one: the other half
    is left for the kernel's threads and whatever else the machine does."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    return max(1, n // 2)
