"""A small pool of spawned worker processes with a wall-clock limit per task.

Why not concurrent.futures: a running task there cannot be stopped without
breaking the whole pool, and one OCC crash takes every in-flight task with
it. Here each worker owns its own task queue and its own result pipe, so a
task past its limit (or a worker that died under it) costs exactly that
task: the worker is killed and replaced, the others carry on, and a
message cut short by the kill corrupts nothing shared.

One Pool serves a whole conversion: its workers start on first use (the
interpreter plus OCP, a few seconds each) and are reused for every task
of every stage, and the clock on a task starts when the worker picks it
up. Tasks are plain picklable objects handed to a module-level function
(spawn needs an importable callable); results come back the same way.
OCC shapes do not pickle, so a task that builds one writes it to a file
and returns the path. A task's stdout goes to a file as it runs and comes
back with the result, so the parent can print one block per task, and a
task that was killed still leaves the lines it had printed. A worker
whose parent is gone exits, idle or mid-task.
"""
import multiprocessing as mp
import multiprocessing.connection
import os
import pickle
import queue
import tempfile
import threading
import time
import traceback

_POLL_S = 0.25


def _worker_main(task_q, conn, parent_pid):
    """The worker loop: run tasks from `task_q` until told to stop.

    Each task's stdout is written to the file the parent named. A task's
    exception is a failed task, not a dead worker; SystemExit is reported
    apart, since the task did not finish its own ladder. A killed parent
    must not leave workers holding hundreds of megabytes each: a watchdog
    thread ends the process when the parent is gone, mid-task or not."""
    import contextlib
    import sys

    def watchdog():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent_pid:
                os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    try:                                  # the start-up every task would pay
        from . import pipeline            # noqa: F401
        import cadquery                   # noqa: F401
    except Exception:
        pass
    while True:
        item = task_q.get()
        if item is None:
            return
        idx, fn, task, log_path = item
        _send(conn, ('start', idx))
        with open(log_path, 'w', buffering=1) as log:
            try:
                with contextlib.redirect_stdout(log):
                    out = fn(task)
                res = {'ok': True, 'kind': 'ok', 'result': out, 'error': None,
                       'exc_type': None}
            except SystemExit as e:
                res = {'ok': False, 'kind': 'exited', 'result': None,
                       'error': f'SystemExit({e.code}) in the worker',
                       'exc_type': 'SystemExit'}
            except BaseException as e:          # noqa: BLE001 - a task must not kill its worker
                log.write(traceback.format_exc())
                res = {'ok': False, 'kind': 'raised', 'result': None,
                       'error': f'{type(e).__name__}: {e}', 'exc_type': type(e).__name__}
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
                                 'exc_type': type(e).__name__,
                                 'error': f'result could not be sent: '
                                          f'{type(e).__name__}: {e}'}))


class _Worker:
    def __init__(self, ctx):
        self.task_q = ctx.Queue()
        self.conn, child_conn = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=_worker_main,
                                args=(self.task_q, child_conn, os.getpid()),
                                daemon=True)
        self.proc.start()
        child_conn.close()            # the child's death must show as EOF here
        self.current = None           # index of the task it holds
        self.started = None           # when the worker picked it up
        self.broken = False

    def send(self, idx, fn, task, log_path):
        self.current, self.started = idx, None
        self.task_q.put((idx, fn, task, log_path))

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


class Pool:
    """`size` spawned workers, started on first use and kept until close().

    run(fn, tasks, ...) runs fn(task) for every task and returns one entry
    per task, in task order: {'ok', 'kind', 'result', 'error', 'exc_type',
    'log'} with kind 'ok', 'raised' (the task raised), 'exited'
    (SystemExit in the task), 'timeout' (killed after `timeout` seconds
    of running; None or 0 means no limit) or 'died' (the worker process
    died under it, a kernel crash). Tasks not yet started by `deadline`
    (a time.monotonic value) are reported as 'timeout' without running.
    `on_start(idx)` and `on_done(idx, entry, n_done, n)` are called as
    tasks start and finish, in that order of events. Tasks start in the
    order given, so put the longest first; `workers` caps how many run at
    once for this call.
    """

    def __init__(self, size, log_dir=None):
        self.size = max(1, int(size))
        self._ctx = mp.get_context('spawn')
        self._pool = []
        self._tmp = None
        self._log_dir = log_dir
        self._n_logs = 0

    @property
    def log_dir(self):
        if self._log_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix='stltosolid-pool-')
            self._log_dir = self._tmp.name
        return self._log_dir

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for w in self._pool:
            w.stop()
        self._pool = []
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def run(self, fn, tasks, timeout=None, deadline=None, on_done=None,
            on_start=None, workers=None):
        tasks = list(tasks)
        n = len(tasks)
        results = [None] * n
        if n == 0:
            return results
        if timeout is not None and timeout <= 0:
            timeout = None
        workers = min(self.size, n, workers or self.size)
        while len(self._pool) < workers:
            self._pool.append(_Worker(self._ctx))
        pool = self._pool[:workers]
        for w in pool:
            w.current = None
        base = self._n_logs
        self._n_logs += n

        def log_path(idx):
            return os.path.join(self.log_dir, f'task{base + idx}.log')

        def read_log(idx):
            try:
                with open(log_path(idx), errors='replace') as f:
                    return f.read()
            except OSError:
                return ''

        pending = list(range(n))
        done = 0

        def finish(idx, entry):
            nonlocal done
            entry['log'] = read_log(idx)
            entry.setdefault('exc_type', None)
            results[idx] = entry
            done += 1
            if on_done is not None:
                on_done(idx, entry, done, n)

        def replace(k):
            w = _Worker(self._ctx)
            pool[k] = w
            self._pool[k] = w

        while done < n:
            now = time.monotonic()
            if deadline is not None and now > deadline:
                while pending:
                    finish(pending.pop(0), {'ok': False, 'kind': 'timeout', 'result': None,
                                            'error': 'not started: past the deadline'})
            for k, w in enumerate(pool):
                if w.current is None and pending:
                    if not w.proc.is_alive():        # died idle: replace, not use
                        w.kill()
                        replace(k)
                        w = pool[k]
                    idx = pending.pop(0)
                    try:
                        pickle.dumps(tasks[idx])
                    except Exception as e:
                        finish(idx, {'ok': False, 'kind': 'raised', 'result': None,
                                     'exc_type': type(e).__name__,
                                     'error': f'task could not be sent: {type(e).__name__}: {e}'})
                        continue
                    w.send(idx, fn, tasks[idx], log_path(idx))
            busy = [w for w in pool if w.current is not None]
            if not busy:
                continue
            ready = mp.connection.wait([w.conn for w in busy], timeout=_POLL_S)
            now = time.monotonic()
            for w in busy:
                if w.conn not in ready:
                    continue
                msg = w.recv()
                if msg is None:
                    continue                        # dead pipe: handled below
                if msg[0] == 'start' and msg[1] == w.current:
                    w.started = now
                    if on_start is not None:
                        on_start(w.current)
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
                    replace(k)
                elif w.broken or not w.proc.is_alive():
                    w.kill()
                    code = w.proc.exitcode
                    finish(w.current, {'ok': False, 'kind': 'died', 'result': None,
                                       'error': f'worker process died (exit code {code})'})
                    replace(k)
        return results


def run_tasks(fn, tasks, workers, timeout=None, on_done=None, log_dir=None,
              deadline=None, on_start=None):
    """One-off: a Pool for this call only."""
    with Pool(workers, log_dir=log_dir) as pool:
        return pool.run(fn, tasks, timeout=timeout, deadline=deadline,
                        on_done=on_done, on_start=on_start)


def default_workers():
    """Half the cores this process may use, at least one: the other half
    is left for the kernel's threads and whatever else the machine does."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    return max(1, n // 2)
