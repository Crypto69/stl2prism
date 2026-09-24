"""The worker pool: results in order, a slow task costs only itself, a
crash or a SystemExit in a worker costs only that task."""
import os
import time
import pytest

from stl_to_solid.parallel import run_tasks


# task functions must be importable by a spawned interpreter, so they live
# at module level here

def _double(x):
    print(f'doubling {x}')
    return 2 * x


def _sleepy(x):
    if x == 'slow':
        time.sleep(30)
    return x


def _sysexit(x):
    if x == 'exit':
        raise SystemExit(3)
    return x


def _crash(x):
    if x == 'crash':
        os._exit(7)
    return x


def test_results_come_back_in_task_order_with_their_logs():
    order = []
    out = run_tasks(_double, [3, 1, 2], workers=2, on_done=lambda i, e, k, n: order.append((i, k, n)))
    assert [e['result'] for e in out] == [6, 2, 4]
    assert all(e['ok'] for e in out)
    assert out[0]['log'].strip() == 'doubling 3'
    assert sorted(i for i, _, _ in order) == [0, 1, 2]
    assert sorted(k for _, k, _ in order) == [1, 2, 3] and all(n == 3 for _, _, n in order)


def test_timeout_kills_only_the_slow_task():
    t0 = time.monotonic()
    out = run_tasks(_sleepy, ['slow', 'a', 'b', 'c'], workers=2, timeout=2.0)
    assert time.monotonic() - t0 < 25
    assert out[0]['ok'] is False and out[0]['kind'] == 'timeout' and 'timed out' in out[0]['error']
    assert [e['result'] for e in out[1:]] == ['a', 'b', 'c']


def test_systemexit_in_a_task_is_a_failed_task_not_a_dead_pool():
    out = run_tasks(_sysexit, ['a', 'exit', 'b'], workers=1)
    assert out[1]['ok'] is False and out[1]['kind'] == 'exited' and 'SystemExit' in out[1]['error']
    assert [out[0]['result'], out[2]['result']] == ['a', 'b']


def test_dead_worker_is_replaced_and_the_rest_still_run():
    out = run_tasks(_crash, ['crash', 'a', 'b', 'c'], workers=2)
    assert out[0]['ok'] is False and out[0]['kind'] == 'died' and 'died' in out[0]['error']
    assert [e['result'] for e in out[1:]] == ['a', 'b', 'c']


def test_no_tasks_is_fine():
    assert run_tasks(_double, [], workers=4) == []


def _raise(x):
    if x == 'bad':
        raise ValueError('no good')
    return x


def _unpicklable(x):
    return (lambda: x) if x == 'lam' else x


def _partial_log(x):
    print('line one')
    print('line two')
    if x == 'slow':
        time.sleep(30)
    return x


def test_a_raising_task_is_reported_with_its_traceback():
    out = run_tasks(_raise, ['bad', 'a'], workers=1)
    assert out[0]['kind'] == 'raised' and 'ValueError: no good' in out[0]['error']
    assert 'Traceback' in out[0]['log']
    assert out[1]['result'] == 'a'


def test_an_unpicklable_result_is_a_failed_task_not_a_hang():
    out = run_tasks(_unpicklable, ['lam', 'a'], workers=1, timeout=20)
    assert out[0]['kind'] == 'raised' and 'could not be sent' in out[0]['error']
    assert out[1]['result'] == 'a'


def test_a_killed_task_keeps_the_lines_it_printed():
    out = run_tasks(_partial_log, ['slow'], workers=1, timeout=2.0)
    assert out[0]['kind'] == 'timeout'
    assert out[0]['log'].splitlines() == ['line one', 'line two']


def test_zero_timeout_means_no_limit():
    out = run_tasks(_partial_log, ['a', 'b'], workers=2, timeout=0)
    assert [e['result'] for e in out] == ['a', 'b']


def test_one_pool_serves_several_runs_and_reports_starts():
    from stl_to_solid.parallel import Pool
    starts = []
    with Pool(2) as pool:
        a = pool.run(_double, [1, 2, 3], on_start=starts.append)
        pids = {w.proc.pid for w in pool._pool}
        b = pool.run(_sleepy, ['a', 'b'])
        assert pids == {w.proc.pid for w in pool._pool}, 'workers are reused'
    assert [e['result'] for e in a] == [2, 4, 6] and [e['result'] for e in b] == ['a', 'b']
    assert sorted(starts) == [0, 1, 2]


def test_deadline_leaves_unstarted_tasks_unrun():
    out = run_tasks(_partial_log, ['a', 'b', 'c', 'd'], workers=1,
                    deadline=time.monotonic() + 1.5)
    kinds = [e['kind'] for e in out]
    assert kinds[0] == 'ok' and 'timeout' in kinds
    assert all(e['error'] == 'not started: past the deadline' for e in out if e['kind'] == 'timeout')


def test_an_unpicklable_task_is_a_failed_task():
    out = run_tasks(_double, [3, lambda: None], workers=1, timeout=10)
    assert out[0]['result'] == 6
    assert out[1]['kind'] == 'raised' and 'could not be sent' in out[1]['error']


def _who(x):
    return os.getpid()


def test_a_dead_pool_worker_is_replaced_between_runs():
    from stl_to_solid.parallel import Pool
    with Pool(1) as pool:
        pool.run(_crash, ['crash'])
        out = pool.run(_who, [1])
    assert out[0]['ok']
