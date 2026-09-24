"""Phase 2 of the parallel plan: shells in a process pool.

The gate: a multi-body file gives the same STEP (modes, faces, volumes per
body) converted in this process and in a pool; a shell past its wall clock
comes back faceted; a worker that raises SystemExit or dies takes one shell
with it, not the job; a shell that raised keeps its error."""
import os
import time
import pytest
import trimesh

from . import synth
from stl_to_solid import pipeline


def _run(path, out, **kw):
    return pipeline.run(path, str(out), verbose=False, **kw)


@pytest.fixture(scope='module')
def parity(tmp_path_factory):
    """The five-shell assembly converted in-process: the reference."""
    d = tmp_path_factory.mktemp('parity')
    stl = synth.assembly(d / 'asm.stl', sphere=True, hollow=True)
    out = d / 'inproc.step'
    r = _run(stl, out, workers=0)
    return stl, out, r


def test_pool_gives_the_same_step_as_in_process(tmp_path, parity):
    stl, out0, r0 = parity
    out4 = tmp_path / 'pooled.step'
    r4 = _run(stl, out4, workers=4, shell_timeout=600)
    assert [b['mode'] for b in r0['bodies']] == [b['mode'] for b in r4['bodies']]
    assert all(b['error'] is None for b in r4['bodies'])
    assert synth.solid_stats(out0) == synth.solid_stats(out4)
    for b0, b4 in zip(r0['bodies'], r4['bodies']):
        for key in ('dev_p95', 'dev_max', 'vol_err_pct'):
            if key in b0['metrics']:
                assert b0['metrics'][key] == pytest.approx(b4['metrics'][key], abs=1e-9)
    assert r4['n_written'] == 4 and r4['metrics']['n_failed'] == 0


def test_one_worker_pool_matches_too(tmp_path, parity):
    stl, out0, _ = parity
    out1 = tmp_path / 'one.step'
    _run(stl, out1, workers=1, shell_timeout=600)
    assert synth.solid_stats(out0) == synth.solid_stats(out1)


def test_pool_logs_one_block_per_shell_and_progress(tmp_path, capsys):
    stl = synth.assembly(tmp_path / 'asm.stl', sphere=True, hollow=True)
    pipeline.run(stl, str(tmp_path / 'asm.step'), verbose=True, workers=2)
    text = capsys.readouterr().out
    assert '[pool] 5 shells on 2 worker(s)' in text
    assert '[progress] 5/5 shells' in text
    assert '[body 4/4 void 1/1] shell -> ' in text
    assert '[body 4/4 void 1/1] shell started (12 faces)' in text
    assert '[body 4/4] -> ' in text
    assert '[time] conversion' in text


# --- failure modes: stand-ins a spawned worker imports from this module ---
# 'b1' is the first body's outer shell; the real task runs in the worker.

def _slow_shell(task):
    if task['name'] == 'b1' and not task.get('faceted_only'):
        time.sleep(60)
    return pipeline._shell_task(task)


def _exiting_shell(task):
    if task['name'] == 'b1' and not task.get('faceted_only'):
        raise SystemExit(2)
    return pipeline._shell_task(task)


def _dying_shell(task):
    if task['name'] == 'b1' and not task.get('faceted_only'):
        os._exit(9)
    return pipeline._shell_task(task)


def _raising_shell(task):
    if task['name'] == 'b1':
        raise RuntimeError('faceted solid volume differs (pretend)')
    return pipeline._shell_task(task)


def _small(tmp_path):
    """Three bodies, four shells: a hollow box, a plate and a cube."""
    return synth.assembly(tmp_path / 'small.stl', sphere=False, hollow=True)


@pytest.mark.parametrize('stand_in,what', [
    (_slow_shell, 'timed out'),
    (_exiting_shell, 'SystemExit'),
    (_dying_shell, 'died'),
])
def test_a_lost_shell_comes_back_faceted_and_the_job_finishes(tmp_path, monkeypatch, stand_in, what):
    monkeypatch.setattr(pipeline, '_shell_task', stand_in)
    t0 = time.monotonic()
    out = tmp_path / 'lost.step'
    r = _run(_small(tmp_path), out, workers=2, shell_timeout=4)
    assert time.monotonic() - t0 < 60
    lost = r['bodies'][0]
    assert lost['error'] is None and lost['mode'] == 'faceted'
    assert what in lost['metrics']['shell_error']
    assert lost['metrics']['timed_out'] is (what == 'timed out')
    assert all(b['mode'] == 'prismatic' and 'shell_error' not in b['metrics']
               for b in r['bodies'][1:])
    assert r['n_written'] == 3
    got = synth.reimport(out)
    assert got['solids'] == 3 and got['naked_edges'] == 0


def test_a_shell_that_raised_keeps_its_error(tmp_path, monkeypatch):
    """The ladder ended in the worker (only the faceted rung's own refusal
    escapes it); the parent must not run it again."""
    monkeypatch.setattr(pipeline, '_shell_task', _raising_shell)
    r = _run(_small(tmp_path), tmp_path / 'raised.step', workers=2)
    assert 'pretend' in r['bodies'][0]['error']
    assert r['bodies'][0]['mode'] is None
    assert r['n_written'] == 2 and r['metrics']['n_failed'] == 1


def _slow_void(task):
    if task['name'].endswith('v1') and not task.get('faceted_only'):   # the one cavity
        time.sleep(60)
    return pipeline._shell_task(task)


def test_a_timed_out_void_is_reported_on_its_body(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, '_shell_task', _slow_void)
    r = _run(_small(tmp_path), tmp_path / 'void.step', workers=2, shell_timeout=4)
    m = next(b for b in r['bodies'] if b['voids'])['metrics']
    assert m['void_modes'] == ['faceted'] and m['timed_out'] is True
    assert m['void_shell_errors'][0].startswith('timed out')


def test_single_shell_and_small_files_stay_in_process(tmp_path, monkeypatch):
    """One shell has nothing to share, and a small file would spend longer
    starting workers than converting; with no explicit count the pool
    must not even start."""
    def boom(*a, **k):
        raise AssertionError('pool used')
    monkeypatch.setattr(pipeline, '_convert_all_pooled', boom)
    monkeypatch.setattr(pipeline, 'WORKERS', None)
    p = str(tmp_path / 'box.stl')
    trimesh.creation.box((20, 10, 5)).export(p)
    assert pipeline.run(p, str(tmp_path / 'box.step'), verbose=False, workers=4)['mode'] == 'prismatic'
    r = _run(synth.assembly(tmp_path / 'asm.stl', sphere=True, hollow=True), tmp_path / 'asm.step')
    assert r['n_written'] == 4


def test_memory_guard_sizes_the_pool_by_shells_in_flight(tmp_path, monkeypatch):
    seen = []
    real = pipeline._convert_all_pooled

    def spy(bodies, force_prismatic, verbose, gates, pool, workers, shell_timeout):
        seen.append(workers)
        return real(bodies, force_prismatic, verbose, gates, pool, workers, shell_timeout)
    monkeypatch.setattr(pipeline, '_convert_all_pooled', spy)
    stl = synth.assembly(tmp_path / 'asm.stl', sphere=True, hollow=True)   # 1280 + 4 x 12 faces
    monkeypatch.setattr(pipeline, 'POOL_ONE_WORKER_FACES', 1300)          # sphere + one box fit
    _run(stl, tmp_path / 'a.step', workers=4)
    monkeypatch.setattr(pipeline, 'POOL_ONE_WORKER_FACES', 1285)          # only the sphere fits
    _run(stl, tmp_path / 'b.step', workers=4)
    assert seen == [2, 1]
