"""Phase 3 of the parallel plan: inside one shell.

A big single shell scores its axis candidates in the pool; the Boundary
Fill dry run checks bodies in the pool. Both must give what the serial
route gives."""
import pytest
import trimesh

from . import synth
from stl2prism import pipeline


def test_pooled_axis_candidates_give_the_same_step(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'POOL_MIN_FACES', 0)
    p = synth.export(synth.slot_hex(), tmp_path / 'part.stl')
    r0 = pipeline.run(p, str(tmp_path / 'serial.step'), verbose=False, workers=0)
    r2 = pipeline.run(p, str(tmp_path / 'pooled.step'), verbose=True, workers=2)
    assert r0['mode'] == r2['mode'] == 'prismatic'
    assert synth.solid_stats(tmp_path / 'serial.step') == synth.solid_stats(tmp_path / 'pooled.step')
    assert r0['metrics']['dev_max'] == pytest.approx(r2['metrics']['dev_max'], abs=1e-9)


def test_pooled_axis_candidates_log_and_respect_the_budget(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'POOL_MIN_FACES', 0)
    p = synth.export(synth.plate_holes(), tmp_path / 'part.stl')
    r = pipeline.run(p, str(tmp_path / 'pooled.step'), verbose=True, workers=2)
    out = capsys.readouterr().out
    assert 'candidates scored on 2 worker(s)' in out
    assert r['mode'] == 'prismatic'
    monkeypatch.setattr(pipeline, 'AXIS_SEARCH_BUDGET_S', 1e-9)
    r = pipeline.run(p, str(tmp_path / 'late.step'), verbose=False, workers=2)
    assert r['mode'] in ('facegroup', 'faceted')


def test_small_shells_score_candidates_in_process(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError('pool used')
    from stl2prism.parallel import Pool
    monkeypatch.setattr(Pool, 'run', boom)
    p = synth.export(synth.plate_holes(), tmp_path / 'part.stl')
    assert pipeline.run(p, str(tmp_path / 'a.step'), verbose=False, workers=4)['mode'] == 'prismatic'


def _script_with_bodies(tmp_path, n):
    from .test_bfill_check import _script_n_bodies
    return _script_n_bodies(tmp_path, n)


def test_pooled_dry_run_matches_serial(tmp_path):
    from stl2prism.bfill_check import check_script, outlook
    from stl2prism.parallel import Pool
    text = _script_with_bodies(tmp_path, 3)
    serial = check_script(text)
    with Pool(2) as pool:
        pooled = check_script(text, pool=pool)
    keys = ('name', 'cells', 'kept_cells', 'enclosed_pct', 'unenclosed', 'error',
            'tools_failed', 'tools_skipped', 'tool_faces')
    assert [{k: r[k] for k in keys} for r in serial] == [{k: r[k] for k in keys} for r in pooled]
    assert pooled[0]['kept'] == [] and len(serial[0]['kept']) == serial[0]['kept_cells']
    assert outlook(serial) == outlook(pooled)


def test_pooled_dry_run_reports_lost_bodies_as_unchecked(tmp_path, monkeypatch):
    """A body past its budget, or one whose worker died, is 'not checked'
    with the reason, never a guessed verdict."""
    from stl2prism.bfill_check import check_script, outlook
    text = _script_with_bodies(tmp_path, 2)

    class FakePool:
        size = 2

        def run(self, fn, tasks, timeout=None, deadline=None, on_done=None,
                on_start=None, workers=None):
            assert len(tasks) == 2 and timeout == 7.0 and deadline is not None
            return [{'ok': False, 'kind': 'timeout', 'error': 'timed out after 7 s', 'log': ''},
                    {'ok': False, 'kind': 'died', 'error': 'worker process died (exit code -9)',
                     'log': ''}]
    res = check_script(text, budget_s=7.0, pool=FakePool())
    assert res[0]['error'] == 'not checked: past the 7 s budget'
    assert res[1]['error'] == 'not checked: worker process died (exit code -9)'
    assert outlook(res)['ok'] is None and res[0]['name'] == res[1]['name']


def test_pool_parity_includes_the_dry_run(tmp_path):
    stl = synth.assembly(tmp_path / 'asm.stl', sphere=True, hollow=True)
    r0 = pipeline.run(stl, str(tmp_path / 'a.step'), verbose=False, workers=0)
    r2 = pipeline.run(stl, str(tmp_path / 'b.step'), verbose=False, workers=2)
    assert r0['bfill_check'] == r2['bfill_check']
