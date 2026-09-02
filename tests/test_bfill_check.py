"""The OCC dry run behind the Boundary Fill outlook: it builds the tools the
Fusion runtime builds (the script's own arc growth, one window for both
ends of a cone), classifies the cells the same way, honours the script's
SKIP list, judges every body on its own, says 'not checked' rather than
guessing when it cannot run, and is fast enough to run on every conversion."""
import importlib.util
import math
import os
import time

import pytest

from . import synth


def _script(tmp_path, name, builder):
    from stl2prism.mesh_prep import load_and_prep_bodies
    from stl2prism import facegroups
    from stl2prism.fusion_boundary_fill import emit_boundary_fill_script
    p = synth.export(builder(), tmp_path / f'{name}.stl')
    bodies, _, _ = load_and_prep_bodies(p, verbose=False)
    _, stats = facegroups.convert(bodies[0].mesh, tol=0.08)
    return emit_boundary_fill_script([dict(stats['export'], mode='facegroup')], name)


def _runtime_ns():
    from stl2prism.fusion_boundary_fill import RUNTIME
    from stl2prism.bfill_check import parse_script
    return parse_script('BODIES = []\nEXPAND = 0.1\nEXPAND_MIN = 0.01\nSKIP = []\n' + RUNTIME)


def test_arc_growth_is_the_runtimes_own():
    """The dry run grows arcs with the script's _arc_span — min(pi, E/r) per
    side, the whole circle within 0.02 rad of closing — the rule the old
    emulation (a pi/6 cap, no closure) had drifted from."""
    from stl2prism.bfill_check import _ax2
    span = _runtime_ns()['_arc_span']
    assert span(1.0, 0.0, 1.0, 0.1) == pytest.approx((-0.1, 1.1))
    assert span(0.05, 0.0, 1.5, 0.1) == pytest.approx((-2.0, 3.5))         # E/r = 2 rad, not pi/6
    assert span(0.01, 0.0, 0.5, 0.1) is None      # growth capped at pi: the whole circle
    assert span(1.0, 0.0, 6.25, 0.1) is None                                # closes to a circle
    assert span(1.0, None, None, 0.1) is None and span(0.0, 0.0, 1.0, 0.1) is None
    ax2, length = _ax2((0, 0, 0), (0, 0, 1), (1, 0, 0), 1.0, span(1.0, 0.0, 1.0, 0.1))
    assert length == pytest.approx(1.2)
    d = ax2.XDirection()
    assert (d.X(), d.Y()) == pytest.approx((math.cos(-0.1), math.sin(-0.1)))
    assert _ax2((0, 0, 0), (0, 0, 1), (1, 0, 0), 1.0, None)[1] == pytest.approx(2 * math.pi)


def test_runtime_helpers_the_dry_run_executes_are_present():
    ns = _runtime_ns()
    assert callable(ns['_expand']) and callable(ns['_offset_convex'])
    assert ns['SLIVER_CM'] > 0 and callable(ns['_cell_is_material'])
    m = ns['_Mesh']([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    assert m.box_outside((2, 2, 2), (3, 3, 3)) and not m.box_outside((0.5, 0.5, 0.5), (3, 3, 3))


def test_old_scripts_are_refused_clearly():
    from stl2prism.fusion_boundary_fill import RUNTIME
    from stl2prism.bfill_check import parse_script
    old = RUNTIME.replace('def _arc_span', 'def _arc_span_gone')
    with pytest.raises(ValueError, match='regenerate'):
        parse_script('BODIES = []\nEXPAND = 0.1\nEXPAND_MIN = 0.01\n' + old)


def test_dry_run_closes_a_clean_part_quickly(tmp_path):
    from stl2prism.bfill_check import check_script, outlook
    text = _script(tmp_path, 'ft', synth.fillet_top)
    t0 = time.time()
    res = check_script(text)
    assert time.time() - t0 < 10.0
    assert len(res) == 1
    r = res[0]
    assert r['error'] is None and r['tools_failed'] == 0 and r['tools_skipped'] == 0
    assert 99.0 <= r['enclosed_pct'] <= 103.0 and r['unenclosed'] == []
    assert r['kept_cells'] >= 1 and not r['fallback']
    v = outlook(res)
    assert v['ok'] is True and 'enclose' in v['reason'] and v['enclosed_pct'] == pytest.approx(r['enclosed_pct'], abs=0.01)


def test_dry_run_honours_the_scripts_skip_list(tmp_path):
    """SKIP indexes the tools that were built, as in the runtime; with two
    walls of the box left out the cells no longer enclose the part."""
    from stl2prism.bfill_check import check_script, outlook
    text = _script(tmp_path, 'ft', synth.fillet_top).replace('SKIP = []', 'SKIP = [0, 1]', 1)
    res = check_script(text)
    r = res[0]
    assert r['tools_skipped'] == 2
    assert r.get('error') or r['unenclosed'] or not (99.0 <= r['enclosed_pct'] <= 103.0)
    assert outlook(res)['ok'] is not True


def _script_n_bodies(tmp_path, n):
    """The fillet_top script with its one body repeated `n` times."""
    text = _script(tmp_path, 'ft', synth.fillet_top)
    head, rt = text.split('\nimport adsk.core')
    return head.replace('BODIES = [', f'BODIES = {n} * [', 1) + '\nimport adsk.core' + rt


def test_dry_run_budget_leaves_later_bodies_unchecked(tmp_path):
    from stl2prism.bfill_check import check_script, outlook
    text = _script(tmp_path, 'ft', synth.fillet_top)
    # one body: a zero budget still checks it (the budget is tested between bodies)
    res = check_script(text, budget_s=0.0)
    assert res[0]['error'] is None
    two = _script_n_bodies(tmp_path, 2)
    res = check_script(two, budget_s=0.0)
    assert res[0]['error'] is None and res[1]['error'].startswith('not checked')
    assert outlook(res)['ok'] is None


def test_outlook_judges_every_body_and_the_dropped_ones():
    from stl2prism.bfill_check import outlook
    good = dict(name='a', enclosed_pct=100.0, unenclosed=[], n_probes=10, error=None,
                probes_by_region={'3 plane': 2, '4 cone': 4, '5 plane': 4})
    assert outlook([good])['ok'] is True
    over = dict(good, name='b', enclosed_pct=140.0)
    v = outlook([good, over])                       # min() over bodies would hide b
    assert v['ok'] is False and 'b: cells enclose 140.0%' in v['reason']
    # a whole region outside every cell (a small boss the tools never enclose): FAIL
    v = outlook([dict(good, name='c', unenclosed=['3 plane', '3 plane'])])
    assert v['ok'] is False and 'no cell holds region 3 plane' in v['reason']
    # a stray probe of a region that has others in cells: mentioned, not a failure
    v = outlook([dict(good, name='c', unenclosed=['4 cone'])])
    assert v['ok'] is True and '1 of 10 probe points' in v['reason']
    v = outlook([dict(good, name='c', unenclosed=['4 cone'], enclosed_pct=97.0)])
    assert v['ok'] is False and 'enclose 97.0%' in v['reason']
    v = outlook([good], dropped=[(1, 250)])
    assert v['ok'] is False and 'body 2: 250 regions' in v['reason']
    v = outlook([good, dict(name='d', error='MakerVolume reports errors')])
    assert v['ok'] is None and v['reason'].startswith('not checked') and 'MakerVolume' in v['reason']
    v = outlook([dict(good, tools_failed=1)])
    assert v['ok'] is True and '1 tool(s) could not be built' in v['reason']
    v = outlook([dict(good, tools_failed=1), over])
    assert v['ok'] is False and 'could not be built' in v['reason']


def test_unbuildable_tools_are_skipped_not_fatal(tmp_path):
    """A degenerate tool record must not abort the dry run (the runtime skips
    it and runs the fill without it): it is counted and the rest is checked."""
    from stl2prism.bfill_check import check_script
    text = _script(tmp_path, 'ft', synth.fillet_top)
    # a record no builder exists for (OCC builds most degenerate records
    # silently; an unknown kind is refused for sure)
    bad = "('blob', (0.0, 0.0, 0.0)),\n"
    assert "'surfaces': [\n" in text
    text = text.replace("'surfaces': [\n", "'surfaces': [\n" + bad, 1)
    res = check_script(text)
    assert res[0]['error'] is None and res[0]['tools_failed'] == 1
    assert 99.0 <= res[0]['enclosed_pct'] <= 103.0


def test_cli_module_has_no_import_side_effects():
    path = os.path.join(os.path.dirname(__file__), '..', 'tools', 'emulate_bfill.py')
    spec = importlib.util.spec_from_file_location('emulate_bfill', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)                  # used to run main() here and die on sys.argv
    assert m.main([]) == 2                      # usage, no crash
