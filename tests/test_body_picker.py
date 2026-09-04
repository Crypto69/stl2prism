"""Picking which bodies to convert: the shell list the UI shows, and the
filter run() applies to it."""
import numpy as np
import pytest
import trimesh

from . import synth
from stl2prism import pipeline
from stl2prism.pipeline import shell_keys, _pick_bodies
from stl2prism.mesh_prep import Body, load_and_prep_bodies
from backend.analysis import body_list


def _box(size, at=(0, 0, 0)):
    b = trimesh.creation.box((size, size, size))
    b.apply_translation(at)
    return b


_tbox = _box          # the mesh-level helper, named for the geometry tests

def _open_sheet(at=(0, 0, 0), w=10.0, h=5.0):
    """A shell that stays open through trimesh's split(): a folded strip.
    A box with one face removed does not work — split() re-processes each
    part and closes it again."""
    V = np.array([[0, 0, 0], [w, 0, 0], [0, w, 0], [w, w, 0],
                  [0, 0, h], [w, 0, h]], float) + np.asarray(at, float)
    F = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 1], [1, 4, 5]], np.int64)
    return trimesh.Trimesh(V, F, process=False)



def _three_parts(path):
    """A 40 cube, a 20 cube and a 10 cube, far apart: three plain bodies."""
    trimesh.util.concatenate(
        [_box(40), _box(20, (100, 0, 0)), _box(10, (200, 0, 0))]).export(str(path))
    return str(path)


# --- the shell list ---------------------------------------------------------

def test_body_list_reports_every_shell_largest_first(tmp_path):
    d = body_list(_three_parts(tmp_path / 'three.stl'))
    assert [b['triangles'] for b in d['bodies']] == [12, 12, 12]
    assert [round(b['volume']) for b in d['bodies']] == [64000, 8000, 1000]
    assert all(b['watertight'] for b in d['bodies'])
    assert not d['truncated']


def test_triangle_map_covers_every_triangle(tmp_path):
    d = body_list(_three_parts(tmp_path / 'three.stl'))
    tri = d['triangle_body']
    assert len(tri) == 36 and min(tri) == 0 and max(tri) == 2
    assert sorted(tri.count(i) for i in range(3)) == [12, 12, 12]


def test_suggested_picks_the_bodies_worth_converting(tmp_path):
    """The default ticks everything down to DEFAULT_PICK_FRAC of the
    largest body: here the 40 and 20 cubes (100% and 12.5%), not the 10
    cube (1.6%)."""
    d = body_list(_three_parts(tmp_path / 'three.stl'))
    assert [b['suggested'] for b in d['bodies']] == [True, True, False]


def test_suggested_leaves_the_small_parts_of_a_real_assembly(tmp_path):
    """The case this exists for: a housing with a scatter of small parts
    should tick the housing alone."""
    parts = [_box(40)] + [_box(4, (100 + 20 * i, 0, 0)) for i in range(5)]
    stl = str(tmp_path / 'asm.stl')
    trimesh.util.concatenate(parts).export(stl)
    d = body_list(stl)
    assert [b['suggested'] for b in d['bodies']] == [True] + [False] * 5


def test_a_hollow_body_lists_its_cavity_as_a_shell(tmp_path):
    stl = str(tmp_path / 'hollow.stl')
    v = _box(20)
    v.invert()
    trimesh.util.concatenate([_box(40), v]).export(stl)
    d = body_list(stl)
    assert len(d['bodies']) == 2          # the UI sees outer and cavity


# --- the filter -------------------------------------------------------------

def test_pick_by_index(tmp_path):
    prep, _, _ = load_and_prep_bodies(_three_parts(tmp_path / 'three.stl'), verbose=False)
    got = _pick_bodies(prep, [0, 2], False)
    assert [round(abs(b.mesh.volume)) for b in got] == [64000, 1000]


def test_pick_by_shell_key_survives_unit_scaling(tmp_path):
    stl = _three_parts(tmp_path / 'three.stl')
    keys = [tuple(b['key']) for b in body_list(stl)['bodies']]
    for units in ('mm', 'cm', 'in'):
        prep, _, _ = load_and_prep_bodies(stl, verbose=False, units=units)
        got = _pick_bodies(prep, [keys[0], keys[1]], False)
        assert [len(b.mesh.faces) for b in got] == [12, 12]
        big = max(abs(b.mesh.volume) for b in got)
        assert big == pytest.approx(64000 * synth_scale(units) ** 3, rel=1e-6)


def synth_scale(units):
    from stl2prism.mesh_prep import UNIT_SCALE
    return UNIT_SCALE[units]


def test_out_of_range_index_raises(tmp_path):
    prep, _, _ = load_and_prep_bodies(_three_parts(tmp_path / 'three.stl'), verbose=False)
    with pytest.raises(ValueError, match='out of range'):
        _pick_bodies(prep, [0, 9], False)


def test_empty_selection_raises(tmp_path):
    prep, _, _ = load_and_prep_bodies(_three_parts(tmp_path / 'three.stl'), verbose=False)
    with pytest.raises(ValueError, match='no bodies selected'):
        _pick_bodies(prep, [], False)


def test_a_cavity_shell_selects_nothing_and_says_so(tmp_path, capsys):
    """The UI lists a cavity as a shell, but preparation folds it into its
    body, so picking it alone must not silently convert the wrong thing."""
    stl = str(tmp_path / 'hollow.stl')
    v = _box(20)
    v.invert()
    trimesh.util.concatenate([_box(40), v]).export(stl)
    keys = [tuple(b['key']) for b in body_list(stl)['bodies']]
    prep, _, _ = load_and_prep_bodies(stl, verbose=False)
    assert len(prep) == 1 and len(prep[0].voids) == 1
    with pytest.raises(ValueError, match='no bodies selected'):
        _pick_bodies(prep, [keys[1]], True)


def test_shell_keys_are_scale_free():
    """The UI reads the file in its own units; the pipeline has scaled it
    to mm by the time it matches. The keys must not notice."""
    parts = [_box(40), _box(20, (100, 0, 0))]
    big = [p.copy() for p in parts]
    for p in big:
        p.apply_scale(10.0)
    assert shell_keys(parts) == shell_keys(big)


def test_shell_keys_separate_same_shape_different_place():
    """Two identical cubes are one key apart only by where they sit."""
    a, b = shell_keys([_box(20), _box(20, (100, 0, 0))])
    assert a != b


def test_shell_keys_separate_different_shapes():
    a, b = shell_keys([_box(40), trimesh.creation.icosphere(subdivisions=1)])
    assert a != b


# --- end to end -------------------------------------------------------------

def test_run_converts_only_the_picked_bodies(tmp_path):
    stl = _three_parts(tmp_path / 'three.stl')
    out = str(tmp_path / 'picked.step')
    r = pipeline.run(stl, out, verbose=False, write_script=False, bodies=[0, 2])
    assert r['n_bodies'] == 2 and r['n_written'] == 2
    assert [round(v) for _, v in synth.solid_stats(out)] == [1000, 64000]


def test_run_without_a_selection_converts_everything(tmp_path):
    stl = _three_parts(tmp_path / 'three.stl')
    out = str(tmp_path / 'all.step')
    r = pipeline.run(stl, out, verbose=False, write_script=False)
    assert r['n_bodies'] == 3 and r['n_written'] == 3


# --- unit sanity ------------------------------------------------------------

def test_unit_warning_flags_a_mesh_ten_times_too_big(tmp_path):
    """The case that cost a real conversion: a cm file written as mm reads
    10x too big, and no input unit divides, so the UI must say rescale."""
    from stl2prism.mesh_prep import unit_warning
    w = unit_warning(_box(1500))
    assert w and w['too'] == 'big'
    assert w['scale'] == 0.1 and w['unit'] is None
    assert w['would_be'] == pytest.approx(150.0)


def test_unit_warning_names_the_unit_when_one_fits(tmp_path):
    from stl2prism.mesh_prep import unit_warning
    w = unit_warning(_box(0.033))          # metres
    assert w and w['too'] == 'small' and w['unit'] == 'm'
    assert w['would_be'] == pytest.approx(33.0)


def test_unit_warning_is_quiet_for_a_normal_part():
    from stl2prism.mesh_prep import unit_warning
    assert unit_warning(_box(40)) is None
    assert unit_warning(_box(500)) is None


def test_upload_stats_carry_the_warning(tmp_path):
    from backend.analysis import mesh_stats
    p = str(tmp_path / 'big.stl')
    _box(1500).export(p)
    assert mesh_stats(p)['unit_warning']['too'] == 'big'


# --- rescaling --------------------------------------------------------------

def test_resolve_scale_combines_units_and_the_free_factor():
    from stl2prism.mesh_prep import resolve_scale
    assert resolve_scale('mm', 1.0) == 1.0
    assert resolve_scale('mm', 0.1) == pytest.approx(0.1)     # the 10x-too-big case
    assert resolve_scale('cm', 1.0) == 10.0
    assert resolve_scale('cm', 0.5) == pytest.approx(5.0)


@pytest.mark.parametrize('bad', [0, -1, 'x', float('nan'), float('inf')])
def test_resolve_scale_refuses_nonsense(bad):
    from stl2prism.mesh_prep import PrepError, resolve_scale
    with pytest.raises(PrepError):
        resolve_scale('mm', bad)


def test_resolve_scale_refuses_an_absurd_total():
    from stl2prism.mesh_prep import PrepError, resolve_scale
    with pytest.raises(PrepError, match='sensible range'):
        resolve_scale('m', 1e9)


def test_scale_shrinks_the_converted_solid(tmp_path):
    """The case the free factor exists for: a mesh written ten times too
    big converts at its real size, with the shape untouched."""
    stl = str(tmp_path / 'big.stl')
    _box(400).export(stl)
    out = str(tmp_path / 'small.step')
    r = pipeline.run(stl, out, verbose=False, write_script=False, scale=0.1)
    (faces, vol), = synth.solid_stats(out)
    assert faces == 6
    assert vol == pytest.approx(40 ** 3, rel=1e-6)
    assert r['metrics']['vol_err_pct'] < 0.01


def test_scale_one_is_unchanged(tmp_path):
    stl = str(tmp_path / 'plain.stl')
    _box(40).export(stl)
    a, b = str(tmp_path / 'a.step'), str(tmp_path / 'b.step')
    pipeline.run(stl, a, verbose=False, write_script=False)
    pipeline.run(stl, b, verbose=False, write_script=False, scale=1.0)
    assert synth.solid_stats(a) == synth.solid_stats(b)


def test_scale_is_refused_before_any_conversion(tmp_path):
    from stl2prism.mesh_prep import PrepError
    stl = str(tmp_path / 'plain.stl')
    _box(40).export(stl)
    with pytest.raises(PrepError):
        pipeline.run(stl, str(tmp_path / 'x.step'), verbose=False, scale=0)


# --- a worker the kernel kills ---------------------------------------------

def test_killed_by_is_quiet_for_a_normal_exit():
    from backend.jobs import _killed_by
    assert _killed_by(0) is None
    assert _killed_by(1) is None          # the worker failed and reported


def test_killed_by_names_the_out_of_memory_case():
    """SIGKILL is what Docker's memory limit and the OOM killer send. The
    worker never writes a result or a traceback, so this is the only thing
    that can explain the empty log."""
    from backend.jobs import _killed_by
    k = _killed_by(-9)
    assert k['kind'] == 'oom' and k['signal'] == 9
    assert 'ran out of memory' in k['message']
    assert 'fewer bodies' in k['message']


def test_killed_by_reports_other_signals_without_guessing():
    from backend.jobs import _killed_by
    k = _killed_by(-15)
    assert k['kind'] == 'stopped' and 'signal 15' in k['message']


def test_public_state_explains_a_killed_worker(tmp_path, monkeypatch):
    """The bug this fixes: a killed worker left the UI saying 'see log',
    and the log stops mid-sentence."""
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'deadbeef'
    d = tmp_path / jid
    d.mkdir()
    (d / 'log.txt').write_text('[prep] 158342 faces\n')
    with jobs._lock:
        jobs._jobs[jid] = {'status': 'error', 'filename': 'x.stl',
                           'killed': jobs._killed_by(-9)}
    st = jobs.public_state(jid)
    assert st['status'] == 'error'
    assert st['result']['ok'] is False
    assert st['result']['failure'] == 'oom'
    assert 'ran out of memory' in st['result']['error']


def test_public_state_keeps_a_real_error_over_the_kill_note(tmp_path, monkeypatch):
    """A worker that reported its own failure keeps that message."""
    import json as _json
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'realfail'
    d = tmp_path / jid
    d.mkdir()
    (d / 'log.txt').write_text('boom\n')
    (d / 'result.json').write_text(_json.dumps({'ok': False, 'error': 'PrepError: bad mesh'}))
    with jobs._lock:
        jobs._jobs[jid] = {'status': 'error', 'filename': 'x.stl',
                           'killed': jobs._killed_by(-9)}
    st = jobs.public_state(jid)
    assert st['result']['error'] == 'PrepError: bad mesh'


# --- memory and determinism in the body split -------------------------------

def test_contains_is_batched_and_matches_the_unbatched_answer():
    """trimesh's contains() allocates points x triangles in one go: 21,833
    vertices against an 18,664-face housing asked for 15.5 GB and the
    kernel killed the conversion before it converted anything."""
    from stl2prism import mesh_prep
    outer = trimesh.creation.icosphere(subdivisions=3, radius=20)
    pts = trimesh.creation.icosphere(subdivisions=4, radius=10).vertices
    assert len(pts) > mesh_prep.CONTAINS_BATCH
    got = mesh_prep._contains(outer, pts)
    assert got.all()                                   # all well inside
    mixed = np.vstack([pts, pts * 10])                 # half outside
    got2 = mesh_prep._contains(outer, mixed)
    assert got2[:len(pts)].all() and not got2[len(pts):].any()


def test_reach_outside_is_zero_for_a_contained_shell():
    from stl2prism.mesh_prep import _reach_outside
    assert _reach_outside(_tbox(40), _tbox(20)) == pytest.approx(0.0, abs=1e-6)


def test_reach_outside_measures_how_far_a_shell_pokes_out():
    from stl2prism.mesh_prep import _reach_outside
    # a 20 cube centred 15 from the origin reaches 5 mm past a 40 cube's wall
    assert _reach_outside(_tbox(40), _tbox(20, (15, 0, 0))) == pytest.approx(5.0, abs=0.05)


def test_reach_outside_is_deterministic():
    """This decides whether a shell is a cavity or a body of its own, so
    the same file must always give the same answer. An unseeded surface
    sample once gave 0.4 mm on one run and 1.5 mm on the next."""
    from stl2prism.mesh_prep import _reach_outside
    a, b = _tbox(40), _tbox(20, (12, 3, 0))
    first = _reach_outside(a, b)
    for _ in range(4):
        assert _reach_outside(a, b) == first


def test_reach_outside_stays_accurate_on_a_dense_shell():
    """The sample must not change the verdict on a shell with far more
    vertices than REACH_SAMPLE."""
    from stl2prism import mesh_prep
    outer = trimesh.creation.icosphere(subdivisions=3, radius=20)
    inner = trimesh.creation.icosphere(subdivisions=5, radius=10)   # 10242 verts
    inner.apply_translation((15, 0, 0))
    assert len(inner.vertices) > mesh_prep.REACH_SAMPLE
    got = mesh_prep._reach_outside(outer, inner)
    exact = max(0.0, -float(trimesh.proximity.signed_distance(outer, inner.vertices).min()))
    assert got == pytest.approx(exact, abs=0.05)


def test_split_bodies_stays_within_memory_on_many_shells(tmp_path):
    """A regression guard for the out-of-memory kill: prep runs over every
    shell whatever the user picked, so it must stay bounded."""
    import resource
    from stl2prism.mesh_prep import classify, split_bodies
    parts = [trimesh.creation.icosphere(subdivisions=4, radius=20)]
    for k in range(6):
        s = trimesh.creation.icosphere(subdivisions=3, radius=3)
        s.apply_translation((k * 8 - 20, 0, 0))
        parts.append(s)
    m = trimesh.util.concatenate(parts)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    split_bodies(m, classify(m, False), verbose=False)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    grew_gb = (after - before) / 1e9        # ru_maxrss is bytes on macOS
    assert grew_gb < 2.0, f'split_bodies grew {grew_gb:.1f} GB'


# --- suggestions on meshes that are not closed ------------------------------

def test_suggested_ranks_open_shells_by_size(tmp_path):
    """An open shell has no volume. Ranking on volume alone made an OBJ
    whose two housing halves are open suggest a single tiny closed screw
    and none of the part the user wanted."""
    big = _open_sheet(w=100.0, h=60.0)                   # open, large
    small = _box(4, (400, 0, 0))                         # closed, tiny
    stl = str(tmp_path / 'mixed.stl')
    trimesh.util.concatenate([big, small]).export(stl)
    d = body_list(stl)
    by = {b['triangles']: b for b in d['bodies']}
    assert by[4]['watertight'] is False and by[4]['volume'] is None
    assert by[4]['suggested'] is True                    # ranked by size
    assert by[12]['suggested'] is False                  # the tiny closed one


# --- filtering before repair ------------------------------------------------

def test_keep_selected_narrows_by_face_count():
    from stl2prism.pipeline import _keep_selected
    keep = _keep_selected([(12, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0)])
    assert keep is not None
    assert keep(Body(_box(40)))                        # 12 faces
    assert not keep(Body(trimesh.creation.icosphere(subdivisions=1)))


def test_keep_selected_keeps_a_body_whose_cavity_was_picked():
    from stl2prism.pipeline import _keep_selected
    keep = _keep_selected([(12, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5)])
    body = Body(trimesh.creation.icosphere(subdivisions=2), [_box(2)])
    assert keep(body)                                  # matched via the void


def test_keep_selected_declines_plain_indices():
    """Indices name positions in the prepared list, which does not exist
    when the filter runs: keep everything and decide afterwards."""
    from stl2prism.pipeline import _keep_selected
    assert _keep_selected([0, 2]) is None


def test_early_filter_and_late_filter_agree(tmp_path):
    """The point of the whole mechanism: repairing only what was asked for
    must give exactly what repairing everything would have given."""
    from stl2prism.mesh_prep import load_and_prep_bodies
    from stl2prism.pipeline import _keep_selected, _pick_bodies
    stl = _three_parts(tmp_path / 'three.stl')
    keys = [tuple(b['key']) for b in body_list(stl)['bodies']]
    want = [keys[0], keys[2]]

    early, _, _ = load_and_prep_bodies(stl, verbose=False, keep=_keep_selected(want))
    a = _pick_bodies(early, want, False)
    late, _, _ = load_and_prep_bodies(stl, verbose=False)
    b = _pick_bodies(late, want, False)
    assert [len(x.mesh.faces) for x in a] == [len(x.mesh.faces) for x in b]
    assert [round(abs(x.mesh.volume)) for x in a] == [round(abs(x.mesh.volume)) for x in b]


def test_run_with_a_selection_still_writes_the_right_solids(tmp_path):
    stl = _three_parts(tmp_path / 'three.stl')
    keys = [tuple(b['key']) for b in body_list(stl)['bodies']]
    out = str(tmp_path / 'picked.step')
    r = pipeline.run(stl, out, verbose=False, write_script=False,
                     bodies=[keys[0], keys[2]])
    assert r['n_written'] == 2
    assert [round(v) for _, v in synth.solid_stats(out)] == [1000, 64000]


def test_body_list_marks_open_shells(tmp_path):
    """The UI warns before a long conversion that an open body cannot
    become a solid. That warning needs the flag to be right."""
    stl = str(tmp_path / 'mixed.stl')
    trimesh.util.concatenate([_box(40), _open_sheet((100, 0, 0))]).export(stl)
    d = body_list(stl)
    by = {b['triangles']: b for b in d['bodies']}
    assert by[12]['watertight'] is True
    assert by[12]['volume'] == pytest.approx(64000)
    assert by[4]['watertight'] is False
    assert by[4]['volume'] is None             # no volume without a closed shell


# --- selections a real file will produce ------------------------------------

def test_picking_among_identical_twins_picks_the_ones_asked_for(tmp_path):
    """Four identical screws differ only in where they sit. Picking the
    second and fourth must convert those two, not any two."""
    parts = [_box(40)] + [_box(4, (100 + 20 * k, 0, 0)) for k in range(4)]
    stl = str(tmp_path / 'screws.stl')
    trimesh.util.concatenate(parts).export(stl)
    keys = [tuple(b['key']) for b in body_list(stl)['bodies']]
    out = str(tmp_path / 'two.step')
    r = pipeline.run(stl, out, verbose=False, write_script=False,
                     bodies=[keys[2], keys[4]], workers=0)
    assert r['n_written'] == 2
    assert [round(v) for _, v in synth.solid_stats(out)] == [64, 64]
    # the two chosen screws sit at x=120 and x=160, so the span is 118..162:
    # proof it took those two and not the block or the first pair
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    BRepBndLib.Add_s(synth.read_step(out), b)
    g = np.array(b.Get())
    assert g[0] == pytest.approx(118.0, abs=0.1)
    assert g[3] == pytest.approx(162.0, abs=0.1)


def test_picking_a_hollow_body_keeps_its_cavity(tmp_path):
    """The cavity is not a separate pick: it travels with its body."""
    v = _box(20)
    v.invert()
    stl = str(tmp_path / 'hollow.stl')
    trimesh.util.concatenate([_box(40), v]).export(stl)
    keys = [tuple(b['key']) for b in body_list(stl)['bodies']]
    out = str(tmp_path / 'hollow.step')
    r = pipeline.run(stl, out, verbose=False, write_script=False,
                     bodies=[keys[0]], workers=0)
    assert r['n_written'] == 1
    assert [round(v) for _, v in synth.solid_stats(out)] == [40 ** 3 - 20 ** 3]


def test_body_list_survives_a_single_body_file(tmp_path):
    stl = str(tmp_path / 'one.stl')
    _box(40).export(stl)
    d = body_list(stl)
    assert len(d['bodies']) == 1 and d['bodies'][0]['suggested'] is True
    assert set(d['triangle_body']) == {0}


def test_body_list_keys_stay_unique_on_the_hard_cases(tmp_path):
    """A key that repeats would let one body be picked and another
    converted. Twins, nesting and slivers are where that would happen."""
    cav = _box(20)
    cav.invert()
    sliver = trimesh.Trimesh(vertices=[[80, 0, 0], [81, 0, 0], [80, 1, 0], [80, 0, 1]],
                             faces=[[0, 1, 2], [0, 2, 3]], process=False)
    parts = [_box(40), cav, sliver] + [_box(4, (100 + 20 * k, 0, 0)) for k in range(4)]
    stl = str(tmp_path / 'hard.stl')
    trimesh.util.concatenate(parts).export(stl)
    d = body_list(stl)
    keys = [tuple(b['key']) for b in d['bodies']]
    assert len(set(keys)) == len(keys)


# --- what the STEP really holds ---------------------------------------------

def test_step_stats_reports_solids_that_are_really_there(tmp_path):
    """A closed body: claimed and real agree."""
    from backend.analysis import step_stats
    stl = str(tmp_path / 'box.stl')
    _box(40).export(stl)
    out = str(tmp_path / 'box.step')
    pipeline.run(stl, out, verbose=False, write_script=False)
    st = step_stats(out)
    assert st['solids'] == 1 and st['solids_claimed'] == 1
    assert st['closed'] is True


def test_step_stats_does_not_call_an_open_shell_a_solid(tmp_path):
    """An open shell is written as MANIFOLD_SOLID_BREP but imports as
    surfaces. Counting the entity told the user '2 solids' for a file that
    holds none — the one number that must not be wrong."""
    from backend.analysis import step_stats
    stl = str(tmp_path / 'open.stl')
    _open_sheet(w=40.0, h=20.0).export(stl)
    out = str(tmp_path / 'open.step')
    try:
        pipeline.run(stl, out, verbose=False, write_script=False)
    except Exception:
        pytest.skip('an open sheet may be refused outright, which is also fine')
    st = step_stats(out)
    assert st['solids'] == 0
    assert st['closed'] is False


def test_solid_check_is_skipped_for_a_very_large_step(tmp_path, monkeypatch):
    """Re-reading costs about a second per 6 MB. On a 371 MB export that is
    a minute of the user's time for a number the file's own text already
    gives, so past a cap the textual count stands."""
    from backend import analysis
    stl = str(tmp_path / 'box.stl')
    _box(40).export(stl)
    out = str(tmp_path / 'box.step')
    pipeline.run(stl, out, verbose=False, write_script=False)
    monkeypatch.setattr(analysis, 'SOLID_CHECK_MAX_BYTES', 1)     # force the skip
    st = analysis.step_stats(out)
    assert st['solids'] == st['solids_claimed'] == 1
    assert st['closed'] is None                                   # not checked


# --- surviving a restart ----------------------------------------------------

def test_a_finished_job_survives_a_restart(tmp_path, monkeypatch):
    """The registry is in memory, so a restart forgot every job while its
    files sat on disk: the user was told 'unknown job' about a conversion
    whose result was right there."""
    import json as _json
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'abc123abc123'
    d = tmp_path / jid
    d.mkdir()
    (d / 'input.stl').write_bytes(b'')
    (d / 'params.json').write_text('{}')
    (d / 'result.json').write_text(_json.dumps({'ok': True, 'mode': 'prismatic'}))
    (d / 'log.txt').write_text('[prep] done\n')
    jobs._jobs.clear()                       # the restart
    st = jobs.public_state(jid)
    assert st is not None and st['status'] == 'done'
    assert st['result']['ok'] is True
    assert '[prep] done' in st['log']


def test_a_job_interrupted_by_a_restart_says_so(tmp_path, monkeypatch):
    """Converting when the server stopped: not 'unknown job', and not a
    job that appears to still be running."""
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'def456def456'
    d = tmp_path / jid
    d.mkdir()
    (d / 'input.stl').write_bytes(b'')
    (d / 'params.json').write_text('{}')     # started, never finished
    (d / 'log.txt').write_text('[body 1/3] shell started\n')
    jobs._jobs.clear()
    st = jobs.public_state(jid)
    assert st['status'] == 'error'
    assert 'server restarted' in st['result']['error']
    assert st['result']['failure'] == 'restart'


def test_an_uploaded_but_unconverted_job_survives(tmp_path, monkeypatch):
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'aaa111bbb222'
    d = tmp_path / jid
    d.mkdir()
    (d / 'input.stl').write_bytes(b'')
    jobs._jobs.clear()
    st = jobs.public_state(jid)
    assert st['status'] == 'uploaded'


def test_recovery_refuses_a_path_that_is_not_a_job_id(tmp_path, monkeypatch):
    """A job id names a directory, so it must never carry a path."""
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jobs._jobs.clear()
    for bad in ('..', '.', '../etc', 'a/b', ''):
        assert jobs.get(bad) is None


def test_unknown_job_is_still_unknown(tmp_path, monkeypatch):
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jobs._jobs.clear()
    assert jobs.get('0123456789ab') is None
    assert jobs.public_state('0123456789ab') is None


# --- not deploying over a running conversion --------------------------------

def test_running_endpoint_is_not_shadowed_by_the_job_id_route(tmp_path, monkeypatch):
    """FastAPI matches in declaration order, so /api/jobs/running has to be
    declared before /api/jobs/{job_id} or it is read as a job called
    'running' and 404s — which is how it was first written."""
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    c = TestClient(app)
    r = c.get('/api/jobs/running')
    assert r.status_code == 200
    assert r.json() == {'running': 0, 'jobs': []}
    assert c.get('/api/jobs/000000000000').status_code == 404


def test_running_summary_counts_only_live_conversions(tmp_path, monkeypatch):
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    with jobs._lock:
        jobs._jobs.clear()
        jobs._jobs['a'] = {'id': 'a', 'status': 'running', 'proc': object(),
                           'filename': 'x.stl'}
        jobs._jobs['b'] = {'id': 'b', 'status': 'done', 'proc': None,
                           'filename': 'y.stl'}
        jobs._jobs['c'] = {'id': 'c', 'status': 'queued', 'proc': object(),
                           'filename': 'z.stl'}
    try:
        got = jobs.running_summary()
        assert got['running'] == 2
        assert {j['id'] for j in got['jobs']} == {'a', 'c'}
    finally:
        with jobs._lock:
            jobs._jobs.clear()


def test_a_second_run_killed_by_a_restart_does_not_report_the_first_run(tmp_path, monkeypatch):
    """A job can be converted twice. params.json is rewritten at the start
    of each run and result.json only at the end, so a params newer than the
    result means a later run never finished — reporting the earlier run's
    success would tell the user their conversion worked when it did not."""
    import json as _json
    import os
    import time as _time
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'twicetwice12'
    d = tmp_path / jid
    d.mkdir()
    (d / 'input.stl').write_bytes(b'')
    (d / 'result.json').write_text(_json.dumps({'ok': True, 'mode': 'prismatic'}))
    (d / 'log.txt').write_text('[progress] 7/18 shells\n')
    (d / 'params.json').write_text('{}')
    # the second run started well after the first one finished
    old = _time.time() - 600
    os.utime(d / 'result.json', (old, old))
    jobs._jobs.clear()
    st = jobs.public_state(jid)
    assert st['status'] == 'error'
    assert st['result']['failure'] == 'restart'
    assert st['result'].get('ok') is False
    assert '[progress] 7/18' in st['log']       # the log still shows the truth


def test_a_single_completed_run_is_still_reported_done(tmp_path, monkeypatch):
    """The guard must not turn every recovered job into a failure."""
    import json as _json
    import os
    import time as _time
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'onceonce1234'
    d = tmp_path / jid
    d.mkdir()
    (d / 'input.stl').write_bytes(b'')
    (d / 'params.json').write_text('{}')
    (d / 'result.json').write_text(_json.dumps({'ok': True, 'mode': 'prismatic'}))
    now = _time.time()
    os.utime(d / 'params.json', (now - 60, now - 60))    # params first, result after
    jobs._jobs.clear()
    st = jobs.public_state(jid)
    assert st['status'] == 'done' and st['result']['ok'] is True


# --- cancelling a conversion ------------------------------------------------

def test_cancel_of_an_unknown_job_is_unknown(tmp_path, monkeypatch):
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jobs._jobs.clear()
    assert jobs.cancel('000000000000') == 'unknown'


def test_cancel_of_a_finished_job_says_it_is_not_running(tmp_path, monkeypatch):
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    with jobs._lock:
        jobs._jobs.clear()
        jobs._jobs['fin'] = {'id': 'fin', 'status': 'done', 'proc': None}
    try:
        assert jobs.cancel('fin') == 'not_running'
    finally:
        with jobs._lock:
            jobs._jobs.clear()


def test_cancel_stops_a_real_process_and_its_children(tmp_path, monkeypatch):
    """The worker runs a pool of its own. Cancelling must take the group,
    or a conversion keeps running invisibly after the user stopped it."""
    import os
    import subprocess
    import time as _time
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    # a process that spawns a child and both sleep, like worker + pool
    script = ('import subprocess, time; '
              'subprocess.Popen(["sleep", "120"]); time.sleep(120)')
    proc = subprocess.Popen([__import__('sys').executable, '-c', script],
                            start_new_session=True)
    _time.sleep(1.5)
    child = subprocess.run(['pgrep', '-P', str(proc.pid)],
                           capture_output=True, text=True).stdout.split()
    assert child, 'the fixture never spawned its child'
    kid = int(child[0])
    with jobs._lock:
        jobs._jobs.clear()
        jobs._jobs['run'] = {'id': 'run', 'status': 'running', 'proc': proc}
    try:
        assert jobs.cancel('run') == 'cancelled'
        _time.sleep(1)
        assert proc.poll() is not None, 'the worker survived the cancel'
        # the child must go with the group, not be orphaned
        alive = subprocess.run(['ps', '-p', str(kid)],
                               capture_output=True, text=True).returncode == 0
        assert not alive, f'pool child {kid} was left running'
    finally:
        for p in (proc.pid, kid):
            try:
                os.kill(p, 9)
            except OSError:
                pass
        with jobs._lock:
            jobs._jobs.clear()


def test_a_cancelled_job_is_not_reported_as_a_failure(tmp_path, monkeypatch):
    """Stopping on purpose is its own state: not an error to explain, and
    never the earlier run's success."""
    import json as _json
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'cancelled123'
    d = tmp_path / jid
    d.mkdir()
    (d / 'log.txt').write_text('[progress] 3/18 shells\n')
    (d / 'result.json').write_text(_json.dumps({'ok': True}))   # an earlier run
    with jobs._lock:
        jobs._jobs.clear()
        jobs._jobs[jid] = {'id': jid, 'status': 'cancelled', 'proc': None,
                           'stale_result': True,
                           'killed': {'signal': None, 'kind': 'cancelled',
                                      'message': 'Conversion cancelled.'}}
    try:
        st = jobs.public_state(jid)
        assert st['status'] == 'cancelled'
        assert st['result']['failure'] == 'cancelled'
        assert st['result']['ok'] is False
        assert '[progress] 3/18' in st['log']
    finally:
        with jobs._lock:
            jobs._jobs.clear()


def test_a_recovered_verdict_is_re_read_when_the_run_finishes(tmp_path, monkeypatch):
    """A recovered entry is a snapshot of the directory at one moment. If
    the run it was waiting on finishes afterwards, reporting the old
    verdict tells the user their conversion failed when it succeeded —
    which is exactly what happened to a 13-body run."""
    import json as _json
    import os
    import time as _time
    from backend import jobs
    monkeypatch.setattr(jobs, 'DATA_DIR', str(tmp_path))
    jid = 'inflight1234'
    d = tmp_path / jid
    d.mkdir()
    (d / 'input.stl').write_bytes(b'')
    (d / 'params.json').write_text('{}')          # converting, no result yet
    (d / 'log.txt').write_text('[progress] 5/18 shells\n')
    jobs._jobs.clear()

    first = jobs.public_state(jid)
    assert first['status'] == 'error'             # in flight, nothing running

    # the run finishes: a result lands, newer than the params
    _time.sleep(0.01)
    (d / 'result.json').write_text(_json.dumps({'ok': True, 'n_written': 13}))
    now = _time.time()
    os.utime(d / 'result.json', (now + 5, now + 5))

    second = jobs.public_state(jid)
    assert second['status'] == 'done'
    assert second['result']['n_written'] == 13
