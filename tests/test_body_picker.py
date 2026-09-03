"""Picking which bodies to convert: the shell list the UI shows, and the
filter run() applies to it."""
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
