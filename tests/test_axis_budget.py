"""Phase 1 of the parallel plan: the extrusion search is finite.

A 12k-face textured cavity offered 772 perpendicular levels on one axis
and kept refine_levels busy for 41 minutes; with the level cap, the slab
cache, the memoised bisections and the search budget the same shell's
axis search takes seconds, and every sample STEP is unchanged."""
import time
import numpy as np
import pytest
import trimesh
import cadquery as cq

from . import synth
from stl_to_solid import extrusion as ex


def _mesh(wp, tmp_path, name='part'):
    return trimesh.load(synth.export(wp, tmp_path / f'{name}.stl'), force='mesh')


# --- the level cap ----------------------------------------------------------

def test_too_many_levels_scores_zero_without_sectioning(tmp_path, monkeypatch):
    m = _mesh(synth.stepped_shaft(), tmp_path)
    ax = np.array([0.0, 0.0, 1.0])
    n_levels = len(ex.slab_levels(m, ax))
    assert n_levels >= 3
    calls = []
    monkeypatch.setattr(ex, '_section_polys',
                        lambda *a, **k: calls.append(1) or [[] for _ in a[2]])
    monkeypatch.setattr(ex, 'MAX_LEVELS', n_levels - 1)
    sc, levels, slabs = ex.score_axis(m, ax)
    assert sc == 0.0 and slabs == [] and len(levels) == n_levels
    assert calls == [], 'a capped candidate must not be sectioned at all'


def test_cap_is_above_real_parts(tmp_path):
    """The cap must sit far above what a real extrusion offers."""
    for wp in (synth.stepped_shaft(), synth.slot_hex(), synth.cross3()):
        m = _mesh(wp, tmp_path)
        for _, ax in ex.dominant_axis(m):
            assert len(ex.slab_levels(m, ax)) <= ex.MAX_LEVELS // 4


# --- the search budget ------------------------------------------------------

def test_deadline_in_the_past_raises(tmp_path):
    m = _mesh(synth.plate_holes(), tmp_path)
    ax = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ex.AxisSearchTimeout):
        ex.score_axis(m, ax, deadline=time.monotonic() - 1.0)


def test_deadline_stops_refinement_gracefully(tmp_path, monkeypatch):
    """Past the deadline inside refine_levels the levels found so far are
    returned with consistent slabs, not an exception."""
    m = _mesh(synth.chamfer_top(), tmp_path)
    ax = np.array([0.0, 0.0, 1.0])
    levels = ex.slab_levels(m, ax)
    slabs = ex.slab_sections(m, ax, levels)
    assert any(not s['constant'] for s in slabs), 'the chamfer slab must need refining'
    checks = []
    real = ex._check_deadline
    monkeypatch.setattr(ex, '_check_deadline',
                        lambda d: checks.append(d) or real(d))
    lv, sl = ex.refine_levels(m, ax, levels, slabs, deadline=time.monotonic() - 1.0)
    assert checks, 'the refinement reached a deadline check and stopped there'
    assert lv == levels
    assert [(s['z0'], s['z1']) for s in sl] == [(s['z0'], s['z1']) for s in slabs]


def test_spent_budget_falls_through_to_the_next_route(tmp_path, monkeypatch):
    """With no time for any candidate the prismatic rung fails and the
    ladder continues; the file still gets a valid solid."""
    from stl_to_solid import pipeline
    monkeypatch.setattr(pipeline, 'AXIS_SEARCH_BUDGET_S', 1e-9)
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    out = str(tmp_path / 'plate.step')
    r = pipeline.run(p, out, verbose=False)
    assert r['mode'] in ('facegroup', 'faceted')
    got = synth.reimport(out)
    assert got['solids'] == 1 and got['naked_edges'] == 0


def test_budget_keeps_the_best_candidate_so_far(tmp_path, monkeypatch, capsys):
    """When the budget runs out after the first candidate, that candidate
    is used rather than the search abandoned."""
    from stl_to_solid import pipeline
    real = ex.score_axis
    calls = []

    def slow(m, axis, **kw):
        if calls:                      # the second candidate finds the budget spent
            kw['deadline'] = time.monotonic() - 1.0
        calls.append(1)
        sc, levels, slabs = real(m, axis, **kw)
        # under-report the first score, or the search would stop there
        # because no later candidate could beat a perfect one
        return 0.5 * sc, levels, slabs
    monkeypatch.setattr(ex, 'score_axis', slow)
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    out = str(tmp_path / 'plate.step')
    r = pipeline.run(p, out, verbose=True)
    assert len(calls) == 2, 'the search stops at the first candidate past the budget'
    assert 'abandoned' in capsys.readouterr().out
    assert r['mode'] == 'prismatic', r['metrics']
    got = synth.reimport(out)
    assert got['solids'] == 1 and got['naked_edges'] == 0


# --- the slab cache and the fast probe polygons are result-identical ---------

def test_slab_cache_returns_the_same_slabs(tmp_path):
    m = _mesh(synth.csk_plate(), tmp_path)
    ax = np.array([0.0, 0.0, 1.0])
    levels = ex.slab_levels(m, ax)
    plain = ex.slab_sections(m, ax, levels)
    cache = {}
    first = ex.slab_sections(m, ax, levels, cache=cache)
    again = ex.slab_sections(m, ax, levels, cache=cache)
    for a, b, c in zip(plain, first, again):
        assert (a['z0'], a['z1']) == (b['z0'], b['z1']) == (c['z0'], c['z1'])
        assert a['constant'] == b['constant'] == c['constant']
        assert a['area'] == b['area'] == c['area']
        assert b is c
    assert len(cache) == len(plain)


@pytest.mark.parametrize('builder', [
    synth.plate_holes, synth.slot_hex, synth.cross_blind, synth.hollow_cube,
    synth.boss_fillet_two, synth.csk_plate,
    lambda: synth.rotate(synth.plate_holes(), (1, 0, 0), 3.0),
])
def test_fast_probe_polygons_match_trimesh(tmp_path, builder):
    """The fast loop chaining gives the same rings nested the same way as
    trimesh's polygons_full: same topology, IoU 1, zero boundary distance.
    Where trimesh's merge leaves a ring touching itself (a corner vertex
    split in two, seen on the tilted plate) the old path dropped the whole
    section; the fast path still returns it, so those are compared for
    area only."""
    m = _mesh(builder(), tmp_path)
    for _, ax in ex.dominant_axis(m)[:2]:
        T = ex._axis_basis(ax)
        h = m.vertices @ ax
        zs = np.linspace(h.min(), h.max(), 9)[1:-1] + 0.0137
        slow = ex._section_polys(m, ax, zs, T)
        fast = ex._section_polys(m, ax, zs, T, fast=True)
        assert all(fast), 'every probe height inside the part has a section'
        for a, b in zip(slow, fast):
            if not a:
                continue
            assert ex._topology(a) == ex._topology(b)
            assert ex._shape_iou(a, b) > 0.999999
            assert ex._shape_dist(a, b) < 1e-6


def test_fast_polygons_bail_out_on_open_chains():
    """Segments that do not close into loops are left to trimesh."""
    seg = np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]], [[1, 1], [0, 1]]], float)
    assert ex._fast_polygons(seg, np.eye(4), np.eye(4)) is None


def test_fast_polygons_leave_crossing_holes_to_trimesh():
    """Two hole rings that overlap make an invalid composite; trimesh's
    repair is the authority there."""
    def square(c, r):
        x, y = c
        pts = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        return [[pts[i], pts[(i + 1) % 4]] for i in range(4)]
    seg = np.array(square((0, 0), 10) + square((-1, 0), 3) + square((1, 0), 3), float)
    assert ex._fast_polygons(seg, np.eye(4), np.eye(4)) is None


def test_fast_polygons_nest_even_odd():
    """A square with a square hole holding a square island: two shells,
    the outer one with one hole."""
    def square(c, r):
        x, y = c
        pts = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        return [[pts[i], pts[(i + 1) % 4]] for i in range(4)]
    seg = np.array(square((0, 0), 10) + square((0, 0), 5) + square((0, 0), 2), float)
    out = ex._fast_polygons(seg, np.eye(4), np.eye(4))
    assert ex._topology(out) == (2, (0, 1))
    assert abs(sum(p.area for p in out) - (400 - 100 + 16)) < 1e-9


def test_same_section_order_is_immaterial(tmp_path):
    """IoU-first with the vertex-distance bound gives the verdict the
    Hausdorff-first test gave."""
    m = _mesh(synth.chamfer_top(), tmp_path)
    ax = np.array([0.0, 0.0, 1.0])
    T = ex._axis_basis(ax)
    h = m.vertices @ ax
    zs = np.linspace(h.min(), h.max(), 12)[1:-1] + 0.0137
    secs = ex._section_polys(m, ax, zs, T)
    for a in secs:
        for b in secs:
            old = (ex._shape_dist(a, b) <= ex.SECTION_DIST_TOL
                   and ex._shape_iou(a, b) >= ex.SHAPE_IOU_MIN)
            assert ex._same_section(a, b) == old
