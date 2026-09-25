"""Sliced loft: plane sections -> fitted outlines -> lofted B-spline solid.

Synthetic parts only (tests/synth.py style, built here), each asserted on
the artefact: mode 'loft', one valid solid, few faces, deviation and
volume against the mesh, and a STEP that re-reads as one solid.
"""
import os
import numpy as np
import pytest
import trimesh
import cadquery as cq

from . import synth
from .test_pipeline import _reimport


def _sphere():
    return trimesh.creation.icosphere(subdivisions=4, radius=15.0)


def _ellipsoid():
    e = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    e.apply_scale([20.0, 12.0, 8.0])
    return e


def _bottle_mesh(tmp_path):
    wp = (cq.Workplane('XY').spline([(0, 0), (14, 0), (16, 10), (13, 30), (6, 42), (5, 50), (5, 60), (0, 60)],
                                    includeCurrent=False).close()
          .revolve(360, (0, 0, 0), (0, 1, 0)))
    p = synth.export(wp, tmp_path / 'bottle.stl', tol=0.01)
    return trimesh.load(p, force='mesh')


def _valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return BRepCheck_Analyzer(shape).IsValid()


def _check(shape, mesh, faces_max, dev_p95=0.1, vol_pct=1.0):
    from stl_to_solid.pipeline import validate
    assert _valid(shape)
    s = cq.Shape.cast(shape)
    assert len(s.Solids()) == 1
    assert len(s.Faces()) <= faces_max, len(s.Faces())
    m = validate(shape, mesh)
    assert m['dev_p95'] <= dev_p95, m
    assert m['rev_dev_p95'] <= dev_p95, m
    assert m['vol_err_pct'] <= vol_pct, m
    return m, s


# --- the section cutter ---------------------------------------------------

def test_section_loops_box_and_hole():
    from stl_to_solid.section_fit import section_loops, signed_area
    m = trimesh.creation.box([40, 30, 10])
    loops, frame = section_loops(m.vertices, m.faces, [0, 0, 1.0], [0, 0, 1])
    assert len(loops) == 1 and not loops[0][1]
    outer = loops[0][0]
    assert signed_area(outer) == pytest.approx(1200.0, rel=1e-6)   # CCW
    # a plate with a hole: one outer, one CW hole
    p = synth.export(synth.plate_holes(), '/tmp/_s2p_plate.stl')
    m = trimesh.load(p, force='mesh')
    loops, _ = section_loops(m.vertices, m.faces, [0, 0, 0.5], [0, 0, 1])
    assert len(loops) == 1 and len(loops[0][1]) == 4
    assert all(signed_area(h) < 0 for h in loops[0][1])
    assert sum(abs(signed_area(h)) for h in loops[0][1]) == pytest.approx(4 * np.pi * 2.5 ** 2, rel=0.02)


def test_section_matches_trimesh():
    from stl_to_solid.section_fit import section_loops, signed_area
    m = _ellipsoid()
    z = 2.0
    loops, _ = section_loops(m.vertices, m.faces, [0, 0, z], [0, 0, 1])
    s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    p2, _ = s.to_2D()
    ref = sum(abs(p.area) for p in p2.polygons_closed if p is not None)
    assert len(loops) == 1
    assert abs(signed_area(loops[0][0])) == pytest.approx(ref, rel=1e-6)


def _sheet(xy, z0=-1.0, z1=1.0):
    """A vertical open sheet (V, F) through the 2-D polyline `xy`: one
    quad per segment, no caps, so a horizontal cut gives an open chain."""
    xy = np.asarray(xy, float)
    n = len(xy)
    V = np.vstack([np.c_[xy, np.full(n, z0)], np.c_[xy, np.full(n, z1)]])
    F = []
    for i in range(n - 1):
        F.append([i, i + 1, n + i + 1])
        F.append([i, n + i + 1, n + i])
    return V, np.asarray(F)


def test_open_chain_stays_open():
    """A U-shaped sheet cuts into one open chain: no loop, no chord, three
    lines whose ends are the sheet's ends, an open preview polyline and a
    script that draws exactly those three lines."""
    from stl_to_solid.section_fit import section_curves, section_loops, fit_section, section_preview
    from stl_to_solid.fusion_export import emit_fusion_sections_script
    V, F = _sheet([[0, 10], [0, 0], [20, 0], [20, 10]])
    loops, opens, _, jst = section_curves(V, F, [0, 0, 0], [0, 0, 1])
    assert loops == [] and len(opens) == 1 and jst['joins'] == 0
    assert np.allclose(opens[0][0], [0, 10]) and np.allclose(opens[0][-1], [20, 10])
    sec = fit_section(V, F, [0, 0, 0], [0, 0, 1], tol=0.05)
    st = sec['stats']
    assert st['loops'] == 0 and st['open'] == 1 and st['lines'] == 3 and st['dev_max'] < 1e-6
    prims = sec['open'][0]
    assert np.allclose(prims[0]['p0'], [0, 10]) and np.allclose(prims[-1]['p1'], [20, 10])
    pv = section_preview(V, F, [0, 0, 0], [0, 0, 1], tol=0.05)
    assert len(pv['polylines']) == 1 and pv['loops'] == [] and len(pv['open']) == 1
    assert not np.allclose(pv['polylines'][0][0], pv['polylines'][0][-1])
    txt = emit_fusion_sections_script([{'origin': pv['origin'], 'normal': pv['normal'],
                                        'name': 'u', 'loops': pv['loops'], 'open': pv['open']}])
    assert txt.count('sketchLines.addByTwoPoints') == 3
    compile(txt, 'u.py', 'exec')
    # outline only: the open chain is left out, but still counted
    pv = section_preview(V, F, [0, 0, 0], [0, 0, 1], tol=0.05, closed_only=True)
    assert pv['polylines'] == [] and pv['open'] == [] and pv['stats']['open'] == 1
    # ... and so are inner loops: a plate with 4 holes keeps only its rim
    p = synth.export(synth.plate_holes(), '/tmp/_s2p_plate2.stl')
    m2 = trimesh.load(p, force='mesh')
    full = section_preview(m2.vertices, m2.faces, [0, 0, 0.5], [0, 0, 1], tol=0.08)
    rim = section_preview(m2.vertices, m2.faces, [0, 0, 0.5], [0, 0, 1], tol=0.08, closed_only=True)
    assert len(full['polylines']) == 5 and full['stats']['inner'] == 0
    assert len(rim['polylines']) == 1 and rim['stats']['inner'] == 4 and rim['loops'][0][1] == []
    # the loft's cutter keeps closing every chain by its chord
    from stl_to_solid.section_fit import signed_area
    lp, _ = section_loops(V, F, [0, 0, 0], [0, 0, 1])
    assert len(lp) == 1 and abs(signed_area(lp[0][0])) == pytest.approx(200.0)


def test_join_chains_bridges_small_gaps_only():
    """Two sheets 0.5 mm apart are two open chains at join 0, one at join
    1; a ring sheet missing one facet closes on itself once the join
    reaches the gap, and becomes a loop."""
    from stl_to_solid.section_fit import section_curves, fit_section
    Va, Fa = _sheet([[0, 0], [10, 0]])
    Vb, Fb = _sheet([[10.5, 0], [20, 0]])
    V = np.vstack([Va, Vb])
    F = np.vstack([Fa, Fb + len(Va)])
    for join, n_open, n_join in [(0.0, 2, 0), (0.4, 2, 0), (1.0, 1, 1)]:
        loops, opens, _, jst = section_curves(V, F, [0, 0, 0], [0, 0, 1], join_mm=join)
        assert (len(loops), len(opens), jst['joins']) == (0, n_open, n_join), join
    _, opens, _, jst = section_curves(V, F, [0, 0, 0], [0, 0, 1], join_mm=1.0)
    assert jst['max_gap'] == pytest.approx(0.5)
    assert np.allclose(opens[0][0], [0, 0]) and np.allclose(opens[0][-1], [20, 0])
    assert np.all(np.diff(opens[0][:, 0]) > 0)          # one chain, in order
    # a ring with one facet missing: gap = one chord of a 32-gon, r = 5
    t = np.linspace(0, 2 * np.pi, 33)[:-1]
    ring = np.c_[5 * np.cos(t), 5 * np.sin(t)]
    V, F = _sheet(ring)                       # open between ring[-1] and ring[0]
    gap = float(np.linalg.norm(ring[-1] - ring[0]))
    loops, opens, _, jst = section_curves(V, F, [0, 0, 0], [0, 0, 1], join_mm=gap * 0.9)
    assert len(loops) == 0 and len(opens) == 1
    loops, opens, _, jst = section_curves(V, F, [0, 0, 0], [0, 0, 1], join_mm=gap * 1.1)
    assert len(loops) == 1 and opens == [] and jst['joins'] == 1
    st = fit_section(V, F, [0, 0, 0], [0, 0, 1], tol=0.1, join_mm=gap * 1.1)['stats']
    assert st['circles'] == 1 and st['open'] == 0


def test_loft_range_gives_a_slab_with_flat_ends():
    """A partial loft of a box (z from -5 to 5) between z = -3 and z = 1 is
    a 40 x 30 x 4 slab, and a range past the body is clipped to it;
    the sketch-path cutter (join / trim set) gives the same sections on a
    watertight body as the chord-closing one."""
    from stl_to_solid.sliced_loft import loft_body, Cutter
    from stl_to_solid.pipeline import accurate_volume
    m = trimesh.creation.box([40, 30, 10])
    shape, info = loft_body(m, 2, 0.5, verbose=False, z_range=(-3.0, 1.0), join_mm=2.5, trim_mm=0.3)
    assert info['range'] == [-3.0, 1.0]
    assert accurate_volume(shape) == pytest.approx(40 * 30 * 4, rel=0.01)
    shape, info = loft_body(m, 2, 0.5, verbose=False, z_range=(2.0, 9.0))
    assert info['range'] == [2.0, 5.0]
    assert accurate_volume(shape) == pytest.approx(40 * 30 * 3, rel=0.01)
    a = Cutter(m, 2).cut(3.0)
    b = Cutter(m, 2, 2.5, 0.3).cut(3.0)
    assert len(a) == len(b) == 1 and np.allclose(a[0][0], b[0][0])


def test_trim_slivers_removes_hairpins_and_twists_only():
    """A rectangle with a 0.1 mm wide, 6 mm deep hairpin and a tiny bow-tie
    twist: trim at 0.3 mm removes both, leaves a real 1 mm slot alone, and
    trim 0 changes nothing."""
    from stl_to_solid.section_fit import trim_slivers, signed_area, _crossings, fit_section
    rect = [[0, 0], [10, 0], [10, 0.0], [10.05, 6], [10.15, 6], [10.2, 0.0],   # hairpin up
            [20, 0], [20, 10], [12, 10], [12, 8], [11, 8], [11, 10],           # 1 mm slot
            [5.2, 10], [5.0, 10.3], [4.8, 10], [0, 10]]                        # tiny twist
    rect = np.asarray(rect, float)
    rect[13] = [5.0, 9.7]                                                      # make the twist cross
    xy = np.vstack([rect, [[5.2, 9.9], [4.8, 10.1]]]) if False else rect
    Q, k = trim_slivers(xy, 0.3)
    assert k >= 1
    assert not any(abs(p[1] - 6) < 1e-6 for p in Q)                            # hairpin gone
    assert any(abs(p[0] - 11) < 1e-6 and abs(p[1] - 8) < 1e-6 for p in Q)      # slot kept
    assert abs(abs(signed_area(Q)) - abs(signed_area(xy))) < 0.3 * 12 + 1.0
    Q0, k0 = trim_slivers(xy, 0.0)
    assert k0 == 0 and np.array_equal(Q0, xy)
    # through fit_section: stats['trimmed'] counts, and no self-crossing remains
    V, F = _sheet(np.vstack([xy, xy[:1]]))
    sec = fit_section(V, F, [0, 0, 0], [0, 0, 1], tol=0.05, join_mm=1.0, trim_mm=0.3)
    assert sec['stats']['trimmed'] >= 1
    sec0 = fit_section(V, F, [0, 0, 0], [0, 0, 1], tol=0.05, join_mm=1.0)
    assert sec0['stats']['trimmed'] == 0


def test_fit_loop_rounded_rect_is_lines_and_arcs():
    from stl_to_solid.section_fit import section_loops, fit_loop
    p = synth.export(synth.rounded_rect(), '/tmp/_s2p_rr.stl')
    m = trimesh.load(p, force='mesh')
    loops, _ = section_loops(m.vertices, m.faces, [0, 0, 0.0], [0, 0, 1])
    prims = fit_loop(loops[0][0], tol=0.08)
    kinds = sorted(p['type'] for p in prims)
    assert kinds.count('line') == 4 and kinds.count('arc') == 4 and 'spline' not in kinds
    assert all(abs(p['r'] - 4.0) < 0.05 for p in prims if p['type'] == 'arc')


def test_fit_loop_spline_fallback_on_ripple():
    """A ring with ripples too short for any line or arc to hold at the
    tolerance comes back as a spline, within tolerance of the raw points."""
    from stl_to_solid.section_fit import fit_loop, prim_points, polyline_deviation
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    r = 10 + 0.25 * np.sin(80 * t)
    xy = np.c_[r * np.cos(t), r * np.sin(t)]
    prims = fit_loop(xy, tol=0.05)
    assert any(p['type'] == 'spline' for p in prims)
    assert polyline_deviation(xy[::5], prim_points(prims)) < 0.1
    from stl_to_solid.section_fit import junction_gap
    assert junction_gap(prims) < 1e-9


def test_fit_loop_holds_tolerance_on_wave():
    """Gentle waves are arcs; whatever the mix, the fit stays within tol."""
    from stl_to_solid.section_fit import fit_loop, prim_points, polyline_deviation
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    r = 10 + 1.5 * np.sin(5 * t)
    xy = np.c_[r * np.cos(t), r * np.sin(t)]
    prims = fit_loop(xy, tol=0.05)
    assert len(prims) > 4
    assert polyline_deviation(xy[::4], prim_points(prims)) < 0.1


def test_trace_rc_n2_section_like_fusion():
    """The section a user cuts in Fusion (plane normal to the depth axis,
    26 mm from the box centre) traced from the mesh: a few hundred lines
    and arcs plus splines on the domes, all within 0.15 mm of the raw cut."""
    from .test_pipeline import _sample
    from stl_to_solid.section_fit import fit_section
    p = _sample('fixed-rc-n2-360.stl')
    m = trimesh.load(p, force='mesh')
    m.apply_scale(0.1)                      # the file is written 10x too big
    z = m.bounding_box.centroid[2] - 26.0
    sec = fit_section(m.vertices, m.faces, [0, 0, z], [0, 0, 1], tol=0.08)
    st = sec['stats']
    assert st['loops'] >= 5 and st['holes'] >= 3
    assert st['lines'] > 50 and st['arcs'] > 50 and st['splines'] > 0
    # every curve meets its neighbour exactly, splines included, or the
    # sketch is open in Fusion's eyes and there is no profile to extrude
    assert st['junction_gap'] < 1e-9
    assert st['dev_max'] < 0.15, st


# --- the loft -------------------------------------------------------------

def test_step_levels_on_shaft(tmp_path):
    from stl_to_solid.sliced_loft import step_levels
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    m = trimesh.load(p, force='mesh')
    lv = step_levels(m, 2)
    assert [round(l, 3) for l in lv] == [0.0, 20.0, 35.0, 45.0]


def test_loft_sphere_one_face_per_run():
    from stl_to_solid.sliced_loft import loft_body
    m = _sphere()
    shape, info = loft_body(m, 2, 0.2, verbose=False)
    assert info['mode'] == 'loft' and info['n_runs'] == 1
    # one smooth face for the body, a few ruled bands where the rings
    # shrink fast near the poles, two apex cones: tens of faces, not the
    # mesh's 5120
    _check(shape, m, faces_max=24, dev_p95=0.08, vol_pct=0.5)


def test_loft_ellipsoid_along_x():
    from stl_to_solid.sliced_loft import loft_body
    m = _ellipsoid()
    shape, info = loft_body(m, 'auto', 0.2, verbose=False)
    assert info['axis_name'] == 'x'
    _check(shape, m, faces_max=30, dev_p95=0.08, vol_pct=0.5)


def test_choose_axis_by_slicing_structure(tmp_path):
    """The whole-body loft picks the axis whose stack has the fewest
    one-section runs, then the fewest runs, then the longest side: a
    round cap wants its short axis, a thin plate its thin one, and an
    ellipsoid (one run whichever way) its longest."""
    from stl_to_solid.sliced_loft import choose_axis
    cap = (cq.Workplane('XY').circle(30).extrude(12).faces('>Z').workplane()
           .circle(30).circle(27).extrude(4))
    m = trimesh.load(synth.export(cap, tmp_path / 'cap.stl'), force='mesh')
    ax, sc = choose_axis(m, 0.2, verbose=False)
    assert ax == 2, sc
    assert sc['z']['runs'] + sc['z']['steep'] < sc['x']['runs'] + sc['x']['steep'], sc
    plate = (cq.Workplane('XZ').box(20, 22, 4).faces('>Y').workplane()
             .rect(10, 12, forConstruction=True).vertices().hole(3))
    m = trimesh.load(synth.export(plate, tmp_path / 'plate.stl'), force='mesh')
    assert abs(m.extents[1] - 4.0) < 0.1
    ax, sc = choose_axis(m, 0.2, verbose=False)
    assert ax == 1, sc
    assert sc['y']['runs'] == 1 and sc['y']['single'] == 0
    ax, sc = choose_axis(_ellipsoid(), 0.2, verbose=False)
    assert ax == 0, sc


def test_loft_merges_identical_rings_into_one_prism(tmp_path):
    """A stack of equal sections is one straight stretch (3 faces), a
    shallow draft is not merged away (it steps to a new group every
    RING_FIT_TOL of drift and stays within tolerance)."""
    from stl_to_solid.sliced_loft import loft_body
    m = trimesh.creation.box([20.0, 20.0, 10.0])
    shape, info = loft_body(m, 2, 0.2, verbose=False)
    assert info['n_merged'] > 40, info
    _check(shape, m, faces_max=3, dev_p95=0.05, vol_pct=0.5)
    p = synth.export(synth.drafted_block(), tmp_path / 'draft.stl')
    m = trimesh.load(p, force='mesh')
    shape, info = loft_body(m, 2, 0.2, verbose=False)
    m2, s = _check(shape, m, faces_max=40, dev_p95=0.05, vol_pct=0.5)
    assert len(s.Faces()) > 3


def test_loft_goes_ruled_directly_when_rings_do_not_fit(monkeypatch):
    """With the pole cap too low for the rings to fit, the smooth loft is
    not even tried (every such attempt blew up before falling back)."""
    from stl_to_solid import sliced_loft as sl
    calls = []
    orig = sl._thru

    def spy(wires, ruled, apex=None, apex_first=False):
        calls.append(bool(ruled))
        return orig(wires, ruled, apex, apex_first)
    monkeypatch.setattr(sl, '_thru', spy)
    monkeypatch.setattr(sl, 'RING_POLES_MIN', 4)
    monkeypatch.setattr(sl, 'RING_POLES_MAX', 4)
    shape, info = sl.loft_body(_sphere(), 2, 0.5, verbose=False)
    assert calls and all(calls), calls
    assert info['n_ruled_runs'] == 1


def test_loft_pairs_extrudes_a_pair_whose_rings_do_not_correspond(monkeypatch):
    """The pair chain: with the per-pair volume check made impossible to
    pass, every pair is replaced by an extrusion of its lower ring, the
    pieces fuse to one solid and the volume is the prism's. (ThruSections
    itself picks compatible origins on closed wires, so a rolled or
    turned ring alone does not make a bad pair.)"""
    from stl_to_solid import sliced_loft as sl
    monkeypatch.setattr(sl, 'RULED_VOL_PCT', 0.0)
    rc = np.array([[10, 5], [-10, 5], [-10, -5], [10, -5]], float)
    ring = sl.resample(rc, 160)
    zs = [0.0, 10.0, 20.0, 30.0]
    rings3d = [np.c_[ring, np.full(len(ring), z)] for z in zs]
    wires, _, _ = sl._ring_wires(rings3d)
    solid, ext = sl._loft_pairs(wires, np.array([0.0, 0.0, 1.0]), [200.0] * 4, zs, 'test', False)
    assert ext == [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]], ext
    assert len(solid.Solids()) == 1
    assert abs(sl._volume(solid) - 6000.0) / 6000.0 < 0.01


def test_loft_never_drops_a_run(tmp_path, monkeypatch):
    """With the run-level volume check made impossible to pass, every run
    falls back to the pair chain instead of vanishing: nothing skipped,
    the material all there."""
    from stl_to_solid import sliced_loft as sl
    monkeypatch.setattr(sl, 'RULED_VOL_PCT', 0.0)
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    m = trimesh.load(p, force='mesh')
    shape, info = sl.loft_body(m, 2, 0.5, verbose=False)
    assert info['skipped'] == [], info['skipped']
    assert info['n_extruded_pairs'] > 0
    from stl_to_solid.pipeline import validate
    mt = validate(shape, m)
    assert mt['vol_err_pct'] < 1.0, mt


def test_validate_can_ignore_internal_caps():
    """Two abutting boxes as one compound against the mesh of their
    union: the shared cap lies inside the part and must not count as a
    solid -> mesh deviation when asked to ignore inside points."""
    from stl_to_solid.pipeline import validate
    a = cq.Workplane('XY').box(20, 20, 10, centered=(True, True, False))
    b = cq.Workplane('XY').box(20, 20, 10, centered=(True, True, False)).translate((0, 0, 10))
    comp = cq.Compound.makeCompound([a.val(), b.val()])
    m = trimesh.creation.box([20, 20, 20])
    m.apply_translation([0, 0, 10])
    plain = validate(comp.wrapped, m)
    assert plain['rev_dev_max'] > 1.0, plain
    fixed = validate(comp.wrapped, m, ignore_inside=True)
    assert fixed['rev_dev_p95'] < 0.05 and fixed['rev_dev_max'] < 0.05, fixed
    assert fixed['rev_internal_pts'] > 0
    assert fixed['dev_max'] < 0.05


def test_loft_prismatic_hint(tmp_path):
    """Flat side walls square to the other axes mark a prismatic part; a
    disc's flat top and bottom, or a plate across its thin axis, do not."""
    from stl_to_solid.sliced_loft import loft_body
    p = synth.export(synth.cross_blind(), tmp_path / 'cb.stl')
    m = trimesh.load(p, force='mesh')
    _, info = loft_body(m, 2, 0.5, verbose=False)
    assert info['prismatic_hint'] is True and info['planar_frac'] > 0.3, info['planar_frac']
    _, info = loft_body(_sphere(), 2, 0.5, verbose=False)
    assert info['prismatic_hint'] is False
    cap = cq.Workplane('XY').circle(30).extrude(12)
    m = trimesh.load(synth.export(cap, tmp_path / 'cap.stl'), force='mesh')
    _, info = loft_body(m, 2, 0.5, verbose=False)
    assert info['prismatic_hint'] is False, info['planar_frac']


def test_loft_stepped_shaft_breaks_at_levels(tmp_path):
    from stl_to_solid.sliced_loft import loft_body
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    m = trimesh.load(p, force='mesh')
    shape, info = loft_body(m, 2, 0.2, verbose=False)
    assert info['n_runs'] == 3 and info['n_levels'] == 4
    m2, s = _check(shape, m, faces_max=3 * 3, dev_p95=0.05, vol_pct=0.5)
    # the shoulders are real planes
    assert sum(1 for f in s.Faces() if f.geomType() == 'PLANE') >= 4


def test_loft_plate_holes_are_cut(tmp_path):
    from stl_to_solid.sliced_loft import loft_body
    p = synth.export(synth.plate_holes(), tmp_path / 'plate.stl')
    m = trimesh.load(p, force='mesh')
    shape, info = loft_body(m, 2, 0.5, verbose=False)
    assert info['n_holes'] == 4
    _check(shape, m, faces_max=3 + 4 + 4, dev_p95=0.1, vol_pct=1.0)


def test_loft_ruled_option(tmp_path):
    from stl_to_solid.sliced_loft import loft_body
    m = _sphere()
    shape, info = loft_body(m, 2, 0.5, ruled=True, verbose=False)
    assert info['ruled'] and info['n_ruled_runs'] == 1
    s = cq.Shape.cast(shape)
    assert len(s.Faces()) > 20            # one face per section pair
    _check(shape, m, faces_max=200, dev_p95=0.1, vol_pct=1.0)


def test_pipeline_loft_mode_writes_step_and_fusion_script(tmp_path):
    from stl_to_solid.pipeline import run
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    out = str(tmp_path / 'shaft.step')
    r = run(p, out, method='loft', slice_mm=0.5, verbose=False)
    assert r['mode'] == 'loft'
    assert r['metrics']['loft']['n_runs'] == 3
    assert r['metrics']['gate_ok'] is True
    got = _reimport(out)
    assert got['solids'] == 1 and got['naked_edges'] == 0
    assert r['script'] is None
    fs = r['fusion_script']
    assert fs and os.path.exists(fs) and fs.endswith('_fusion.py')
    txt = open(fs).read()
    assert 'loftFeatures' in txt and 'sketchFittedSplines' in txt
    assert txt.count('_loft(lofts, newBody if first else join') == 3
    assert 'isTangentEdgesMerged' not in txt
    compile(txt, fs, 'exec')              # at least valid Python


def test_fusion_loft_script_rebuilds_every_piece(tmp_path):
    """The script draws what the STEP holds: every outline of a run (not
    only the first), an Extrude where a run bridges to the next, and a
    Loft to a sketch point for a dome tip."""
    from stl_to_solid.sliced_loft import loft_body
    from stl_to_solid.fusion_export import emit_fusion_loft_script
    p = synth.export(synth.cross_blind(), tmp_path / 'cb.stl')
    m = trimesh.load(p, force='mesh')
    _, info = loft_body(m, 2, 0.5, verbose=False)
    txt = emit_fusion_loft_script([info])
    compile(txt, 'cb_fusion.py', 'exec')
    assert '# bridge to the next run' in txt
    assert 'outline 2' in txt                     # the run split by the sideways hole
    assert 'isTangentEdgesMerged' not in txt
    n_chains = sum(len(r['chains']) for r in info['runs'] if r['n'] >= 2 or r.get('bridge_to') is not None)
    assert txt.count('_loft(lofts, newBody if first else join') + txt.count('# bridge to the next run') >= n_chains
    _, info = loft_body(_sphere(), 2, 0.5, verbose=False)
    assert len(info['cones']) == 2
    txt = emit_fusion_loft_script([info])
    compile(txt, 'sphere_fusion.py', 'exec')
    assert txt.count('sketchPoints.add') == 2


def test_pipeline_loft_gate_is_reported_not_enforced(tmp_path):
    """A sideways hole smears under a loft; the loft is still written."""
    from stl_to_solid.pipeline import run
    p = synth.export(synth.cross_blind(), tmp_path / 'cb.stl')
    out = str(tmp_path / 'cb.step')
    r = run(p, out, method='loft', slice_mm=0.5, slice_axis='z', verbose=False)
    assert r['mode'] == 'loft'
    assert r['metrics']['gate_ok'] is False          # the smeared holes
    # the runs share their planes exactly (bridges are repeated rings
    # inside a run), so the pieces fuse to one solid across the holes
    assert _reimport(out)['solids'] == 1
    assert r['metrics']['vol_err_pct'] < 1.0


def test_loft_side_boss_is_one_solid(tmp_path):
    """A block with a cylinder boss across the axis: the boss changes the
    outline between two slices (no flat step), which used to leave a
    0.2 mm bridge sliver as its own solid. One solid, volume exact."""
    from stl_to_solid.sliced_loft import loft_body
    wp = cq.Workplane('XY').box(40, 30, 20).union(cq.Workplane('YZ').circle(5).extrude(30))
    m = trimesh.load(synth.export(wp, tmp_path / 'boss.stl'), force='mesh')
    shape, info = loft_body(m, 2, 0.5, verbose=False)
    assert info['skipped'] == [] and info['n_solids'] == 1
    m2, s = _check(shape, m, faces_max=40, dev_p95=0.05, vol_pct=0.5)
    # no piece thinner than the interval survives as a run of its own
    assert all(r['z1'] - r['z0'] >= 0.5 - 1e-6 or r['n'] == 1 for r in info['runs'])


def test_cli_flags_parse(tmp_path):
    import subprocess, sys
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    out = str(tmp_path / 'shaft.step')
    res = subprocess.run([sys.executable, '-m', 'stl_to_solid.pipeline', p, out,
                          '--method', 'loft', '--slice-mm', '1.0', '--slice-axis', 'z',
                          '--workers', '0', '--quiet'], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert os.path.exists(out)


# --- single slice: API + Fusion sketch script ------------------------------

def test_section_api_and_sketch_script(tmp_path, monkeypatch):
    """Upload a stepped shaft, trace one slice through its middle step,
    get the curves back as JSON and the same slice as a Fusion sketch."""
    pytest.importorskip('fastapi')
    pytest.importorskip('httpx')
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch.setenv('STLTOSOLID_DATA', str(tmp_path / 'data'))
    from backend import jobs, main
    importlib.reload(jobs)
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl')
    with TestClient(main.app) as c:
        with open(p, 'rb') as fh:
            r = c.post('/api/jobs', files={'file': ('shaft.stl', fh, 'application/octet-stream')})
        assert r.status_code == 200, r.text
        job = r.json()['id']
        # the middle step is r=7 between z=20 and 35; the box centre is z=22.5
        r = c.get(f'/api/jobs/{job}/loft-axis', params={'slice_mm': 0.5})
        assert r.status_code == 200, r.text
        assert r.json()['axis'] == 'z' and set(r.json()['scores']) == {'x', 'y', 'z'}
        r = c.get(f'/api/jobs/{job}/section', params={'axis': 'z', 'offset': 5.0, 'tol': 0.08})
        assert r.status_code == 200, r.text
        sec = r.json()
        assert sec['axis'] == 'z' and sec['at'] == pytest.approx(27.5, abs=0.01)
        assert sec['stats']['loops'] == 1 and sec['stats']['circles'] == 1
        assert len(sec['polylines']) == 1
        r = c.get(f'/api/jobs/{job}/section-script', params={'axis': 'z', 'offset': 5.0})
        assert r.status_code == 200
        txt = r.text
        assert 'sketchCircles.addByCenterRadius' in txt and 'root.sketches.add' in txt
        compile(txt, 'section.py', 'exec')
        assert 'attachment' in r.headers['content-disposition']
        r = c.get(f'/api/jobs/{job}/section', params={'axis': 'q'})
        assert r.status_code == 400


def test_sections_script_draws_lines_arcs_and_splines():
    from stl_to_solid.section_fit import section_preview
    from stl_to_solid.fusion_export import emit_fusion_sections_script
    p = synth.export(synth.rounded_rect(), '/tmp/_s2p_rr2.stl')
    m = trimesh.load(p, force='mesh')
    sec = section_preview(m.vertices, m.faces, [0, 0, 0], [0, 0, 1], tol=0.08)
    txt = emit_fusion_sections_script([{'origin': sec['origin'], 'normal': sec['normal'],
                                        'name': 'rr', 'loops': sec['loops']}])
    assert txt.count('sketchLines.addByTwoPoints') == 4
    assert txt.count('sketchArcs.addByThreePoints') == 4
    compile(txt, 'rr.py', 'exec')
