"""Fusion 360 Boundary Fill script for face-group results: the engine hands
out every region's surface and extent, the emitter turns them into a script
that parses, is in centimetres, carries one tool per region and one
boundaryFillFeatures call, and the pipeline writes it only for face-group
bodies. The script itself can only run inside Fusion."""
import ast
import math
import numpy as np
import pytest

from . import synth


def _regions(wp, tmp_path, name):
    from stl_to_solid.mesh_prep import load_and_prep_bodies
    from stl_to_solid import facegroups
    p = synth.export(wp, tmp_path / f'{name}.stl')
    bodies, _, _ = load_and_prep_bodies(p, verbose=False)
    m = bodies[0].mesh
    _, stats = facegroups.convert(m, tol=0.08)
    return m, stats


def _script_ns(text):
    """Execute the data part of a generated script (everything before the
    adsk import) and return its namespace."""
    head = text.split('\nimport adsk.core')[0]
    ns = {}
    exec(head, ns)
    return ns


@pytest.mark.parametrize('name,builder,kinds', [
    ('fillet_top', synth.fillet_top, {'plane': 6, 'cylinder': 4}),
    ('sphere_boss', synth.sphere_boss, {'plane': 6, 'sphere': 1}),
    ('csk_plate', synth.csk_plate, {'plane': 6, 'cylinder': 4, 'cone': 4}),
    ('boss_fillet', synth.boss_fillet, {'plane': 7, 'cylinder': 1, 'torus': 1}),
    ('pencil', synth.pencil, {'plane': 1, 'cylinder': 1, 'cone': 1}),
])
def test_export_regions_and_script(tmp_path, name, builder, kinds):
    from collections import Counter
    from stl_to_solid.fusion_boundary_fill import emit_boundary_fill_script
    m, stats = _regions(builder(), tmp_path, name)
    ex = stats['export']
    regs = ex['regions']
    assert Counter(r['kind'] for r in regs) == kinds
    assert abs(ex['mesh_volume'] - m.volume) < 1e-6
    if name == 'fillet_top':
        # four quarter-round fillets: each spans ~90 degrees, none is full
        spans = [r['a1'] - r['a0'] for r in regs if r['kind'] == 'cylinder']
        assert all(abs(sp - math.pi / 2) < 0.2 for sp in spans), spans
    if name == 'csk_plate':
        # through-holes go all the way round
        assert all(r['a0'] is None for r in regs if r['kind'] == 'cylinder')
    if name == 'pencil':
        # the tip cone reaches its apex: t0 = 0, full ring, one solid cone tool
        c = [r for r in regs if r['kind'] == 'cone'][0]
        assert c['t0'] == 0.0 and c['a0'] is None and not c['concave'], c
    for r in regs:
        assert r['inside'], f"region {r['id']} ({r['kind']}) has no inside point"
        assert m.contains(np.asarray(r['inside'])).all()
        assert r['src'] and all(isinstance(i, int) for i in r['src'])
        assert isinstance(r['approx'], bool) and 'seal' not in r
        if r['kind'] == 'plane':
            assert len(r['hull']) >= 3
            n, o = np.asarray(r['normal']), np.asarray(r['point'])
            assert np.abs((np.asarray(r['hull']) - o) @ n).max() < 1e-3
        elif r['kind'] in ('cylinder', 'cone'):
            assert r['t1'] > r['t0']
            if r['kind'] == 'cone':
                assert r['t0'] >= -1e-6           # data on the +axis side of the apex
    # every value must be finite and JSON-able
    import json
    json.dumps(ex)

    build = dict(ex, mode='facegroup')
    text = emit_boundary_fill_script([build], name)
    ast.parse(text)
    assert 'nan' not in text and 'inf' not in text.replace('info', '')
    assert 'boundaryFillFeatures' in text and 'TemporaryBRepManager' in text
    assert 'TINY_CELL' not in text and 'WARNING' not in text
    # both ends of a cone tool share one angular window (a ruled sheet
    # between a circle and an arc fails), computed by the shared _arc_span
    assert 'span = _arc_span(r1, a0, a1, slant)' in text
    assert '_arc_or_circle(p2, ax, xref, r2, span)' in text
    ns = _script_ns(text)
    assert len(ns['BODIES']) == 1
    bd = ns['BODIES'][0]
    assert bd['name'] == name
    assert 0 < len(bd["surfaces"]) <= len(regs)
    assert abs(bd['volume'] - m.volume / 1000.0) < 1e-6      # cm^3
    # units: every surface lies inside the mesh bounds expressed in cm
    lo, hi = m.bounds[0] / 10.0 - 0.01, m.bounds[1] / 10.0 + 0.01
    for s in bd['surfaces']:
        if s[0] == 'plane':
            o, u, v = (np.asarray(x) for x in s[1:4])
            pts = np.asarray([o + x * u + y * v for x, y in s[4]])
            assert (pts >= lo).all() and (pts <= hi).all()
            assert abs(np.linalg.norm(np.cross(u, v)) - 1) < 1e-5
            assert 0 <= s[5] <= 0.2                       # emin, cm
        elif s[0] == 'cyl':
            _, o, ax, r, t0, t1, emin, xref, a0, a1 = s
            assert abs(np.linalg.norm(ax) - 1) < 1e-5 and r > 0 and t1 > t0
            assert abs(np.dot(ax, xref)) < 1e-5
            assert (a0 is None and a1 is None) or 0 < a1 - a0 < 2 * math.pi
        elif s[0] == 'cone':
            _, apex, ax, half, t0, t1, emin, xref, a0, a1 = s
            assert 0 < half < math.pi / 2 and t1 > t0 >= 0
        elif s[0] == 'sphere':
            assert s[2] > 0
        elif s[0] == 'torus':
            _, c, ax, R, r = s
            assert abs(np.linalg.norm(ax) - 1) < 1e-5 and R > r > 0
            assert (np.asarray(c) >= lo).all() and (np.asarray(c) <= hi).all()
    if 'torus' in kinds:
        assert sum(1 for s in bd['surfaces'] if s[0] == 'torus') == kinds['torus']
        assert 'createTorus' in text
    for p in bd['inside']:
        q = np.asarray(p[:3], float)
        assert (q >= lo).all() and (q <= hi).all()
        assert isinstance(p[3], str)                    # '<region id> <kind>' label


def test_script_refuses_huge_and_skips_other_modes(tmp_path):
    from stl_to_solid.fusion_boundary_fill import emit_boundary_fill_script, TooManyRegions
    _, stats = _regions(synth.fillet_top(), tmp_path, 'ft')
    build = dict(stats['export'], mode='facegroup')
    with pytest.raises(TooManyRegions):
        emit_boundary_fill_script([build], 'ft', max_regions=3)
    text = emit_boundary_fill_script([{'mode': 'prismatic'}, build], 'ft')
    ns = _script_ns(text)
    assert len(ns['BODIES']) == 1 and ns['BODIES'][0]['name'] == 'ft_body2'
    assert '# body 1: prismatic' in text
    with pytest.raises(ValueError):
        emit_boundary_fill_script([{'mode': 'prismatic'}], 'ft')


def test_pipeline_writes_bfill_script_for_facegroup_only(tmp_path):
    from stl_to_solid import run
    # fillet_top is not an extrusion along any axis: face-group route
    p = synth.export(synth.fillet_top(), tmp_path / 'ft.stl')
    r = run(str(p), str(tmp_path / 'ft.step'), verbose=False)
    assert r['mode'] == 'facegroup'
    assert 'build' not in r['metrics']
    assert r['bfill_script'] == str(tmp_path / 'ft_fusion_bfill.py')
    assert (tmp_path / 'ft_fusion_bfill.py').exists()
    assert not (tmp_path / 'ft_fusion.py').exists()
    ast.parse((tmp_path / 'ft_fusion_bfill.py').read_text())
    # a plain extrusion gets the prismatic scripts and no Boundary Fill one
    p = synth.export(synth.plate_holes(), tmp_path / 'ph.stl')
    r = run(str(p), str(tmp_path / 'ph.step'), verbose=False)
    assert r['mode'] == 'prismatic'
    assert (tmp_path / 'ph_fusion.py').exists()
    assert r['bfill_script'] is None
    assert not (tmp_path / 'ph_fusion_bfill.py').exists()


def test_tangent_fillet_snapped_to_its_plane():
    """A fillet fitted a few microns short of its tangent plane must be moved
    so the emitted cylinder touches the emitted plane exactly; a cylinder
    that merely crosses a plane is left alone."""
    from stl_to_solid.fusion_boundary_fill import _surface_record, _snap_tangencies
    plane = {'id': 0, 'kind': 'plane', 'normal': [0, 0, 1], 'point': [0, 0, 10.0],
             'hull': [[-10, -10, 10.0], [10, -10, 10.0], [10, 10, 10.0], [-10, 10, 10.0]],
             'area': 400.0, 'adjacent': {1: list(range(12)), 2: list(range(12)), 3: list(range(8))}}
    # fillet r=2 along y, tangent to z=10 from below: axis should be at z=8,
    # fitted 4 microns low
    fillet = {'id': 1, 'kind': 'cylinder', 'axis': [0, 1, 0], 'point': [5.0, 0, 7.996],
              'r': 2.0, 't0': -10, 't1': 10, 'area': 60.0, 'adjacent': {0: list(range(12)), 3: list(range(6))}}
    # a hole through the plate: crosses the plane, not tangent
    hole = {'id': 2, 'kind': 'cylinder', 'axis': [0, 0, 1], 'point': [-5.0, 0, 0],
            'r': 1.5, 't0': 0, 't1': 10, 'area': 90.0, 'adjacent': {0: list(range(12))}}
    # a smaller fillet running into the big one (outer tangency, parallel
    # axes): must end up touching the *snapped* big fillet
    small = {'id': 3, 'kind': 'cylinder', 'axis': [0, 1, 0], 'point': [7.831, 0, 9.0],
             'r': 1.0, 't0': -10, 't1': 10, 'area': 30.0, 'adjacent': {0: list(range(8)), 1: list(range(6))}}
    regs = [plane, fillet, hole, small]
    S = [_surface_record(r) for r in regs]
    before = list(S)
    snapped = _snap_tangencies(regs, S)
    assert snapped == {(1, 0), (0, 1), (3, 0), (0, 3), (3, 1), (1, 3)}
    assert S[2] == before[2]
    o, u, v = (np.asarray(x) for x in S[0][1:4])
    n = np.cross(u, v)
    p, r = np.asarray(S[1][1]), S[1][3]
    assert abs(abs((p - o) @ n) - r) < 1e-8          # cm: exact tangency
    assert abs(p[2] - 0.8) < 1e-8 and abs(p[0] - 0.5) < 1e-9
    q, rq = np.asarray(S[3][1]), S[3][3]
    assert abs(abs((q - o) @ n) - rq) < 1e-8         # tangent to the plane
    assert abs(np.linalg.norm((q - p)[[0, 2]]) - (r + rq)) < 1e-8   # and to the fillet


def test_same_fit_pieces_become_one_tool():
    """Two pieces of one torus/sphere/cone fit (the engine's pinch repair
    gives the peeled piece the core's params) must not become two
    coincident whole tools — a kernel failure — but one."""
    from stl_to_solid.fusion_boundary_fill import _merge_same_surface
    t = {'kind': 'torus', 'built': 'analytic', 'center': [0, 0, 5.0], 'axis': [0, 0, 1],
         'R': 6.0, 'r': 1.5, 'area': 30.0, 'inside': [[6, 0, 4]], 'adjacent': {1: [0, 1]}}
    a = dict(t, id=0)
    b = dict(t, id=1, area=2.0, inside=[[0, 6, 4]], adjacent={0: [0, 1]})
    out, n = _merge_same_surface([a, b])
    assert n == 1 and len(out) == 1 and out[0]['kind'] == 'torus'
    assert len(out[0]['inside']) == 2 and out[0]['area'] == 32.0
    c = dict(t, id=1, center=[0, 0, 9.0], adjacent={0: [0, 1]})          # another torus
    out, n = _merge_same_surface([a, c])
    assert n == 0 and len(out) == 2
    s = {'kind': 'sphere', 'built': 'analytic', 'center': [1, 2, 3.0], 'r': 4.0, 'area': 10.0,
         'inside': [[1, 2, 6]], 'adjacent': {1: [0, 1]}}
    out, n = _merge_same_surface([dict(s, id=0), dict(s, id=1, adjacent={0: [0, 1]})])
    assert n == 1 and len(out) == 1
    k = {'kind': 'cone', 'built': 'analytic', 'apex': [0, 0, 0.0], 'axis': [0, 0, 1],
         'half_angle': 0.3, 't0': 2.0, 't1': 5.0, 'xref': [1, 0, 0], 'a0': 0.0, 'a1': 1.0,
         'area': 10.0, 'inside': [[0, 0, 3]], 'adjacent': {1: [0, 1]}}
    k2 = dict(k, id=1, t0=5.0, t1=8.0, a0=1.0, a1=2.0, adjacent={0: [0, 1]})
    out, n = _merge_same_surface([dict(k, id=0), k2])
    assert n == 1 and len(out) == 1
    assert out[0]['t0'] == 2.0 and out[0]['t1'] == 8.0          # extents united
    assert out[0]['a0'] == pytest.approx(0.0, abs=0.03) and out[0]['a1'] == pytest.approx(2.0, abs=0.03)


def test_tapered_fillet_cone_snapped_to_its_planes():
    """An approximate cone tool blending two planes is tangent to each along
    a ruling: after the snap the apex lies on both planes and the axis makes
    exactly (90 - half) degrees with each normal — a fit a hair off is
    corrected, not left with a growing gap along the ruling."""
    from stl_to_solid.fusion_boundary_fill import _surface_record, _snap_tangencies
    half = math.radians(10.0)
    s = math.sin(half)
    axis = np.array([math.sqrt(1 - 2 * s * s), s, s])          # tangent to z=0 and y=0
    th = math.radians(0.3)                                    # ... fitted 0.3 deg off
    rot = np.array([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]])
    fitted = rot @ axis
    xref = np.cross(fitted, [0, 0, 1.0])
    xref /= np.linalg.norm(xref)
    big = [[-50, -50, 0.0], [50, -50, 0.0], [50, 50, 0.0], [-50, 50, 0.0]]
    p1 = {'id': 0, 'kind': 'plane', 'normal': [0, 0, 1], 'point': [0, 0, 0.0], 'hull': big,
          'area': 10000.0, 'adjacent': {2: [0, 1, 2]}}
    p2 = {'id': 1, 'kind': 'plane', 'normal': [0, 1, 0], 'point': [0, 0, 0.0],
          'hull': [[-50, 0, -50.0], [50, 0, -50.0], [50, 0, 50.0], [-50, 0, 50.0]],
          'area': 10000.0, 'adjacent': {2: [3, 4, 5]}}
    cone = {'id': 2, 'kind': 'cone', 'apex': [0.0, 0.003, -0.002], 'axis': fitted.tolist(),
            'half_angle': half, 't0': 1.0, 't1': 20.0, 'xref': xref.tolist(), 'a0': 0.0,
            'a1': 1.0, 'area': 50.0, 'approx': True, 'adjacent': {0: [0, 1, 2], 1: [3, 4, 5]}}
    regs = [p1, p2, cone]
    S = [_surface_record(r) for r in regs]
    pairs = _snap_tangencies(regs, S)
    assert {(2, 0), (0, 2), (2, 1), (1, 2)} <= pairs
    apex, ax, xr = (np.asarray(S[2][k], float) for k in (1, 2, 7))
    assert abs(apex[2]) < 1e-8 and abs(apex[1]) < 1e-8             # on both planes
    assert abs(np.linalg.norm(ax) - 1) < 1e-8
    assert abs(ax @ [0, 0, 1] - s) < 1e-6 and abs(ax @ [0, 1, 0] - s) < 1e-6
    assert abs(ax @ xr) < 1e-6 and abs(np.linalg.norm(xr) - 1) < 1e-6
    assert S[2][3:7] == tuple(_surface_record(cone)[3:7])          # half, t0, t1, emin kept


def test_same_surface_neighbours_are_merged():
    from stl_to_solid.fusion_boundary_fill import _merge_same_surface
    a = {'id': 0, 'kind': 'plane', 'normal': [0, 0, 1], 'point': [0, 0, 10.0], 'area': 100.0,
         'hull': [[0, 0, 10.0], [10, 0, 10.0], [10, 10, 10.0], [0, 10, 10.0]],
         'inside': [[5, 5, 9.7]], 'adjacent': {1: [0, 1, 2, 3], 2: [4, 5, 6]}}
    b = {'id': 1, 'kind': 'plane', 'normal': [0, 0, 1], 'point': [0, 0, 10.006], 'area': 50.0,
         'hull': [[10, 0, 10.006], [15, 0, 10.006], [15, 10, 10.006], [10, 10, 10.006]],
         'inside': [[12, 5, 9.7]], 'adjacent': {0: [0, 1, 2, 3], 2: [7]}}
    c = {'id': 2, 'kind': 'plane', 'normal': [0, 0, 1], 'point': [0, 0, 12.0], 'area': 50.0,
         'hull': [[0, 10, 12.0], [15, 10, 12.0], [15, 20, 12.0], [0, 20, 12.0]],
         'inside': [[5, 15, 11.7]], 'adjacent': {0: [4, 5, 6], 1: [7]}}
    out, n = _merge_same_surface([a, b, c])
    assert n == 1 and len(out) == 2
    m = next(r for r in out if r['id'] == 0)
    assert m['area'] == 150.0 and len(m['inside']) == 2
    hull = np.asarray(m['hull'])
    assert hull[:, 0].max() == 15 and np.allclose(hull[:, 2], 10.0)   # on a's plane
    assert m['adjacent'] == {2: [4, 5, 6, 7]}
    assert next(r for r in out if r['id'] == 2)['adjacent'] == {0: [4, 5, 6, 7]}


def test_pipeline_outlook_is_the_dry_run_and_the_header_agrees(tmp_path, monkeypatch):
    """The API's outlook comes from the OCC dry run, per body; a FAIL is
    written into the script's own header; when the dry run cannot run the
    outlook says so (ok None) instead of a heuristic verdict; and a run
    without scripts carries no outlook at all (nothing stale from the
    run before)."""
    from stl_to_solid import run, bfill_check
    p = synth.export(synth.fillet_top(), tmp_path / 'ft.stl')
    r = run(str(p), str(tmp_path / 'ft.step'), verbose=False)
    c = r['bfill_check']
    assert c['ok'] is True and 'enclose' in c['reason']
    assert c['enclosed_pct'] == pytest.approx(100.0, abs=3.0)
    text = (tmp_path / 'ft_fusion_bfill.py').read_text()
    assert 'WARNING' not in text
    r2 = run(str(p), str(tmp_path / 'ft_noscript.step'), verbose=False, write_script=False)
    assert r2['bfill_check'] is None and r2['bfill_script'] is None

    def failing(text, budget_s=None, **kw):
        return [dict(name='ft', enclosed_pct=80.0, unenclosed=[], n_probes=5, error=None)]
    monkeypatch.setattr(bfill_check, 'check_script', failing)
    r = run(str(p), str(tmp_path / 'ft_fail.step'), verbose=False)
    c = r['bfill_check']
    assert c['ok'] is False and 'ft: cells enclose 80.0%' in c['reason']
    text = (tmp_path / 'ft_fail_fusion_bfill.py').read_text()
    assert 'WARNING: likely to fail in Fusion' in text and c['reason'] in text
    ast.parse(text)

    def broken(text, budget_s=None, **kw):
        raise RuntimeError('no kernel')
    monkeypatch.setattr(bfill_check, 'check_script', broken)
    r = run(str(p), str(tmp_path / 'ft_nocheck.step'), verbose=False)
    c = r['bfill_check']
    assert c['ok'] is None and c['reason'].startswith('not checked') and 'RuntimeError' in c['reason']
    assert 'WARNING' not in (tmp_path / 'ft_nocheck_fusion_bfill.py').read_text()


def test_outlook_flags_band_blends(tmp_path):
    """The outlook check must pass cleanly fitted parts — including a torus
    blend, which is one torus region since the torus fit — and flag a
    blend kept as a chain of short cylinder bands, the one thing Fusion's
    Boundary Fill is known to choke on."""
    from stl_to_solid.fusion_boundary_fill import assess
    _, stats = _regions(synth.fillet_top(), tmp_path, 'ft')
    c = assess(stats['export'])
    assert c['ok'] and c['bands'] == 0 and c['unfitted'] == 0
    _, stats = _regions(synth.boss_fillet(), tmp_path, 'torus')
    c = assess(stats['export'])
    assert c['ok'] and c['bands'] == 0, c
    assert sum(1 for r in stats['export']['regions'] if r['kind'] == 'torus') == 1
    # an apex cone is one clean cone region since the cone fits (v0.3.3);
    # the stubby shank+cone pair still trips the band COUNTER (2 <= 3), but
    # the verdict must stay ok and nothing may be left as facets
    _, stats = _regions(synth.pencil(), tmp_path, 'pencil')
    c = assess(stats['export'])
    assert c['ok'] and c['bands'] <= 3 and c['unfitted'] == 0, c
    assert sum(1 for r in stats['export']['regions'] if r['kind'] == 'cone') == 1
    # a chain of 24 short bands round a circle, each 15 deg from the next
    regs = []
    for i in range(24):
        a = math.radians(15 * i)
        regs.append({'id': i, 'kind': 'cylinder', 'built': 'analytic', 'area': 1.0,
                     'axis': [-math.sin(a), math.cos(a), 0.0], 'point': [6.5 * math.cos(a), 6.5 * math.sin(a), 0.0],
                     'r': 1.5, 't0': -0.8, 't1': 0.8, 'adjacent': {(i + 1) % 24: [0, 1], (i - 1) % 24: [2, 3]}})
    c = assess({'regions': regs})
    assert not c['ok'] and c['bands'] == 24 and 'blend bands' in c['reason'], c
