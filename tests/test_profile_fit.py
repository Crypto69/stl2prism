"""B1: the profile fitter (F2/F6/F10) — exact primitives from CAD sections."""
import numpy as np
import pytest
import trimesh

from . import synth


def _ring(wp, tmp_path, name, tol=0.08):
    """Fit the mid-slab outer ring of a single-slab part; return prims."""
    from stl2prism.extrusion import dominant_axis, score_axis
    from stl2prism.rebuild import _fit_ring, _slab_vertices_2d, _axis_basis
    from stl2prism.profile_fit import solve_junctions
    m = trimesh.load(synth.export(wp, tmp_path / f'{name}.stl'), force='mesh')
    best = None
    for frac, ax in dominant_axis(m):
        sc, levels, slabs = score_axis(m, ax)
        if best is None or sc > best[0] + 1e-9:
            best = (sc, ax, levels, slabs)
    sc, ax, levels, slabs = best
    basis = _axis_basis(ax)
    slab = slabs[0]
    pts2d = _slab_vertices_2d(m, ax, basis, slab)
    poly = max(slab['polygons'], key=lambda p: p.area)
    ring = _fit_ring(np.array(poly.exterior.coords), tol, pts2d)
    if isinstance(ring, list):
        solve_junctions(ring)
    return ring, poly


def _types(ring):
    return ''.join('L' if p['type'] == 'line' else 'A' for p in ring)


def test_obround_is_two_lines_two_arcs(tmp_path):
    ring, poly = _ring(synth.obround(), tmp_path, 'ob')
    assert isinstance(ring, list) and len(ring) == 4
    assert sorted(_types(ring)) == ['A', 'A', 'L', 'L']
    for p in ring:
        if p['type'] == 'arc':
            assert p['r'] == pytest.approx(6.0, abs=0.01)   # slot2D 40x12 -> r 6
            assert abs(p['sweep'] - np.pi) < 0.05


def test_rounded_rect_is_four_lines_four_arcs(tmp_path):
    ring, poly = _ring(synth.rounded_rect(), tmp_path, 'rr')
    assert len(ring) == 8 and sorted(_types(ring)) == list('AAAALLLL')
    for p in ring:
        if p['type'] == 'arc':
            assert p['r'] == pytest.approx(4.0, abs=0.005)


def test_fillets_are_tangent_and_lines_axis_aligned(tmp_path):
    from stl2prism.profile_fit import _line_dir, _arc_tangent_at
    ring, poly = _ring(synth.rounded_rect(), tmp_path, 'rr2')
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        assert np.allclose(a['p1'], b['p0'], atol=1e-9)          # exact closure
        if a['type'] == 'line' and b['type'] == 'arc':
            t = _arc_tangent_at(b, b['p0'])
            assert abs(abs(t @ _line_dir(a)) - 1) < 1e-6         # tangent
        if a['type'] == 'line':
            d = _line_dir(a)
            assert min(abs(d[0]), abs(d[1])) < 1e-9              # exactly axis-aligned


def test_bogus_big_arc_rejected():
    """A straight wall sampled only at its ends: a big circle through the
    two endpoints has zero vertex deviation but the chord interior sags by
    r(1-cos) — the polyline deviation must see it."""
    from stl2prism.profile_fit import polyline_circle_dev
    r = 52.0
    c = np.array([-7.0, -6.0 + np.sqrt(r * r - 7.0 ** 2)])   # circle through (0,-6),(-14,-6)
    pts = np.array([[0.0, -6.0], [-14.0, -6.0]])
    d = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
    assert np.abs(d - r).max() < 1e-9                        # vertices: perfect
    dev = polyline_circle_dev(pts, c, r)
    assert dev > 0.4                                          # sagitta ~0.47 mm


def test_end_to_end_face_counts(tmp_path):
    from stl2prism.pipeline import run
    for name, wp, faces in [('obround', synth.obround(), 10),
                            ('rounded_rect', synth.rounded_rect(), 10),
                            ('plate_fillets', synth.plate_holes_fillets(), 14),
                            ('slot_hex', synth.slot_hex(), 17)]:
        p = synth.export(wp, tmp_path / f'{name}.stl')
        out = str(tmp_path / f'{name}.step')
        r = run(p, out, verbose=False)
        assert r['mode'] == 'prismatic', (name, r['metrics'])
        assert synth.step_faces(out)[0] == faces, name
        assert r['metrics']['dev_max'] < 0.05, name
        assert r['metrics']['vol_err_pct'] < 0.5, name
        got = synth.reimport(out)
        assert got['solids'] == 1 and got['valid'], name


def test_arc_radius_refined_on_vertices(tmp_path):
    """Coarse tessellation: section vertices sit on chords (radius biased
    low); refinement on mesh vertices recovers the true radius."""
    from stl2prism.pipeline import run
    p = synth.export(synth.stepped_shaft(), tmp_path / 'shaft.stl', tol=0.5, ang=0.5)
    out = str(tmp_path / 'shaft.step')
    r = run(p, out, verbose=False)
    assert r['mode'] == 'prismatic'
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    rd = STEPControl_Reader(); rd.ReadFile(out); rd.TransferRoots(); s = rd.OneShape()
    radii = []
    e = TopExp_Explorer(s, TopAbs_FACE)
    while e.More():
        ad = GeomAdaptor_Surface(BRep_Tool.Surface_s(TopoDS.Face_s(e.Current())))
        if ad.GetType() == GeomAbs_Cylinder:
            radii.append(ad.Cylinder().Radius())
        e.Next()
    assert sorted(round(x, 2) for x in radii) == [4.0, 7.0, 10.0]


def _tessellated_arc(c, r, a0, a1, n, diag=True):
    """Section-like points along an arc: vertices on the circle plus, when
    `diag`, a point on each chord where a facet diagonal would be crossed."""
    ang = np.linspace(a0, a1, n + 1)
    V = np.column_stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)])
    if not diag:
        return V
    out = [V[0]]
    for a, b in zip(V[:-1], V[1:]):
        out.append(a + 0.35 * (b - a))
        out.append(b)
    return np.array(out)


def test_exact_wall_is_not_swallowed_by_a_gentle_arc():
    """A 5 mm straight run (two facets, five section points) followed by
    20 degrees of an r=12 arc: one circle through all of it stays within
    0.08 mm, but the wall must come out as a line and the arc as an arc of
    about r=12 — not one r=50..70 arc."""
    from stl2prism.profile_fit import segment_polyline, fit_line
    r = 12.0
    c = np.array([5.0, -r])                             # arc starts tangent at (5, 0)
    arc = _tessellated_arc(c, r, np.pi / 2, np.pi / 2 - np.radians(20), 8)
    wall = np.array([[0.0, 0.0], [0.52, 0.0], [2.75, 0.0], [3.27, 0.0], [5.0, 0.0]])
    pts = np.vstack([wall, arc[1:]])
    prims = segment_polyline(pts, tol=0.08, closed=False)
    kinds = [p['type'] for p in prims]
    assert kinds == ['line', 'arc'], kinds
    line, a = prims
    assert fit_line(line['_pts'])['dev'] < 1e-6
    assert abs(line['p0'][1]) < 1e-6 and abs(line['p1'][1]) < 1e-6   # exactly on the wall
    assert abs(a['r'] - r) < 0.5, a['r']


def test_true_arc_with_uneven_chords_stays_one_arc():
    """An r=8.45 fillet whose section alternates long and short chords
    (facet edge, facet diagonal) is one arc, not a chain of pieces."""
    from stl2prism.profile_fit import segment_polyline
    r = 8.45
    pts = _tessellated_arc(np.array([0.0, 0.0]), r, 0.0, np.pi / 2, 14)
    prims = segment_polyline(pts, tol=0.08, closed=False)
    assert [p['type'] for p in prims] == ['arc'], [p['type'] for p in prims]
    assert abs(prims[0]['r'] - r) < 0.05


def test_line_with_transition_facet_keeps_the_wall_exact():
    """A wall whose section ends with one short facet tilted by a degree
    (the tessellator's first step into a blend) is still the wall's own
    line: endpoints projected onto it, not a chord tilted towards the
    facet."""
    from stl2prism.profile_fit import _mk_line, _collinear_runs
    pts = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0], [3.5, 0.0], [6.0, 0.0],
                    [7.2, 0.021]])
    assert len(_collinear_runs(pts)) == 2
    line = _mk_line(pts)
    assert abs(line['p0'][1]) < 1e-9 and abs(line['p1'][1]) < 1e-9
    assert abs(line['p1'][0] - 7.2) < 1e-9


def test_walls_are_unified_across_slabs():
    """Two slabs whose fitted walls sit 4 microns apart (a 0.1 degree
    export tilt sampled at two heights) share one line afterwards."""
    from stl2prism.rebuild import _snap_lines_across_slabs
    def ring(y):
        pts = np.array([[0.0, y], [10.0, y]])
        return [{'type': 'line', 'p0': pts[0].copy(), 'p1': pts[1].copy(), '_pts': pts, 'snapped': True},
                {'type': 'line', 'p0': np.array([10.0, y]), 'p1': np.array([10.0, y + 5]),
                 '_pts': np.array([[10.0, y], [10.0, y + 5]]), 'snapped': True},
                {'type': 'line', 'p0': np.array([10.0, y + 5]), 'p1': np.array([0.0, y + 5]),
                 '_pts': np.array([[10.0, y + 5], [0.0, y + 5]]), 'snapped': True},
                {'type': 'line', 'p0': np.array([0.0, y + 5]), 'p1': np.array([0.0, y]),
                 '_pts': np.array([[0.0, y + 5], [0.0, y]]), 'snapped': True}]
    a, b = ring(0.0), ring(0.004)
    fitted = [[(a, [])], [(b, [])]]
    _snap_lines_across_slabs(fitted)
    assert abs(a[0]['p0'][1] - b[0]['p0'][1]) < 1e-9
    assert abs(a[2]['p0'][1] - b[2]['p0'][1]) < 1e-9
    # orientation preserved: the ring still runs the same way round
    assert a[0]['p1'][0] > a[0]['p0'][0] and a[2]['p1'][0] < a[2]['p0'][0]
