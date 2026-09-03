"""v0.3.8: a body's cavities go in as inner shells of its solid, not as a
boolean cut. Every test asserts on the artefact (OCC round trip)."""
import time
import pytest
import numpy as np
import trimesh
import cadquery as cq

from . import synth
from stl2prism.rebuild import add_cavities, _volume
from stl2prism.mesh_prep import Body
from stl2prism import pipeline


def _run(wp_or_path, tmp_path, name, **kw):
    src = (wp_or_path if isinstance(wp_or_path, str)
           else synth.export(wp_or_path, tmp_path / f'{name}.stl'))
    out = str(tmp_path / f'{name}.step')
    return pipeline.run(src, out, **{'verbose': False, **kw}), out


def _box(size, at=(0, 0, 0)):
    return cq.Workplane('XY').box(size, size, size).translate(at).val().wrapped


def _faceted(mesh):
    """An OCC solid with one planar face per triangle, like the faceted rung."""
    from stl2prism.rebuild import faceted_solid
    shape, _ = faceted_solid(mesh, verbose=False)
    return shape


# --- the function -------------------------------------------------------------

def test_box_in_box_is_one_valid_solid_with_exact_volume():
    shape, info = add_cavities(_box(40), [_box(20)])
    assert shape is not None, info
    assert info == {'solids': 1, 'cavities': 1, 'volume': pytest.approx(40 ** 3 - 20 ** 3)}
    s = cq.Shape.cast(shape)
    assert s.ShapeType() == 'Solid' and s.isValid()
    assert len(s.Shells()) == 2 and len(s.Faces()) == 12


def test_faceted_sphere_in_sphere_takes_milliseconds():
    outer = _faceted(trimesh.creation.icosphere(subdivisions=5, radius=10))    # 20480 faces
    inner = _faceted(trimesh.creation.icosphere(subdivisions=4, radius=4))     # 5120 faces
    t0 = time.time()
    shape, info = add_cavities(outer, [inner])
    dt = time.time() - t0
    assert shape is not None, info
    assert info['volume'] == pytest.approx(_volume(outer) - _volume(inner), rel=1e-9)
    assert cq.Shape.cast(shape).isValid()
    assert dt < 5.0, f'{dt:.1f} s'     # the classifier on 25k faces; the cut takes seconds


def test_two_cavities_in_one_body():
    shape, info = add_cavities(_box(40), [_box(8, (-10, 0, 0)), _box(8, (10, 0, 0))])
    assert shape is not None, info
    assert info['cavities'] == 2
    assert info['volume'] == pytest.approx(40 ** 3 - 2 * 8 ** 3)
    assert len(cq.Shape.cast(shape).Shells()) == 3


def test_cavity_poking_through_the_wall_is_refused():
    shape, why = add_cavities(_box(40), [_box(20, (15, 0, 0))])
    assert shape is None and 'not strictly inside' in why


def test_cavity_touching_the_wall_is_refused():
    shape, why = add_cavities(_box(40), [_box(20, (10, 0, 0))])     # face on face
    assert shape is None and 'not strictly inside' in why


def test_cavity_inside_a_cavity_is_refused():
    shape, why = add_cavities(_box(40), [_box(20), _box(5)])
    assert shape is None and 'not outside' in why


def test_already_inverted_cavity_is_not_flipped_twice():
    inner = cq.Shape.cast(_box(20)).wrapped.Reversed()
    shape, info = add_cavities(_box(40), [inner])
    assert shape is not None, info
    assert info['volume'] == pytest.approx(40 ** 3 - 20 ** 3)


def test_compound_outer_picks_the_solid_that_holds_the_cavity():
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    b.Add(comp, _box(40))
    b.Add(comp, _box(40, (100, 0, 0)))
    shape, info = add_cavities(comp, [_box(20, (100, 0, 0))])
    assert shape is not None, info
    assert info['solids'] == 2 and info['cavities'] == 1
    sol = sorted(cq.Shape.cast(shape).Solids(), key=lambda s: s.Center().x)
    assert len(sol) == 2
    assert sol[0].Volume() == pytest.approx(40 ** 3)
    assert sol[1].Volume() == pytest.approx(40 ** 3 - 20 ** 3)
    assert len(sol[1].Shells()) == 2


def test_cavity_in_no_solid_of_a_compound_is_refused():
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    b.Add(comp, _box(40))
    b.Add(comp, _box(40, (100, 0, 0)))
    shape, why = add_cavities(comp, [_box(20, (50, 0, 0))])
    assert shape is None and 'not strictly inside' in why


# --- the pipeline -----------------------------------------------------------

def test_hollow_cube_goes_through_inner_shells(tmp_path):
    r, out = _run(synth.hollow_cube(), tmp_path, 'hollow')
    m = r['metrics']
    assert m['voids'] == 1 and m['void_method'] == 'inner_shell'
    assert 'void_reason' not in m and m['void_s'] < 2.0
    assert m['vol_err_pct'] < 0.01
    got = synth.reimport(out)
    assert got['solids'] == 1 and got['valid'] and got['naked_edges'] == 0
    assert got['faces'] == 12
    assert got['volume'] == pytest.approx(40 ** 3 - 20 ** 3, rel=1e-6)


def test_sphere_in_sphere_round_trips_hollow(tmp_path):
    outer = trimesh.creation.icosphere(subdivisions=4, radius=10)
    inner = trimesh.creation.icosphere(subdivisions=3, radius=4)
    inner.invert()
    stl = str(tmp_path / 'ss.stl')
    trimesh.util.concatenate([outer, inner]).export(stl)
    r, out = _run(stl, tmp_path, 'ss')
    m = r['metrics']
    assert m['voids'] == 1 and m['void_method'] == 'inner_shell'
    assert m['vol_err_pct'] < 2.0
    got = synth.reimport(out)
    assert got['solids'] == 1 and got['valid'] and got['naked_edges'] == 0
    assert got['volume'] == pytest.approx(m['vol_solid'], rel=1e-6)


def test_assemble_group_falls_back_to_the_boolean(capsys):
    """A cavity the inner shell refuses (it pokes through the wall) is cut
    and the metrics say so."""
    outer = trimesh.creation.box((40, 40, 40))
    void = trimesh.creation.box((20, 20, 20))
    void.apply_translation((15, 0, 0))
    body = Body(outer, [void])
    shape, mode, m = pipeline._assemble_group(
        body, (_box(40), 'prismatic', {'x': 1}),
        [(_box(20, (15, 0, 0)), 'prismatic', {})], verbose=True)
    assert m['void_method'] == 'boolean' and 'not strictly inside' in m['void_reason']
    assert m['void_modes'] == ['prismatic'] and m['x'] == 1
    assert _volume(shape) == pytest.approx(40 ** 3 - 20 * 20 * 15)
    assert 'inner shells refused' in capsys.readouterr().out


def test_two_shell_assembly_keeps_the_hollow_box(tmp_path):
    stl = synth.assembly(tmp_path / 'asm.stl', sphere=False, hollow=True)
    r, out = _run(stl, tmp_path, 'asm')
    hollow = next(b for b in r['bodies'] if b['voids'])
    assert hollow['metrics']['void_method'] == 'inner_shell'
    stats = synth.solid_stats(out)
    assert (12, pytest.approx(30 ** 3 - 10 ** 3)) in [(f, v) for f, v in stats]


# --- nesting: a shell that crosses its container is a body, not a void ------

def _shells(*meshes):
    from stl2prism.mesh_prep import nest_shells
    return nest_shells(sorted(meshes, key=lambda m: -len(m.faces)))


def _tbox(size, at=(0, 0, 0)):
    b = trimesh.creation.box((size, size, size))
    b.apply_translation(at)
    return b


def _tube_block():
    """40 cube with a D10 hole through it along Z, as a mesh."""
    from stl2prism.pipeline import tessellate_solid
    m = tessellate_solid(cq.Workplane('XY').box(40, 40, 40).faces('>Z').workplane().hole(10))
    return trimesh.Trimesh(m.vertices, m.faces)          # merge the per-face vertices: closed


def _crossing_sphere():
    """r4 sphere at x=6: its bounding box sits inside the block's, its
    first vertex is in the material, and its near side reaches 3 mm into
    the hole."""
    s = trimesh.creation.icosphere(subdivisions=2, radius=4)
    s.apply_translation((6, 0, 0))
    order = np.argsort(-s.vertices[:, 0])                   # vertex 0 = the +x pole
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return trimesh.Trimesh(s.vertices[order], inv[s.faces], process=False)


def test_shell_crossing_the_wall_is_its_own_body():
    from stl2prism.mesh_prep import nest_shells
    bodies = _shells(_tube_block(), _crossing_sphere())
    assert len(bodies) == 2 and all(not b.voids for b in bodies)
    (i, j, reach), = nest_shells.last_crossings
    assert reach == pytest.approx(3.0, abs=0.05)


def test_shell_touching_the_wall_is_still_a_cavity():
    bodies = _shells(_tbox(40), _tbox(20, (10, 0, 0)))      # one face on the wall
    assert len(bodies) == 1 and len(bodies[0].voids) == 1


def test_shell_inside_is_still_a_cavity():
    from stl2prism.mesh_prep import nest_shells
    bodies = _shells(_tbox(40), _tbox(20))
    assert len(bodies) == 1 and len(bodies[0].voids) == 1
    assert nest_shells.last_crossings == []


def test_pipeline_keeps_a_crossing_shell_as_a_solid(tmp_path, capsys):
    stl = str(tmp_path / 'cross.stl')
    trimesh.util.concatenate([_tube_block(), _crossing_sphere()]).export(stl)
    r, out = _run(stl, tmp_path, 'cross', verbose=True)
    assert r['n_bodies'] == 2 and all(b['voids'] == 0 for b in r['bodies'])
    assert synth.reimport(out)['solids'] == 2
    assert 'crosses its surface by 3.0 mm: not a cavity' in capsys.readouterr().out


def test_add_cavities_asks_the_oracle():
    asked = []

    def clear(k, pts):
        asked.append(k)
        return [False] * len(pts)
    shape, why = add_cavities(_box(40), [_box(20)], clear=clear)
    assert shape is None and 'not strictly inside' in why and asked == [None]


def test_mesh_oracle_uses_deviation_as_clearance():
    body = Body(_tbox(40), [_tbox(20)])
    clear = pipeline._mesh_clearance(body, {'dev_max': 0.5}, [{'dev_max': 0.25}])
    pts = [(0, 0, 0), (19.0, 0, 0), (19.5, 0, 0), (25, 0, 0)]
    assert clear(None, pts) == [True, True, False, False]           # 1.0 mm from the wall
    assert clear(0, [(0, 0, 0), (10.5, 0, 0), (12, 0, 0)]) == [False, False, True]


def test_mesh_oracle_is_none_for_open_meshes():
    open_box = _tbox(40)
    open_box.update_faces(np.arange(len(open_box.faces)) > 0)
    assert pipeline._mesh_clearance(Body(open_box, [_tbox(20)]), {}, [{}]) is None
