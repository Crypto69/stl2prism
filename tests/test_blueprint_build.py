"""Blueprint CadQuery compiler: plane orientation per direction, the shape
kinds, through-all cuts, and compile_recipe's files on the SG90 fixture."""
import json
import math
import os

import pytest

from stl_to_solid.blueprint.build_cq import build, BlueprintError
from stl_to_solid.blueprint.validate import validate

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'sg90_recipe.json')


def _recipe(features, params=None, overall=None):
    return {'name': 't', 'units': 'mm', 'overall': overall or {'w': 0, 'd': 0, 'h': 0},
            'params': params or [], 'features': features}


def _feat(fid, plane, shapes, op='new_body', offset=0, direction='+', distance=2, through_all=False):
    return {'id': fid, 'plane': plane, 'shapes': shapes, 'op': op, 'offset': offset,
            'direction': direction, 'distance': distance, 'through_all': through_all}


def _rect(cu, cv, w, h):
    return {'kind': 'rect', 'center': {'u': cu, 'v': cv}, 'w': w, 'h': h}


def _build(recipe):
    rep = validate(recipe)
    assert rep.errors == [], rep.errors
    return build(rep.resolved)


@pytest.mark.parametrize('plane,direction,offset,expect', [
    ('XY', '+', 3, ([-2, -3, 3], [2, 3, 5])),
    ('XY', '-', 3, ([-2, -3, 1], [2, 3, 3])),
    ('XZ', '+', 3, ([-2, 3, -3], [2, 5, 3])),        # u=x, v=z, normal +Y
    ('XZ', '-', 3, ([-2, 1, -3], [2, 3, 3])),
    ('YZ', '+', 22.5, ([22.5, -2, -3], [24.5, 2, 3])),  # u=y, v=z, normal +X
    ('YZ', '-', 22.5, ([20.5, -2, -3], [22.5, 2, 3])),
    ('XY', 'symmetric', 3, ([-2, -3, 2], [2, 3, 4])),
])
def test_plane_and_direction(plane, direction, offset, expect):
    b = _build(_recipe([_feat('a', plane, [_rect(0, 0, 4, 6)], offset=offset, direction=direction)]))
    assert b.bbox['min'] == pytest.approx(expect[0], abs=1e-6)
    assert b.bbox['max'] == pytest.approx(expect[1], abs=1e-6)
    assert b.n_solids == 1 and b.volume_mm3 == pytest.approx(48, rel=1e-6)


def test_shapes_and_cuts():
    feats = [
        _feat('body', 'XY', [_rect(10, 10, 20, 20)], distance=10),
        _feat('hole', 'XY', [{'kind': 'circle', 'center': {'u': 10, 'v': 10}, 'd': 2}], op='cut',
              distance=0, through_all=True),
        _feat('slot', 'XY', [{'kind': 'slot', 'p1': {'u': 3, 'v': 3}, 'p2': {'u': 8, 'v': 3}, 'width': 2}],
              op='cut', offset=10, direction='-', distance=1),
        _feat('tri', 'XY', [{'kind': 'polygon', 'points': [{'u': 20, 'v': 0}, {'u': 24, 'v': 0}, {'u': 20, 'v': 3}]}],
              op='join', distance=10),
        _feat('rr', 'XY', [dict(_rect(10, 10, 4, 4), corner_radius=1)], op='join', offset=10, distance=2),
    ]
    b = _build(_recipe(feats))
    v = b.per_feature
    assert v[0]['volume_delta_mm3'] == pytest.approx(4000, rel=1e-6)
    assert v[1]['volume_delta_mm3'] == pytest.approx(-math.pi * 1 * 1 * 10, rel=1e-4)
    # slot: 5 long centre to centre + round ends of diameter 2, 1 deep
    assert v[2]['volume_delta_mm3'] == pytest.approx(-(5 * 2 + math.pi) * 1, rel=1e-4)
    assert v[3]['volume_delta_mm3'] == pytest.approx(0.5 * 4 * 3 * 10, rel=1e-6)
    # rounded 4x4 square, r=1: 16 - (4 - pi)
    assert v[4]['volume_delta_mm3'] == pytest.approx((16 - (4 - math.pi)) * 2, rel=1e-4)
    assert b.n_solids == 1
    assert b.bbox['max'] == pytest.approx([24, 20, 12], abs=1e-6)


def test_two_disjoint_bodies_warn():
    b = _build(_recipe([_feat('a', 'XY', [_rect(0, 0, 2, 2)]),
                        _feat('b', 'XY', [_rect(10, 0, 2, 2)], op='new_body')]))
    assert b.n_solids == 2
    assert not b.warnings                  # two new_body features, two bodies: as asked
    b = _build(_recipe([_feat('a', 'XY', [_rect(0, 0, 2, 2)]),
                        _feat('b', 'XY', [_rect(10, 0, 2, 2)], op='join', offset=0)]))
    assert b.n_solids == 2 and any('2 separate bodies' in w for w in b.warnings)


def test_sg90_fixture_builds_to_one_solid():
    with open(FIX) as f:
        rec = json.load(f)
    b = _build(rec)
    assert b.n_solids == 1
    assert b.bbox['min'] == pytest.approx([-4.85, 0, 0], abs=1e-3)
    assert b.bbox['max'] == pytest.approx([27.35, 11.8, 29.9], abs=1e-3)
    assert 6700 < b.volume_mm3 < 7050
    assert [p['op'] for p in b.per_feature] == ['new_body', 'join', 'cut', 'join', 'join', 'join', 'cut', 'cut']


def test_compile_recipe_writes_everything(tmp_path):
    from stl_to_solid.blueprint.compile import compile_recipe
    from backend.analysis import step_stats
    import ast
    import trimesh
    with open(FIX) as f:
        rec = json.load(f)
    r = compile_recipe(rec, str(tmp_path), title='SG90')
    for n in ('output.step', 'preview.stl', 'output_fusion.py', 'recipe.json'):
        assert (tmp_path / n).exists(), n
    assert r['mode'] == 'blueprint' and r['solids'] == 1 and r['has_fusion_script']
    assert r['bbox']['size'] == pytest.approx([32.2, 11.8, 29.9], abs=1e-3)
    assert r['overall_check']['dev_pct'] == pytest.approx([0, 0, 0], abs=0.05)
    assert step_stats(str(tmp_path / 'output.step'))['solids'] == 1
    m = trimesh.load(str(tmp_path / 'preview.stl'))
    assert m.is_watertight and abs(m.volume - r['volume_mm3']) / r['volume_mm3'] < 0.02
    ast.parse((tmp_path / 'output_fusion.py').read_text())
    rj = json.loads((tmp_path / 'recipe.json').read_text())
    assert rj['params'][0]['name'] == 'body_w'


def test_compile_refuses_a_bad_recipe(tmp_path):
    from stl_to_solid.blueprint.compile import compile_recipe
    bad = _recipe([_feat('a', 'XY', [_rect(0, 0, 0, 6)])])
    with pytest.raises(BlueprintError) as e:
        compile_recipe(bad, str(tmp_path))
    assert 'not buildable' in str(e.value) and e.value.report is not None
    assert not (tmp_path / 'output.step').exists()
