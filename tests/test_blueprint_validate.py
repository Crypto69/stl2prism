"""Blueprint recipe schema and validator on the hand-written SG90 recipe
and one-change mutations of it."""
import copy
import json
import os

import pytest

from stl_to_solid.blueprint.schema import SCHEMA, normalize, param_map
from stl_to_solid.blueprint.validate import validate, shape_area

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'sg90_recipe.json')


@pytest.fixture
def sg90():
    with open(FIX) as f:
        return json.load(f)


def _feature(r, fid):
    return next(f for f in r['features'] if f['id'] == fid)


def test_schema_is_strict_everywhere():
    """Structured-output modes refuse an object without additionalProperties:
    false or with an optional key; a later edit must not break that."""
    def walk(node, path='$'):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node.get('additionalProperties') is False, path
                assert sorted(node['required']) == sorted(node['properties']), path
            for k, v in node.items():
                walk(v, f'{path}.{k}')
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f'{path}[{i}]')
    walk(SCHEMA)


def test_fixture_validates_and_boxes_to_the_drawing(sg90):
    rep = validate(sg90)
    assert rep.errors == []
    assert rep.bbox['size'] == pytest.approx([32.2, 11.8, 29.9], abs=1e-6)
    assert rep.bbox['min'] == pytest.approx([-4.85, 0, 0], abs=1e-6)
    # inferred features are pointed out, nothing else
    assert any('deduced' in w for w in rep.warnings)
    assert not any('overlap' in w for w in rep.warnings)
    res = rep.resolved
    assert res['features'][0]['shapes'][0]['center'] == {'u': 11.25, 'v': 5.9}
    assert shape_area(res['features'][0]['shapes'][0]) == pytest.approx(22.5 * 11.8)


def test_normalize_is_idempotent_and_fills_defaults(sg90):
    n1 = normalize(sg90)
    assert normalize(n1) == n1
    assert param_map(n1)['body_w'] == 22.5
    loose = {'features': [{'plane': 'XY', 'shapes': [{'kind': 'rect', 'center': {'u': 0, 'v': 0}, 'w': 1, 'h': 1}],
                           'op': 'new_body', 'distance': 2}]}
    n = normalize(loose)
    f = n['features'][0]
    assert f['id'] == 'f1' and f['offset'] == 0 and f['direction'] == '+' and f['through_all'] is False
    assert f['shapes'][0]['corner_radius'] == 0 and f['confidence'] == 1.0
    assert n['params'] == [] and n['overall'] == {'w': None, 'd': None, 'h': None}
    # a params dict (older shape) becomes the list
    assert normalize({'params': {'a': 1}})['params'] == [{'name': 'a', 'value': 1, 'source': '', 'inferred': False}]


def _errors(recipe):
    return validate(recipe).errors


def test_first_feature_must_be_a_body(sg90):
    _feature(sg90, 'body')['op'] = 'join'
    assert any('first feature must be a new_body' in e for e in _errors(sg90))


def test_zero_size_shape(sg90):
    _feature(sg90, 'body')['shapes'][0]['w'] = 0
    assert any('body.shapes[0]' in e and 'w and h > 0' in e for e in _errors(sg90))


def test_cut_that_removes_nothing(sg90):
    _feature(sg90, 'screw')['offset'] = 100
    assert any('screw' in e and 'removes nothing' in e for e in _errors(sg90))


def test_nested_shapes_in_one_sketch(sg90):
    _feature(sg90, 'body')['shapes'].append({'kind': 'circle', 'center': {'u': 5, 'v': 5}, 'd': 2})
    assert any('nested' in e for e in _errors(sg90))


def test_duplicate_feature(sg90):
    sg90['features'].append(copy.deepcopy(_feature(sg90, 'tabs')))
    assert any('duplicate of tabs' in e for e in _errors(sg90))


def test_overall_mismatch_is_an_error_and_names_the_axis(sg90):
    sg90['overall']['w'] = 35
    errs = _errors(sg90)
    assert any('along X' in e and '32.20' in e and '35.00' in e for e in errs)
    # two axes off: hint at a swapped view
    sg90['overall']['h'] = 40
    assert any('wrong plane' in e for e in _errors(sg90))


def test_overall_small_deviation_is_a_warning(sg90):
    sg90['overall']['w'] = 32.8           # about 1.8 %
    rep = validate(sg90)
    assert rep.errors == []
    assert any('along X' in w for w in rep.warnings)


def test_through_all_on_a_join(sg90):
    _feature(sg90, 'boss')['through_all'] = True
    assert any('through_all only makes sense for a cut' in e for e in _errors(sg90))


def test_bad_expression_names_the_field(sg90):
    _feature(sg90, 'tabs')['shapes'][1]['center']['u'] = 'body_w + nope'
    errs = _errors(sg90)
    assert any(e.startswith('tabs.shapes[1].center.u') and 'nope' in e for e in errs)


def test_reserved_parameter_name(sg90):
    sg90['params'].append({'name': 'sin', 'value': 1, 'source': '', 'inferred': False})
    assert any("'sin'" in e for e in _errors(sg90))


def test_bad_enums_and_missing_shapes():
    rep = validate({'params': [{'name': 'w', 'value': -1}],
                    'features': [{'op': 'melt', 'plane': 'QQ', 'shapes': [], 'extrude': 'nope'}]})
    text = ' '.join(rep.errors)
    assert 'plane must be' in text and 'op must be' in text and 'no shapes' in text


def test_loose_join_and_sticking_out_hole_are_warnings(sg90):
    _feature(sg90, 'hub')['offset'] = 'body_h + boss_h + 5'
    rep = validate(sg90)
    assert any('hub' in w and 'touches no existing material' in w for w in rep.warnings)
    with open(FIX) as f:
        fresh = json.load(f)
    _feature(fresh, 'screw')['shapes'][0]['center']['u'] = 'boss_x + 2.5'
    rep = validate(fresh)
    assert any('screw' in w and 'sticks out' in w for w in rep.warnings)
