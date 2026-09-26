"""Blueprint expressions: the safe evaluator and the Fusion rewriter."""
import pytest

from stl_to_solid.blueprint.expr import ExprError, check_name, evaluate, names, to_fusion

P = {'body_w': 22.5, 'tab_z': 15.9, 'a': 2.0, 'b': 3.0}


def test_evaluate_numbers_and_arithmetic():
    assert evaluate(2.5, P) == 2.5
    assert evaluate(3, P) == 3.0
    assert evaluate('body_w/2', P) == 11.25
    assert evaluate('-(a+b)*2', P) == -10.0
    assert evaluate('(a + b) / 2 - 1', P) == 1.5
    assert evaluate('+a', P) == 2.0


@pytest.mark.parametrize('bad', [
    'nope', '__import__("os")', 'a**2', 'a.b', 'a[0]', '1 if a else 2', 'lambda: 0',
    'a % b', 'a // b', 'True', '"x"', 'max(a, b)', '', 'a +', 'x' * 300, '1/0', 'a/(b-3)',
])
def test_evaluate_rejects(bad):
    with pytest.raises(ExprError):
        evaluate(bad, P)


def test_evaluate_rejects_non_numbers():
    with pytest.raises(ExprError):
        evaluate(True, P)
    with pytest.raises(ExprError):
        evaluate(None, P)
    with pytest.raises(ExprError):
        evaluate(float('inf'), P)


def test_names():
    assert names('body_w/2 + tab_z') == {'body_w', 'tab_z'}
    assert names(3.0) == set()


def test_to_fusion():
    assert to_fusion(2.5) == '2.5 mm'
    assert to_fusion(0) == '0 mm'
    assert to_fusion('body_w') == 'body_w'
    assert to_fusion('body_w/2') == 'body_w / 2'
    assert to_fusion('tab_z + 1.5') == 'tab_z + 1.5 mm'
    assert to_fusion('-tab_len/2') == '-(tab_len) / 2'
    assert to_fusion('body_w + 2*tab_len') == 'body_w + 2 * tab_len'
    assert to_fusion('(a + b) * 2') == '(a + b) * 2'
    assert to_fusion('2*3') == '6 mm'            # no names: folded


def test_check_name():
    check_name('body_w')
    for bad in ('pi', 'sin', 'for', '2x', 'a-b', '', 'x' * 65):
        with pytest.raises(ExprError):
            check_name(bad)
