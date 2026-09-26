"""Numbers in a recipe may be expressions over the named parameters
("body_w/2", "tab_z + tab_t"). This evaluates them safely (a whitelist
over the ast: + - * /, unary sign, parentheses, numbers and parameter
names; nothing else) and rewrites them as Fusion 360 expressions.
"""
import ast
import keyword
import math
import operator
import re

MAX_LEN = 200
NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$')
# Fusion's own functions and constants: a parameter with one of these
# names would shadow them in the Parameters table
RESERVED = {'pi', 'e', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'abs', 'min', 'max',
            'floor', 'ceil', 'round', 'sqrt', 'ln', 'log', 'exp', 'pow', 'sign'}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_SYM = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}


class ExprError(ValueError):
    """An expression that cannot be used: names what is wrong with it."""


def is_expr(value):
    return isinstance(value, str)


def check_name(name):
    """A parameter name Fusion accepts and nothing in the evaluator confuses."""
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ExprError(f'{name!r} is not a valid parameter name (letters, digits and _, '
                        'starting with a letter, at most 64 characters)')
    if keyword.iskeyword(name) or name in RESERVED:
        raise ExprError(f'{name!r} cannot be a parameter name (it is a keyword or a Fusion '
                        'built-in)')


def _parse(text):
    if len(text) > MAX_LEN:
        raise ExprError(f'expression longer than {MAX_LEN} characters')
    if not text.strip():
        raise ExprError('empty expression')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as e:
        raise ExprError(f'cannot parse {text!r}: {e.msg}') from None
    _check(tree.body, text)
    return tree.body


def _check(node, text):
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN:
            raise ExprError(f'operator {type(node.op).__name__} is not allowed in {text!r} '
                            '(only + - * /)')
        _check(node.left, text)
        _check(node.right, text)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UN:
            raise ExprError(f'operator {type(node.op).__name__} is not allowed in {text!r}')
        _check(node.operand, text)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExprError(f'{node.value!r} is not a number in {text!r}')
    elif isinstance(node, ast.Name):
        pass
    else:
        raise ExprError(f'{type(node).__name__} is not allowed in {text!r} (only numbers, '
                        'parameter names, + - * / and parentheses)')


def _eval(node, params):
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, params), _eval(node.right, params)
        try:
            return _BIN[type(node.op)](a, b)
        except ZeroDivisionError:
            raise ExprError('division by zero') from None
    if isinstance(node, ast.UnaryOp):
        return _UN[type(node.op)](_eval(node.operand, params))
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in params:
            raise ExprError(f'unknown parameter {node.id!r}')
        return float(params[node.id])
    raise ExprError(f'{type(node).__name__} is not allowed')


def evaluate(value, params):
    """A number, or an expression string over `params` ({name: mm}), as a
    finite float. Raises ExprError."""
    if isinstance(value, bool):
        raise ExprError(f'{value!r} is not a number')
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        out = _eval(_parse(value), params)
    else:
        raise ExprError(f'{value!r} is not a number or an expression')
    if not math.isfinite(out):
        raise ExprError(f'{value!r} does not evaluate to a finite number')
    return out


def names(value):
    """The parameter names an expression refers to (empty for a number)."""
    if not isinstance(value, str):
        return set()
    return {n.id for n in ast.walk(_parse(value)) if isinstance(n, ast.Name)}


def _fmt(v):
    """A number the way Fusion likes it: no exponent, no trailing zeros."""
    s = f'{float(v):.6f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-', '-0') else '0'


def _render(node, unit):
    """Fusion text for `node`. `unit`: this value stands for a length, so
    a bare literal gets ' mm' (a bare number in Fusion is read in the
    document's unit, which may be inches). Inside * and / a literal is a
    dimensionless factor and stays bare."""
    if isinstance(node, ast.Constant):
        return _fmt(node.value) + (' mm' if unit else '')
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.UnaryOp):
        inner = _render(node.operand, unit)
        return ('-(%s)' if isinstance(node.op, ast.USub) else '(%s)') % inner
    op = type(node.op)
    if op in (ast.Add, ast.Sub):
        return f'{_render(node.left, True)} {_SYM[op]} {_render(node.right, True)}'
    # * and /: one side a length, the other a factor. A literal is the
    # factor; a name is a length; both names is mm*mm and Fusion will say so.
    left = _render(node.left, False if isinstance(node.left, ast.Constant) else unit)
    right = _render(node.right, False if isinstance(node.right, ast.Constant) else unit)
    if isinstance(node.left, ast.BinOp) and type(node.left.op) in (ast.Add, ast.Sub):
        left = f'({left})'
    if isinstance(node.right, ast.BinOp):
        right = f'({right})'
    return f'{left} {_SYM[op]} {right}'


def to_fusion(value, params=None):
    """A Fusion 360 expression (mm) for a recipe value: 2.5 -> '2.5 mm',
    'body_w' -> 'body_w', 'body_w/2' -> 'body_w / 2', 'tab_z + 1.5' ->
    'tab_z + 1.5 mm'. An expression without parameter names is folded to
    a number."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _fmt(value) + ' mm'
    node = _parse(value)
    if not names(value):
        return _fmt(_eval(node, {})) + ' mm'
    return _render(node, True)
