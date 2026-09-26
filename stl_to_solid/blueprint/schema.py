"""The recipe's JSON Schema, in the shape the LLM structured-output
modes accept (every object closed with additionalProperties: false and
every key required; optional values as anyOf [T, null]; no numeric range
keywords, the validator owns those; no free-keyed maps, so the parameters
are a list), plus normalize(), which fills defaults and tidies a recipe
that came from a model or from the UI's table.
"""
import copy
import re

from .expr import check_name, ExprError

# Every value is text: a number ("22.5") or an expression over the
# parameters ("body_w/2"). Plain numbers are accepted by the evaluator as
# well; the model is asked for strings because the structured-output
# compilers cap the number of union-typed fields (Anthropic: 16) and a
# recipe has more numeric fields than that.
NUM = {'type': 'string',
       'description': 'a length in mm as text: a number ("22.5") or an expression over the params ("body_w/2")'}
NUM_OR_NULL = {'type': 'string',
               'description': 'as NUM; "0" for none'}

PLANES = ('XY', 'XZ', 'YZ')
OPS = ('new_body', 'join', 'cut')
DIRECTIONS = ('+', '-', 'symmetric')
VIEWS = ('front', 'top', 'right', 'left', 'back', 'bottom', 'isometric', 'section', 'other',
         'none')
SHAPE_KINDS = ('rect', 'circle', 'slot', 'polygon')


def _obj(props, description=None):
    out = {'type': 'object', 'additionalProperties': False,
           'required': list(props), 'properties': props}
    if description:
        out['description'] = description
    return out


def _enum(values, description=None):
    out = {'type': 'string', 'enum': list(values)}
    if description:
        out['description'] = description
    return out


PT = _obj({'u': NUM, 'v': NUM},
          'a point on the sketch plane, mm: (u, v) are the plane\'s two world axes in '
          'alphabetical order (XY: x,y; XZ: x,z; YZ: y,z)')

RECT = _obj({'kind': {'const': 'rect'}, 'center': PT, 'w': NUM, 'h': NUM,
             'corner_radius': NUM_OR_NULL},
            'axis-aligned rectangle: w along u, h along v; corner_radius rounds the corners')
CIRCLE = _obj({'kind': {'const': 'circle'}, 'center': PT, 'd': NUM}, 'circle by diameter')
SLOT = _obj({'kind': {'const': 'slot'}, 'p1': PT, 'p2': PT, 'width': NUM},
            'a slot: the line p1-p2 thickened to `width` with round ends')
POLYGON = _obj({'kind': {'const': 'polygon'}, 'points': {'type': 'array', 'items': PT}},
               'closed polygon (3 or more points, in order, not self-crossing)')
SHAPE = {'anyOf': [RECT, CIRCLE, SLOT, POLYGON]}

SOURCE = _obj({
    'view': _enum(VIEWS, 'which view of the drawing this feature was read from'),
    'labels': {'type': 'array', 'items': {'type': 'string'},
               'description': 'the dimension texts used, as printed (e.g. "22.5mm", "DIA 2.0mm")'},
    'inferred': {'type': 'boolean',
                 'description': 'true when any number of this feature was deduced rather than read from a label'},
})

FEATURE = _obj({
    'id': {'type': 'string', 'description': 'short unique id, e.g. "body", "tab_holes"'},
    'name': {'type': 'string', 'description': 'label for the UI and the Fusion sketch'},
    'plane': _enum(PLANES, 'the sketch plane: XY = top view, XZ = front view, YZ = right-side view'),
    'offset': NUM,
    'shapes': {'type': 'array', 'items': SHAPE},
    'op': _enum(OPS, 'new_body starts a body, join adds material to it, cut removes'),
    'direction': _enum(DIRECTIONS, "'+' along the plane's positive normal (XY:+Z, XZ:+Y, YZ:+X), '-' the other way, symmetric both ways"),
    'distance': NUM,
    'through_all': {'type': 'boolean', 'description': 'cut through the whole part (distance ignored)'},
    'source': SOURCE,
    'confidence': {'type': 'number', 'description': '0..1: how sure the reading is'},
})

PARAM = _obj({
    'name': {'type': 'string', 'description': 'snake_case, e.g. body_w, tab_z, hole_d'},
    'value': {'type': 'number', 'description': 'mm'},
    'source': {'type': 'string', 'description': 'where the number came from: the view and label, or how it was deduced'},
    'inferred': {'type': 'boolean', 'description': 'true when not read from a label'},
})

SCHEMA = _obj({
    'name': {'type': 'string', 'description': 'what the part is'},
    'units': _enum(('mm',)),
    'overall': _obj({'w': NUM, 'd': NUM, 'h': NUM},
                    'bounding box of the whole part including tabs and bosses: width (X), depth (Y), height (Z)'),
    'params': {'type': 'array', 'items': PARAM},
    'features': {'type': 'array', 'items': FEATURE},
    'views_found': {'type': 'array', 'items': _enum(VIEWS)},
    'notes': {'type': 'array', 'items': {'type': 'string'}},
})

_ID_RE = re.compile(r'[^A-Za-z0-9_]+')


def _num(v, default):
    return default if v is None else v


def normalize(recipe):
    """A deep copy with every optional field filled, parameters as a list
    of {name, value, source, inferred}, confidence clamped to [0, 1], a
    missing corner radius as 0, ids present and unique. Idempotent: the
    UI's table round-trips this form. Structural mistakes are left for
    validate() to name."""
    r = copy.deepcopy(recipe) if isinstance(recipe, dict) else {}
    out = {'name': str(r.get('name') or 'part'), 'units': r.get('units', 'mm')}
    ov = r.get('overall') if isinstance(r.get('overall'), dict) else {}
    out['overall'] = {k: ov.get(k) for k in ('w', 'd', 'h')}
    params = r.get('params')
    if isinstance(params, dict):
        params = [{'name': k, 'value': v} if not isinstance(v, dict) else {'name': k, **v}
                  for k, v in params.items()]
    plist = []
    for p in params if isinstance(params, list) else []:
        if not isinstance(p, dict):
            continue
        plist.append({'name': str(p.get('name', '')), 'value': p.get('value'),
                      'source': str(p.get('source') or ''), 'inferred': bool(p.get('inferred', False))})
    out['params'] = plist
    feats = []
    seen = set()
    for i, f in enumerate(r.get('features') if isinstance(r.get('features'), list) else []):
        if not isinstance(f, dict):
            continue
        fid = _ID_RE.sub('_', str(f.get('id') or '')).strip('_') or f'f{i + 1}'
        base, k = fid, 2
        while fid in seen:
            fid = f'{base}_{k}'
            k += 1
        seen.add(fid)
        shapes = []
        for s in f.get('shapes') if isinstance(f.get('shapes'), list) else []:
            if not isinstance(s, dict):
                continue
            s = dict(s)
            if s.get('kind') == 'rect':
                s['corner_radius'] = _num(s.get('corner_radius'), 0)
            shapes.append(s)
        src = f.get('source') if isinstance(f.get('source'), dict) else {}
        conf = f.get('confidence')
        try:
            conf = min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            conf = 1.0
        feats.append({
            'id': fid, 'name': str(f.get('name') or fid.replace('_', ' ')),
            'plane': f.get('plane'), 'offset': _num(f.get('offset'), 0),
            'shapes': shapes, 'op': f.get('op'),
            'direction': _num(f.get('direction'), '+'),
            'distance': _num(f.get('distance'), 0),
            'through_all': bool(f.get('through_all', False)),
            'source': {'view': src.get('view', 'none'),
                       'labels': [str(x) for x in src.get('labels', []) or []],
                       'inferred': bool(src.get('inferred', False))},
            'confidence': conf,
        })
    out['features'] = feats
    out['views_found'] = [str(v) for v in r.get('views_found', []) or []]
    out['notes'] = [str(n) for n in r.get('notes', []) or []]
    return out


def param_map(recipe):
    """{name: value} from a normalized recipe's parameter list. Names are
    not checked here (validate() does that); a duplicate keeps its last
    value."""
    out = {}
    for p in recipe.get('params', []):
        out[p['name']] = p['value']
    return out


def check_param_names(recipe):
    """ExprError for the first bad parameter name, else None."""
    for p in recipe.get('params', []):
        check_name(p['name'])
