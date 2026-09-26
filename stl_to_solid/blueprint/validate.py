"""Deterministic checks on a recipe before anything is built: structure,
parameter names, every expression resolvable, shapes that make sense, a
feature order a solid can follow (a body first, cuts that remove
something, joins that touch something), and a bounding-box simulation
against the drawing's stated overall size. Errors stop the build;
warnings are shown next to the result.

Also the 2-D helpers both compilers share: a shape as a shapely polygon,
its area, its bounds, and which world axes a plane's (u, v, normal) are.
"""
import math
from dataclasses import dataclass, field

from shapely.geometry import LineString, Point, Polygon, box

from .expr import ExprError, check_name, evaluate
from .schema import DIRECTIONS, OPS, PLANES, SHAPE_KINDS, normalize, param_map

# world axis index of (u, v, normal) for each plane
PLANE_AXES = {'XY': (0, 1, 2), 'XZ': (0, 2, 1), 'YZ': (1, 2, 0)}
AXIS_NAMES = 'XYZ'

OVERALL_ERR = 0.03      # beyond +-3 %: error
OVERALL_WARN = 0.015    # beyond +-1.5 %: warning (the SG90 drawing's own tolerance)
MAX_PARAMS = 64
TOUCH = 0.01            # mm: boxes closer than this touch
THROUGH_MARGIN = 1.0    # mm past the material on each side for a through-all cut


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    # {'min': [x,y,z], 'max': [x,y,z], 'size': [w,d,h]} from the box simulation
    bbox: dict = None
    # the normalized recipe with every number evaluated to a float
    resolved: dict = None

    @property
    def ok(self):
        return not self.errors


# --- 2-D shape helpers -----------------------------------------------------------

def shape_polygon(shape):
    """A resolved (all-float) shape as a shapely polygon in (u, v) mm."""
    k = shape['kind']
    if k == 'rect':
        cu, cv = shape['center']['u'], shape['center']['v']
        w, h = shape['w'], shape['h']
        poly = box(cu - w / 2, cv - h / 2, cu + w / 2, cv + h / 2)
        r = shape.get('corner_radius') or 0
        if r > 0:
            poly = box(cu - w / 2 + r, cv - h / 2 + r, cu + w / 2 - r, cv + h / 2 - r).buffer(r, 64)
        return poly
    if k == 'circle':
        return Point(shape['center']['u'], shape['center']['v']).buffer(shape['d'] / 2, 128)
    if k == 'slot':
        a, b = shape['p1'], shape['p2']
        return LineString([(a['u'], a['v']), (b['u'], b['v'])]).buffer(shape['width'] / 2, 64)
    if k == 'polygon':
        return Polygon([(p['u'], p['v']) for p in shape['points']])
    raise ValueError(f'unknown shape kind {k!r}')


def shape_area(shape):
    return float(shape_polygon(shape).area)


def shape_bounds(shape):
    """(umin, vmin, umax, vmax)."""
    return tuple(float(x) for x in shape_polygon(shape).bounds)


# --- the checks ---------------------------------------------------------------------

def _resolve(recipe, params, report):
    """The recipe with every NUM field evaluated. Errors name the field."""
    out = {'name': recipe['name'], 'units': recipe['units'], 'params': params,
           'views_found': recipe['views_found'], 'notes': recipe['notes']}

    def num(value, path, allow_none=False):
        if value is None and allow_none:
            return None
        try:
            return evaluate(value, params)
        except ExprError as e:
            report.errors.append(f'{path}: {e}')
            return None

    out['overall'] = {k: num(recipe['overall'].get(k), f'overall.{k}') for k in ('w', 'd', 'h')}
    feats = []
    for f in recipe['features']:
        fid = f['id']
        g = {'id': fid, 'name': f['name'], 'plane': f['plane'], 'op': f['op'],
             'direction': f['direction'], 'through_all': f['through_all'],
             'source': f['source'], 'confidence': f['confidence'],
             'offset': num(f['offset'], f'{fid}.offset'),
             'distance': num(f['distance'], f'{fid}.distance'), 'shapes': []}
        for j, s in enumerate(f['shapes']):
            p = f'{fid}.shapes[{j}]'
            k = s.get('kind')
            if k not in SHAPE_KINDS:
                report.errors.append(f'{p}: unknown shape kind {k!r} (rect, circle, slot or polygon)')
                continue
            t = {'kind': k}

            def pt(v, name):
                if not isinstance(v, dict):
                    report.errors.append(f'{p}.{name}: a point needs u and v')
                    return {'u': 0.0, 'v': 0.0}
                return {'u': num(v.get('u'), f'{p}.{name}.u'), 'v': num(v.get('v'), f'{p}.{name}.v')}

            if k == 'rect':
                t['center'] = pt(s.get('center'), 'center')
                t['w'], t['h'] = num(s.get('w'), f'{p}.w'), num(s.get('h'), f'{p}.h')
                t['corner_radius'] = num(s.get('corner_radius', 0), f'{p}.corner_radius')
            elif k == 'circle':
                t['center'] = pt(s.get('center'), 'center')
                t['d'] = num(s.get('d'), f'{p}.d')
            elif k == 'slot':
                t['p1'], t['p2'] = pt(s.get('p1'), 'p1'), pt(s.get('p2'), 'p2')
                t['width'] = num(s.get('width'), f'{p}.width')
            else:
                pts = s.get('points') if isinstance(s.get('points'), list) else []
                t['points'] = [pt(q, f'points[{i}]') for i, q in enumerate(pts)]
            g['shapes'].append(t)
        feats.append(g)
    out['features'] = feats
    return out


def _has_none(x):
    if x is None:
        return True
    if isinstance(x, dict):
        return any(_has_none(v) for v in x.values())
    if isinstance(x, list):
        return any(_has_none(v) for v in x)
    return False


def _check_shapes(f, report):
    fid = f['id']
    for j, s in enumerate(f['shapes']):
        p = f'{fid}.shapes[{j}]'
        k = s['kind']
        if k == 'rect':
            if s['w'] <= 0 or s['h'] <= 0:
                report.errors.append(f'{p}: a rect needs w and h > 0 (got {s["w"]:g} x {s["h"]:g})')
            elif s['corner_radius'] < 0 or s['corner_radius'] > min(s['w'], s['h']) / 2 + 1e-9:
                report.errors.append(f'{p}: corner_radius must be between 0 and half the shorter side')
        elif k == 'circle':
            if s['d'] <= 0:
                report.errors.append(f'{p}: a circle needs d > 0')
        elif k == 'slot':
            if s['width'] <= 0:
                report.errors.append(f'{p}: a slot needs width > 0')
            if math.hypot(s['p2']['u'] - s['p1']['u'], s['p2']['v'] - s['p1']['v']) < 1e-9:
                report.errors.append(f'{p}: a slot needs two different end points (use a circle for a round hole)')
        else:
            if len(s['points']) < 3:
                report.errors.append(f'{p}: a polygon needs at least 3 points')
            else:
                poly = Polygon([(q['u'], q['v']) for q in s['points']])
                if not poly.is_valid or poly.area <= 1e-9:
                    report.errors.append(f'{p}: the polygon crosses itself or has no area')


def _n_range(f, offset, distance, material_lo, material_hi):
    """[lo, hi] along the normal for this feature's extrude."""
    if f['through_all']:
        return material_lo - THROUGH_MARGIN, material_hi + THROUGH_MARGIN
    if f['direction'] == '+':
        return offset, offset + distance
    if f['direction'] == '-':
        return offset - distance, offset
    return offset - distance / 2, offset + distance / 2


def _box_of(f, shape_bounds_uv, nlo, nhi):
    u, v, n = PLANE_AXES[f['plane']]
    lo, hi = [0.0] * 3, [0.0] * 3
    lo[u], hi[u] = shape_bounds_uv[0], shape_bounds_uv[2]
    lo[v], hi[v] = shape_bounds_uv[1], shape_bounds_uv[3]
    lo[n], hi[n] = nlo, nhi
    return lo, hi


def _overlap(a, b, tol=TOUCH):
    return all(a[0][i] <= b[1][i] + tol and b[0][i] <= a[1][i] + tol for i in range(3))


def _contains_uv(host, fb, plane, tol=TOUCH):
    u, v, _ = PLANE_AXES[plane]
    return (host[0][u] - tol <= fb[0][u] and fb[1][u] <= host[1][u] + tol
            and host[0][v] - tol <= fb[0][v] and fb[1][v] <= host[1][v] + tol)


def validate(recipe):
    """Report for `recipe` (raw or normalized)."""
    report = Report()
    r = normalize(recipe)

    # --- structure -----------------------------------------------------------
    if r['units'] != 'mm':
        report.errors.append(f"units must be 'mm' (got {r['units']!r})")
    if not r['features']:
        report.errors.append('the recipe has no features')
    if len(r['params']) > MAX_PARAMS:
        report.errors.append(f'{len(r["params"])} parameters is more than {MAX_PARAMS}')
    seen = set()
    for p in r['params']:
        try:
            check_name(p['name'])
        except ExprError as e:
            report.errors.append(str(e))
            continue
        if p['name'] in seen:
            report.errors.append(f'parameter {p["name"]!r} is defined twice')
        seen.add(p['name'])
        v = p['value']
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            report.errors.append(f'parameter {p["name"]!r} needs a plain number (got {v!r})')
    for f in r['features']:
        if f['plane'] not in PLANES:
            report.errors.append(f"{f['id']}: plane must be one of {', '.join(PLANES)} (got {f['plane']!r})")
        if f['op'] not in OPS:
            report.errors.append(f"{f['id']}: op must be one of {', '.join(OPS)} (got {f['op']!r})")
        if f['direction'] not in DIRECTIONS:
            report.errors.append(f"{f['id']}: direction must be '+', '-' or 'symmetric' (got {f['direction']!r})")
        if not f['shapes']:
            report.errors.append(f"{f['id']}: the feature has no shapes")
    if report.errors:
        return report

    # --- expressions ---------------------------------------------------------
    params = {p['name']: float(p['value']) for p in r['params']}
    res = _resolve(r, params, report)
    if report.errors:
        return report
    report.resolved = res

    # --- shapes and feature semantics --------------------------------------------
    feats = res['features']
    for f in feats:
        _check_shapes(f, report)
    if report.errors:
        return report
    if feats[0]['op'] != 'new_body':
        report.errors.append(f"{feats[0]['id']}: the first feature must be a new_body (it is a {feats[0]['op']})")
    for i, f in enumerate(feats):
        fid = f['id']
        if f['through_all'] and f['op'] != 'cut':
            report.errors.append(f'{fid}: through_all only makes sense for a cut')
        if not f['through_all'] and f['distance'] <= 0:
            report.errors.append(f'{fid}: distance must be > 0 (got {f["distance"]:g})')
        if i > 0 and f['op'] == 'new_body':
            report.warnings.append(f'{fid}: a second new_body makes a separate body')
        if f['confidence'] < 0.5:
            report.warnings.append(f'{fid}: low confidence ({f["confidence"]:.2f}); check its numbers against the drawing')
        if f['source'].get('inferred'):
            report.warnings.append(f'{fid}: some of its numbers were deduced, not read from a label')
        # shapes within one sketch: nesting breaks the profile pick in Fusion
        polys = [shape_polygon(s) for s in f['shapes']]
        for a in range(len(polys)):
            for b in range(a + 1, len(polys)):
                pa, pb = polys[a], polys[b]
                if pa.contains(pb) or pb.contains(pa):
                    report.errors.append(f'{fid}: shapes {a} and {b} are nested in one sketch; make the inner '
                                         'shape its own cut feature')
                elif pa.intersects(pb) and pa.intersection(pb).area > 1e-9:
                    report.warnings.append(f'{fid}: shapes {a} and {b} overlap in one sketch; Fusion will split '
                                           'them into more profiles than expected')
    if report.errors:
        return report

    # --- bounding-box simulation ----------------------------------------------------
    material = []          # (lo, hi) boxes of material added so far
    box_owner = []         # the feature id that added each box
    for f in feats:
        fid = f['id']
        _, _, n = PLANE_AXES[f['plane']]
        if material:
            mlo, mhi = min(b[0][n] for b in material), max(b[1][n] for b in material)
        else:
            mlo, mhi = f['offset'], f['offset']
        nlo, nhi = _n_range(f, f['offset'], f['distance'], mlo, mhi)
        boxes = [_box_of(f, shape_bounds(s), nlo, nhi) for s in f['shapes']]
        if f['op'] in ('new_body', 'join'):
            if f['op'] == 'join' and material and not any(_overlap(b, m) for b in boxes for m in material):
                report.warnings.append(f'{fid}: this join touches no existing material (it would be a loose '
                                       'body in Fusion)')
            material.extend(boxes)
            box_owner.extend([fid] * len(boxes))
        else:
            if not material:
                report.errors.append(f'{fid}: a cut before any body')
                continue
            hits = [m for b in boxes for m in material if _overlap(b, m)]
            if not hits:
                report.errors.append(f'{fid}: this cut removes nothing (its box meets no material)')
                continue
            for b in boxes:
                # the faces the cut really goes into (a box it only touches
                # is not the one it is cut from)
                hosts = [m for m in material if _overlap(b, m, tol=-TOUCH)]
                if hosts and not any(_contains_uv(m, b, f['plane']) for m in hosts):
                    report.warnings.append(f'{fid}: a hole sticks out of the face it is cut into')
                    break
            if f['through_all']:
                # a through-all cut goes the whole way: everything in line is
                # cut, not just the face it was aimed at
                hit = []
                for b in boxes:
                    for m, owner in zip(material, box_owner):
                        if _overlap(b, m, tol=-TOUCH) and owner not in hit:
                            hit.append(owner)
                if len(hit) > 1:
                    report.warnings.append(f'{fid}: this through-all cut passes through {", ".join(hit)}; if '
                                           'only one of them should be cut, give it a distance instead')

    # --- duplicates -----------------------------------------------------------------
    keys = {}
    for f in feats:
        key = (f['plane'], f['op'], round(f['offset'], 6), f['direction'], round(f['distance'], 6),
               f['through_all'], repr(sorted(repr(_round_shape(s)) for s in f['shapes'])))
        if key in keys:
            report.errors.append(f"{f['id']}: duplicate of {keys[key]} (same plane, offset, shapes and extrude)")
        keys.setdefault(key, f['id'])

    # --- overall -----------------------------------------------------------------------
    if material:
        lo = [min(b[0][i] for b in material) for i in range(3)]
        hi = [max(b[1][i] for b in material) for i in range(3)]
        size = [hi[i] - lo[i] for i in range(3)]
        report.bbox = {'min': lo, 'max': hi, 'size': size}
        want = [res['overall']['w'], res['overall']['d'], res['overall']['h']]
        devs = []
        for i in range(3):
            if want[i] is None or want[i] <= 0:
                devs.append(None)
                continue
            dev = (size[i] - want[i]) / want[i]
            devs.append(dev)
        bad = [i for i, d in enumerate(devs) if d is not None and abs(d) > OVERALL_ERR]
        for i in bad:
            msg = (f'the built part is {size[i]:.2f} mm along {AXIS_NAMES[i]} but the drawing says '
                   f'{want[i]:.2f} ({devs[i] * 100:+.1f} %)')
            if len(bad) >= 2:
                msg += '; two axes are off, so a view may be mapped to the wrong plane'
            report.errors.append(msg)
        for i, d in enumerate(devs):
            if d is not None and OVERALL_WARN < abs(d) <= OVERALL_ERR:
                report.warnings.append(f'along {AXIS_NAMES[i]} the part comes out {size[i]:.2f} mm, the '
                                       f'drawing says {want[i]:.2f} ({d * 100:+.1f} %)')
    return report


def _round_shape(s, nd=6):
    if isinstance(s, dict):
        return {k: _round_shape(v, nd) for k, v in s.items()}
    if isinstance(s, list):
        return [_round_shape(v, nd) for v in s]
    if isinstance(s, float):
        return round(s, nd)
    return s
