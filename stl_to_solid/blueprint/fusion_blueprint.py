"""Emit the Fusion 360 script for a Blueprint recipe, the X-Ray way: the
recipe as data literals, then a fixed runtime that draws it. Unlike the
mesh tools the result is meant to be edited afterwards, so the runtime
makes it parametric: every recipe parameter becomes a User Parameter,
every rect / circle / slot gets driving sketch dimensions whose
expressions are the recipe's ("body_w / 2"), every extrude distance is
an expression, and construction-plane offsets are bound where the API
allows it. Change body_h in the Parameters table and the boss moves.

Every value is emitted twice: `*_mm` for placing the geometry (numbers,
so the runtime never evaluates anything) and `*_expr` for the driving
dimension. Fusion works in cm, so mm values are divided by 10 when drawn.

Fusion API calls the runtime leans on that were not yet checked inside
Fusion (a first run of tests/fixtures/sg90_recipe.json's script settles
them; each is wrapped so a refusal degrades to fixed geometry):
  - addTwoPointRectangle returning 4 unconstrained lines (else switch to
    addCenterPointRectangle and dimension the centre),
  - ValueInput.createByString('-(body_h)') as a negative extrude distance,
  - geometricConstraints.addHorizontalPoints / addVerticalPoints,
  - ConstructionPlaneOffsetDefinition.offset.expression being writable,
  - ExtrudeFeatureInput.setOneSideExtent(ThroughAllExtentDefinition.create(), dir).
"""
import math

from ..fusion_export import _XRAY_RUNTIME, _xray_round, emit_fusion_script
from .expr import to_fusion
from .schema import normalize
from .validate import PLANE_AXES, shape_area, validate

_AXIS_VEC = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])


def _src(v):
    """The recipe's own text for a value, safe to embed in a bigger expression."""
    return f'({v})' if isinstance(v, str) else repr(float(v))


def _expr(text):
    return to_fusion(text)


def _shape_data(raw, res):
    """Data for one shape from its raw (expressions) and resolved (floats) forms."""
    k = res['kind']
    if k == 'rect':
        cu, cv, w, h = res['center']['u'], res['center']['v'], res['w'], res['h']
        r = res.get('corner_radius') or 0.0
        rcu, rcv, rw, rh = raw['center']['u'], raw['center']['v'], raw['w'], raw['h']
        d = {'kind': 'rect', 'center_mm': [cu, cv], 'w_mm': w, 'h_mm': h, 'r_mm': r,
             'center_expr': [_expr(rcu), _expr(rcv)], 'w_expr': _expr(rw), 'h_expr': _expr(rh),
             'r_expr': _expr(raw.get('corner_radius') or 0),
             'corner_mm': [cu - w / 2, cv - h / 2],
             'corner_expr': [_expr(f'{_src(rcu)} - {_src(rw)}/2'), _expr(f'{_src(rcv)} - {_src(rh)}/2')]}
        return d
    if k == 'circle':
        return {'kind': 'circle', 'center_mm': [res['center']['u'], res['center']['v']],
                'center_expr': [_expr(raw['center']['u']), _expr(raw['center']['v'])],
                'd_mm': res['d'], 'd_expr': _expr(raw['d'])}
    if k == 'slot':
        a, b, w = res['p1'], res['p2'], res['width']
        du, dv = b['u'] - a['u'], b['v'] - a['v']
        L = math.hypot(du, dv)
        nu, nv = -dv / L * w / 2, du / L * w / 2          # half-width normal
        # straight sides, then the two end arcs (three points each)
        s1 = ([a['u'] + nu, a['v'] + nv], [b['u'] + nu, b['v'] + nv])
        s2 = ([b['u'] - nu, b['v'] - nv], [a['u'] - nu, a['v'] - nv])
        arc_b = (s1[1], [b['u'] + du / L * w / 2, b['v'] + dv / L * w / 2], s2[0])
        arc_a = (s2[1], [a['u'] - du / L * w / 2, a['v'] - dv / L * w / 2], s1[0])
        ra, rb = raw['p1'], raw['p2']
        if abs(dv) < 1e-9:
            length_expr = _expr(f'{_src(rb["u"])} - {_src(ra["u"])}' if du > 0 else f'{_src(ra["u"])} - {_src(rb["u"])}')
        elif abs(du) < 1e-9:
            length_expr = _expr(f'{_src(rb["v"])} - {_src(ra["v"])}' if dv > 0 else f'{_src(ra["v"])} - {_src(rb["v"])}')
        else:
            length_expr = _expr(L)                      # no sqrt in the recipe language
        return {'kind': 'slot', 'p1_mm': [a['u'], a['v']], 'p2_mm': [b['u'], b['v']],
                'p1_expr': [_expr(ra['u']), _expr(ra['v'])],
                'width_mm': w, 'width_expr': _expr(raw['width']), 'length_mm': L,
                'length_expr': length_expr, 'axis_aligned': abs(dv) < 1e-9 or abs(du) < 1e-9,
                'lines': [list(s1), list(s2)], 'arcs': [list(arc_b), list(arc_a)]}
    return {'kind': 'polygon', 'points_mm': [[p['u'], p['v']] for p in res['points']]}


def blueprint_data(recipe):
    """The literals the script embeds: (params, features). `recipe` may be
    raw or normalized; it must validate."""
    r = normalize(recipe)
    rep = validate(r)
    if not rep.ok:
        raise ValueError('recipe does not validate: ' + '; '.join(rep.errors[:3]))
    res = rep.resolved
    params = [{'name': p['name'], 'value_mm': float(p['value']),
               'comment': (p.get('source') or '') + (' (inferred)' if p.get('inferred') else '')}
              for p in r['params']]
    feats = []
    for raw, f in zip(r['features'], res['features']):
        u, v, n = PLANE_AXES[f['plane']]
        shapes = [_shape_data(rs, s) for rs, s in zip(raw['shapes'], f['shapes'])]
        feats.append({
            'id': f['id'], 'name': f['name'], 'plane': f['plane'],
            'axes': [_AXIS_VEC[u], _AXIS_VEC[v], _AXIS_VEC[n]],
            'offset_mm': f['offset'], 'offset_expr': _expr(raw['offset']),
            'op': f['op'], 'direction': f['direction'],
            'distance_mm': f['distance'], 'distance_expr': _expr(raw['distance']),
            'through_all': bool(f['through_all']),
            'areas_mm2': [shape_area(s) for s in f['shapes']],
            'shapes': shapes,
        })
    return params, feats


def emit_fusion_blueprint_script(recipe, title='Blueprint'):
    params, feats = blueprint_data(recipe)
    L = [
        '"""Generated by stlToSolid — Blueprint: a dimensioned drawing rebuilt as a',
        'parametric Fusion 360 part. Every recipe parameter becomes a User Parameter,',
        'every feature a sketch with driving dimensions plus an Extrude (new body /',
        'join / cut). Edit a parameter in Modify > Change Parameters and the part',
        'follows. Fusion API units are centimetres; the data below is mm and is',
        'divided by 10 when drawn."""',
        'import adsk.core, adsk.fusion, traceback',
        '',
        f'TITLE = {title!r}',
        '',
        '# user parameters: name, value (mm), comment',
        'PARAMS = [',
    ]
    L += [f'    {_xray_round(p, 6)!r},' for p in params]
    L += [']', '',
          '# features in order: plane axes (u, v, normal as world vectors), offset along',
          '# the normal, shapes with mm values for placing and Fusion expressions for',
          '# the driving dimensions, extrude op / direction / distance, expected',
          '# material area per shape (to pick the profiles)',
          'FEATURES = [']
    L += [f'    {_xray_round(f, 6)!r},' for f in feats]
    L += [']', '', '']
    base = emit_fusion_script([])
    L.append(base[base.index('def _plane('):base.index('def _pick(')].rstrip('\n'))
    L.append('')
    L.append(_XRAY_RUNTIME[_XRAY_RUNTIME.index('def _token'):_XRAY_RUNTIME.index('def _extrude_slice')].rstrip('\n'))
    L.append(_BLUEPRINT_RUNTIME)
    return '\n'.join(L)


_BLUEPRINT_RUNTIME = '''

def _ensure_params(design, notes):
    """One User Parameter per recipe parameter. One that already exists is
    kept (the user's edits survive a re-run). A direct-modelling design
    has no parameters: the geometry is then drawn fixed."""
    if design.designType != adsk.fusion.DesignTypes.ParametricDesignType:
        notes.append('direct-modelling design: no User Parameters, the geometry is fixed')
        return False
    ups = design.userParameters
    made = 0
    for p in PARAMS:
        if ups.itemByName(p['name']) is not None:
            continue
        ups.add(p['name'], adsk.core.ValueInput.createByString('%s mm' % _num(p['value_mm'])),
                'mm', p['comment'])
        made += 1
    return True


def _num(v):
    s = ('%.6f' % v).rstrip('0').rstrip('.')
    return s if s not in ('', '-', '-0') else '0'


def _world(feat, u, v):
    """A point (u, v) mm on the feature's plane as a model Point3D (cm)."""
    au, av, an = feat['axes']
    o = feat['offset_mm']
    return adsk.core.Point3D.create(*[(au[i] * u + av[i] * v + an[i] * o) / 10.0 for i in range(3)])


def _sk_pt(sk, feat, u, v):
    return sk.modelToSketchSpace(_world(feat, u, v))


def _text_near(pt, dx=0.3, dy=0.3):
    g = pt.geometry
    return adsk.core.Point3D.create(g.x + dx, g.y + dy, 0)


def _sketch(root, feat, notes):
    """Construction plane + sketch for a feature. Returns (sketch, sgn,
    hor_is_u): sgn from _plane (the plane normal against the recipe's),
    hor_is_u whether the sketch's x axis runs along the recipe's u axis
    (which way a horizontal dimension goes)."""
    an = feat['axes'][2]
    pl, sgn = _plane(root, tuple(c * feat['offset_mm'] / 10.0 for c in an), tuple(an))
    pl.name = 'plane ' + feat['name']
    try:
        par = pl.definition.offset               # ConstructionPlaneOffsetDefinition
        same = (par.value >= 0) == (feat['offset_mm'] >= 0)
        if abs(feat['offset_mm']) > 1e-9:
            par.expression = feat['offset_expr'] if same else '-(%s)' % feat['offset_expr']
    except Exception:
        pass                                     # a fixed offset is still right
    sk = root.sketches.add(pl)
    sk.name = feat['name']
    try:
        xd = sk.xDirection
        au = feat['axes'][0]
        hor_is_u = abs(xd.x * au[0] + xd.y * au[1] + xd.z * au[2]) > 0.9
    except Exception:
        hor_is_u = True
    return sk, sgn, hor_is_u


def _bind(dim, expr, param_ok, notes):
    if not param_ok:
        return
    try:
        dim.parameter.expression = expr
    except Exception:
        notes.append('could not bind a dimension to "%s"' % expr)


def _dim_pos(sk, pt, u_mm, v_mm, u_expr, v_expr, hor_is_u, param_ok, notes):
    """Pin a sketch point to the sketch origin: a driving distance
    dimension per axis with the recipe's expression, or a horizontal /
    vertical alignment where the value is zero (a zero-length dimension is
    not allowed)."""
    D = adsk.fusion.DimensionOrientations
    gc, dims, org = sk.geometricConstraints, sk.sketchDimensions, sk.originPoint
    for val, expr, along_u in ((u_mm, u_expr, True), (v_mm, v_expr, False)):
        horizontal = along_u == hor_is_u
        try:
            if abs(val) < 1e-6:
                if horizontal:
                    gc.addVerticalPoints(pt, org)      # same x: aligned vertically
                else:
                    gc.addHorizontalPoints(pt, org)
                continue
            d = dims.addDistanceDimension(org, pt, D.HorizontalDimensionOrientation if horizontal
                                          else D.VerticalDimensionOrientation, _text_near(pt))
            _bind(d, expr if val > 0 else '-(%s)' % expr, param_ok, notes)
        except Exception:
            notes.append('%s: a position could not be dimensioned' % sk.name)


def _draw_rect(sk, feat, sh, hor_is_u, param_ok, notes):
    c, w, h = sh['center_mm'], sh['w_mm'], sh['h_mm']
    p1 = _sk_pt(sk, feat, c[0] - w / 2.0, c[1] - h / 2.0)
    p2 = _sk_pt(sk, feat, c[0] + w / 2.0, c[1] + h / 2.0)
    lines = sk.sketchCurves.sketchLines.addTwoPointRectangle(p1, p2)
    gc = sk.geometricConstraints
    items = [lines.item(i) for i in range(lines.count)]
    for i, ln in enumerate(items):
        try:
            (gc.addHorizontal if i % 2 == 0 else gc.addVertical)(ln)
        except Exception:
            pass
    if len(items) < 2:
        return
    # the side whose length is nearest w carries the w expression, whatever
    # way round Fusion drew the rectangle
    l0, l1 = items[0], items[1]
    D = adsk.fusion.DimensionOrientations
    try:
        w_line, h_line = (l0, l1) if abs(l0.length - w / 10.0) <= abs(l1.length - w / 10.0) else (l1, l0)
        for ln, expr in ((w_line, sh['w_expr']), (h_line, sh['h_expr'])):
            d = sk.sketchDimensions.addDistanceDimension(ln.startSketchPoint, ln.endSketchPoint,
                                                         D.AlignedDimensionOrientation,
                                                         _text_near(ln.startSketchPoint))
            _bind(d, expr, param_ok, notes)
    except Exception:
        notes.append('%s: a rectangle could not be dimensioned' % sk.name)
    _dim_pos(sk, l0.startSketchPoint, sh['corner_mm'][0], sh['corner_mm'][1],
             sh['corner_expr'][0], sh['corner_expr'][1], hor_is_u, param_ok, notes)
    if sh['r_mm'] > 0:
        try:
            r_cm = sh['r_mm'] / 10.0
            for i in range(4):
                a, b = items[i], items[(i + 1) % 4]
                arc = sk.sketchCurves.sketchArcs.addFillet(a, a.endSketchPoint.geometry,
                                                           b, b.startSketchPoint.geometry, r_cm)
                if i == 0:
                    d = sk.sketchDimensions.addRadialDimension(arc, _text_near(arc.centerSketchPoint))
                    _bind(d, sh['r_expr'], param_ok, notes)
        except Exception:
            notes.append('%s: the corner fillets could not be drawn' % sk.name)


def _draw_circle(sk, feat, sh, hor_is_u, param_ok, notes):
    c = sk.sketchCurves.sketchCircles.addByCenterRadius(
        _sk_pt(sk, feat, sh['center_mm'][0], sh['center_mm'][1]), sh['d_mm'] / 20.0)
    try:
        d = sk.sketchDimensions.addDiameterDimension(c, _text_near(c.centerSketchPoint))
        _bind(d, sh['d_expr'], param_ok, notes)
    except Exception:
        notes.append('%s: a circle could not be dimensioned' % sk.name)
    _dim_pos(sk, c.centerSketchPoint, sh['center_mm'][0], sh['center_mm'][1],
             sh['center_expr'][0], sh['center_expr'][1], hor_is_u, param_ok, notes)


def _draw_slot(sk, feat, sh, hor_is_u, param_ok, notes):
    P = lambda p: _sk_pt(sk, feat, p[0], p[1])
    lines = [sk.sketchCurves.sketchLines.addByTwoPoints(P(a), P(b)) for a, b in sh['lines']]
    arcs = [sk.sketchCurves.sketchArcs.addByThreePoints(P(a), P(m), P(b)) for a, m, b in sh['arcs']]
    gc = sk.geometricConstraints
    for ln in lines:
        for arc in arcs:
            try:
                gc.addTangent(ln, arc)
            except Exception:
                pass
    D = adsk.fusion.DimensionOrientations
    try:
        c0, c1 = arcs[1].centerSketchPoint, arcs[0].centerSketchPoint       # at p1, at p2
        d = sk.sketchDimensions.addDistanceDimension(c0, c1, D.AlignedDimensionOrientation, _text_near(c0))
        _bind(d, sh['length_expr'], param_ok, notes)
        r = sk.sketchDimensions.addRadialDimension(arcs[0], _text_near(arcs[0].centerSketchPoint))
        _bind(r, '(%s) / 2' % sh['width_expr'], param_ok, notes)
        if sh['axis_aligned']:
            if abs(sh['p2_mm'][1] - sh['p1_mm'][1]) < 1e-9:
                (gc.addHorizontalPoints if hor_is_u else gc.addVerticalPoints)(c0, c1)
            else:
                (gc.addVerticalPoints if hor_is_u else gc.addHorizontalPoints)(c0, c1)
        _dim_pos(sk, c0, sh['p1_mm'][0], sh['p1_mm'][1], sh['p1_expr'][0], sh['p1_expr'][1],
                 hor_is_u, param_ok, notes)
    except Exception:
        notes.append('%s: a slot could not be dimensioned' % sk.name)


def _draw_polygon(sk, feat, sh, hor_is_u, param_ok, notes):
    pts = sh['points_mm']
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        sk.sketchCurves.sketchLines.addByTwoPoints(_sk_pt(sk, feat, a[0], a[1]),
                                                   _sk_pt(sk, feat, b[0], b[1]))
    notes.append('%s: polygon drawn without driving dimensions' % sk.name)


_DRAW = {'rect': _draw_rect, 'circle': _draw_circle, 'slot': _draw_slot, 'polygon': _draw_polygon}


def _extrude(root, sk, feat, sgn, bodies, seen, notes):
    """Extrude the feature's material profiles with the recipe's op and
    an expression-driven distance. Returns whether the expected areas
    matched the profiles (else every profile was taken)."""
    ops = adsk.fusion.FeatureOperations
    ext = root.features.extrudeFeatures
    if sk.profiles.count == 0:
        raise RuntimeError('no closed profile to extrude')
    profs, matched = _material_profiles(sk, [a / 100.0 for a in feat['areas_mm2']])
    coll = adsk.core.ObjectCollection.create()
    for p in profs:
        coll.add(p)
    op = {'new_body': ops.NewBodyFeatureOperation, 'join': ops.JoinFeatureOperation,
          'cut': ops.CutFeatureOperation}[feat['op']]
    flip = (sgn < 0) != (feat['direction'] == '-')
    dist_cm = feat['distance_mm'] / 10.0

    def make(op_):
        inp = ext.createInput(coll, op_)
        if feat['through_all']:
            Dirs = adsk.fusion.ExtentDirections
            direction = (Dirs.SymmetricExtentDirection if feat['direction'] == 'symmetric'
                         else Dirs.NegativeExtentDirection if flip else Dirs.PositiveExtentDirection)
            try:
                inp.setOneSideExtent(adsk.fusion.ThroughAllExtentDefinition.create(), direction)
            except Exception:
                inp.setAllExtent(direction)
        elif feat['direction'] == 'symmetric':
            try:
                inp.setSymmetricExtent(adsk.core.ValueInput.createByString(feat['distance_expr']), True)
            except Exception:
                inp.setSymmetricExtent(adsk.core.ValueInput.createByReal(dist_cm), True)
        else:
            expr = feat['distance_expr']
            try:
                inp.setDistanceExtent(False, adsk.core.ValueInput.createByString(
                    '-(%s)' % expr if flip else expr))
            except Exception:
                inp.setDistanceExtent(False, adsk.core.ValueInput.createByReal(
                    -dist_cm if flip else dist_cm))
        if op_ != ops.NewBodyFeatureOperation:
            live = _live(bodies)
            if live:
                inp.participantBodies = live
        return inp

    try:
        f = ext.add(make(op))
    except Exception:
        if op != ops.JoinFeatureOperation:
            raise
        f = ext.add(make(ops.NewBodyFeatureOperation))     # a join that met nothing
        notes.append('%s: joined nothing, made a new body' % feat['name'])
    _track(f, bodies, seen)
    return matched


def run(context):
    ui = None
    prog = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        root = design.rootComponent
        notes = []
        param_ok = _ensure_params(design, notes)
        n = len(FEATURES)
        prog = ui.createProgressDialog()
        prog.isCancelButtonShown = True
        prog.show(TITLE, 'Feature %v of %m', 0, n, 0)
        bodies, seen, made, unmatched, cancelled = [], set(), 0, 0, False
        for i, feat in enumerate(FEATURES):
            prog.progressValue = i
            adsk.doEvents()
            if prog.wasCancelled:
                cancelled = True
                break
            sk, sgn, hor_is_u = _sketch(root, feat, notes)
            sk.isComputeDeferred = True
            try:
                for sh in feat['shapes']:
                    _DRAW[sh['kind']](sk, feat, sh, hor_is_u, param_ok, notes)
            finally:
                sk.isComputeDeferred = False
            try:
                if not _extrude(root, sk, feat, sgn, bodies, seen, notes):
                    unmatched += 1
                made += 1
            except Exception:
                why = traceback.format_exc().strip().splitlines()[-1]
                notes.append('%s: extrude failed: %s' % (feat['name'], why))
        prog.hide()
        live = _live(bodies)
        msg = 'stlToSolid Blueprint %s: %d of %d feature(s), %d parameter(s), %d body%s' % (
            TITLE, made, n, len(PARAMS), len(live), '' if len(live) == 1 else 'ies')
        if cancelled:
            msg += ' (stopped)'
        if unmatched:
            notes.append('%d feature(s): no profile matched the expected area, so every '
                         'profile was extruded there' % unmatched)
        if notes:
            msg += '\\n' + '\\n'.join(notes[:12])
        ui.messageBox(msg)
    except:
        if prog is not None:
            try:
                prog.hide()
            except Exception:
                pass
        if ui:
            ui.messageBox('stlToSolid Blueprint script failed:\\n{}'.format(traceback.format_exc()))
'''
