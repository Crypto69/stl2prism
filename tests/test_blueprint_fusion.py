"""The Blueprint Fusion script: its data literals, and its runtime run
against a stand-in adsk that records User Parameters, sketch curves,
dimensions, constraints and extrudes the way Fusion exposes them."""
import ast
import json
import math
import os
import re
import sys
import types
from types import SimpleNamespace

import pytest

from stl_to_solid.blueprint.fusion_blueprint import blueprint_data, emit_fusion_blueprint_script

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'sg90_recipe.json')


@pytest.fixture
def sg90():
    with open(FIX) as f:
        return json.load(f)


def _literals(txt):
    ns = {}
    exec(txt[txt.index('TITLE ='):txt.index('def _plane(')], ns)
    return ns


def test_script_parses_and_embeds_the_recipe(sg90):
    txt = emit_fusion_blueprint_script(sg90, title='SG90')
    ast.parse(txt)
    ns = _literals(txt)
    assert ns['TITLE'] == 'SG90'
    assert len(ns['PARAMS']) == 20 and ns['PARAMS'][0]['name'] == 'body_w'
    F = ns['FEATURES']
    assert len(F) == 8
    body = F[0]
    assert body['axes'] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert body['areas_mm2'][0] == pytest.approx(22.5 * 11.8)
    sh = body['shapes'][0]
    assert sh['w_expr'] == 'body_w' and sh['center_expr'] == ['body_w / 2', 'body_d / 2']
    assert sh['corner_mm'] == pytest.approx([0, 0])
    assert F[2]['through_all'] and F[2]['op'] == 'cut'
    assert F[6]['direction'] == '-' and F[6]['distance_expr'] == 'hub_h'
    slot = F[7]
    assert slot['axes'] == [[0, 1, 0], [0, 0, 1], [1, 0, 0]] and slot['offset_expr'] == 'body_w'
    # every expression is Fusion text: names, numbers with units, + - * / ( ) only
    for f in F:
        for key in ('offset_expr', 'distance_expr'):
            assert re.fullmatch(r'[A-Za-z0-9_ .+\-*/()]+(mm)?', f[key]), f[key]


# --- a stand-in Fusion -------------------------------------------------------------

class Coll(list):
    @property
    def count(self):
        return len(self)

    def item(self, i):
        return self[i]

    def add(self, x):
        self.append(x)


class P3:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @staticmethod
    def create(x, y, z=0.0):
        return P3(x, y, z)


class SkPoint:
    def __init__(self, geom):
        self.geometry = geom


class Line:
    def __init__(self, a, b):
        self.startSketchPoint, self.endSketchPoint = SkPoint(a), SkPoint(b)
        self.length = math.dist((a.x, a.y), (b.x, b.y))


class Arc:
    def __init__(self, a, m, b):
        self.centerSketchPoint = SkPoint(P3((a.x + b.x) / 2, (a.y + b.y) / 2))


class Circle:
    def __init__(self, c, r):
        self.centerSketchPoint, self.r = SkPoint(c), r


class Dim:
    def __init__(self, kind):
        self.kind = kind
        self.parameter = SimpleNamespace(expression=None)


class Body:
    _n = 0

    def __init__(self):
        Body._n += 1
        self.entityToken = 'b%d' % Body._n
        self.isValid = True


class Sketch:
    """Records what is drawn; its profiles are one per closed shape with
    the shape's area (cm^2), like Fusion's."""
    def __init__(self, plane, log):
        self.plane = plane
        self.name = None
        self.isComputeDeferred = False
        self.log = log
        self.areas = []
        self.dims = []
        self.constraints = []
        self.xDirection = SimpleNamespace(x=1.0, y=0.0, z=0.0) if plane.normal[2] else \
            (SimpleNamespace(x=1.0, y=0.0, z=0.0) if plane.normal[1] else SimpleNamespace(x=0.0, y=1.0, z=0.0))
        self.originPoint = SkPoint(P3(0, 0))
        sk = self

        def rect(p1, p2):
            w, h = abs(p2.x - p1.x), abs(p2.y - p1.y)
            sk.areas.append(w * h)
            c = [P3(p1.x, p1.y), P3(p2.x, p1.y), P3(p2.x, p2.y), P3(p1.x, p2.y)]
            return Coll([Line(c[0], c[1]), Line(c[1], c[2]), Line(c[2], c[3]), Line(c[3], c[0])])

        def circle(c, r):
            sk.areas.append(math.pi * r * r)
            return Circle(c, r)

        self.sketchCurves = SimpleNamespace(
            sketchLines=SimpleNamespace(addTwoPointRectangle=rect, addByTwoPoints=lambda a, b: Line(a, b)),
            sketchCircles=SimpleNamespace(addByCenterRadius=circle),
            sketchArcs=SimpleNamespace(addByThreePoints=lambda a, m, b: Arc(a, m, b),
                                       addFillet=lambda *a: Arc(a[1], a[1], a[3])),
        )

        def dim(kind):
            def f(*a):
                d = Dim(kind)
                sk.dims.append(d)
                return d
            return f

        self.sketchDimensions = SimpleNamespace(
            addDistanceDimension=dim('distance'), addDiameterDimension=dim('diameter'),
            addRadialDimension=dim('radial'))

        def con(kind):
            def f(*a):
                sk.constraints.append(kind)
            return f

        self.geometricConstraints = SimpleNamespace(
            addHorizontal=con('h'), addVertical=con('v'), addHorizontalPoints=con('hp'),
            addVerticalPoints=con('vp'), addTangent=con('t'))

    def modelToSketchSpace(self, p):
        # sketch x, y = the two in-plane world axes in alphabetical order
        n = self.plane.normal
        if n[2]:
            return P3(p.x, p.y)
        if n[1]:
            return P3(p.x, p.z)
        return P3(p.y, p.z)

    @property
    def profiles(self):
        return Coll([SimpleNamespace(areaProperties=lambda a=a: SimpleNamespace(area=a)) for a in self.areas])


class Plane:
    def __init__(self, origin, normal):
        self.geometry = SimpleNamespace(origin=P3(*origin), normal=SimpleNamespace(x=normal[0], y=normal[1], z=normal[2]))
        self.normal = normal
        self.name = None
        self.definition = SimpleNamespace(offset=SimpleNamespace(value=sum(o * n for o, n in zip(origin, normal)),
                                                                 expression=None))

    def deleteMe(self):
        pass


class Planes:
    def __init__(self):
        self.made = []

    def createInput(self):
        return SimpleNamespace(setByOffset=lambda base, off: setattr(self, '_pending', (base, off)),
                               setByPlane=lambda pl: setattr(self, '_pending', ('plane', pl)))

    def add(self, inp):
        base, off = self._pending
        if base == 'plane':                  # direct design: (Point3D, normal vector)
            origin, normal = off
            pl = Plane([origin.x, origin.y, origin.z], list(normal))
            self.made.append(pl)
            return pl
        off = off[1] if isinstance(off, tuple) else off        # a ValueInput of the fake
        n = base.normal
        pl = Plane([n[i] * off for i in range(3)], n)
        self.made.append(pl)
        return pl


class Extrudes:
    def __init__(self, log):
        self.log = log

    def createInput(self, coll, op):
        inp = SimpleNamespace(coll=coll, op=op, dist=None, extent=None, participantBodies=None)
        inp.setDistanceExtent = lambda sym, v: setattr(inp, 'dist', v)
        inp.setSymmetricExtent = lambda v, full: setattr(inp, 'extent', ('sym', v))
        inp.setOneSideExtent = lambda ext, d: setattr(inp, 'extent', ('through', d))
        inp.setAllExtent = lambda d: setattr(inp, 'extent', ('all', d))
        return inp

    def add(self, inp):
        b = Body()
        self.log.append({'op': inp.op, 'dist': inp.dist, 'extent': inp.extent,
                         'profiles': len(inp.coll), 'participants': len(inp.participantBodies or [])})
        return SimpleNamespace(bodies=Coll([b]))


class UserParams:
    def __init__(self):
        self.added = []

    def itemByName(self, n):
        return object() if n in self.added else None

    def add(self, name, value, unit, comment):
        self.added.append(name)


def _fake_fusion(monkeypatch, parametric=True):
    adsk = types.ModuleType('adsk')
    core, fusion = types.ModuleType('adsk.core'), types.ModuleType('adsk.fusion')
    core.ValueInput = SimpleNamespace(createByReal=staticmethod(lambda v: ('real', v)),
                                      createByString=staticmethod(lambda s: ('str', s)))
    core.ObjectCollection = SimpleNamespace(create=staticmethod(Coll))
    core.Point3D = P3
    core.Vector3D = SimpleNamespace(create=staticmethod(lambda *a: a))
    core.Plane = SimpleNamespace(create=staticmethod(lambda *a: a))
    fusion.FeatureOperations = SimpleNamespace(NewBodyFeatureOperation='new', JoinFeatureOperation='join',
                                               CutFeatureOperation='cut')
    fusion.DesignTypes = SimpleNamespace(ParametricDesignType='param', DirectDesignType='direct')
    fusion.DimensionOrientations = SimpleNamespace(HorizontalDimensionOrientation='H',
                                                   VerticalDimensionOrientation='V',
                                                   AlignedDimensionOrientation='A')
    fusion.ExtentDirections = SimpleNamespace(PositiveExtentDirection='+', NegativeExtentDirection='-',
                                              SymmetricExtentDirection='s')
    fusion.ThroughAllExtentDefinition = SimpleNamespace(create=staticmethod(lambda: 'through'))
    log = {'extrudes': [], 'sketches': [], 'messages': []}
    ups = UserParams()
    design = SimpleNamespace(designType='param' if parametric else 'direct', userParameters=ups)
    planes = Planes()
    sketches = SimpleNamespace(add=lambda pl: log['sketches'].append(Sketch(pl, log)) or log['sketches'][-1])
    root = SimpleNamespace(
        parentDesign=design, constructionPlanes=planes, sketches=sketches,
        yZConstructionPlane=Plane([0, 0, 0], [1, 0, 0]), xZConstructionPlane=Plane([0, 0, 0], [0, 1, 0]),
        xYConstructionPlane=Plane([0, 0, 0], [0, 0, 1]),
        features=SimpleNamespace(extrudeFeatures=Extrudes(log['extrudes'])))
    design.rootComponent = root
    fusion.Design = SimpleNamespace(cast=staticmethod(lambda p: design))
    prog = SimpleNamespace(isCancelButtonShown=False, progressValue=0, wasCancelled=False,
                           show=lambda *a: None, hide=lambda: None)
    ui = SimpleNamespace(createProgressDialog=lambda: prog, messageBox=lambda m: log['messages'].append(m))
    core.Application = SimpleNamespace(get=staticmethod(lambda: SimpleNamespace(userInterface=ui, activeProduct=None)))
    adsk.doEvents = lambda: None
    adsk.core, adsk.fusion = core, fusion
    for name, mod in (('adsk', adsk), ('adsk.core', core), ('adsk.fusion', fusion)):
        monkeypatch.setitem(sys.modules, name, mod)
    return log, ups, planes


def _run(txt, monkeypatch, parametric=True):
    log, ups, planes = _fake_fusion(monkeypatch, parametric)
    ns = {}
    exec(txt, ns)
    ns['run'](None)
    return ns, log, ups, planes


def test_runtime_builds_the_sg90_parametrically(sg90, monkeypatch):
    txt = emit_fusion_blueprint_script(sg90, title='SG90')
    ns, log, ups, planes = _run(txt, monkeypatch)
    assert log['messages'] and 'failed' not in log['messages'][0], log['messages']
    assert len(ups.added) == 20
    ex = log['extrudes']
    assert [e['op'] for e in ex] == ['new', 'join', 'cut', 'join', 'join', 'join', 'cut', 'cut']
    assert ex[0]['dist'] == ('str', 'body_h') and ex[0]['participants'] == 0
    assert ex[1]['profiles'] == 2 and ex[1]['participants'] >= 1       # both tabs, joined to the body
    assert ex[2]['extent'] == ('through', '+')                         # tab holes through all
    assert ex[6]['dist'] == ('str', '-(hub_h)')                          # screw cut goes down
    assert ex[7]['dist'] == ('str', '-(slot_depth)')                     # slot cut goes into the side
    sks = log['sketches']
    assert [s.name for s in sks][:2] == ['body', 'mounting tabs']
    body = sks[0]
    exprs = [d.parameter.expression for d in body.dims]
    assert 'body_w' in exprs and 'body_d' in exprs
    # the body's corner sits on the origin: aligned by constraints, no zero dimension
    assert 'vp' in body.constraints and 'hp' in body.constraints
    assert len([d for d in body.dims if d.kind == 'distance']) == 2
    boss = sks[3]
    dexpr = [d.parameter.expression for d in boss.dims]
    assert 'boss_d' in dexpr and 'boss_x' in dexpr and 'body_d / 2' in dexpr
    # the slot sketch is on YZ: its plane normal is +X at body_w
    slot_plane = planes.made[-1]
    assert slot_plane.normal == [1, 0, 0] and slot_plane.geometry.origin.x == pytest.approx(2.25)
    assert slot_plane.definition.offset.expression == 'body_w'
    # a second run adds no parameters
    ns['run'](None)
    assert len(ups.added) == 20


def test_runtime_in_a_direct_design_draws_fixed(sg90, monkeypatch):
    txt = emit_fusion_blueprint_script(sg90)
    ns, log, ups, planes = _run(txt, monkeypatch, parametric=False)
    assert ups.added == []
    assert 'direct-modelling' in log['messages'][0]
    assert all(d.parameter.expression is None for s in log['sketches'] for d in s.dims)
    assert len(log['extrudes']) == 8


def test_blueprint_data_slot_and_rounded_rect():
    rec = {'name': 't', 'units': 'mm', 'overall': {'w': 20, 'd': 20, 'h': 10},
           'params': [{'name': 'L', 'value': 20}, {'name': 'sw', 'value': 2}],
           'features': [
               {'id': 'b', 'plane': 'XY', 'op': 'new_body', 'distance': 10,
                'shapes': [{'kind': 'rect', 'center': {'u': 'L/2', 'v': 'L/2'}, 'w': 'L', 'h': 'L', 'corner_radius': 2}]},
               {'id': 's', 'plane': 'XY', 'op': 'cut', 'offset': 10, 'direction': '-', 'distance': 1,
                'shapes': [{'kind': 'slot', 'p1': {'u': 4, 'v': 4}, 'p2': {'u': 12, 'v': 4}, 'width': 'sw'}]}]}
    params, feats = blueprint_data(rec)
    rr = feats[0]['shapes'][0]
    assert rr['r_mm'] == 2 and rr['r_expr'] == '2 mm' and rr['corner_expr'] == ['L / 2 - L / 2', 'L / 2 - L / 2']
    sl = feats[1]['shapes'][0]
    assert sl['length_expr'] == '8 mm' and sl['width_expr'] == 'sw' and sl['axis_aligned']
    assert len(sl['lines']) == 2 and len(sl['arcs']) == 2
    assert feats[1]['areas_mm2'][0] == pytest.approx(8 * 2 + math.pi, rel=1e-3)
