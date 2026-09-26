"""A resolved recipe as a CadQuery solid: one tool solid per shape,
unioned into the feature's tool, then new_body / join / cut against the
result so far. Every feature's effect is measured (volume delta, solid
count) for the report.

CadQuery's named workplanes (checked on cq 2.8.0): 'XY' has normal +Z,
'YZ' normal +X, but 'XZ' has normal -Y. The recipe's direction '+' means
along the plane's positive world normal, so the XZ distance is flipped.
A plane at an offset is made with Workplane(plane, origin=...) rather
than .workplane(offset=), which would move along cq's own normal.
"""
import math
from dataclasses import dataclass, field

# plane -> (u axis, v axis, normal axis, sign of cq's normal vs the recipe's)
_CQ = {'XY': (0, 1, 2, 1.0), 'XZ': (0, 2, 1, -1.0), 'YZ': (1, 2, 0, 1.0)}
THROUGH_MARGIN = 10.0


class BlueprintError(RuntimeError):
    """A failure with a sentence for the user (errors.describe returns
    str(exc) as is). `report` carries the validator's Report when the
    recipe itself was refused."""
    readable = True

    def __init__(self, message, kind='blueprint', report=None):
        super().__init__(message)
        self.kind = kind
        self.report = report


@dataclass
class Build:
    solid: object                  # cq.Workplane holding the result
    bbox: dict                     # {'min', 'max', 'size'} mm
    volume_mm3: float
    n_solids: int
    per_feature: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    tools: list = field(default_factory=list)      # one cq.Workplane per feature: what it added or cut
    features: list = field(default_factory=list)   # [{'id', 'name', 'op'}] in build order


def _workplane(plane, offset):
    import cadquery as cq
    _, _, n, _ = _CQ[plane]
    origin = [0.0, 0.0, 0.0]
    origin[n] = float(offset)
    return cq.Workplane(plane, origin=tuple(origin))


def _shape_tool(plane, offset, shape, dist_cq, both):
    """One shape extruded: a cq.Workplane with a solid."""
    wp = _workplane(plane, offset)
    k = shape['kind']
    if k == 'rect':
        cu, cv = shape['center']['u'], shape['center']['v']
        r = shape.get('corner_radius') or 0.0
        if r > 0:
            wp = wp.center(cu, cv).sketch().rect(shape['w'], shape['h']).vertices().fillet(r).finalize()
        else:
            wp = wp.pushPoints([(cu, cv)]).rect(shape['w'], shape['h'])
    elif k == 'circle':
        wp = wp.pushPoints([(shape['center']['u'], shape['center']['v'])]).circle(shape['d'] / 2)
    elif k == 'slot':
        a, b = shape['p1'], shape['p2']
        du, dv = b['u'] - a['u'], b['v'] - a['v']
        L = math.hypot(du, dv)
        wp = wp.pushPoints([((a['u'] + b['u']) / 2, (a['v'] + b['v']) / 2)]).slot2D(
            L + shape['width'], shape['width'], math.degrees(math.atan2(dv, du)))
    elif k == 'polygon':
        wp = wp.polyline([(p['u'], p['v']) for p in shape['points']]).close()
    else:
        raise BlueprintError(f'unknown shape kind {k!r}')
    return wp.extrude(dist_cq, both=both)


def _bbox(shape):
    bb = shape.val().BoundingBox()
    lo = [bb.xmin, bb.ymin, bb.zmin]
    hi = [bb.xmax, bb.ymax, bb.zmax]
    return {'min': lo, 'max': hi, 'size': [hi[i] - lo[i] for i in range(3)]}


def _volume(shape):
    from ..pipeline import accurate_volume
    try:
        return accurate_volume(shape)
    except Exception:
        return 0.0


def build(resolved):
    """Build a validated, resolved recipe (validate.Report.resolved)."""
    result = None
    per = []
    warnings = []
    tools = []
    n_bodies_expected = sum(1 for f in resolved['features'] if f['op'] == 'new_body')
    for f in resolved['features']:
        fid = f['id']
        plane = f['plane']
        _, _, n, cq_sign = _CQ[plane]
        dir_sign = {'+': 1.0, '-': -1.0}.get(f['direction'], 1.0)
        both = f['direction'] == 'symmetric'
        if f['through_all']:
            if result is None:
                raise BlueprintError(f'{fid}: a through-all cut before any body')
            bb = _bbox(result)
            dist = max(bb['size']) + abs(f['offset']) + max(abs(bb['min'][n]), abs(bb['max'][n])) + THROUGH_MARGIN
            both = True
        elif both:
            dist = f['distance'] / 2.0
        else:
            dist = f['distance'] * dir_sign * cq_sign
        tool = None
        for j, s in enumerate(f['shapes']):
            try:
                t = _shape_tool(plane, f['offset'], s, dist, both)
            except BlueprintError:
                raise
            except Exception as e:
                raise BlueprintError(f'{fid}: shape {j} could not be drawn ({type(e).__name__}: {e})') from e
            tool = t if tool is None else tool.union(t)
        tools.append(tool)
        before = _volume(result) if result is not None else 0.0
        try:
            if f['op'] == 'new_body' or result is None:
                if f['op'] != 'new_body':
                    raise BlueprintError(f'{fid}: a {f["op"]} before any body')
                result = tool if result is None else result.union(tool)
            elif f['op'] == 'join':
                result = result.union(tool)
            else:
                result = result.cut(tool)
        except BlueprintError:
            raise
        except Exception as e:
            raise BlueprintError(f'{fid}: the {f["op"]} failed in the geometry kernel '
                                 f'({type(e).__name__}: {e})') from e
        after = _volume(result)
        n_sol = len(result.solids().vals())
        per.append({'id': fid, 'op': f['op'], 'volume_delta_mm3': after - before, 'solids': n_sol})
    if result is None:
        raise BlueprintError('the recipe built nothing')
    n_sol = len(result.solids().vals())
    if n_sol != max(1, n_bodies_expected):
        warnings.append(f'the part came out as {n_sol} separate bodies (the recipe has '
                        f'{n_bodies_expected} new_body feature{"s" if n_bodies_expected != 1 else ""}); '
                        'a join that touches nothing, or a cut that split the part')
    return Build(solid=result, bbox=_bbox(result), volume_mm3=_volume(result), n_solids=n_sol,
                 per_feature=per, warnings=warnings, tools=tools,
                 features=[{'id': f['id'], 'name': f['name'], 'op': f['op']} for f in resolved['features']])


PREVIEW_TOL = 0.02
PREVIEW_ANG = 0.1
OWNER_TOL = 0.1        # mm: a triangle whose centre is this close to a tool's surface is that tool's


def preview_mesh(b):
    """The result as a trimesh, the way the web viewer and the feature map
    both see it (one tessellation, so triangle order matches)."""
    from ..pipeline import tessellate_solid
    return tessellate_solid(b.solid, PREVIEW_TOL, PREVIEW_ANG)


def feature_map(b, mesh, tol=OWNER_TOL):
    """Which feature made each triangle's surface: the last feature (in
    build order) whose tool surface passes through the triangle's centre.
    A boss's sides belong to the boss, a hole's wall to the cut that made
    it, everything else to the body. Returns a list, one int per triangle
    (an index into b.features)."""
    import numpy as np
    import trimesh
    from ..pipeline import tessellate_solid
    if len(mesh.faces) == 0:
        return []
    centres = mesh.triangles_center
    owner = np.zeros(len(centres), dtype=int)
    for i, tool in enumerate(b.tools):
        if i == 0:
            continue                          # the body owns whatever nothing else claims
        try:
            tm = tessellate_solid(tool, PREVIEW_TOL, PREVIEW_ANG)
            if len(tm.faces) == 0:
                continue
            _, dist, _ = trimesh.proximity.ProximityQuery(tm).on_surface(centres)
        except Exception:
            continue
        owner[dist <= tol] = i
    return owner.tolist()
