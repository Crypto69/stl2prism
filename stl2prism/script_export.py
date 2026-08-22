"""Emit an editable CadQuery script that rebuilds the recognised solid.

Fusion's Prismatic conversion returns a plain B-rep; this tool knows the
sketch + extrude structure it recognised, so it can hand back a program:
one workplane per slab, lines/arcs/circles for the profile, extrude or
ruled loft, cylinder/cone cuts for cross-axis holes and countersinks, void
subtraction. Numbers are rounded to 4 decimals; recurring radii and slab
heights are hoisted into named parameters at the top so the part can be
edited by changing a value.
"""
import numpy as np


def _f(x, nd=4):
    v = round(float(x), nd)
    if v == 0:
        v = 0.0
    return repr(v)


def _vec(v, nd=6):
    return '(' + ', '.join(_f(c, nd) for c in np.asarray(v, float)) + ')'


class _Params:
    """Collects named parameters (radii, heights) with de-duplication."""

    def __init__(self):
        self.items = {}      # name -> value
        self.by_value = {}   # (kind, rounded value) -> name

    def get(self, kind, value, nd=4):
        key = (kind, round(float(value), nd))
        if key in self.by_value:
            return self.by_value[key]
        name = f'{kind}_{sum(1 for k in self.by_value if k[0] == kind) + 1}'
        self.by_value[key] = name
        self.items[name] = round(float(value), nd)
        return name

    def lines(self):
        return [f'{k} = {v}' for k, v in self.items.items()]


def _ring_code(ring, params, indent='    '):
    """Chained CadQuery calls (starting after a Workplane) for one ring."""
    if isinstance(ring, dict):                       # full circle
        c = ring['center']
        r = params.get('R', ring['r'])
        return [f"{indent}.moveTo({_f(c[0])}, {_f(c[1])}).circle({r})"]
    out = []
    p0 = ring[0]['p0']
    out.append(f"{indent}.moveTo({_f(p0[0])}, {_f(p0[1])})")
    for p in ring:
        if p['type'] == 'line':
            out.append(f"{indent}.lineTo({_f(p['p1'][0])}, {_f(p['p1'][1])})")
        else:
            from .rebuild import _arc_mid
            mid = _arc_mid(p)
            out.append(f"{indent}.threePointArc(({_f(mid[0])}, {_f(mid[1])}), "
                       f"({_f(p['p1'][0])}, {_f(p['p1'][1])}))")
    out.append(f"{indent}.close()")
    return out


def body_script(info, params, body_name='body', indent=''):
    """Lines of code building `body_name` from a build-info dict:
    {'axis', 'xdir', 'slabs': [{'z0','z1','rings','kind','loft'}],
     'cross_cyls': [...], 'cones': [...], 'voids': [info...]}"""
    L = []
    axis = np.asarray(info['axis'], float)
    xdir = np.asarray(info['xdir'], float)
    L.append(f"{indent}# extrusion axis and in-plane X direction (workplane frame)")
    L.append(f"{indent}AXIS = {_vec(axis)}")
    L.append(f"{indent}XDIR = {_vec(xdir)}")
    L.append(f"{indent}def plane_at(h):")
    L.append(f"{indent}    return cq.Plane(origin=tuple(np.array(AXIS) * h), xDir=XDIR, normal=AXIS)")
    L.append("")
    L.append(f"{indent}{body_name} = None")
    for si, slab in enumerate(info['slabs']):
        z0, z1 = slab['z0'], slab['z1']
        h = params.get('H', z1 - z0)
        L.append(f"{indent}# slab {si}: h = {_f(z0)} .. {_f(z1)} ({slab.get('kind', 'extrude')})")
        if slab.get('kind') == 'loft' and slab.get('loft'):
            ra, rb = slab['loft'][0], slab['loft'][1]
            L.append(f"{indent}_wp = cq.Workplane(plane_at({_f(z0)}))")
            for pi, ((oa, ha), (ob, hb)) in enumerate(zip(ra, rb)):
                L.append(f"{indent}_s = (_wp")
                L += _ring_code(oa, params, indent + '      ')
                L.append(f"{indent}      .workplane(offset={h})")
                L += _ring_code(ob, params, indent + '      ')
                L.append(f"{indent}      .loft(ruled=True))")
                for x, y in zip(ha, hb):
                    L.append(f"{indent}_h = (_wp.workplane(offset=-0.02)")
                    L += _ring_code(x, params, indent + '      ')
                    L.append(f"{indent}      .workplane(offset={h} + 0.04)")
                    L += _ring_code(y, params, indent + '      ')
                    L.append(f"{indent}      .loft(ruled=True))")
                    L.append(f"{indent}_s = _s.cut(_h)")
                L.append(f"{indent}{body_name} = _s if {body_name} is None else {body_name}.union(_s, tol=1e-4)")
        else:
            for pi, (outer, holes) in enumerate(slab['rings']):
                L.append(f"{indent}_s = (cq.Workplane(plane_at({_f(z0)}))")
                L += _ring_code(outer, params, indent + '      ')
                for hring in holes:
                    L += _ring_code(hring, params, indent + '      ')
                L.append(f"{indent}      .extrude({h}))")
                L.append(f"{indent}{body_name} = _s if {body_name} is None else {body_name}.union(_s, tol=1e-4)")
    for c in info.get('cross_cyls', []):
        ax = np.asarray(c['axis'], float)
        bx, by = c['basis']
        c3 = c['center2'][0] * np.asarray(bx) + c['center2'][1] * np.asarray(by)
        blind = c.get('blind', [False, False])
        e0 = 0.0 if blind[0] else 1.0
        e1 = 0.0 if blind[1] else 1.0
        p0 = c3 + ax * (c['h0'] - e0)
        depth = (c['h1'] - c['h0']) + e0 + e1
        r = params.get('R', c['r'])
        L.append(f"{indent}# cross-axis hole r={_f(c['r'])} ({'through' if e0 and e1 else 'blind'})")
        L.append(f"{indent}{body_name} = {body_name}.cut(cq.Workplane(cq.Plane(origin={_vec(p0)}, normal={_vec(ax)}))"
                 f".circle({r}).extrude({_f(depth)}), tol=1e-4)")
    for c in info.get('cones', []):
        ax = np.asarray(c['axis'], float)
        apex = np.asarray(c['apex'], float)
        t = np.tan(c['half_angle'])
        a0, a1 = max(c['h0'] - 0.0, 0.0), c['h1'] + 1.0
        r0, r1 = a0 * t, a1 * t
        pnt = apex + ax * a0
        L.append(f"{indent}# cross-axis cone (countersink) half-angle {_f(np.degrees(c['half_angle']), 1)} deg")
        L.append(f"{indent}{body_name} = {body_name}.cut(cq.Workplane('XY').newObject([cq.Solid.makeCone({_f(r0)}, {_f(r1)}, "
                 f"{_f(a1 - a0)}, cq.Vector{_vec(pnt)}, cq.Vector{_vec(ax)})]), tol=1e-4)")
    return L


def emit_script(bodies_info, out_step_name='part.step'):
    """Full script text for a list of body infos (each may carry
    'voids': [info, ...] and 'mode' / 'note')."""
    params = _Params()
    chunks = []
    names = []
    for i, info in enumerate(bodies_info):
        name = f'body{i + 1}'
        names.append(name)
        if info.get('mode') != 'prismatic' or 'slabs' not in info:
            chunks.append([f"# {name}: {info.get('note', 'not recognised as an extrusion; see the STEP')}",
                           f"{name} = None"])
            continue
        lines = body_script(info, params, body_name=name)
        for k, vinfo in enumerate(info.get('voids', [])):
            if vinfo.get('mode') == 'prismatic' and 'slabs' in vinfo:
                vname = f'{name}_void{k + 1}'
                lines.append(f"# internal cavity {k + 1}")
                lines += body_script(vinfo, params, body_name=vname)
                lines.append(f"{name} = {name}.cut({vname}, tol=1e-4)")
        chunks.append(lines)
    head = [
        '"""Generated by stl2prism: the extrusion structure recognised in the mesh,',
        'as an editable CadQuery program. Edit the parameters below and re-run',
        'to regenerate the STEP."""',
        'import numpy as np',
        'import cadquery as cq',
        '',
        '# ---- parameters (radii R_n, slab heights H_n) ----',
    ]
    head += params.lines()
    head.append('')
    body = []
    for ch in chunks:
        body += ch
        body.append('')
    tail = [
        'bodies = [b for b in [' + ', '.join(names) + '] if b is not None]',
        "assert bodies, 'no body in this file was recognised as an extrusion'",
        'result = bodies[0]',
        'for b in bodies[1:]:',
        '    result = result.add(b)',
        f"cq.exporters.export(result, {out_step_name!r})",
        "print('wrote', " + repr(out_step_name) + ")",
    ]
    return '\n'.join(head + body + tail) + '\n'
