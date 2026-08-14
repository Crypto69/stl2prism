"""Rebuild a parametric solid from fitted slabs and export STEP."""
import numpy as np
import cadquery as cq
from .profile_fit import segment_polyline, try_full_circle, snap_profile


def build_solid(slabs, axis, tol=0.08, verbose=True):
    """Union of one extrusion per slab. Profiles fitted as lines/arcs,
    then globally constraint-snapped (radii/centers unified across slabs)."""
    basis = _axis_basis(axis)
    # pass 1: fit every ring
    fitted = []
    for slab in slabs:
        rings = []
        for poly in slab['polygons']:
            outer = _fit_ring(np.array(poly.exterior.coords), tol)
            holes = [_fit_ring(np.array(r.coords), tol) for r in poly.interiors]
            rings.append((outer, holes))
        fitted.append(rings)
    _global_snap(fitted)
    # pass 2: build
    solid = None
    report = []
    for si, (slab, rings) in enumerate(zip(slabs, fitted)):
        h = slab['z1'] - slab['z0']
        wp = cq.Workplane(cq.Plane(
            origin=tuple(np.asarray(axis) * slab['z0']),
            xDir=tuple(basis[:3, 0]),
            normal=tuple(axis)))
        slab_solid = None
        for outer, holes in rings:
            s = _extrude_profile(wp, outer, holes, h)
            slab_solid = s if slab_solid is None else slab_solid.union(s)
        report.append({'slab': si, 'z0': slab['z0'], 'z1': slab['z1'],
                       'profiles': len(rings)})
        if verbose:
            print(f"[build] slab {si}: z {slab['z0']:.2f}..{slab['z1']:.2f} "
                  f"({len(rings)} profile(s))")
        solid = slab_solid if solid is None else solid.union(slab_solid)
    return solid, report


def _iter_arcs(fitted):
    for rings in fitted:
        for outer, holes in rings:
            for ring in [outer] + holes:
                if isinstance(ring, dict):
                    yield ring
                else:
                    for p in ring:
                        if p['type'] == 'arc':
                            yield p


def _global_snap(fitted, radius_tol=0.12, center_tol=0.35):
    fitted_rings_ref = [fitted]
    """Cluster radii and centers across ALL slabs and snap to cluster means.
    This turns facet-noise families like 5.242..5.257 into one radius."""
    arcs = list(_iter_arcs(fitted))
    if not arcs:
        return
    # radii clustering (1D, sort + gap split)
    idx = np.argsort([a['r'] for a in arcs])
    cluster = [arcs[idx[0]]]
    clusters = []
    for i in idx[1:]:
        if arcs[i]['r'] - cluster[-1]['r'] <= radius_tol:
            cluster.append(arcs[i])
        else:
            clusters.append(cluster)
            cluster = [arcs[i]]
    clusters.append(cluster)
    for cl in clusters:
        r = float(np.mean([a['r'] for a in cl]))
        rr = round(r, 1)
        if abs(rr - r) < 0.02:      # snap to 0.1mm grid only when very close
            r = rr
        for a in cl:
            a['r'] = r
    # center clustering (2D greedy)
    done = [False] * len(arcs)
    for i, a in enumerate(arcs):
        if done[i]:
            continue
        grp = [a]
        for j in range(i + 1, len(arcs)):
            if not done[j] and np.linalg.norm(
                    arcs[j]['center'] - a['center']) < center_tol:
                grp.append(arcs[j])
                done[j] = True
        c = np.mean([g['center'] for g in grp], axis=0)
        for g in grp:
            g['center'] = c
    # project arc endpoints onto their snapped circles, then re-close
    # each ring chain (threePointArc refits through points, so endpoints
    # must lie exactly on the snapped circle to preserve the radius)
    for rings in fitted_rings_ref[0]:
        for outer, holes in rings:
            for ring in [outer] + holes:
                if not isinstance(ring, list):
                    continue
                for p in ring:
                    if p['type'] == 'arc':
                        for key in ('p0', 'p1'):
                            v = np.asarray(p[key]) - p['center']
                            L = np.linalg.norm(v)
                            if L > 1e-9:
                                p[key] = p['center'] + v / L * p['r']
                for prev, cur in zip(ring, ring[1:] + ring[:1]):
                    if cur['type'] == 'arc':
                        # arcs own their endpoints; move the line to meet it
                        prev['p1'] = np.asarray(cur['p0'])
                    else:
                        cur['p0'] = np.asarray(prev['p1'])


def _count(w):
    outer, holes = w
    n = len(outer) if isinstance(outer, list) else 1
    for h in holes:
        n += len(h) if isinstance(h, list) else 1
    return n


def _fit_ring(coords, tol):
    """Ring of 2D coords -> full circle dict or list of line/arc prims."""
    pts = np.array(coords)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    circ = try_full_circle(pts, tol)
    if circ:
        return circ
    prims = segment_polyline(pts, tol=tol, closed=True)
    return snap_profile(prims)


def _extrude_profile(wp, outer, holes, h):
    w = _draw(wp, outer)
    for hole in holes:
        w = _draw(w, hole)
    return w.extrude(h)


def _draw(wp, ring):
    if isinstance(ring, dict) and ring['type'] == 'arc':  # full circle
        c = ring['center']
        return wp.moveTo(float(c[0]), float(c[1])).circle(float(ring['r']))
    # chain of prims
    start = ring[0]['p0']
    w = wp.moveTo(float(start[0]), float(start[1]))
    for p in ring:
        if p['type'] == 'line':
            w = w.lineTo(float(p['p1'][0]), float(p['p1'][1]))
        else:  # arc through mid point for robustness
            mid = _arc_mid(p)
            w = w.threePointArc(
                (float(mid[0]), float(mid[1])),
                (float(p['p1'][0]), float(p['p1'][1])))
    return w.close()


def _arc_mid(p):
    c, r = p['center'], p['r']
    a0 = np.arctan2(p['p0'][1] - c[1], p['p0'][0] - c[0])
    a1 = np.arctan2(p['p1'][1] - c[1], p['p1'][0] - c[0])
    if p.get('ccw', True):
        while a1 <= a0:
            a1 += 2 * np.pi
    else:
        while a1 >= a0:
            a1 -= 2 * np.pi
    am = (a0 + a1) / 2
    return c + r * np.array([np.cos(am), np.sin(am)])


def _axis_basis(axis):
    z = np.asarray(axis, float)
    z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    return T


def export_step(solid, path):
    cq.exporters.export(solid, path)


def faceted_fallback(mesh, path, angular_tol=5e-3):
    """Guaranteed output: sew triangles, unify coplanar faces, write STEP."""
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid)
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.STEPControl import STEPControl_Writer, STEPControl_ManifoldSolidBrep
    from OCP.Interface import Interface_Static

    sew = BRepBuilderAPI_Sewing(1e-3)
    V = mesh.vertices
    for tri in mesh.faces:
        p = BRepBuilderAPI_MakePolygon()
        for vi in tri:
            p.Add(gp_Pnt(*map(float, V[vi])))
        p.Close()
        f = BRepBuilderAPI_MakeFace(p.Wire())
        if f.IsDone():
            sew.Add(f.Face())
    sew.Perform()
    exp = TopExp_Explorer(sew.SewedShape(), TopAbs_SHELL)
    shell = TopoDS.Shell_s(exp.Current())
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    fx = ShapeFix_Solid(solid); fx.Perform(); solid = fx.Solid()
    up = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
    up.SetLinearTolerance(1e-4); up.SetAngularTolerance(angular_tol)
    up.Build()
    exp = TopExp_Explorer(up.Shape(), TopAbs_SHELL)
    shell2 = TopoDS.Shell_s(exp.Current())
    solid2 = BRepBuilderAPI_MakeSolid(shell2).Solid()
    fx2 = ShapeFix_Solid(solid2); fx2.Perform(); solid2 = fx2.Solid()
    w = STEPControl_Writer()
    Interface_Static.SetCVal_s('write.step.schema', 'AP214')
    w.Transfer(solid2, STEPControl_ManifoldSolidBrep)
    w.Write(path)
