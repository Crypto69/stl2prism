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
    # Radii clustering (1D, sort + gap split). The spread of a cluster is
    # capped as well as the gap between neighbours: comparing only against the
    # previous member lets radii chain (5.0, 5.1, ... 5.4 all within 0.12 of
    # their predecessor) and collapse to one mean, silently resizing holes by
    # far more than radius_tol.
    idx = np.argsort([a['r'] for a in arcs])
    cluster = [arcs[idx[0]]]
    clusters = []
    for i in idx[1:]:
        r = arcs[i]['r']
        if (r - cluster[-1]['r'] <= radius_tol
                and r - cluster[0]['r'] <= radius_tol):
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


class FacetedError(RuntimeError):
    """Faceted export could not produce a solid representing the mesh."""


def _count(shape, kind):
    from OCP.TopExp import TopExp_Explorer
    exp = TopExp_Explorer(shape, kind)
    n = 0
    while exp.More():
        n += 1
        exp.Next()
    return n


def _naked_edges(shape):
    """Edges bounded by fewer than two faces — a shell with any of these is
    not closed, whatever the STEP file calls it."""
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return sum(1 for i in range(1, m.Extent() + 1)
               if m.FindFromIndex(i).Extent() < 2)


def _volume(shape):
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    return abs(g.Mass())


def _largest_solid(shape, what):
    """Build a solid per shell and return the one enclosing the most volume.

    Sewing a non-watertight mesh yields many disjoint shells. Taking whichever
    the explorer happens to surface first can silently export a two-triangle
    sliver while reporting success, so pick deliberately and report the drop.
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp

    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    best, best_vol, best_faces, n_shells, total_faces = None, -1.0, 0, 0, 0
    while exp.More():
        shell = TopoDS.Shell_s(exp.Current())
        n_shells += 1
        total_faces += _count(shell, TopAbs_FACE)
        solid = BRepBuilderAPI_MakeSolid(shell).Solid()
        fx = ShapeFix_Solid(solid)
        fx.Perform()
        solid = fx.Solid()
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(solid, g)
        vol = abs(g.Mass())
        if vol > best_vol:
            best, best_vol = solid, vol
            best_faces = _count(solid, TopAbs_FACE)
        exp.Next()
    if best is None:
        raise FacetedError(f"{what}: sewing produced no shell")
    return best, best_vol, best_faces, n_shells, total_faces


def faceted_fallback(mesh, path, angular_tol=5e-3, min_face_frac=0.5,
                     verbose=True):
    """Sew triangles, unify coplanar faces, write STEP.

    Returns stats describing what was actually written. Raises FacetedError
    rather than emitting a fragment that would pass as a valid STEP file.
    """
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing)
    from OCP.TopAbs import TopAbs_FACE
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.STEPControl import STEPControl_Writer, STEPControl_ManifoldSolidBrep
    from OCP.Interface import Interface_Static

    sew = BRepBuilderAPI_Sewing(1e-3)
    V = mesh.vertices
    n_added = 0
    for tri in mesh.faces:
        p = BRepBuilderAPI_MakePolygon()
        for vi in tri:
            p.Add(gp_Pnt(*map(float, V[vi])))
        p.Close()
        f = BRepBuilderAPI_MakeFace(p.Wire())
        if f.IsDone():
            sew.Add(f.Face())
            n_added += 1
    sew.Perform()
    free_edges = sew.NbFreeEdges()

    solid, _, kept_faces, n_shells, _ = _largest_solid(sew.SewedShape(), 'sew')
    if n_shells > 1 and verbose:
        print(f"[faceted] sewing produced {n_shells} shells "
              f"({free_edges} free edges); keeping the largest by volume")
    frac = kept_faces / n_added if n_added else 0.0
    if frac < min_face_frac:
        raise FacetedError(
            f"largest sewn shell holds {kept_faces} of {n_added} faces "
            f"({frac:.1%}); mesh is too fragmented to export as one solid")

    from OCP.BRepCheck import BRepCheck_Analyzer
    # Not fatal: a scan that no repair route could close still sews into a
    # usable open shell, and refusing to write it helps nobody. The face-count
    # gate above is what guards against genuine garbage.
    valid = BRepCheck_Analyzer(solid).IsValid()
    if not valid and verbose:
        print('[faceted] warning: sewn shape failed BRepCheck validation')
    naked = _naked_edges(solid)

    # Coplanar-face merging is cosmetic; it must not cost closure. On dense
    # scan tessellation it can leave naked edges, which makes the written
    # CLOSED_SHELL a lie and re-reads as a shell rather than a solid.
    up = ShapeUpgrade_UnifySameDomain(solid, True, True, True)
    up.SetLinearTolerance(1e-4)
    up.SetAngularTolerance(angular_tol)
    up.Build()
    solid2, vol, faces2, _, _ = _largest_solid(up.Shape(), 'unify')
    naked2 = _naked_edges(solid2)
    if naked2 > naked:
        if verbose:
            print(f"[faceted] coplanar merge opened {naked2 - naked} edge(s); "
                  f"keeping the unmerged solid ({kept_faces} faces)")
        solid2, faces2, naked2 = solid, kept_faces, naked
        vol = _volume(solid2)

    if naked2 > 0 and verbose:
        # Worth writing — an open shell still imports — but it must not be
        # reported as a solid: OCC re-reads it as a shell, and the caller
        # needs to know that before trusting a volume from it.
        print(f"[faceted] warning: {naked2} naked edge(s) remain; the result "
              f"is an open shell, not a closed solid")

    w = STEPControl_Writer()
    Interface_Static.SetCVal_s('write.step.schema', 'AP214')
    w.Transfer(solid2, STEPControl_ManifoldSolidBrep)
    w.Write(path)
    return {'faces_in': n_added, 'faces_out': faces2, 'shells': n_shells,
            'free_edges': free_edges, 'naked_edges': naked2, 'volume': vol,
            'is_solid': bool(valid and naked2 == 0)}
