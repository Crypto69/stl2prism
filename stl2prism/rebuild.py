"""Rebuild a parametric solid from fitted slabs and export STEP."""
import numpy as np
import trimesh
import cadquery as cq
from .profile_fit import (segment_polyline, try_full_circle, snap_profile,
                          solve_junctions, refine_arcs_with_points)


def build_solid(slabs, axis, tol=0.08, verbose=True, mesh=None):
    """Union of one extrusion per slab. Profiles fitted as lines/arcs,
    refined against the mesh vertices (which lie exactly on the CAD
    surface), then globally constraint-snapped (radii/centers unified
    across slabs) and their junctions solved exactly.

    Consecutive slabs with identical profiles are merged into one extrusion
    (fewer Booleans, no seam faces); slabs are fused with glue since they
    only touch."""
    basis = _axis_basis(axis)
    # pass 1: fit every ring. A slab whose section varies (taper, draft,
    # chamfer, countersink) is fitted at BOTH ends and lofted; a constant
    # slab is fitted at its mid section and extruded.
    fitted = []
    lofts = []
    # all mesh vertices in the section frame: a thin slab (a 0.5 mm ledge)
    # holds too few of its own to refine an arc that runs through the whole
    # part, so refinement falls back to every vertex near the circle
    pts_all = None
    if mesh is not None:
        Vm = np.asarray(mesh.vertices, float)
        pts_all = np.column_stack([Vm @ basis[:3, 0], Vm @ basis[:3, 1]])
    for slab in slabs:
        pts2d = _slab_vertices_2d(mesh, axis, basis, slab) if mesh is not None else None
        rings = _fit_polys(slab['polygons'], tol, pts2d, pts_all)
        fitted.append(rings)
        loft = None
        from .extrusion import _loftable
        if not slab.get('constant', True) and slab.get('ends') and _loftable(slab):
            (za, pa), (zb, pb) = slab['ends']
            if pa and pb and len(pa) == len(pb):
                ra = _fit_polys(pa, tol, pts2d)
                rb_ = _fit_polys(pb, tol, pts2d)
                if _compatible(ra, rb_):
                    loft = (ra, rb_, za, zb)
        lofts.append(loft)
    # global radius/centre clustering over the constant (extruded) slabs
    # only: a taper's end radii must not be averaged into them
    _global_snap(fitted)
    for rings in fitted + [r for lf in lofts if lf for r in lf[:2]]:
        for outer, holes in rings:
            for ring in [outer] + holes:
                if isinstance(ring, list):
                    solve_junctions(ring, closed=True)
                    _drop_degenerate(ring)
    for ring in _align_arcs_across_slabs(fitted):
        solve_junctions(ring, closed=True, freeze_arcs=True)
        _drop_degenerate(ring)
    # merge consecutive slabs whose fitted profiles are identical
    merged = _merge_equal_slabs(slabs, fitted, lofts)
    # pass 2: build
    solid = None
    report = []
    for si, (z0, z1, rings, loft) in enumerate(merged):
        h = z1 - z0
        wp = cq.Workplane(cq.Plane(
            origin=tuple(np.asarray(axis) * z0),
            xDir=tuple(basis[:3, 0]),
            normal=tuple(axis)))
        slab_solid = None
        kind = 'extrude'
        if loft is not None:
            try:
                ra, rb_, za, zb = loft
                # the end sections were taken slightly inside the slab;
                # extrapolate the (linear) profiles to the true slab ends
                ta = (z0 - za) / (zb - za)
                tb = (z1 - za) / (zb - za)
                ra_e = _lerp_rings(ra, rb_, ta)
                rb_e = _lerp_rings(ra, rb_, tb)
                # continuity with the neighbouring extruded slabs: where the
                # lofted end matches the neighbour's fitted profile within
                # tolerance, take the neighbour's numbers exactly (no
                # sliver step faces at the interface)
                if si > 0 and merged[si - 1][3] is None:
                    ra_e = _snap_rings_to(ra_e, merged[si - 1][2], tol)
                if si + 1 < len(merged) and merged[si + 1][3] is None:
                    rb_e = _snap_rings_to(rb_e, merged[si + 1][2], tol)
                slab_solid = _loft_slab(wp, ra_e, rb_e, h)
                kind = 'loft'
            except Exception as e:
                if verbose:
                    print(f"[build] slab {si}: loft failed ({type(e).__name__}); extruding")
                slab_solid = None
        if slab_solid is None:
            for outer, holes in rings:
                s = _extrude_profile(wp, outer, holes, h)
                slab_solid = s if slab_solid is None else slab_solid.union(s, tol=FUZZY)
        report.append({'slab': si, 'z0': z0, 'z1': z1, 'profiles': len(rings),
                       'rings': rings, 'kind': kind,
                       'loft': loft})
        if verbose:
            print(f"[build] slab {si}: z {z0:.2f}..{z1:.2f} "
                  f"({len(rings)} profile(s), {kind})")
        solid = slab_solid if solid is None else solid.union(slab_solid, tol=FUZZY)
    solid = finish_solid(solid, verbose=verbose)
    return solid, report


def _fit_polys(polys, tol, pts2d, pts_all=None):
    rings = []
    for poly in polys:
        outer = _fit_ring(np.array(poly.exterior.coords), tol, pts2d, pts_all)
        holes = [_fit_ring(np.array(r.coords), tol, pts2d, pts_all) for r in poly.interiors]
        rings.append((outer, holes))
    return rings


def _ring_types(ring):
    if isinstance(ring, dict):
        return 'O'
    return ''.join('L' if p['type'] == 'line' else 'A' for p in ring)


def _compatible(ra, rb):
    """Two fitted sections can be lofted edge-to-edge: same number of
    profiles, each with the same number of holes and the same primitive
    sequence (up to cyclic rotation)."""
    if len(ra) != len(rb):
        return False
    for (oa, ha), (ob, hb) in zip(ra, rb):
        if len(ha) != len(hb):
            return False
        for x, y in [(oa, ob)] + list(zip(ha, hb)):
            tx, ty = _ring_types(x), _ring_types(y)
            if tx == 'O' or ty == 'O':
                if tx != ty:
                    return False
                continue
            if len(tx) != len(ty) or tx not in ty + ty:
                return False
    return True


def _align_ring(ref, ring):
    """Rotate `ring` (list of prims) so its first primitive corresponds to
    ref's first primitive: same type sequence and nearest start point."""
    if isinstance(ring, dict) or isinstance(ref, dict):
        return ring
    tr = _ring_types(ref)
    n = len(ring)
    best = None
    for k in range(n):
        rot = ring[k:] + ring[:k]
        if _ring_types(rot) != tr:
            continue
        d = np.linalg.norm(np.asarray(rot[0]['p0'], float) - np.asarray(ref[0]['p0'], float))
        if best is None or d < best[0]:
            best = (d, rot)
    return best[1] if best else ring


def _loft_slab(wp, rings_a, rings_b, h):
    """Ruled loft from the profiles at the slab bottom to those at the
    top; holes are lofted separately and cut. Ruled lofts between matched
    lines give planes, between matched arcs cones — a linear taper,
    chamfer or countersink becomes exact analytic geometry."""
    solid = None
    for (oa, ha), (ob, hb) in zip(rings_a, rings_b):
        ob = _align_ring(oa, ob)
        body = _loft_ring(wp, oa, ob, h)
        for x, y in zip(ha, hb):
            y = _align_ring(x, y)
            hole = _loft_ring(wp, x, y, h, grow=0.02)
            body = body.cut(hole, tol=FUZZY)
        solid = body if solid is None else solid.union(body, tol=FUZZY)
    return solid


def _loft_ring(wp, ring_a, ring_b, h, grow=0.0):
    """Loft one wire pair. Analytic construction first (planes between
    matched lines, cones/cylinders between matched arcs or full circles),
    B-spline ruled loft as the fallback. `grow` extends a hole loft
    slightly past both ends so the cut is clean."""
    try:
        s = _loft_ring_analytic(wp.plane, ring_a, ring_b, h, grow)
        if s is not None:
            return s
    except Exception:
        pass
    if grow:
        wp0 = wp.workplane(offset=-grow)
        w = _draw(wp0, ring_a)
        w = _draw(w.workplane(offset=h + 2 * grow), ring_b)
    else:
        w = _draw(wp, ring_a)
        w = _draw(w.workplane(offset=h), ring_b)
    return w.loft(combine=True, ruled=True)


def _loft_ring_analytic(plane, ring_a, ring_b, h, grow=0.0):
    """Ruled solid between two fitted rings at heights 0 and h of `plane`
    (extended by `grow` at both ends), built from analytic faces."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.ShapeFix import ShapeFix_Face, ShapeFix_Solid
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.BRepCheck import BRepCheck_Analyzer
    za, zb = -grow, h + grow

    def W(x, y, z):
        return plane.toWorldCoords(cq.Vector(float(x), float(y), float(z)))

    # full circles: cone frustum / cylinder
    if isinstance(ring_a, dict) and isinstance(ring_b, dict):
        ca, ra = np.asarray(ring_a['center'], float), float(ring_a['r'])
        cb, rb = np.asarray(ring_b['center'], float), float(ring_b['r'])
        if np.linalg.norm(ca - cb) > 0.05:
            return None
        if grow:
            t = (rb - ra) / h
            ra_, rb_ = ra - t * grow, rb + t * grow
        else:
            ra_, rb_ = ra, rb
        pnt = W(ca[0], ca[1], za)
        d = plane.zDir
        if abs(ra_ - rb_) < 1e-6:
            return cq.Workplane('XY').newObject([cq.Solid.makeCylinder(ra_, zb - za, pnt, d)])
        return cq.Workplane('XY').newObject([cq.Solid.makeCone(ra_, rb_, zb - za, pnt, d)])
    if isinstance(ring_a, dict) or isinstance(ring_b, dict):
        return None
    if len(ring_a) != len(ring_b):
        return None
    if grow:
        # extrapolate both rings along the ruling direction: p(z) linear
        t0 = -grow / h
        t1 = 1 + grow / h
        ra_ = _lerp_ring(ring_a, ring_b, t0)
        rb_ = _lerp_ring(ring_a, ring_b, t1)
    else:
        ra_, rb_ = ring_a, ring_b
    faces = []
    for pa, pb in zip(ra_, rb_):
        if pa['type'] != pb['type']:
            return None
        A0, A1 = W(*pa['p0'], za), W(*pa['p1'], za)
        B0, B1 = W(*pb['p0'], zb), W(*pb['p1'], zb)
        if pa['type'] == 'line':
            f = _quad_face(A0, A1, B1, B0)
        else:
            f = _cone_face(plane, pa, pb, za, zb, A0, A1, B0, B1)
        if f is None:
            return None
        faces.append(f)
    # caps
    bottom = _cap_face(plane, ra_, za)
    top = _cap_face(plane, rb_, zb)
    if bottom is None or top is None:
        return None
    faces += [bottom, top]
    sew = BRepBuilderAPI_Sewing(1e-3)
    for f in faces:
        sew.Add(f.wrapped)
    sew.Perform()
    sewed = sew.SewedShape()
    exp = TopExp_Explorer(sewed, TopAbs_SHELL)
    if not exp.More():
        return None
    shell = TopoDS.Shell_s(exp.Current())
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    fx = ShapeFix_Solid(solid)
    fx.Perform()
    solid = fx.Solid()
    if not BRepCheck_Analyzer(solid).IsValid():
        return None
    out = cq.Shape.cast(solid)
    if out.Volume() <= 0:
        return None
    return cq.Workplane('XY').newObject([out])


def _snap_rings_to(rings, ref_rings, tol):
    """Copy primitives of `ref_rings` onto `rings` where they coincide
    within 2*tol (circles: centre and radius; chains: matched primitives'
    endpoints, centres and radii)."""
    band = 2 * tol
    out = []
    for o, hs in rings:
        o2 = _snap_ring_to(o, ref_rings, band)
        out.append((o2, [_snap_ring_to(h, ref_rings, band) for h in hs]))
    return out


def _snap_ring_to(ring, ref_rings, band):
    refs = [r for o, hs in ref_rings for r in [o] + hs]
    if isinstance(ring, dict):
        for r in refs:
            if isinstance(r, dict) and abs(r['r'] - ring['r']) < band \
                    and np.linalg.norm(np.asarray(r['center']) - np.asarray(ring['center'])) < band:
                c = dict(ring)
                c['center'] = np.asarray(r['center'], float).copy()
                c['r'] = float(r['r'])
                return c
        return ring
    for r in refs:
        if isinstance(r, dict) or len(r) != len(ring):
            continue
        r_al = _align_ring(ring, r)
        if _ring_types(r_al) != _ring_types(ring):
            continue
        close = all(np.linalg.norm(np.asarray(a['p0'], float) - np.asarray(b['p0'], float)) < band
                    and np.linalg.norm(np.asarray(a['p1'], float) - np.asarray(b['p1'], float)) < band
                    for a, b in zip(ring, r_al))
        if close:
            out = []
            for a, b in zip(ring, r_al):
                c = dict(a)
                c['p0'] = np.asarray(b['p0'], float).copy()
                c['p1'] = np.asarray(b['p1'], float).copy()
                if a['type'] == 'arc':
                    c['center'] = np.asarray(b['center'], float).copy()
                    c['r'] = float(b['r'])
                out.append(c)
            return out
    return ring


def _lerp_rings(RA, RB, t):
    """Interpolate/extrapolate whole fitted sections (list of (outer,
    holes)); rings are aligned first so primitives correspond."""
    out = []
    for (oa, ha), (ob, hb) in zip(RA, RB):
        ob = _align_ring(oa, ob)
        o = _lerp_one(oa, ob, t)
        hs = []
        for x, y in zip(ha, hb):
            y = _align_ring(x, y)
            hs.append(_lerp_one(x, y, t))
        out.append((o, hs))
    return out


def _lerp_one(a, b, t):
    if isinstance(a, dict) and isinstance(b, dict):
        c = dict(a)
        c['center'] = (1 - t) * np.asarray(a['center'], float) + t * np.asarray(b['center'], float)
        c['r'] = (1 - t) * a['r'] + t * b['r']
        return c
    if isinstance(a, dict) or isinstance(b, dict) or len(a) != len(b):
        return a if t < 0.5 else b
    return _lerp_ring(a, b, t)


def _lerp_ring(ra, rb, t):
    """Linear interpolation/extrapolation of two matched rings."""
    out = []
    for pa, pb in zip(ra, rb):
        p = dict(pa)
        for k in ('p0', 'p1'):
            p[k] = (1 - t) * np.asarray(pa[k], float) + t * np.asarray(pb[k], float)
        if pa['type'] == 'arc':
            p['center'] = (1 - t) * np.asarray(pa['center'], float) + t * np.asarray(pb['center'], float)
            p['r'] = (1 - t) * pa['r'] + t * pb['r']
        out.append(p)
    return out


def _quad_face(A0, A1, B1, B0):
    """Planar face through four (near-)coplanar points; the last two are
    projected onto the plane of the first three so the face is exact."""
    P = np.array([[v.x, v.y, v.z] for v in (A0, A1, B1, B0)])
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    dev = np.abs((P - c) @ n).max()
    if dev > 0.05:
        return None
    Pp = P - np.outer((P - c) @ n, n)
    pts = [cq.Vector(*q) for q in Pp]
    wire = cq.Wire.makePolygon(pts + [pts[0]])
    return cq.Face.makeFromWires(wire)


def _cone_face(plane, pa, pb, za, zb, A0, A1, B0, B1):
    """Conical (or cylindrical) face between two matched coaxial arcs."""
    from OCP.Geom import Geom_ConicalSurface, Geom_CylindricalSurface
    from OCP.gp import gp_Ax3, gp_Pnt, gp_Dir
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.ShapeFix import ShapeFix_Face
    ca, ra = np.asarray(pa['center'], float), float(pa['r'])
    cb, rb = np.asarray(pb['center'], float), float(pb['r'])
    if np.linalg.norm(ca - cb) > 0.05:
        return None
    c = 0.5 * (ca + cb)
    apex_pt = plane.toWorldCoords(cq.Vector(float(c[0]), float(c[1]), float(za)))
    d = plane.zDir
    ax3 = gp_Ax3(gp_Pnt(apex_pt.x, apex_pt.y, apex_pt.z), gp_Dir(d.x, d.y, d.z))
    hh = zb - za
    if abs(rb - ra) < 1e-6:
        surf = Geom_CylindricalSurface(ax3, ra)
    else:
        semi = np.arctan2(rb - ra, hh)
        surf = Geom_ConicalSurface(ax3, float(semi), ra)
    # boundary wire: arc a, ruling A1->B1, arc b reversed, ruling B0->A0
    ma = _arc_mid(pa)
    mb = _arc_mid(pb)
    MA = plane.toWorldCoords(cq.Vector(float(ma[0]), float(ma[1]), float(za)))
    MB = plane.toWorldCoords(cq.Vector(float(mb[0]), float(mb[1]), float(zb)))
    e1 = cq.Edge.makeThreePointArc(A0, MA, A1)
    e2 = cq.Edge.makeLine(A1, B1)
    e3 = cq.Edge.makeThreePointArc(B1, MB, B0)
    e4 = cq.Edge.makeLine(B0, A0)
    wire = cq.Wire.assembleEdges([e1, e2, e3, e4])
    mk = BRepBuilderAPI_MakeFace(surf, wire.wrapped, True)
    if not mk.IsDone():
        return None
    face = mk.Face()
    fx = ShapeFix_Face(face)
    fx.Perform()
    return cq.Face(fx.Face())


def _cap_face(plane, ring, z):
    """Planar cap bounded by the ring at height z."""
    def W(x, y):
        return plane.toWorldCoords(cq.Vector(float(x), float(y), float(z)))
    edges = []
    for p in ring:
        A, B = W(*p['p0']), W(*p['p1'])
        if p['type'] == 'line':
            edges.append(cq.Edge.makeLine(A, B))
        else:
            m = _arc_mid(p)
            edges.append(cq.Edge.makeThreePointArc(A, W(*m), B))
    wire = cq.Wire.assembleEdges(edges)
    return cq.Face.makeFromWires(wire)


# Fuzzy tolerance for Booleans between fitted slabs. Consecutive slabs share
# an end plane, but their boundary curves differ by fit noise; below this
# OCC sees two touching solids and leaves both end faces inside the fuse.
FUZZY = 1e-4


def finish_solid(solid, verbose=True):
    """Cleanliness pass on a CadQuery Workplane solid: make sure it is ONE
    solid (re-fuse with a coarser fuzzy value if not), drop micro-edges,
    unify same-domain faces (guarded: must stay valid and keep volume),
    check validity."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.ShapeFix import ShapeFix_Wireframe
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    shape = solid.val()
    solids = shape.Solids()
    if len(solids) > 1:
        # touching pieces that the per-slab fuse left apart
        merged = solids[0].fuse(*solids[1:], tol=1e-3).clean()
        if len(merged.Solids()) == 1 and abs(merged.Volume() - shape.Volume()) < 1e-6 * max(1.0, shape.Volume()):
            shape = merged
        elif verbose:
            print(f"[build] warning: result holds {len(solids)} touching solids "
                  f"that would not fuse")
    vol0 = shape.Volume()
    # micro-edges from junction solving / Booleans
    try:
        wf = ShapeFix_Wireframe(shape.wrapped)
        wf.SetPrecision(1e-4)
        wf.SetMaxTolerance(1e-3)
        wf.ModeDropSmallEdges = True
        wf.FixSmallEdges()
        wf.FixWireGaps()
        fixed = cq.Shape.cast(wf.Shape())
        if BRepCheck_Analyzer(fixed.wrapped).IsValid() and abs(fixed.Volume() - vol0) < 1e-4 * max(1.0, vol0):
            shape = fixed
    except Exception:
        pass
    try:
        up = ShapeUpgrade_UnifySameDomain(shape.wrapped, True, True, True)
        up.SetLinearTolerance(1e-4)
        up.SetAngularTolerance(1e-4)
        up.Build()
        uni = cq.Shape.cast(up.Shape())
        if (BRepCheck_Analyzer(uni.wrapped).IsValid()
                and abs(uni.Volume() - vol0) < 1e-4 * max(1.0, vol0)
                and len(uni.Faces()) <= len(shape.Faces())):
            shape = uni
    except Exception:
        pass
    if verbose and not BRepCheck_Analyzer(shape.wrapped).IsValid():
        print('[build] warning: solid failed BRepCheck validation')
    return cq.Workplane('XY').newObject([shape])


def _slab_vertices_2d(mesh, axis, basis, slab, eps=1e-6):
    """Mesh vertices whose height lies in the slab, projected into the
    section frame (used to refine arc radii/centres)."""
    V = mesh.vertices
    h = V @ np.asarray(axis, float)
    sel = (h >= slab['z0'] - eps) & (h <= slab['z1'] + eps)
    if not sel.any():
        return None
    P = V[sel]
    return np.column_stack([P @ basis[:3, 0], P @ basis[:3, 1]])


def _drop_degenerate(ring, eps=1e-6):
    """Remove zero-length lines / zero-sweep arcs left by junction solving."""
    keep = []
    for p in ring:
        L = np.linalg.norm(np.asarray(p['p1'], float) - np.asarray(p['p0'], float))
        if L > eps:
            keep.append(p)
    if len(keep) != len(ring):
        ring[:] = keep
        # re-close after dropping
        for prev, cur in zip(ring, ring[1:] + ring[:1]):
            cur['p0'] = np.asarray(prev['p1'], float).copy()


def _ring_signature(ring, nd=4):
    if isinstance(ring, dict):
        return ('circle', tuple(np.round(ring['center'], nd)), round(ring['r'], nd))
    out = []
    for p in ring:
        if p['type'] == 'line':
            out.append(('L', tuple(np.round(p['p0'], nd)), tuple(np.round(p['p1'], nd))))
        else:
            out.append(('A', tuple(np.round(p['center'], nd)), round(p['r'], nd),
                        tuple(np.round(p['p0'], nd)), tuple(np.round(p['p1'], nd))))
    return tuple(out)


def _slab_signature(rings):
    return tuple(sorted((repr(_ring_signature(o)),
                         tuple(sorted(repr(_ring_signature(h)) for h in hs)))
                        for o, hs in rings))


def _merge_equal_slabs(slabs, fitted, lofts=None, gap_tol=1e-6):
    """Consecutive extruded slabs with identical fitted profiles become one
    extrusion. Lofted slabs are never merged."""
    lofts = lofts or [None] * len(slabs)
    out = []
    for slab, rings, loft in zip(slabs, fitted, lofts):
        sig = _slab_signature(rings) if loft is None else object()
        if (loft is None and out and out[-1][4] is None and out[-1][3] == sig
                and abs(out[-1][1] - slab['z0']) < gap_tol):
            z0, _, r, _, _ = out[-1]
            out[-1] = (z0, slab['z1'], r, sig, None)
        else:
            out.append((slab['z0'], slab['z1'], rings, sig, loft))
    return [(z0, z1, rings, loft) for z0, z1, rings, _, loft in out]


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
    """Cluster radii and centers across ALL slabs and snap to cluster means.
    This turns facet-noise families like 5.242..5.257 into one radius.
    Junctions are re-solved afterwards by the caller."""
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
        # refined (vertex-fitted) radii are trusted more than chord fits
        ref = [a['r'] for a in cl if a.get('refined')]
        r = float(np.mean(ref if ref else [a['r'] for a in cl]))
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
        ref = [g['center'] for g in grp if g.get('refined')]
        c = np.mean(ref if ref else [g['center'] for g in grp], axis=0)
        for g in grp:
            g['center'] = c
    _snap_lines_across_slabs(fitted)


def _iter_lines(fitted):
    for rings in fitted:
        for outer, holes in rings:
            for ring in [outer] + holes:
                if isinstance(ring, list):
                    for p in ring:
                        if p['type'] == 'line':
                            yield p


# Lines from different slabs that are parallel within this angle and offset
# within this distance are one wall: each slab samples the wall at its own
# height, and a wall tilted a tenth of a degree in the mesh lands a few
# microns apart in each, which the union turns into sliver faces.
LINE_ANGLE_TOL = np.radians(0.3)
LINE_OFFSET_TOL = 0.02


def _snap_lines_across_slabs(fitted, angle_tol=LINE_ANGLE_TOL,
                             offset_tol=LINE_OFFSET_TOL):
    """Cluster line directions and offsets across ALL slabs and snap each
    cluster to one line. Direction: the frame-snapped member with most
    length if any (exact axis), else the length-weighted mean. Offset: the
    weighted mean, exactly straight lines (their points collinear to
    float precision) weighing ten times a line that absorbed a transition
    facet. Lines rotate about their midpoints; endpoints are re-solved by
    the caller's junction pass."""
    from .profile_fit import EXACT_EPS, fit_line
    lines = list(_iter_lines(fitted))
    if len(lines) < 2:
        return
    info = []
    for p in lines:
        p0, p1 = np.asarray(p['p0'], float), np.asarray(p['p1'], float)
        v = p1 - p0
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            continue
        u = v / L
        sgn = 1.0
        if u[0] < 0 or (abs(u[0]) < 1e-12 and u[1] < 0):
            u, sgn = -u, -1.0                     # canonical half-plane
        pts = p.get('_pts')
        exact = pts is not None and len(pts) >= 2 and fit_line(np.asarray(pts, float))['dev'] <= 4 * EXACT_EPS
        info.append({'p': p, 'u': u, 'L': L, 'sgn': sgn, 'ang': float(np.arctan2(u[1], u[0])),
                     'w': L * (10.0 if exact else 1.0), 'snapped': bool(p.get('snapped'))})
    if len(info) < 2:
        return
    # direction clusters (angles live on a half-circle; sort + gap split with
    # a capped spread, wrap handled by testing the first/last pair)
    info.sort(key=lambda d: d['ang'])
    clusters, cur = [], [info[0]]
    for d in info[1:]:
        if d['ang'] - cur[-1]['ang'] <= angle_tol and d['ang'] - cur[0]['ang'] <= angle_tol:
            cur.append(d)
        else:
            clusters.append(cur)
            cur = [d]
    clusters.append(cur)
    if len(clusters) > 1 and (clusters[0][0]['ang'] + np.pi - clusters[-1][-1]['ang']) <= angle_tol \
            and (clusters[0][-1]['ang'] + np.pi - clusters[-1][0]['ang']) <= angle_tol:
        for d in clusters[-1]:
            d['u'] = -d['u']
            d['sgn'] = -d['sgn']
            d['ang'] -= np.pi
        clusters[0] = clusters.pop() + clusters[0]
    for cl in clusters:
        if len(cl) < 2:
            continue
        snapped = [d for d in cl if d['snapped']]
        if snapped:
            u = max(snapped, key=lambda d: d['L'])['u']
        else:
            u = np.sum([d['u'] * d['w'] for d in cl], axis=0)
            u = u / np.linalg.norm(u)
        nrm = np.array([-u[1], u[0]])
        for d in cl:
            mid = 0.5 * (np.asarray(d['p']['p0'], float) + np.asarray(d['p']['p1'], float))
            d['off'] = float(mid @ nrm)
            d['mid'] = mid
        cl.sort(key=lambda d: d['off'])
        groups, cur = [], [cl[0]]
        for d in cl[1:]:
            if d['off'] - cur[-1]['off'] <= offset_tol and d['off'] - cur[0]['off'] <= offset_tol:
                cur.append(d)
            else:
                groups.append(cur)
                cur = [d]
        groups.append(cur)
        for g in groups:
            off = sum(d['off'] * d['w'] for d in g) / sum(d['w'] for d in g)
            for d in g:
                p = d['p']
                mid = d['mid'] + (off - d['off']) * nrm
                p['p0'] = mid - d['sgn'] * u * d['L'] / 2
                p['p1'] = mid + d['sgn'] * u * d['L'] / 2
                p['snapped'] = True


def _align_arcs_across_slabs(fitted, center_tol=0.35):
    """After junction solving, fillet and tangent-nudged arcs sit where
    their lines put them — and a fillet (two lines) and a tangent nudge
    (one line) of the same boss in neighbouring slabs land microns apart;
    a full circle or a free arc of the same radius still has the
    vertex-fitted centre. The slab union turns every such mismatch into a
    hairline wedge on the interface plane. Unify each group of equal-radius
    arcs on one centre (mean of the line-constrained members, else of all)
    and return the rings that changed; the caller re-solves their junctions
    with the arcs frozen."""
    arcs = []
    for rings in fitted:
        for outer, holes in rings:
            for ring in [outer] + holes:
                if isinstance(ring, dict):
                    arcs.append((ring, ring))
                else:
                    for p in ring:
                        if p['type'] == 'arc':
                            arcs.append((p, ring))
    dirty = []
    done = [False] * len(arcs)
    for i, (a, _) in enumerate(arcs):
        if done[i]:
            continue
        grp = [i]
        for j in range(i + 1, len(arcs)):
            b = arcs[j][0]
            if not done[j] and abs(b['r'] - a['r']) < 1e-6 \
                    and np.linalg.norm(np.asarray(b['center']) - np.asarray(a['center'])) < center_tol:
                grp.append(j)
                done[j] = True
        if len(grp) < 2:
            continue
        fil = [arcs[k][0] for k in grp if arcs[k][0].get('fillet') or arcs[k][0].get('tangent')]
        c = np.mean([f['center'] for f in (fil or [arcs[k][0] for k in grp])], axis=0)
        for k in grp:
            p, ring = arcs[k]
            if np.linalg.norm(np.asarray(p['center']) - c) < 1e-9:
                continue
            p['center'] = c
            if not isinstance(ring, dict) and not any(r is ring for r in dirty):
                dirty.append(ring)
    return dirty


def _fit_ring(coords, tol, pts2d=None, pts_all=None):
    """Ring of 2D coords -> full circle dict or list of line/arc prims.
    Arcs are refined on the slab's own mesh vertices, then — those that
    could not be — on all mesh vertices near their circle."""
    pts = np.array(coords)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    circ = try_full_circle(pts, tol)
    if circ:
        refine_arcs_with_points([circ], pts2d, tol)
        if not circ.get('refined') and pts_all is not None:
            refine_arcs_with_points([circ], pts_all, tol)
        return circ
    prims = segment_polyline(pts, tol=tol, closed=True)
    refine_arcs_with_points(prims, pts2d, tol)
    if pts_all is not None:
        rest = [p for p in prims if p['type'] == 'arc' and not p.get('refined')]
        if rest:
            refine_arcs_with_points(rest, pts_all, tol)
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


# Face colours by surface type (RGB 0-1): a quick visual audit in any CAD
# viewer of what was recognised analytically vs left faceted/free-form.
SURFACE_COLOURS = {
    'plane': (0.80, 0.80, 0.82),
    'cylinder': (0.25, 0.55, 0.95),
    'cone': (0.95, 0.60, 0.15),
    'sphere': (0.30, 0.75, 0.35),
    'torus': (0.65, 0.35, 0.85),
    'other': (0.90, 0.25, 0.25),
}


def _surface_kind(face):
    from OCP.GeomAdaptor import GeomAdaptor_Surface
    from OCP.BRep import BRep_Tool
    from OCP.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                             GeomAbs_Sphere, GeomAbs_Torus)
    t = GeomAdaptor_Surface(BRep_Tool.Surface_s(face)).GetType()
    return {GeomAbs_Plane: 'plane', GeomAbs_Cylinder: 'cylinder',
            GeomAbs_Cone: 'cone', GeomAbs_Sphere: 'sphere',
            GeomAbs_Torus: 'torus'}.get(t, 'other')


def write_step(shapes, path, names=None, colours=True, schema='AP214'):
    """Write one or more OCC solids to a single STEP file.

    With `names` (one per shape) the file is written through XDE so each
    body carries its name and every face a colour by surface type; without,
    the plain writer is used. Several solids go into one compound / one
    assembly, so a multi-body part imports as multiple bodies of one
    component (Fusion, FreeCAD, Onshape all do this). Explicit
    MANIFOLD_SOLID_BREP per solid.
    """
    from OCP.TopoDS import TopoDS_Compound
    from OCP.BRep import BRep_Builder
    from OCP.STEPControl import STEPControl_Writer, STEPControl_ManifoldSolidBrep
    from OCP.Interface import Interface_Static
    shapes = list(shapes)
    if not shapes:
        raise ValueError('nothing to write')
    if names:
        try:
            _write_step_xde(shapes, path, names, colours, schema)
            return
        except Exception as e:      # never fail an export over metadata
            print(f"[out] named STEP export failed ({type(e).__name__}: {e}); "
                  f"writing plain STEP")
    if len(shapes) == 1:
        shape = shapes[0]
    else:
        shape = TopoDS_Compound()
        b = BRep_Builder()
        b.MakeCompound(shape)
        for s in shapes:
            b.Add(shape, s)
    w = STEPControl_Writer()
    Interface_Static.SetCVal_s('write.step.schema', schema)
    w.Transfer(shape, STEPControl_ManifoldSolidBrep)
    w.Write(path)


def _write_step_xde(shapes, path, names, colours, schema):
    from OCP.TDocStd import TDocStd_Document
    from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorSurf
    from OCP.TDataStd import TDataStd_Name
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.Interface import Interface_Static
    from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS
    doc = TDocStd_Document(TCollection_ExtendedString('stl2prism'))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    for shape, name in zip(shapes, names):
        label = shape_tool.AddShape(shape, False)
        TDataStd_Name.Set_s(label, TCollection_ExtendedString(str(name)))
        if colours:
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            while exp.More():
                face = TopoDS.Face_s(exp.Current())
                rgb = SURFACE_COLOURS[_surface_kind(face)]
                sub = shape_tool.AddSubShape(label, face)
                if not sub.IsNull():
                    color_tool.SetColor(sub, Quantity_Color(*rgb, Quantity_TOC_RGB),
                                        XCAFDoc_ColorSurf)
                exp.Next()
    Interface_Static.SetCVal_s('write.step.schema', schema)
    w = STEPCAFControl_Writer()
    w.SetColorMode(bool(colours))
    w.SetNameMode(True)
    w.Transfer(doc, STEPControl_AsIs)
    w.Write(path)


def export_step(solid, path):
    """Write a CadQuery solid (Workplane) as STEP."""
    write_step([solid.val().wrapped], path)


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
    not closed, whatever the STEP file calls it. Degenerate edges (a sphere
    pole, a cone apex) are parametric artefacts, not boundaries, and are not
    counted; a periodic face's seam edge is listed twice by its face."""
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopExp import TopExp
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopoDS import TopoDS
    from OCP.BRep import BRep_Tool
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, m)
    return sum(1 for i in range(1, m.Extent() + 1)
               if m.FindFromIndex(i).Extent() < 2
               and not BRep_Tool.Degenerated_s(TopoDS.Edge_s(m.FindKey(i))))


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
    shape, stats = faceted_solid(mesh, angular_tol=angular_tol,
                                 min_face_frac=min_face_frac, verbose=verbose)
    write_step([shape], path)
    return stats


def planar_groups(mesh, angle_tol=1e-3):
    """Coplanar-connected face groups (list of index arrays), via the
    face-adjacency graph restricted to adjacencies with a dihedral below
    angle_tol (radians)."""
    import networkx as nx
    adj = mesh.face_adjacency
    ang = np.abs(mesh.face_adjacency_angles)
    G = nx.Graph()
    G.add_nodes_from(range(len(mesh.faces)))
    G.add_edges_from(adj[ang <= angle_tol])
    return [np.array(sorted(c)) for c in nx.connected_components(G)]


def _group_loops(mesh, faces):
    """Boundary loops (lists of vertex indices) of a face group; None if the
    boundary is not a set of simple closed loops."""
    fset = set(faces.tolist())
    tri = mesh.faces[faces]
    # boundary edges: appear once among the group's directed edges
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    bnd = edges[idx[cnt == 1]]                     # directed, consistent with face winding
    if len(bnd) < 3:
        return None
    nxt = {}
    for a, b in bnd:
        if a in nxt:
            return None                             # vertex with two outgoing edges: not simple
        nxt[int(a)] = int(b)
    loops = []
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start:
            if cur in seen or cur not in nxt:
                return None
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def _planar_face_from_group(mesh, faces, normal):
    """One planar OCC face (with holes) for a coplanar face group, or None."""
    from OCP.gp import gp_Pnt, gp_Pln, gp_Dir, gp_Vec
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
    from OCP.ShapeFix import ShapeFix_Face
    from OCP.BRepCheck import BRepCheck_Analyzer
    loops = _group_loops(mesh, faces)
    if not loops:
        return None
    V = mesh.vertices
    # areas (projected on the plane) to find the outer loop
    n = np.array(normal, float)
    n = n / np.linalg.norm(n)
    b0 = np.cross([0, 1, 0], n)
    if np.linalg.norm(b0) < 1e-6:
        b0 = np.cross([1, 0, 0], n)
    b0 /= np.linalg.norm(b0)
    b1 = np.cross(n, b0)

    def area2d(loop):
        P = V[loop]
        u, v = P @ b0, P @ b1
        return 0.5 * np.sum(u * np.roll(v, -1) - np.roll(u, -1) * v)
    loops.sort(key=lambda l: -abs(area2d(l)))
    outer, inners = loops[0], loops[1:]

    def wire(loop):
        mp = BRepBuilderAPI_MakePolygon()
        for vi in loop:
            mp.Add(gp_Pnt(*map(float, V[vi])))
        mp.Close()
        return mp.Wire()
    c = V[outer].mean(axis=0)
    pln = gp_Pln(gp_Pnt(*map(float, c)), gp_Dir(*map(float, n)))
    mk = BRepBuilderAPI_MakeFace(pln, wire(outer), True)
    if not mk.IsDone():
        return None
    for il in inners:
        mk.Add(wire(il))
    face = mk.Face()
    fx = ShapeFix_Face(face)
    fx.Perform()
    face = fx.Face()
    if not BRepCheck_Analyzer(face).IsValid():
        return None
    return face


def reduce_mesh(mesh, reduce_tol, verbose=True, targets=(0.5, 0.7, 0.85, 0.93, 0.97)):
    """Tolerance-driven decimation (Fusion 'Reduce by Tolerance'): the
    strongest QEM reduction whose deviation from the original stays within
    reduce_tol (both directions, p95 and max) and that stays watertight.
    Returns (mesh, info)."""
    try:
        import fast_simplification as fs
    except ImportError:
        return mesh, {'reduced': False, 'reason': 'fast_simplification not installed'}
    if reduce_tol <= 0 or len(mesh.faces) < 200:
        return mesh, {'reduced': False}
    from .pipeline import SAMPLE_SEED
    best = None
    ref_pts = mesh.sample(min(20000, max(4000, len(mesh.faces))), seed=SAMPLE_SEED)
    for t in targets:
        try:
            v, f = fs.simplify(mesh.vertices, mesh.faces, target_reduction=t)
        except Exception:
            break
        if len(f) >= 0.98 * len(mesh.faces):
            break                                  # decimator made no progress
        d = trimesh.Trimesh(v, f, process=True)
        if len(d.faces) < 12 or not d.is_watertight or d.body_count != mesh.body_count:
            break
        # deviation both ways
        _, d1, _ = trimesh.proximity.closest_point(d, ref_pts)
        pts2 = d.sample(min(20000, max(4000, len(d.faces))), seed=SAMPLE_SEED)
        _, d2, _ = trimesh.proximity.closest_point(mesh, pts2)
        mx = max(d1.max(), d2.max())
        if mx <= reduce_tol and abs(d.volume - mesh.volume) <= 0.005 * abs(mesh.volume):
            best = (d, t, mx)
        else:
            break
    if best is None:
        return mesh, {'reduced': False}
    d, t, mx = best
    if verbose:
        print(f"[reduce] {len(mesh.faces)} -> {len(d.faces)} triangles "
              f"(max deviation {mx:.3f} mm <= {reduce_tol})")
    return d, {'reduced': True, 'faces_before': int(len(mesh.faces)),
               'faces_after': int(len(d.faces)), 'max_dev': float(mx)}


def faceted_solid(mesh, angular_tol=5e-3, min_face_frac=0.5, verbose=True,
                  merge_planar=True):
    """Sew the mesh into a solid; nothing written.

    Coplanar triangle groups become ONE planar face each (with holes)
    before sewing — far fewer OCC faces than one-per-triangle, and no
    reliance on UnifySameDomain to merge them afterwards; curved regions
    stay triangles. Returns (TopoDS_Shape, stats). Raises FacetedError
    rather than returning a fragment that would pass as a valid solid.
    """
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing)
    from OCP.TopAbs import TopAbs_FACE
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    sew = BRepBuilderAPI_Sewing(1e-3)
    V = mesh.vertices
    n_added = 0
    n_planar_faces = 0
    as_triangles = np.ones(len(mesh.faces), bool)
    if merge_planar:
        for grp in planar_groups(mesh):
            if len(grp) < 3:                       # pairs (quads) are left to unify
                continue
            f = _planar_face_from_group(mesh, grp, mesh.face_normals[grp[0]])
            if f is not None:
                sew.Add(f)
                n_added += 1
                n_planar_faces += 1
                as_triangles[grp] = False
    for tri in mesh.faces[as_triangles]:
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

    solid, vol_sewn, kept_faces, n_shells, _ = _largest_solid(
        sew.SewedShape(), 'sew')
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
    # ...nor may it change the enclosed volume: on paper-thin bodies (decals,
    # zero-thickness sheets) unify collapses opposite skins into each other.
    vol_drift = (abs(vol - vol_sewn) / vol_sewn) if vol_sewn > 0 else 0.0
    if naked2 > naked or vol_drift > 0.01:
        if verbose:
            why = (f"opened {naked2 - naked} edge(s)" if naked2 > naked
                   else f"changed the volume by {vol_drift:.1%}")
            print(f"[faceted] coplanar merge {why}; "
                  f"keeping the unmerged solid ({kept_faces} faces)")
        solid2, faces2, naked2 = solid, kept_faces, naked
        vol = _volume(solid2)

    if naked2 > 0 and verbose:
        # Worth writing — an open shell still imports — but it must not be
        # reported as a solid: OCC re-reads it as a shell, and the caller
        # needs to know that before trusting a volume from it.
        print(f"[faceted] warning: {naked2} naked edge(s) remain; the result "
              f"is an open shell, not a closed solid")

    return solid2, {
        'faces_in': int(len(mesh.faces)), 'faces_out': faces2, 'shells': n_shells,
        'free_edges': free_edges, 'naked_edges': naked2, 'volume': vol,
        'planar_faces_merged': n_planar_faces,
        'is_solid': bool(valid and naked2 == 0)}
