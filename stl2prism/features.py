"""Detect cylindrical bores whose axis differs from the primary extrusion
axis, and subtract them from the rebuilt solid.

Cylinder recovery per curved region:
  * axis = eigenvector of the face-normal covariance with the smallest
    eigenvalue (all cylinder normals are perpendicular to its axis)
  * center/radius = Taubin circle fit of face centers projected onto the
    plane perpendicular to that axis
  * accept if radial residual is small and normals are consistent
"""
import numpy as np
import trimesh
from .profile_fit import fit_circle_taubin


def _curved_regions(mesh, min_faces=8, resid_tol=0.12, depth=0):
    """Connected curved facet regions, split where one region holds two
    coaxial primitives (a countersink cone running into its hole cylinder,
    a chamfer cone on a boss): faces are separated by their normal's
    component along the region's axis — ~0 for a cylinder, sin(alpha) for a
    cone — when neither a single cylinder nor a single cone explains the
    region. Yields arrays of face indices."""
    import networkx as nx
    n = mesh.face_normals
    a = mesh.area_faces
    curved = _curved_face_mask(mesh)
    if curved.sum() < min_faces:
        return []
    adj = mesh.face_adjacency
    both = curved[adj].all(axis=1)
    G = nx.Graph()
    G.add_nodes_from(np.where(curved)[0])
    G.add_edges_from(adj[both])
    out = []
    for comp in nx.connected_components(G):
        comp = np.array(sorted(comp))
        if len(comp) < min_faces:
            continue
        out.extend(_split_region(mesh, comp, min_faces, resid_tol))
    return out


def _split_region(mesh, comp, min_faces, resid_tol, depth=0):
    n = mesh.face_normals
    a = mesh.area_faces
    nn, ww = n[comp], a[comp]
    verts = mesh.vertices[np.unique(mesh.faces[comp])]
    C0 = (nn * ww[:, None]).T @ nn
    w0, v0 = np.linalg.eigh(C0)
    axis = v0[:, 0]
    cyl = fit_cylinder(verts, axis)
    if cyl is not None and cyl['resid'] <= resid_tol:
        return [comp]
    mean_n = (nn * ww[:, None]).sum(axis=0) / ww.sum()
    cn = nn - mean_n
    C = (cn * ww[:, None]).T @ cn
    w, v = np.linalg.eigh(C)
    ax_c = v[:, 0]
    s_al = float((nn @ ax_c * ww).sum() / ww.sum())
    half = np.degrees(np.arcsin(np.clip(abs(s_al), 0, 1)))
    if half > 5:
        cone = fit_cone(verts, ax_c if s_al >= 0 else -ax_c, np.radians(half))
        if cone is not None and cone['resid'] <= 0.5 * resid_tol:
            return [comp]
    if depth >= 2:
        return [comp]
    # split by |n . axis| at the largest gap
    val = np.abs(nn @ axis)
    order = np.argsort(val)
    sv = val[order]
    gaps = np.diff(sv)
    if len(gaps) == 0 or gaps.max() < 0.15:
        return [comp]
    k = int(np.argmax(gaps)) + 1
    parts = [comp[order[:k]], comp[order[k:]]]
    out = []
    import networkx as nx
    adj = mesh.face_adjacency
    for part in parts:
        if len(part) < min_faces:
            continue
        pset = set(part.tolist())
        sub = adj[np.array([x in pset and y in pset for x, y in adj])] if len(adj) else adj
        G = nx.Graph()
        G.add_nodes_from(part.tolist())
        G.add_edges_from(sub)
        for cc in nx.connected_components(G):
            cc = np.array(sorted(cc))
            if len(cc) >= min_faces:
                out.extend(_split_region(mesh, cc, min_faces, resid_tol, depth + 1))
    return out or [comp]


def find_cross_cylinders(mesh, primary_axis, angle_tol_deg=12.0,
                         max_r=200.0, min_faces=8, resid_tol=0.12,
                         exclude_parallel=True, min_span_deg=100.0):
    """Return list of dicts {axis, center2, basis, r, h0, h1, ...} for
    cylindrical regions.

    Per curved region: axis from the area-weighted normal covariance (all
    cylinder normals are perpendicular to the axis), then centre/radius from
    a Taubin fit of the region's *vertices* projected along the axis, then a
    joint nonlinear least-squares refinement of axis, centre and radius on
    the vertices (Eberly-style geometric distance). Vertices lie exactly on
    the CAD surface; triangle centroids sit inside it by the facet sagitta,
    which is why they are not used.

    A region whose normals span less than `min_span_deg` around the axis
    (a shallow fillet fragment) is skipped: its axis is ill-determined and
    it is not a bore anyway.

    With exclude_parallel=True (default) only cross-axis bores are returned —
    the ones that must be subtracted, since axis-parallel holes are already
    interior rings of the extruded profile. exclude_parallel=False keeps the
    parallel ones too (tagged 'parallel'), which validation needs: a hole is
    a hole regardless of which stage of the rebuild produced it."""
    n = mesh.face_normals
    a = mesh.area_faces
    out = []
    for comp in _curved_regions(mesh, min_faces, resid_tol):
        nn = n[comp]
        ww = a[comp]
        # cylinder axis: null direction of the (area-weighted) normals
        C = (nn * ww[:, None]).T @ nn
        w, v = np.linalg.eigh(C)
        axis = v[:, 0]
        if w[0] > 0.05 * w[2]:          # normals not planar enough
            continue
        # angular coverage of the normals around the axis
        b0, b1 = _plane_basis(axis)
        ang = np.degrees(np.arctan2(nn @ b1, nn @ b0))
        span = _angular_span(ang)
        if span < min_span_deg:
            continue
        cos = abs(float(axis @ primary_axis))
        parallel = cos > np.cos(np.radians(angle_tol_deg))
        if parallel and exclude_parallel:
            continue                     # parallel to primary: already built
        verts = mesh.vertices[np.unique(mesh.faces[comp])]
        fit = fit_cylinder(verts, axis)
        if fit is None or fit['resid'] > resid_tol or fit['r'] > max_r:
            continue
        axis, c3, r = fit['axis'], fit['point'], fit['r']
        basis = _plane_basis(axis)
        centers = mesh.triangles_center[comp]
        # verify normals point radially
        radial = centers - c3 - np.outer((centers - c3) @ axis, axis)
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        signed = (radial * nn).sum(axis=1)
        if np.abs(signed).mean() < 0.95:
            continue
        # Outward mesh normals pointing back toward the cylinder axis mean
        # the surface encloses a void: a hole. Pointing away means a boss or
        # fillet — an outer surface, not a mating feature.
        concave = float(signed.mean()) < 0.0
        vh = verts @ axis
        center2 = np.array([c3 @ basis[0], c3 @ basis[1]])
        out.append({'axis': axis, 'center2': center2, 'basis': basis,
                    'r': float(r), 'parallel': bool(parallel),
                    'concave': concave, 'span_deg': float(span),
                    'h0': float(vh.min()), 'h1': float(vh.max()),
                    'faces': len(comp), 'resid': float(fit['resid'])})
    return _merge_coaxial(out)


def find_cross_cones(mesh, primary_axis, angle_tol_deg=12.0, min_faces=8,
                     resid_tol=0.12, min_half_angle_deg=5.0, min_span_deg=150.0,
                     exclude_parallel=True):
    """Conical regions (countersinks, chamfered hole mouths, cone bosses)
    whose axis is not the extrusion axis.

    Per curved region: the cone axis is the smallest eigenvector of the
    *centred* (area-weighted) normal covariance — the normals of a cone lie
    on a small circle around its axis, those of a cylinder on a great
    circle; the mean normal component along the axis gives the half-angle
    (0 for a cylinder, which is skipped here). Apex, axis and half-angle
    are then refined jointly by least squares on the region's vertices.
    Returns dicts {axis, apex, half_angle, h0, h1, concave, ...} with h the
    coordinate along the axis measured from the apex.
    """
    n = mesh.face_normals
    a = mesh.area_faces
    out = []
    for comp in _curved_regions(mesh, min_faces, resid_tol):
        nn = n[comp]
        ww = a[comp]
        mean_n = (nn * ww[:, None]).sum(axis=0) / ww.sum()
        cn = nn - mean_n
        C = (cn * ww[:, None]).T @ cn
        w, v = np.linalg.eigh(C)
        axis = v[:, 0]
        if w[0] > 0.05 * w[2]:
            continue
        s_al = float((nn @ axis * ww).sum() / ww.sum())
        half = np.degrees(np.arcsin(np.clip(abs(s_al), 0, 1)))
        if half < min_half_angle_deg or half > 85:
            continue                       # a cylinder (or a plane): not a cone
        b0, b1 = _plane_basis(axis)
        ang = np.degrees(np.arctan2(nn @ b1, nn @ b0))
        if _angular_span(ang) < min_span_deg:
            continue
        cos = abs(float(axis @ primary_axis))
        parallel = cos > np.cos(np.radians(angle_tol_deg))
        if parallel and exclude_parallel:
            continue
        verts = mesh.vertices[np.unique(mesh.faces[comp])]
        # cylinder first: a region that a cylinder explains is not a cone
        # (a 6-parameter cone can fit a handful of cylinder vertices)
        C0 = (nn * ww[:, None]).T @ nn
        w0, v0 = np.linalg.eigh(C0)
        cyl = fit_cylinder(verts, v0[:, 0])
        if cyl is not None and cyl['resid'] <= resid_tol:
            continue
        # orient the axis so that normals' axial component is positive:
        # then a concave cone (countersink) opens along +axis
        if s_al < 0:
            axis = -axis
        fit = fit_cone(verts, axis, np.radians(half))
        if fit is None or fit['resid'] > 0.5 * resid_tol:
            continue
        axis, apex, alpha = fit['axis'], fit['apex'], fit['half_angle']
        centers = mesh.triangles_center[comp]
        rel = centers - apex
        h = rel @ axis
        radial = rel - np.outer(h, axis)
        rn = np.linalg.norm(radial, axis=1, keepdims=True)
        rn[rn == 0] = 1
        radial /= rn
        # the fitted cone's normal at each facet: cos(a) radial - sin(a) axial
        # (up to sign) must agree with the facet normal
        cone_n = np.cos(alpha) * radial - np.sin(alpha) * axis
        agree = np.abs((cone_n * nn).sum(axis=1))
        if agree.mean() < 0.98:
            continue
        signed = (radial * nn).sum(axis=1)
        concave = float(signed.mean()) < 0.0
        vh = (verts - apex) @ axis
        out.append({'axis': axis, 'apex': apex, 'half_angle': float(alpha),
                    'h0': float(vh.min()), 'h1': float(vh.max()),
                    'concave': concave, 'parallel': bool(parallel),
                    'faces': len(comp), 'resid': float(fit['resid'])})
    return out


def fit_cone(pts, axis0, alpha0, iters=2, max_nfev=300):
    """Least-squares cone through 3-D points: axis (2 angles), apex (3) and
    half-angle refined jointly from an initial axis/half-angle. The apex is
    initialised from the points' mean radius and height along the axis.
    Returns {'axis', 'apex', 'half_angle', 'resid'} or None."""
    pts = np.asarray(pts, float)
    if len(pts) < 6:
        return None
    axis = np.asarray(axis0, float) / np.linalg.norm(axis0)
    b0, b1 = _plane_basis(axis)
    # initial axis line through the centroid of the projected points, apex
    # from mean radius / tan(alpha)
    c0 = pts.mean(axis=0)
    rel = pts - c0
    h = rel @ axis
    rad = np.linalg.norm(rel - np.outer(h, axis), axis=1)
    t = np.tan(alpha0) if abs(np.tan(alpha0)) > 1e-6 else 1e-6
    # r = (h - h_apex) * tan(alpha) with sign convention: opening along +axis
    h_apex = float(np.mean(h - rad / t))
    apex0 = c0 + h_apex * axis

    def unpack(x):
        da, db, ax_, ay_, az_, al = x
        ax = axis + da * b0 + db * b1
        ax /= np.linalg.norm(ax)
        return ax, np.array([ax_, ay_, az_]), al

    def resid(x):
        ax, apex, al = unpack(x)
        rel = pts - apex
        hh = rel @ ax
        rr = np.linalg.norm(rel - np.outer(hh, ax), axis=1)
        # distance from a point to the cone surface (locally): (r cos a - h sin a)
        return rr * np.cos(al) - hh * np.sin(al)

    x0 = np.array([0.0, 0.0, apex0[0], apex0[1], apex0[2], alpha0])
    try:
        from scipy.optimize import least_squares
        sol = least_squares(resid, x0, method='lm', max_nfev=max_nfev)
        x = sol.x
    except Exception:
        x = x0
    ax, apex, al = unpack(x)
    if not (np.radians(2) < abs(al) < np.radians(88)):
        return None
    if al < 0:
        al, ax = -al, -ax
        apex = apex
    res = resid(x)
    return {'axis': ax, 'apex': apex, 'half_angle': float(al),
            'resid': float(np.abs(res).max())}


def subtract_cones(solid, cones, mesh=None, extend=1.0, verbose=True):
    """Cut concave conical regions (countersinks, chamfered hole mouths) as
    cone frustums between their measured extents, extended where an end
    breaks out of the body."""
    import cadquery as cq
    for c in cones:
        axis, apex, al = c['axis'], c['apex'], c['half_angle']
        t = np.tan(al)
        h0, h1 = c['h0'], c['h1']
        # exit probes: a point just beyond each end on the axis
        def exits(hh):
            if mesh is None or not mesh.is_watertight:
                return True
            p = apex + axis * hh
            try:
                return not bool(mesh.contains(np.array([p]))[0])
            except Exception:
                return True
        e0 = extend if exits(h0 - 0.3) else 0.0
        e1 = extend if exits(h1 + 0.3) else 0.0
        a0, a1 = h0 - e0, h1 + e1
        r0, r1 = max(a0 * t, 0.0), max(a1 * t, 0.0)
        if a0 < 0:                         # cannot extend past the apex
            a0, r0 = 0.0, 0.0
        if a1 - a0 < 1e-6:
            continue
        pnt = cq.Vector(*(apex + axis * a0))
        cone = cq.Solid.makeCone(r0, r1, a1 - a0, pnt, cq.Vector(*axis))
        solid = solid.cut(cq.Workplane('XY').newObject([cone]), tol=1e-4)
        if verbose:
            print(f"[feature] cut cone half-angle {np.degrees(al):.1f}deg axis "
                  f"{np.round(axis,2)} r {r0:.2f}->{r1:.2f} depth {h1 - h0:.2f} "
                  f"({c['faces']} facets)")
    return solid


def _angular_span(ang_deg):
    """Extent (deg) of a set of angles on the circle: 360 minus the largest
    gap between sorted angles."""
    if len(ang_deg) == 0:
        return 0.0
    s = np.sort(np.mod(ang_deg, 360.0))
    gaps = np.diff(np.concatenate([s, [s[0] + 360.0]]))
    return float(360.0 - gaps.max())


def fit_cylinder(pts, axis0, iters=2):
    """Least-squares cylinder through 3-D points.

    Start: axis0 (from the normals); centre and radius from a Taubin circle
    fit of the points projected along the axis. Then refine axis
    (2 angles), centre (2 in-plane coords) and radius jointly with
    scipy least_squares on the geometric residual |dist(p, axis) - r|.
    Returns {'axis', 'point' (on the axis, at the points' mean height),
    'r', 'resid' (max abs residual)} or None."""
    pts = np.asarray(pts, float)
    if len(pts) < 6:
        return None
    axis = np.asarray(axis0, float) / np.linalg.norm(axis0)
    b0, b1 = _plane_basis(axis)
    uv = np.column_stack([pts @ b0, pts @ b1])
    circ = fit_circle_taubin(uv, polyline=False)
    if circ is None:
        return None
    cu, cv = circ['center']
    r = circ['r']

    def unpack(x):
        # axis perturbed by rotations about b0/b1; centre in the plane; r
        da, db, u, v_, rr = x
        ax = axis + da * b0 + db * b1
        ax /= np.linalg.norm(ax)
        c = u * b0 + v_ * b1
        return ax, c, rr

    def resid(x):
        ax, c, rr = unpack(x)
        rel = pts - c
        along = rel @ ax
        radial = rel - np.outer(along, ax)
        return np.linalg.norm(radial, axis=1) - rr

    x0 = np.array([0.0, 0.0, cu, cv, r])
    try:
        from scipy.optimize import least_squares
        sol = least_squares(resid, x0, method='lm', max_nfev=200)
        x = sol.x
    except Exception:
        x = x0
    ax, c, rr = unpack(x)
    if rr <= 0:
        return None
    res = resid(x)
    return {'axis': ax, 'point': c, 'r': float(rr),
            'resid': float(np.abs(res).max())}


def _curved_face_mask(mesh, lo_deg=4.0, hi_deg=62.0, planar_min_faces=3,
                      planar_max_band_frac=0.5):
    """Faces participating in gently-curved regions: at least one facet
    dihedral in a band (bigger than coplanar noise, smaller than sharp
    edges), excluding faces of *planar groups* — coplanar-connected groups
    whose boundary is mostly sharp edges. A planar face's triangles that
    border a countersink or fillet have one in-band neighbour but belong
    to a group whose perimeter is dominated by 90-degree edges; a cylinder
    facet strip is bounded mostly by in-band edges."""
    import networkx as nx
    adj = mesh.face_adjacency
    ang = np.degrees(np.abs(mesh.face_adjacency_angles))
    band = (ang > lo_deg) & (ang < hi_deg)
    coplanar = ang <= lo_deg
    nf = len(mesh.faces)
    in_band = np.zeros(nf, bool)
    in_band[adj[band].ravel()] = True
    if not in_band.any():
        return in_band
    # coplanar groups
    G = nx.Graph()
    G.add_nodes_from(range(nf))
    G.add_edges_from(adj[coplanar])
    group_of = np.empty(nf, dtype=np.int64)
    for gi, comp in enumerate(nx.connected_components(G)):
        group_of[list(comp)] = gi
    ngroups = int(group_of.max()) + 1
    # boundary adjacencies (between different groups), length-weighted
    e = mesh.face_adjacency_edges
    L = np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
    g0, g1 = group_of[adj[:, 0]], group_of[adj[:, 1]]
    bnd = g0 != g1
    tot = np.zeros(ngroups)
    inb = np.zeros(ngroups)
    for gg in (g0, g1):
        np.add.at(tot, gg[bnd], L[bnd])
        np.add.at(inb, gg[bnd & band], L[bnd & band])
    sizes = np.bincount(group_of, minlength=ngroups)
    frac = np.where(tot > 0, inb / np.maximum(tot, 1e-12), 0.0)
    planar_group = (sizes >= planar_min_faces) & (frac < planar_max_band_frac)
    return in_band & ~planar_group[group_of]


def _plane_basis(axis):
    x = np.cross([0, 1, 0], axis)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1, 0, 0], axis)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    return x, y


def _merge_coaxial(cyls, center_tol=0.5, r_tol=0.15):
    merged = []
    for c in cyls:
        hit = None
        for m in merged:
            if (abs(abs(float(c['axis'] @ m['axis'])) - 1) < 1e-3 and
                    abs(c['r'] - m['r']) < r_tol and
                    np.linalg.norm(c['center2'] - m['center2']) < center_tol):
                hit = m
                break
        if hit:
            hit['h0'] = min(hit['h0'], c['h0'])
            hit['h1'] = max(hit['h1'], c['h1'])
            hit['faces'] += c['faces']
        else:
            merged.append(dict(c))
    return merged


def hole_end_exits(mesh, c, end, probe=0.3):
    """True if the hole's `end` ('h0'|'h1') opens to the outside of the body.

    Points just beyond the end — on the axis and at half the radius in four
    directions — are inside the mesh when the hole is blind there (the floor
    is solid material) and outside when the hole breaks through; a majority
    vote tolerates holes that graze an outer face. Non-watertight meshes
    cannot answer, so both ends are treated as exits (the historical
    behaviour).
    """
    if not mesh.is_watertight:
        return True
    axis = c['axis']
    bx, by = c['basis']
    c3 = c['center2'][0] * bx + c['center2'][1] * by
    h = c['h1'] + probe if end == 'h1' else c['h0'] - probe
    base = c3 + axis * h
    r = 0.5 * c['r']
    pts = np.array([base, base + bx * r, base - bx * r, base + by * r, base - by * r])
    try:
        inside = mesh.contains(pts)
    except Exception:
        return True
    return int(inside.sum()) * 2 <= len(pts)      # majority outside => exit


def subtract_cylinders(solid, cyls, extend=1.0, mesh=None, verbose=True):
    """Cut each detected bore out of the solid.

    Ends that break out of the body are extended by `extend` so the cut is
    clean; a blind end (hole floor) is cut exactly to the measured depth —
    extending it would carve into the floor.
    """
    import cadquery as cq
    for c in cyls:
        axis = c['axis']
        bx, by = c['basis']
        c3 = c['center2'][0] * bx + c['center2'][1] * by
        ext0 = extend if (mesh is None or hole_end_exits(mesh, c, 'h0')) else 0.0
        ext1 = extend if (mesh is None or hole_end_exits(mesh, c, 'h1')) else 0.0
        p0 = c3 + axis * (c['h0'] - ext0)
        h = (c['h1'] - c['h0']) + ext0 + ext1
        cut = (cq.Workplane(cq.Plane(origin=tuple(p0), normal=tuple(axis)))
               .circle(c['r']).extrude(h))
        solid = solid.cut(cut, tol=1e-4)
        c['blind'] = [ext0 == 0.0, ext1 == 0.0]
        if verbose:
            kind = ('through' if ext0 and ext1 else
                    'blind' if not (ext0 or ext1) else 'blind one end')
            print(f"[feature] cut cylinder r={c['r']:.3f} axis "
                  f"{np.round(axis,2)} depth {c['h1'] - c['h0']:.2f} {kind} "
                  f"({c['faces']} facets)")
    return solid
