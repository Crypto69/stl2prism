"""Detect extrusion (prismatic) structure in a mesh.

Approach (Point2Cyl-inspired, classical implementation):
  1. Cluster face normals on the Gaussian sphere to find the dominant
     extrusion axis: the direction whose +/- aligned faces carry the
     most area (top/bottom faces of slabs).
  2. Collect the heights (along the axis) of those perpendicular planar
     faces -> cluster into discrete Z-levels.
  3. Between consecutive levels, cross-section the mesh -> slab profiles.
"""
import numpy as np
import trimesh


def dominant_axis(m, align_tol_deg=2.0, snap_deg=5.0, max_candidates=4):
    """Return candidate extrusion axes ranked by perpendicular-face area.

    Each raw axis (area-weighted mean normal of a cluster) is offered as is,
    and — when within `snap_deg` of a global axis — also in its snapped
    form, snapped first. The caller scores every candidate and keeps the
    best; snapping is therefore a *hypothesis*, not a decision: a part
    tilted 3 degrees keeps its true axis if the snapped one fits worse.
    """
    n = m.face_normals
    a = m.area_faces
    cands = []
    used = np.zeros(len(n), bool)
    order = np.argsort(-a)
    cos_tol = np.cos(np.radians(align_tol_deg))
    for i in order:
        if used[i]:
            continue
        d = n @ n[i]
        grp = np.abs(d) > cos_tol
        # area-weighted mean normal, sign-aligned with the seed
        w = (a[grp] * np.sign(d[grp]))[:, None]
        ax = (n[grp] * w).sum(axis=0)
        L = np.linalg.norm(ax)
        ax = ax / L if L > 1e-12 else n[i]
        cands.append((a[grp].sum(), ax))
        used |= grp
        if len(cands) > 40:
            break
    cands.sort(key=lambda t: -t[0])
    out = []

    def add(frac, ax):
        ax = ax / np.linalg.norm(ax)
        if not any(abs(ax @ o) > 0.99999 for _, o in out):
            out.append((frac, ax))

    cos_snap = np.cos(np.radians(snap_deg))
    for area, ax in cands[:max_candidates]:
        frac = area / m.area
        snapped = None
        for g in np.eye(3):
            if abs(ax @ g) > cos_snap:
                snapped = g * np.sign(ax @ g)
                break
        if snapped is not None:
            add(frac, snapped)
            if abs(snapped @ ax) < 0.99999:
                add(frac, ax)
        else:
            add(frac, ax)
    return out


def score_axis(m, axis, keep_small_levels=True, adaptive=True):
    """Score an axis by the volume fraction living in constant slabs.

    Returns (score, levels, slabs). With `adaptive`, slabs whose section
    changes abruptly somewhere inside them get a level inserted at the step
    (found by bisection), so small features that left no perpendicular face
    big enough to vote still become their own slab.
    """
    levels = slab_levels(m, axis, keep_small=keep_small_levels)
    if len(levels) < 2:
        return 0.0, [], []
    slabs = slab_sections(m, axis, levels)
    if adaptive:
        levels, slabs = refine_levels(m, axis, levels, slabs)
    if not slabs:
        return 0.0, levels, slabs
    tot = sum(s['area'] * (s['z1'] - s['z0']) for s in slabs)
    const = sum(s['area'] * (s['z1'] - s['z0']) for s in slabs
                if s['constant'] or _loftable(s))
    return (const / tot if tot > 0 else 0.0), levels, slabs


def _loftable(s):
    """A varying slab whose two end sections have the same topology (same
    number of profiles and holes) and are not wildly different can be
    lofted (taper, draft, chamfer, countersink); it counts as explained
    when scoring an axis."""
    ends = s.get('ends')
    if not ends:
        return False
    (za, pa), (zb, pb) = ends
    if not pa or not pb or len(pa) != len(pb):
        return False
    ha = sorted(len(p.interiors) for p in pa)
    hb = sorted(len(p.interiors) for p in pb)
    if ha != hb:
        return False
    if _shape_iou(pa, pb) < 0.6:
        return False
    # linear change only: the section area must vary monotonically through
    # the slab (a taper, chamfer or countersink does; a slab cut by a
    # cross hole dips in the middle and cannot be a ruled loft)
    areas = [sum(p.area for p in pa)]
    areas += [sum(p.area for p in polys) for _, polys in s.get('sections', [])]
    areas += [sum(p.area for p in pb)]
    d = np.diff(areas)
    tol_a = 1e-3 * max(areas)
    mono = np.all(d >= -tol_a) or np.all(d <= tol_a)
    return bool(mono)


# Two perpendicular faces whose heights differ by less than this are one
# plane (float noise, a 0.1-degree export tilt); farther apart and not
# edge-connected they are a real step, however shallow.
COPLANAR_TOL = 0.02


def slab_levels(m, axis, align_tol_deg=2.0, min_level_area_frac=0.002,
                merge_tol=0.25, keep_small=True, min_level_area_abs=1.0,
                min_level_faces=2):
    """Heights along axis of planar faces perpendicular to it, clustered
    into discrete levels. Returns sorted list of level heights.

    Facets are first grouped into edge-connected faces (a face tilted a
    tenth of a degree spans a range of heights but is one face); faces are
    then merged into one level only when their heights agree within
    COPLANAR_TOL. A disconnected face a quarter of a millimetre away is a
    ledge and gets its own level — `merge_tol` no longer swallows it; it
    only bounds the extremes check below.

    A level is kept if it carries >= min_level_area_frac of the total area,
    or (keep_small) at least `min_level_faces` facets and
    `min_level_area_abs` mm^2 — a 3 mm boss on a 100 mm plate is a
    feature, not noise.
    """
    n = m.face_normals
    a = m.area_faces
    cos_tol = np.cos(np.radians(align_tol_deg))
    perp = np.abs(n @ axis) > cos_tol
    tri_h = m.triangles_center @ axis
    if not perp.any():
        return []
    # connected faces among the perpendicular facets
    import networkx as nx
    G = nx.Graph()
    idx = np.where(perp)[0]
    G.add_nodes_from(idx.tolist())
    adj = m.face_adjacency
    both = perp[adj[:, 0]] & perp[adj[:, 1]]
    G.add_edges_from(adj[both].tolist())
    faces = []           # (weighted height, weight, count)
    for comp in nx.connected_components(G):
        ci = np.fromiter(comp, int)
        w = a[ci]
        faces.append((float((tri_h[ci] * w).sum() / w.sum()), float(w.sum()), len(ci)))
    faces.sort()
    levels = []
    cur_h, cur_w, cur_n = faces[0][0] * faces[0][1], faces[0][1], faces[0][2]
    last = faces[0][0]
    for h, w, c in faces[1:]:
        if h - last > COPLANAR_TOL:
            levels.append((cur_h / cur_w, cur_w, cur_n))
            cur_h, cur_w, cur_n = 0.0, 0.0, 0
        cur_h += h * w
        cur_w += w
        cur_n += c
        last = h
    levels.append((cur_h / cur_w, cur_w, cur_n))
    total = m.area
    keep = [h for h, w, c in levels
            if w > min_level_area_frac * total
            or (keep_small and c >= min_level_faces and w >= min_level_area_abs)]
    # ensure extremes are present
    lo = float((m.vertices @ axis).min())
    hi = float((m.vertices @ axis).max())
    if not keep or abs(keep[0] - lo) > merge_tol:
        keep.insert(0, lo)
    if abs(keep[-1] - hi) > merge_tol:
        keep.append(hi)
    return keep


# Two sections are "the same shape" when their intersection-over-union is at
# least this. 0.985 tolerates facet noise on curved boundaries yet still
# rejects a 3-degree tilt over a few mm of height, or a small pocket
# appearing.
SHAPE_IOU_MIN = 0.985


def _section_polys(m, axis, zs, T):
    """Cross-sections at heights `zs` (along axis), as shapely polygons in
    the common 2-D basis `T`. One pass over the mesh for all heights."""
    from shapely.geometry import Polygon
    origin = np.zeros(3)
    try:
        lines, to3ds, _ = trimesh.intersections.mesh_multiplane(
            m, origin, axis, np.asarray(zs, float))
    except Exception:
        lines, to3ds = None, None
    out = []
    for i, z in enumerate(zs):
        polys = None
        if lines is not None:
            seg = lines[i]
            if len(seg):
                try:
                    path = trimesh.load_path(seg)
                    polys = [_reproject(p, to3ds[i], T)
                             for p in path.polygons_full]
                except Exception:
                    polys = None
        if not polys:                    # fall back to the per-plane path
            sec = m.section(plane_origin=axis * z, plane_normal=axis)
            if sec is None:
                out.append([])
                continue
            planar, to3d = sec.to_2D()
            polys = [_reproject(p, to3d, T) for p in planar.polygons_full]
        out.append([p for p in polys if p.is_valid and p.area > 0])
    return out


def _shape_iou(pa, pb):
    """Intersection-over-union of two polygon sets."""
    from shapely.ops import unary_union
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    ua, ub = unary_union(pa), unary_union(pb)
    inter = ua.intersection(ub).area
    union = ua.union(ub).area
    return inter / union if union > 0 else 0.0


def slab_sections(m, axis, levels, n_check=3):
    """For each slab between consecutive levels, extract the cross-section
    polygons and verify the section is constant through the slab.

    Constancy compares the *shapes* of the sections (IoU), not only their
    areas: a tilted plate has constant area but a drifting section.

    Returns list of dicts: {z0, z1, polygons(shapely), constant(bool),
    area, sections: [(z, polys)]} with polygons in the axis basis.
    """
    T = _axis_basis(axis)
    slabs = []
    for z0, z1 in zip(levels[:-1], levels[1:]):
        if z1 - z0 < 1e-6:
            continue
        nc = n_check if z1 - z0 >= 0.3 else 1     # thin slab: mid section only
        zs = np.linspace(z0, z1, nc + 2)[1:-1]
        # nudge off exact mid-heights: a plane through mesh vertices yields
        # degenerate segments and no closed loops
        zs = zs + 0.0137 * (z1 - z0) / (nc + 2)
        # end sections just inside the slab: chamfers, countersinks and
        # drafts show up there but not at the interior samples
        d = min(END_DELTA, 0.1 * (z1 - z0))
        ends = [z0 + d, z1 - d]
        polys_all = _section_polys(m, axis, list(zs) + ends, T)
        polys_per_z, end_polys = polys_all[:len(zs)], polys_all[len(zs):]
        good = [(z, p) for z, p in zip(zs, polys_per_z) if p]
        if not good:
            continue
        areas = np.array([sum(p.area for p in polys) for _, polys in good])
        mid_z, mid_polys = good[len(good) // 2]
        area_ok = areas.std() < max(0.01 * areas.mean(), 0.5)
        shape_ok = all(_shape_iou(good[i][1], good[i + 1][1]) >= SHAPE_IOU_MIN
                       for i in range(len(good) - 1))
        interior_const = bool(area_ok and shape_ok and len(good) == len(zs))
        ends_ok = all(_shape_iou(mid_polys, ep) >= SHAPE_IOU_MIN for ep in end_polys if ep)
        slabs.append({
            'z0': float(z0), 'z1': float(z1),
            'polygons': mid_polys,
            'constant': bool(interior_const and ends_ok),
            'interior_constant': interior_const,
            'area': float(areas.mean()),
            'sections': good,
            'ends': [(ends[0], end_polys[0]), (ends[1], end_polys[1])],
        })
    return slabs


# how far inside a slab the end sections are taken (mm)
END_DELTA = 0.1


# Two sections are "the same" for level refinement when their boundaries
# are within this distance (mm) — an absolute, tolerance-like criterion, so
# the start of a chamfer or countersink is found where the geometry begins
# to move, not where an area ratio happens to cross a threshold.
SECTION_DIST_TOL = 0.04


def _shape_dist(pa, pb):
    """Hausdorff distance between two polygon sets' boundaries (inf if one
    is empty and the other is not)."""
    from shapely.ops import unary_union
    if not pa and not pb:
        return 0.0
    if not pa or not pb:
        return float('inf')
    ua, ub = unary_union(pa), unary_union(pb)
    try:
        return float(ua.boundary.hausdorff_distance(ub.boundary))
    except Exception:
        return float('inf')


def _same_section(pa, pb):
    return _shape_dist(pa, pb) <= SECTION_DIST_TOL and _shape_iou(pa, pb) >= SHAPE_IOU_MIN


def _topology(polys):
    """(number of polygons, sorted hole counts) of a section."""
    return (len(polys), tuple(sorted(len(p.interiors) for p in polys)))


def _snap_to_vertex_height(z, heights, tol=0.15):
    """Snap a level to the nearest mesh vertex height (feature edges lie on
    vertices), if one is within tol."""
    if len(heights) == 0:
        return z
    i = int(np.argmin(np.abs(heights - z)))
    return float(heights[i]) if abs(heights[i] - z) <= tol else float(z)


def refine_levels(m, axis, levels, slabs, max_extra=16, min_gap=0.08):
    """Insert levels where a slab's section starts or stops changing.

    A slab is sampled at its ends and at interior heights. Where a run of
    identical sections meets a differing one, the height at which the
    change begins (or ends) is found by bisection on the boundary distance
    and snapped to the nearest mesh-vertex height. The constant part is
    then extruded and the varying part lofted (chamfers, countersinks,
    drafts, tapered ribs). A slab that varies from end to end is left whole
    and lofted. Returns (levels, slabs) recomputed with the extra levels."""
    T = _axis_basis(axis)
    levels = list(levels)
    heights = np.unique(np.round(m.vertices @ axis, 6))
    inserted = set()
    added = 0
    for _ in range(max_extra):
        new_level = None
        for s in (slab_sections(m, axis, levels) if added else slabs):
            if s['constant']:
                continue
            samples = []
            if s.get('ends'):
                ze, pe = s['ends'][0]
                if pe:
                    samples.append((ze, pe))
            samples += list(s['sections'])
            if s.get('ends'):
                ze, pe = s['ends'][1]
                if pe:
                    samples.append((ze, pe))
            if len(samples) < 2:
                continue
            # a topology change (a boss ends, a hole starts) is a level in
            # its own right, even inside a slab that varies end to end
            topo = [_topology(p) for _, p in samples]
            for i in range(len(samples) - 1):
                if topo[i] == topo[i + 1]:
                    continue
                lo, hi = samples[i][0], samples[i + 1][0]
                t0 = topo[i]
                for _ in range(16):
                    mid = 0.5 * (lo + hi)
                    pm = _section_polys(m, axis, [mid], T)[0]
                    if _topology(pm) == t0:
                        lo = mid
                    else:
                        hi = mid
                    if abs(hi - lo) < 0.002:
                        break
                zc = _snap_to_vertex_height(0.5 * (lo + hi), heights)
                if min(abs(zc - l) for l in levels) > min_gap:
                    new_level = zc
                    break
            if new_level is not None:
                break
            same = [_same_section(samples[i][1], samples[i + 1][1])
                    for i in range(len(samples) - 1)]
            if all(same) or not any(same):
                continue                     # constant (noise) or varying end to end
            # every transition: constant->varying (a change begins) or
            # varying->constant (a change ends); take the first that yields
            # a level not already present
            for k in range(1, len(same)):
                if same[k] == same[k - 1]:
                    continue
                if same[k - 1]:
                    # samples k-1, k alike; k+1 differs: change begins in (k, k+1)
                    ref = samples[k][1]
                    lo, hi = samples[k][0], samples[k + 1][0]
                else:
                    # varying up to k; samples k, k+1 alike: change ends in (k-1, k)
                    ref = samples[k][1]
                    lo, hi = samples[k][0], samples[k - 1][0]
                # bisect: `lo` side matches ref, `hi` side does not
                for _ in range(14):
                    mid = 0.5 * (lo + hi)
                    pm = _section_polys(m, axis, [mid], T)[0]
                    if _same_section(ref, pm):
                        lo = mid
                    else:
                        hi = mid
                    if abs(hi - lo) < 0.003:
                        break
                zc = _snap_to_vertex_height(lo, heights)
                if min(abs(zc - l) for l in levels) > min_gap:
                    new_level = zc
                    break
            if new_level is not None:
                break
        if new_level is None:
            break
        levels = sorted(levels + [float(new_level)])
        inserted.add(float(new_level))
        added += 1
    if added:
        # keep an inserted level only if it made things explainable: a slab
        # that is neither constant nor loftable (a cross hole seen as a
        # notch, a fillet) is better left whole for the extrusion + feature
        # cut, so drop the inserted levels bounding it and recompute
        for _ in range(len(inserted) + 1):
            slabs = slab_sections(m, axis, levels)
            drop = set()
            for sl in slabs:
                if sl['constant'] or _loftable(sl):
                    continue
                for z in (sl['z0'], sl['z1']):
                    if z in inserted:
                        drop.add(z)
            if not drop:
                break
            levels = [l for l in levels if l not in drop]
            inserted -= drop
        slabs = slab_sections(m, axis, levels)
    return levels, slabs


def _reproject(poly, to3d, T):
    """Map a shapely polygon from trimesh's planar frame -> axis-basis 2D."""
    from shapely.geometry import Polygon

    def ring(coords):
        c = np.array(coords)
        h = np.column_stack([c, np.zeros(len(c)), np.ones(len(c))])
        world = (to3d @ h.T).T[:, :3]
        return np.column_stack([world @ T[:3, 0], world @ T[:3, 1]])

    return Polygon(ring(poly.exterior.coords),
                   [ring(r.coords) for r in poly.interiors])


def _axis_basis(axis):
    """Right-handed basis with `axis` as +Z."""
    z = axis / np.linalg.norm(axis)
    x = np.cross([0, 1, 0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1, 0, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    return T
