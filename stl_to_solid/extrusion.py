"""Detect extrusion (prismatic) structure in a mesh.

Approach (Point2Cyl-inspired, classical implementation):
  1. Cluster face normals on the Gaussian sphere to find the dominant
     extrusion axis: the direction whose +/- aligned faces carry the
     most area (top/bottom faces of slabs).
  2. Collect the heights (along the axis) of those perpendicular planar
     faces -> cluster into discrete Z-levels.
  3. Between consecutive levels, cross-section the mesh -> slab profiles.
"""
import time
import numpy as np
import trimesh


# A candidate with more levels than this is not an extrusion: it is a
# faceted curved shell whose facets happen to lie perpendicular to the
# axis (a 12k-face cavity offered 768 of them). Scoring it would mean
# sectioning every slab on every refinement round, for hours; a real
# extrusion has tens of levels. Such a candidate scores 0 and is skipped.
MAX_LEVELS = 120


class AxisSearchTimeout(Exception):
    """The axis-search deadline passed inside a section pass."""


def _check_deadline(deadline):
    if deadline is not None and time.monotonic() > deadline:
        raise AxisSearchTimeout('axis search budget spent')


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


def score_axis(m, axis, keep_small_levels=True, adaptive=True, deadline=None):
    """Score an axis by the volume fraction living in constant slabs.

    Returns (score, levels, slabs). With `adaptive`, slabs whose section
    changes abruptly somewhere inside them get a level inserted at the step
    (found by bisection), so small features that left no perpendicular face
    big enough to vote still become their own slab.

    A candidate with more than MAX_LEVELS levels scores 0 without being
    sectioned (its levels are still returned, so the caller can say why).
    `deadline` is a time.monotonic() value: past it, the first section
    pass raises AxisSearchTimeout and the refinement stops where it is.
    """
    levels = slab_levels(m, axis, keep_small=keep_small_levels)
    if len(levels) < 2:
        return 0.0, [], []
    if len(levels) > MAX_LEVELS:
        return 0.0, levels, []
    slabs = slab_sections(m, axis, levels, deadline=deadline)
    if adaptive:
        levels, slabs = refine_levels(m, axis, levels, slabs, deadline=deadline)
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
    areas = [sum(p.area for p in polys) for _, polys in _slab_samples(s)]
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


def _section_polys(m, axis, zs, T, deadline=None, fast=False):
    """Cross-sections at heights `zs` (along axis), as shapely polygons in
    the common 2-D basis `T`. One pass over the mesh for all heights.
    Raises AxisSearchTimeout when `deadline` has passed.

    With `fast`, the segments are chained into loops here instead of by
    trimesh's path machinery (the same rings, nested the same way, but a
    ring may start at a different vertex): for sections that are only
    compared — a bisection probe — never for a profile that gets built.
    """
    from shapely.geometry import Polygon
    if not len(zs):
        return []
    _check_deadline(deadline)
    origin = np.zeros(3)
    try:
        lines, to3ds, _ = trimesh.intersections.mesh_multiplane(
            m, origin, axis, np.asarray(zs, float))
    except Exception:
        lines, to3ds = None, None
    out = []
    for i, z in enumerate(zs):
        _check_deadline(deadline)
        polys = None
        if lines is not None:
            seg = lines[i]
            if len(seg):
                if fast:
                    polys = _fast_polygons(seg, to3ds[i], T)
                if polys is None:
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


def _fast_polygons(seg, to3d, T):
    """The polygons of one section from its (n, 2, 2) segments: endpoints
    merged by trimesh's own rule, chained into closed loops,
    reprojected into the axis basis and nested even-odd (a ring inside an
    odd number of others is a hole of its innermost container, one inside
    an even number is a shell of its own) — the same rule as trimesh's
    polygons_full. Returns None whenever the segments are not a clean set
    of closed loops (a branch vertex, an open chain, a self-touching ring):
    the caller then takes the trimesh path, which repairs such cases."""
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree
    from trimesh.constants import tol_path
    pts = np.asarray(seg, float).reshape(-1, 2)
    ui, inv = trimesh.grouping.unique_rows(pts, digits=tol_path.merge_digits)
    uniq = pts[ui]
    edges = np.asarray(inv).reshape(-1, 2)
    edges = edges[edges[:, 0] != edges[:, 1]]
    n = len(uniq)
    if n < 3 or len(edges) < 3:
        return None
    deg = np.bincount(edges.ravel(), minlength=n)
    if (deg != 2).any():
        return None
    # every vertex has exactly two edge ends: sorting the ends by vertex
    # lines the two neighbours of each vertex up side by side
    order = np.argsort(edges.ravel(), kind='stable')
    nbr = edges[:, ::-1].ravel()[order].reshape(n, 2)
    xy = _to_axis_xy(uniq, to3d, T)
    seen = np.zeros(n, bool)
    rings = []
    for start in range(n):
        if seen[start]:
            continue
        loop = [start]
        seen[start] = True
        prev, cur = start, int(nbr[start, 0])
        while cur != start:
            if seen[cur]:
                return None
            seen[cur] = True
            loop.append(cur)
            a, b = int(nbr[cur, 0]), int(nbr[cur, 1])
            prev, cur = cur, (b if a == prev else a)
        if len(loop) < 3:
            return None
        rings.append(Polygon(xy[loop]))
    if not rings or not all(r.is_valid for r in rings):
        return None
    tree = STRtree(rings)
    outer, inner = tree.query(rings, predicate='contains')
    keep = outer != inner
    outer, inner = outer[keep], inner[keep]
    depth = np.bincount(inner, minlength=len(rings))
    out = []
    for r in np.where(depth % 2 == 0)[0]:
        holes = inner[(outer == r) & (depth[inner] == depth[r] + 1)]
        out.append(Polygon(rings[r].exterior.coords,
                           [rings[j].exterior.coords[::-1] for j in holes]))
    if not all(p.is_valid for p in out):
        return None                   # crossing holes: trimesh repairs those
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
    union = ua.area + ub.area - inter        # one overlay, not two
    return inter / union if union > 0 else 0.0


def slab_sections(m, axis, levels, n_check=3, deadline=None, cache=None):
    """For each slab between consecutive levels, extract the cross-section
    polygons and verify the section is constant through the slab.

    Constancy compares the *shapes* of the sections (IoU), not only their
    areas: a tilted plate has constant area but a drifting section.

    Every slab's probe heights are cut in one pass over the mesh. A slab's
    result depends only on its (z0, z1), so with `cache` (a dict the
    caller keeps between calls) slabs already sectioned are reused and
    only the new ones are cut: level refinement then costs two slabs per
    round instead of all of them.

    Returns list of dicts: {z0, z1, polygons(shapely), constant(bool),
    area, sections: [(z, polys)]} with polygons in the axis basis.
    """
    T = _axis_basis(axis)
    if cache is None:
        cache = {}
    plan = []                      # (key, zs, ends) per slab to cut
    keys = []                      # every slab, in order
    for z0, z1 in zip(levels[:-1], levels[1:]):
        if z1 - z0 < 1e-6:
            continue
        key = (float(z0), float(z1))
        keys.append(key)
        if key in cache:
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
        plan.append((key, list(zs), ends))
    all_z = [z for _, zs, ends in plan for z in zs + ends]
    polys_flat = _section_polys(m, axis, all_z, T, deadline=deadline)
    pos = 0
    for key, zs, ends in plan:
        polys_per_z = polys_flat[pos:pos + len(zs)]
        end_polys = polys_flat[pos + len(zs):pos + len(zs) + 2]
        pos += len(zs) + 2
        cache[key] = _slab_record(key, zs, polys_per_z, ends, end_polys)
    return [cache[k] for k in keys if cache[k] is not None]


def _slab_samples(s):
    """A slab's sections bottom to top, [(z, polys)]: the lower end section,
    the interior ones, the upper end section — an end only when it closed."""
    out = []
    ends = s.get('ends') or [(None, None), (None, None)]
    if ends[0][1]:
        out.append(tuple(ends[0]))
    out += list(s.get('sections', []))
    if ends[1][1]:
        out.append(tuple(ends[1]))
    return out


def _slab_record(key, zs, polys_per_z, ends, end_polys):
    """The slab dict for one (z0, z1) from its sections; None when no
    interior section closed (nothing to represent)."""
    z0, z1 = key
    good = [(z, p) for z, p in zip(zs, polys_per_z) if p]
    if not good:
        return None
    areas = np.array([sum(p.area for p in polys) for _, polys in good])
    _, mid_polys = good[len(good) // 2]
    area_ok = areas.std() < max(0.01 * areas.mean(), 0.5)
    shape_ok = all(_shape_iou(good[i][1], good[i + 1][1]) >= SHAPE_IOU_MIN
                   for i in range(len(good) - 1))
    interior_const = bool(area_ok and shape_ok and len(good) == len(zs))
    ends_ok = all(_shape_iou(mid_polys, ep) >= SHAPE_IOU_MIN for ep in end_polys if ep)
    return {
        'z0': z0, 'z1': z1,
        'polygons': mid_polys,
        'constant': bool(interior_const and ends_ok),
        'interior_constant': interior_const,
        'area': float(areas.mean()),
        'sections': good,
        'ends': [(ends[0], end_polys[0]), (ends[1], end_polys[1])],
    }


# how far inside a slab the end sections are taken (mm)
END_DELTA = 0.1


# Two sections are "the same" for level refinement when their boundaries
# are within this distance (mm) — an absolute, tolerance-like criterion, so
# the start of a chamfer or countersink is found where the geometry begins
# to move, not where an area ratio happens to cross a threshold.
SECTION_DIST_TOL = 0.04


def _boundaries(pa, pb):
    """The two sets' union boundaries, or a distance verdict when one set
    is empty: (ba, bb, None) to compare, (None, None, d) to return d."""
    from shapely.ops import unary_union
    if not pa and not pb:
        return None, None, 0.0
    if not pa or not pb:
        return None, None, float('inf')
    return unary_union(pa).boundary, unary_union(pb).boundary, None


def _shape_dist(pa, pb):
    """Hausdorff distance between two polygon sets' boundaries (inf if one
    is empty and the other is not)."""
    ba, bb, d = _boundaries(pa, pb)
    if d is not None:
        return d
    try:
        return float(ba.hausdorff_distance(bb))
    except Exception:
        return float('inf')


def _same_section(pa, pb):
    """Same shape (IoU) and same boundary (Hausdorff distance). The IoU
    goes first because it is cheap and usually the one that fails; the
    exact Hausdorff distance (quadratic in the vertex count) only runs
    when the vertex-to-vertex distance, an upper bound on it, is not
    already within tolerance."""
    if _shape_iou(pa, pb) < SHAPE_IOU_MIN:
        return False
    if _vertex_dist_bound(pa, pb) <= SECTION_DIST_TOL:
        return True
    return _shape_dist(pa, pb) <= SECTION_DIST_TOL


def _vertex_dist_bound(pa, pb):
    """Largest distance from a boundary vertex of one set to the nearest
    boundary vertex of the other, both ways: at least the discrete
    Hausdorff distance (which measures vertices against whole segments),
    so a value within tolerance settles _same_section without it."""
    import shapely
    from scipy.spatial import cKDTree
    ba, bb, d = _boundaries(pa, pb)
    if d is not None:
        return d
    try:
        va, vb = shapely.get_coordinates(ba), shapely.get_coordinates(bb)
        if not len(va) or not len(vb):
            return float('inf')
        da, _ = cKDTree(vb).query(va)
        db, _ = cKDTree(va).query(vb)
        return float(max(da.max(), db.max()))
    except Exception:
        return float('inf')


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


def refine_levels(m, axis, levels, slabs, max_extra=16, min_gap=0.08,
                  deadline=None):
    """Insert levels where a slab's section starts or stops changing.

    A slab is sampled at its ends and at interior heights. Where a run of
    identical sections meets a differing one, the height at which the
    change begins (or ends) is found by bisection on the boundary distance
    and snapped to the nearest mesh-vertex height. The constant part is
    then extruded and the varying part lofted (chamfers, countersinks,
    drafts, tapered ribs). A slab that varies from end to end is left whole
    and lofted. Returns (levels, slabs) recomputed with the extra levels.

    Slabs are cached by (z0, z1): a round only sections the two slabs the
    new level made. Past `deadline` the refinement stops with the levels
    it has (their slabs are consistent, since a level is only added once
    its bisection finished)."""
    T = _axis_basis(axis)
    levels = list(levels)
    heights = np.unique(np.round(m.vertices @ axis, 6))
    cache = {(s['z0'], s['z1']): s for s in slabs}
    # a slab's candidate levels depend on the slab alone, and a candidate
    # too close to an existing level stays too close as levels are added,
    # so a slab's bisections are done once and never repeated in a later
    # round (the slab that gets split is a new slab with new candidates)
    cands = {}
    inserted = set()
    for _ in range(max_extra):
        try:
            new_level = _next_level(m, axis, levels, slabs, heights, T, min_gap,
                                    deadline, cands)
        except AxisSearchTimeout:
            return levels, slabs      # what there is, without the tidy-up below
        if new_level is None:
            break
        levels = sorted(levels + [float(new_level)])
        inserted.add(float(new_level))
        slabs = slab_sections(m, axis, levels, cache=cache)
    if inserted:
        # keep an inserted level only if it made things explainable: a slab
        # that is neither constant nor loftable (a cross hole seen as a
        # notch, a fillet) is better left whole for the extrusion + feature
        # cut, so drop the inserted levels bounding it and recompute
        for _ in range(len(inserted) + 1):
            slabs = slab_sections(m, axis, levels, cache=cache)
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
        slabs = slab_sections(m, axis, levels, cache=cache)
    return levels, slabs


def _next_level(m, axis, levels, slabs, heights, T, min_gap, deadline, cands):
    """One refinement round: the first height at which a varying slab's
    section starts or stops changing (None when there is none), found by
    bisection. `cands` memoises each slab's candidate generator between
    rounds. Raises AxisSearchTimeout past `deadline`."""
    for s in slabs:
        if s['constant']:
            continue
        key = (s['z0'], s['z1'])
        if key not in cands:
            cands[key] = _slab_level_candidates(m, axis, s, heights, T, deadline)
        for zc in cands[key]:
            if min(abs(zc - l) for l in levels) > min_gap:
                return zc
    return None


def _slab_level_candidates(m, axis, s, heights, T, deadline):
    """The heights at which slab `s` changes section, in the order the
    refinement considers them: topology changes (a boss ends, a hole
    starts) first, then the starts and ends of a gradual change. Each is
    found by bisection on probe sections and snapped to a vertex height.
    A generator, so a candidate is only bisected when it is needed."""
    samples = _slab_samples(s)
    if len(samples) < 2:
        return
    # a probe inside the slab only meets the faces that span it: on a
    # 160k-face part that is a percent of the mesh, and the sections are
    # the same segments
    m = _slab_submesh(m, axis, s['z0'], s['z1'])
    # a topology change (a boss ends, a hole starts) is a level in
    # its own right, even inside a slab that varies end to end
    topo = [_topology(p) for _, p in samples]
    for i in range(len(samples) - 1):
        if topo[i] == topo[i + 1]:
            continue
        t0 = topo[i]
        lo, hi = _bisect(m, axis, T, samples[i][0], samples[i + 1][0],
                         lambda pm: _topology(pm) == t0, 16, 0.002, deadline)
        yield _snap_to_vertex_height(0.5 * (lo + hi), heights)
    _check_deadline(deadline)         # the Hausdorff work below is quadratic
    same = [_same_section(samples[i][1], samples[i + 1][1])
            for i in range(len(samples) - 1)]
    if all(same) or not any(same):
        return                       # constant (noise) or varying end to end
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
        # `lo` side matches ref, `hi` side does not; a looser stop than the
        # topology search, since the start of a gradual change is soft
        lo, hi = _bisect(m, axis, T, lo, hi,
                         lambda pm: _same_section(ref, pm), 14, 0.003, deadline)
        yield _snap_to_vertex_height(lo, heights)


def _slab_submesh(m, axis, z0, z1):
    """The faces of `m` that reach into (z0, z1) along `axis`, as a mesh on
    the same vertices: a plane between the two heights cuts exactly these
    faces, so a section of the sub-mesh is the section of the mesh."""
    fz = (m.vertices @ axis)[m.faces]
    keep = (fz.min(axis=1) <= z1) & (fz.max(axis=1) >= z0)
    if keep.all():
        return m
    return trimesh.Trimesh(m.vertices, m.faces[keep], process=False)


def _bisect(m, axis, T, lo, hi, like_lo, iters, stop, deadline):
    """Narrow (lo, hi) on probe sections: `like_lo(section)` holds at lo and
    not at hi. Stops after `iters` halvings or once the gap is under `stop`.
    Returns the narrowed (lo, hi)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pm = _section_polys(m, axis, [mid], T, deadline=deadline, fast=True)[0]
        if like_lo(pm):
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < stop:
            break
    return lo, hi


def _to_axis_xy(coords, to3d, T):
    """Points of trimesh's planar section frame -> the axis-basis 2-D."""
    c = np.asarray(coords, float)[:, :2]
    h = np.column_stack([c, np.zeros(len(c)), np.ones(len(c))])
    world = (to3d @ h.T).T[:, :3]
    return np.column_stack([world @ T[:3, 0], world @ T[:3, 1]])


def _reproject(poly, to3d, T):
    """Map a shapely polygon from trimesh's planar frame -> axis-basis 2D."""
    from shapely.geometry import Polygon
    return Polygon(_to_axis_xy(poly.exterior.coords, to3d, T),
                   [_to_axis_xy(r.coords, to3d, T) for r in poly.interiors])


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
