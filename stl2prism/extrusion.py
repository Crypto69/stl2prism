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


def dominant_axis(m, align_tol_deg=2.0):
    """Return candidate extrusion axes ranked by perpendicular-face area,
    each snapped to global XYZ when within 5 degrees."""
    n = m.face_normals
    a = m.area_faces
    cands = []
    used = np.zeros(len(n), bool)
    order = np.argsort(-a)
    cos_tol = np.cos(np.radians(align_tol_deg))
    for i in order:
        if used[i]:
            continue
        d = np.abs(n @ n[i])
        grp = d > cos_tol
        cands.append((a[grp].sum(), n[i]))
        used |= grp
        if len(cands) > 40:
            break
    cands.sort(key=lambda t: -t[0])
    out = []
    for area, ax in cands[:4]:
        for g in np.eye(3):
            if abs(ax @ g) > np.cos(np.radians(5.0)):
                ax = g * np.sign(ax @ g)
                break
        ax = ax / np.linalg.norm(ax)
        if not any(abs(ax @ o) > 0.999 for _, o in out):
            out.append((area / m.area, ax))
    return out


def score_axis(m, axis):
    """Score an axis by the volume fraction living in constant slabs."""
    levels = slab_levels(m, axis)
    if len(levels) < 2:
        return 0.0, [], []
    slabs = slab_sections(m, axis, levels)
    if not slabs:
        return 0.0, levels, slabs
    tot = sum(s['area'] * (s['z1'] - s['z0']) for s in slabs)
    const = sum(s['area'] * (s['z1'] - s['z0']) for s in slabs if s['constant'])
    return (const / tot if tot > 0 else 0.0), levels, slabs


def slab_levels(m, axis, align_tol_deg=2.0, min_level_area_frac=0.002,
                merge_tol=0.25):
    """Heights along axis of planar faces perpendicular to it, clustered
    into discrete levels. Returns sorted list of level heights."""
    n = m.face_normals
    a = m.area_faces
    cos_tol = np.cos(np.radians(align_tol_deg))
    perp = np.abs(n @ axis) > cos_tol
    tri_h = m.triangles_center @ axis
    hs = tri_h[perp]
    ws = a[perp]
    if len(hs) == 0:
        return []
    order = np.argsort(hs)
    hs, ws = hs[order], ws[order]
    levels = []          # (weighted height, weight)
    cur_h, cur_w = hs[0] * ws[0], ws[0]
    last = hs[0]
    for h, w in zip(hs[1:], ws[1:]):
        if h - last > merge_tol:
            levels.append((cur_h / cur_w, cur_w))
            cur_h, cur_w = 0.0, 0.0
        cur_h += h * w
        cur_w += w
        last = h
    levels.append((cur_h / cur_w, cur_w))
    total = m.area
    keep = [h for h, w in levels if w > min_level_area_frac * total]
    # ensure extremes are present
    lo = float((m.vertices @ axis).min())
    hi = float((m.vertices @ axis).max())
    if not keep or abs(keep[0] - lo) > merge_tol:
        keep.insert(0, lo)
    if abs(keep[-1] - hi) > merge_tol:
        keep.append(hi)
    return keep


def slab_sections(m, axis, levels, n_check=3):
    """For each slab between consecutive levels, extract the cross-section
    polygons and verify the section is constant through the slab.

    Returns list of dicts: {z0, z1, polygons(shapely), constant(bool), to3d}
    where to3d is the 4x4 transform mapping section 2D coords -> world.
    """
    from shapely.geometry import Polygon
    T = _axis_basis(axis)
    slabs = []
    for z0, z1 in zip(levels[:-1], levels[1:]):
        if z1 - z0 < 0.15:
            continue
        zs = np.linspace(z0, z1, n_check + 2)[1:-1]
        polys_per_z, areas = [], []
        for z in zs:
            sec = m.section(plane_origin=axis * z, plane_normal=axis)
            if sec is None:
                polys_per_z.append(None)
                areas.append(0.0)
                continue
            planar, to3d = sec.to_2D()
            polys = list(planar.polygons_full)
            polys_per_z.append((polys, to3d))
            areas.append(sum(p.area for p in polys))
        good = [p for p in polys_per_z if p]
        if not good:
            continue
        areas = np.array(areas)
        mid_polys, mid_to3d = good[len(good) // 2]
        constant = areas.std() < max(0.01 * areas.mean(), 0.5)
        # re-project polygons into the consistent axis basis so downstream
        # rebuilding shares one 2D frame across all slabs
        T = _axis_basis(axis)
        proj = [_reproject(p, mid_to3d, T) for p in mid_polys]
        slabs.append({
            'z0': float(z0), 'z1': float(z1),
            'polygons': proj,
            'constant': bool(constant),
            'area': float(areas.mean()),
        })
    return slabs


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
