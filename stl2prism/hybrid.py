"""Local repair of a prismatic solid that fails validation.

Fusion keeps the face groups it could convert and leaves the rest as
surfaces; the industry norm is never all-or-nothing. Here: find where the
rebuilt solid deviates from the mesh, box those regions, and replace the
solid inside the boxes with the exact faceted geometry of the mesh:

    result = (P - B) U (F ∩ B)

P = prismatic solid, F = faceted solid of the mesh, B = union of patch
boxes. Outside the boxes the clean analytic faces survive; inside, the
part is exact. The patched solid is re-validated by the caller.
"""
import numpy as np
import trimesh


def bad_regions(mesh, metrics_pts, dist, rev_pts, rev_dist, threshold,
                link_radius, margin, max_regions=40):
    """Cluster points that deviate more than `threshold` (from either
    direction) into regions; return a list of (lo, hi) boxes."""
    pts = []
    if len(metrics_pts):
        pts.append(metrics_pts[dist > threshold])
    if rev_pts is not None and len(rev_pts):
        pts.append(rev_pts[rev_dist > threshold])
    if not pts:
        return []
    P = np.vstack(pts)
    if len(P) == 0:
        return []
    from scipy.spatial import cKDTree
    pairs = cKDTree(P).query_pairs(link_radius, output_type='ndarray')
    parent = np.arange(len(P))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(P))])
    boxes = []
    for r in np.unique(roots):
        Q = P[roots == r]
        if len(Q) < 3:
            continue
        lo = Q.min(axis=0) - margin
        hi = Q.max(axis=0) + margin
        boxes.append((lo, hi))
    # merge overlapping boxes
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            lo, hi = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                lo2, hi2 = boxes[j]
                if np.all(lo <= hi2) and np.all(lo2 <= hi):
                    lo, hi = np.minimum(lo, lo2), np.maximum(hi, hi2)
                    used[j] = True
                    merged = True
            out.append((lo, hi))
        boxes = out
    boxes.sort(key=lambda b: -np.prod(b[1] - b[0]))
    return boxes[:max_regions]


_FACETED_CACHE = {}


def faceted_reference(mesh, verbose=False):
    """Exact faceted solid of the mesh, cached per mesh object (patching
    runs several rounds; sewing thousands of triangles is the slow part)."""
    import cadquery as cq
    from .rebuild import faceted_solid
    key = id(mesh)
    hit = _FACETED_CACHE.get(key)
    if hit is not None and hit[0] is mesh:
        return hit[1], hit[2]
    fshape, stats = faceted_solid(mesh, verbose=verbose)
    F = cq.Shape.cast(fshape)
    _FACETED_CACHE.clear()
    _FACETED_CACHE[key] = (mesh, F, stats)
    return F, stats


def patch_solid(solid, mesh, boxes, verbose=True, fuzzies=(5e-2, 2e-2, 1e-1)):
    """(P - B) U (F ∩ B) for the union of the boxes, tried with several
    fuzzy tolerances; the first attempt that yields one valid solid is
    returned as a CadQuery Workplane, else None."""
    import cadquery as cq
    from .rebuild import finish_solid
    from OCP.BRepCheck import BRepCheck_Analyzer
    P = solid.val() if hasattr(solid, 'val') else cq.Shape.cast(solid)
    F, stats = faceted_reference(mesh)
    if not stats.get('is_solid', False):
        if verbose:
            print('[patch] faceted reference is not a closed solid; cannot patch')
        return None
    box_shapes = [cq.Solid.makeBox(*(hi - lo), pnt=cq.Vector(*lo)) for lo, hi in boxes]
    B = box_shapes[0] if len(box_shapes) == 1 else box_shapes[0].fuse(*box_shapes[1:])
    for fuzzy in fuzzies:
        try:
            outside = P.cut(B, tol=fuzzy)
            inside = F.intersect(B, tol=fuzzy)
            R = outside.fuse(inside, tol=fuzzy).clean()
        except Exception as e:
            if verbose:
                print(f'[patch] boolean failed at fuzzy {fuzzy} ({type(e).__name__})')
            continue
        if len(R.Solids()) > 1:
            try:
                R2 = R.Solids()[0].fuse(*R.Solids()[1:], tol=2 * fuzzy).clean()
                if len(R2.Solids()) == 1:
                    R = R2
            except Exception:
                pass
        if len(R.Solids()) == 1 and not BRepCheck_Analyzer(R.wrapped).IsValid():
            try:
                from OCP.ShapeFix import ShapeFix_Shape
                fx = ShapeFix_Shape(R.wrapped)
                fx.SetPrecision(fuzzy)
                fx.SetMaxTolerance(10 * fuzzy)
                fx.Perform()
                R3 = cq.Shape.cast(fx.Shape())
                if len(R3.Solids()) == 1 and BRepCheck_Analyzer(R3.wrapped).IsValid():
                    R = R3
            except Exception:
                pass
        if len(R.Solids()) == 1 and BRepCheck_Analyzer(R.wrapped).IsValid():
            return finish_solid(cq.Workplane('XY').newObject([R]), verbose=False)
    if verbose:
        print('[patch] no fuzzy value produced one valid solid')
    return None


def try_hybrid(solid, mesh, metrics, accept_max, tol, verbose=True,
               max_bad_frac=0.25, rounds=3):
    """Attempt a local patch. Returns (patched_workplane, info) or
    (None, info) when patching is not sensible / failed.

    Up to `rounds` passes: regions still deviating after a patch are added
    to the box set and the patch is rebuilt from the original solid."""
    from .pipeline import tessellate_solid, sample_points
    threshold = max(0.6 * accept_max, 2 * tol)
    diag = float(np.linalg.norm(mesh.bounding_box.primitive.extents))
    link = max(1.0, 4 * tol, 0.01 * diag)
    margin = max(0.5, 3 * tol)
    boxes = []
    info = {'bad_frac': None, 'patches': 0}
    current = solid
    for rnd in range(rounds):
        rb = tessellate_solid(current)
        pts, n_uni = sample_points(mesh)
        _, dist, _ = trimesh.proximity.closest_point(rb, pts)
        rpts, rdist = None, None
        if mesh.is_watertight:
            rpts, _ = sample_points(rb, max(2000, n_uni // 2), include_vertices=False)
            _, rdist, _ = trimesh.proximity.closest_point(mesh, rpts)
        bad_frac = float((dist[:n_uni] > threshold).mean())
        if rnd == 0:
            info['bad_frac'] = bad_frac
            if bad_frac > max_bad_frac:
                if verbose:
                    print(f'[patch] {bad_frac:.0%} of the surface deviates; too much to patch')
                return None, info
        new_boxes = bad_regions(mesh, pts, dist, rpts, rdist, threshold, link, margin)
        if not new_boxes:
            break
        boxes = _merge_boxes(boxes + new_boxes)
        if verbose:
            print(f'[patch] round {rnd + 1}: {len(boxes)} region(s) deviate > {threshold:.2f} mm '
                  f'({bad_frac:.1%} of the surface); replacing them with exact faceted geometry')
        patched = patch_solid(solid, mesh, boxes, verbose=verbose)
        if patched is None:
            break
        current = patched
        info['patches'] = len(boxes)
        info['boxes'] = [[np.round(lo, 2).tolist(), np.round(hi, 2).tolist()] for lo, hi in boxes]
        # good enough? (the caller re-validates properly; this is a cheap check)
        rb2 = tessellate_solid(current)
        _, d2, _ = trimesh.proximity.closest_point(rb2, pts)
        if d2.max() <= accept_max:
            break
    if current is solid:
        return None, info
    return current, info


def _merge_boxes(boxes):
    merged = True
    while merged and len(boxes) > 1:
        merged = False
        out = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            lo, hi = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                lo2, hi2 = boxes[j]
                if np.all(lo <= hi2) and np.all(lo2 <= hi):
                    lo, hi = np.minimum(lo, lo2), np.maximum(hi, hi2)
                    used[j] = True
                    merged = True
            out.append((lo, hi))
        boxes = out
    return boxes
