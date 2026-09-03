"""File statistics for the web UI: input mesh (STL/OBJ) and output STEP overviews."""
import math
import os
import re


def sanitize(obj):
    """Replace NaN/inf with None so the payload is strict JSON."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def mesh_stats(path):
    # Same loader as the pipeline, so watertight/vertex/volume numbers here
    # describe the geometry the conversion will actually see (an OBJ with
    # per-corner normals would otherwise report as open with 4x the vertices).
    from stl2prism.mesh_prep import load_mesh, suggest_units, unit_warning
    m = load_mesh(path)
    ext = m.bounding_box.primitive.extents
    stats = {
        'units_suggestion': suggest_units(m),
        # None unless the mesh, read as mm, is an implausible size for a
        # part: a wrong unit is cheap to fix here and expensive later.
        'unit_warning': unit_warning(m),
        'file_size': os.path.getsize(path),
        'triangles': int(len(m.faces)),
        'vertices': int(len(m.vertices)),
        'watertight': bool(m.is_watertight),
        'bbox_mm': [round(float(v), 2) for v in ext],
        'surface_area_mm2': round(float(m.area), 2),
        'volume_mm3': round(float(m.volume), 2) if m.is_watertight else None,
        'bodies': int(m.body_count),
        'edge_mm': {
            'min': round(float(m.edges_unique_length.min()), 3),
            'mean': round(float(m.edges_unique_length.mean()), 3),
            'max': round(float(m.edges_unique_length.max()), 3),
        } if len(m.edges_unique_length) else None,
    }
    return stats


_FACE_KINDS = ('PLANE', 'CYLINDRICAL_SURFACE', 'CONICAL_SURFACE',
               'SPHERICAL_SURFACE', 'TOROIDAL_SURFACE', 'B_SPLINE_SURFACE')


def step_stats(path):
    """Cheap textual scan of the STEP file: face count and surface types.

    STEP is line-oriented; ADVANCED_FACE entities map 1:1 to BREP faces,
    which is the number CAD users compare against the triangle count.
    """
    faces = 0
    solids = 0
    kinds = {k: 0 for k in _FACE_KINDS}
    pat = re.compile(r'=\s*([A-Z_0-9]+)\s*\(')
    with open(path, errors='replace') as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            name = m.group(1)
            if name == 'ADVANCED_FACE':
                faces += 1
            elif name in ('MANIFOLD_SOLID_BREP', 'BREP_WITH_VOIDS'):
                solids += 1
            elif name in kinds:
                kinds[name] += 1
            elif name.startswith('B_SPLINE_SURFACE'):
                kinds['B_SPLINE_SURFACE'] += 1
    return {
        'file_size': os.path.getsize(path),
        'faces': faces,
        'solids': solids,
        'surface_types': {
            'planes': kinds['PLANE'],
            'cylinders': kinds['CYLINDRICAL_SURFACE'],
            'cones': kinds['CONICAL_SURFACE'],
            'spheres': kinds['SPHERICAL_SURFACE'],
            'tori': kinds['TOROIDAL_SURFACE'],
            'freeform': kinds['B_SPLINE_SURFACE'],
        },
    }


# Bodies smaller than this share of the largest body's volume are left
# unticked by default: on a 72-shell controller that selects the two
# housing halves and none of the 70 screws and buttons.
DEFAULT_PICK_FRAC = 0.05


def body_list(path, max_bodies=400):
    """One entry per connected shell of the mesh, largest first.

    Returns {'bodies': [...], 'triangle_body': [...]} where `triangle_body`
    maps every triangle of the mesh, in the order load_mesh yields them, to
    the index of the body it belongs to. That mapping is what lets the
    viewer colour bodies and turn a ray-cast triangle into a selection
    without loading a mesh per body.

    Volumes and sizes are in file units; the caller scales by the chosen
    unit. A shell that is not closed reports volume None.
    """
    import numpy as np
    from trimesh.graph import connected_components
    from stl2prism.mesh_prep import load_mesh
    m = load_mesh(path)
    n_faces = len(m.faces)
    # split() does not hand back which face went where, so label the faces
    # directly from the same connectivity split() uses.
    comps = connected_components(m.face_adjacency, nodes=np.arange(n_faces))
    tri_body = np.full(n_faces, -1, dtype=np.int64)
    order = sorted(range(len(comps)), key=lambda i: -len(comps[i]))
    subs = [m.submesh([comps[i]], append=True, repair=False)
            for i in order[:max_bodies]]
    # Keys are measured against the whole file's box, so they must all be
    # built together (see pipeline.shell_keys).
    from stl2prism.pipeline import shell_keys
    keys = shell_keys(subs)
    bodies = []
    for new_i, old_i in enumerate(order[:max_bodies]):
        faces = comps[old_i]
        tri_body[faces] = new_i
        sub = subs[new_i]
        closed = bool(sub.is_watertight)
        ext = sub.bounds[1] - sub.bounds[0]
        bodies.append({
            'index': new_i,
            'key': list(keys[new_i]),
            'triangles': int(len(faces)),
            'watertight': closed,
            'volume': round(float(abs(sub.volume)), 4) if closed else None,
            'size': [round(float(v), 4) for v in ext],
            'center': [round(float(v), 4) for v in (sub.bounds[0] + ext / 2)],
        })
    vols = [b['volume'] or 0.0 for b in bodies]
    biggest = max(vols) if vols else 0.0
    for b, v in zip(bodies, vols):
        b['suggested'] = bool(biggest > 0 and v >= DEFAULT_PICK_FRAC * biggest)
    return {'bodies': bodies, 'triangle_body': tri_body.tolist(),
            'truncated': len(comps) > max_bodies}
