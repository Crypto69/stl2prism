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
    from stl2prism.mesh_prep import load_mesh, suggest_units
    m = load_mesh(path)
    ext = m.bounding_box.primitive.extents
    stats = {
        'units_suggestion': suggest_units(m),
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
