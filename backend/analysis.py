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


def _solid_check(path):
    """(solids, closed) by reading the STEP back, or (None, None) if it
    cannot be read.

    `closed` is False when the file carries open shells: geometry a CAD
    package imports as surfaces rather than bodies you can model against.
    """
    try:
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.STEPControl import STEPControl_Reader

        reader = STEPControl_Reader()
        if reader.ReadFile(path) != 1:      # IFSelect_RetDone
            return None, None
        reader.TransferRoots()
        shape = reader.OneShape()
        if shape.IsNull():
            return None, None
        n = 0
        e = TopExp_Explorer(shape, TopAbs_SOLID)
        while e.More():
            n += 1
            e.Next()
        from stl2prism.rebuild import _naked_edges
        return n, _naked_edges(shape) == 0
    except Exception:
        return None, None


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
    # The entity count says what the writer *claimed*; reading the file back
    # says what a CAD package will actually find. They disagree exactly
    # where it matters: an open shell is written as a MANIFOLD_SOLID_BREP
    # but imports as surfaces, so reporting "2 solids" for a file holding
    # none is the one number a user must not be given wrongly.
    real_solids, closed = _solid_check(path)
    return {
        'file_size': os.path.getsize(path),
        'faces': faces,
        'solids': real_solids if real_solids is not None else solids,
        'solids_claimed': solids,
        'closed': closed,
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
    # Label faces from the connectivity split() uses, because split() does
    # not hand back which face went where — but build each body's mesh with
    # split() itself. Its parts are what the pipeline will convert, and it
    # re-processes each one (merging and dropping faces), so a submesh of
    # the same component can differ by a few faces. Reporting the component
    # count made a selection silently lose a body whose count moved by 4.
    comps = connected_components(m.face_adjacency, nodes=np.arange(n_faces))
    tri_body = np.full(n_faces, -1, dtype=np.int64)
    order = sorted(range(len(comps)), key=lambda i: -len(comps[i]))
    parts = sorted(m.split(only_watertight=False), key=lambda p: -len(p.faces))
    subs = []
    for rank, i in enumerate(order[:max_bodies]):
        # split() sorts by size as we do, so ranks line up; fall back to a
        # submesh if the two disagree in length for any reason.
        if rank < len(parts) and abs(len(parts[rank].faces) - len(comps[i])) <= 8:
            subs.append(parts[rank])
        else:
            subs.append(m.submesh([comps[i]], append=True, repair=False))
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
            # the body the pipeline will convert, not the raw component:
            # split() re-processes each part, and reporting the component's
            # count made the shell key miss by a few faces
            'triangles': int(len(sub.faces)),
            'watertight': closed,
            'volume': round(float(abs(sub.volume)), 4) if closed else None,
            'size': [round(float(v), 4) for v in ext],
            'center': [round(float(v), 4) for v in (sub.bounds[0] + ext / 2)],
        })
    # Rank by volume where the shell is closed, and by the cube of its
    # longest side where it is not. An open shell has no volume, and
    # ranking on volume alone made an OBJ whose two housing halves are open
    # suggest a single 2 mm3 screw and none of the part the user wanted.
    def weight(b):
        if b['volume']:
            return float(b['volume'])
        return float(max(b['size'])) ** 3 if b['size'] else 0.0
    ws = [weight(b) for b in bodies]
    biggest = max(ws) if ws else 0.0
    for b, w in zip(bodies, ws):
        b['suggested'] = bool(biggest > 0 and w >= DEFAULT_PICK_FRAC * biggest)
    return {'bodies': bodies, 'triangle_body': tri_body.tolist(),
            'truncated': len(comps) > max_bodies}
