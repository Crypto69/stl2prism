"""One traced mesh section for the web app: the slider's plane, cut and
fitted, as JSON for the 3D preview and as a Fusion sketch script.

The mesh is loaded once per job and kept (one entry; a second job evicts
the first), because a slider generates many cuts of the same file and the
cut itself takes milliseconds while loading a 150k-face STL takes a
second."""
import threading

import numpy as np

from stl2prism.mesh_prep import load_mesh, UNIT_SCALE
from stl2prism.section_fit import section_preview

_lock = threading.Lock()
_cache = {'key': None, 'mesh': None}


def _mesh(path, units, scale):
    key = (path, units, float(scale))
    with _lock:
        if _cache['key'] == key:
            return _cache['mesh']
    m = load_mesh(path)
    k = UNIT_SCALE.get(units, 1.0) * float(scale)
    if k != 1.0:
        m.apply_scale(k)
    with _lock:
        _cache.update(key=key, mesh=m)
    return m


def trace(path, axis, offset, tol=0.08, units='mm', scale=1.0, join=2.5, outline=False, trim=0.0):
    """Section of the mesh at `offset` mm from its bounding-box centre
    along `axis` ('x'|'y'|'z'), in the converted (mm) frame. Open chains
    stay open (a leaky mesh), but free ends within `join` mm of each other
    are joined first; `outline` drops the open chains altogether. Returns the section_preview dict plus 'at' (absolute
    mm coordinate), 'centre' and 'extent' along the axis, so the UI can
    place its slider."""
    m = _mesh(path, units, scale)
    k = 'xyz'.index(axis)
    c = m.bounding_box.centroid
    ext = float(m.extents[k])
    at = float(c[k] + offset)
    origin = np.zeros(3)
    origin[k] = at
    normal = np.zeros(3)
    normal[k] = 1.0
    sec = section_preview(m.vertices, m.faces, origin, normal, tol=tol, join_mm=join,
                          closed_only=outline, trim_mm=trim)
    sec.update(axis=axis, at=at, offset=float(offset), centre=float(c[k]), extent=ext,
               bbox_centre=np.asarray(c).tolist())
    return sec
