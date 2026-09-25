"""Traced mesh sections for the web app: the slider's plane (or the x-ray's
stack of planes), cut and fitted, as JSON for the 3D preview and as a
Fusion sketch script.

The mesh is loaded once per job and kept (one entry; a second job evicts
the first), because a slider generates many cuts of the same file and the
cut itself takes milliseconds while loading a 150k-face STL takes a
second. A stack of planes cuts each one with only the faces whose extent
along the axis spans that plane: the per-face extents are worked out once
and kept with the mesh."""
import threading
import time

import numpy as np

from stl_to_solid.mesh_prep import load_mesh, UNIT_SCALE
from stl_to_solid.section_fit import section_preview

# the most sketches one x-ray script may hold (the UI mirrors this)
XRAY_MAX = 500

_lock = threading.Lock()
_cache = {'key': None, 'mesh': None, 'spans': {}}

_ZERO_STATS = {'loops': 0, 'holes': 0, 'open': 0, 'inner': 0, 'lines': 0, 'arcs': 0,
               'circles': 0, 'splines': 0, 'joins': 0, 'max_gap': 0.0, 'dev_max': 0.0,
               'junction_gap': 0.0, 'trimmed': 0}


def _mesh(path, units, scale):
    key = (path, units, float(scale))
    # the lock is held across the load, so two traces on a cold cache (the
    # x-ray's start and end run in parallel) load the file once
    with _lock:
        if _cache['key'] == key:
            return _cache['mesh']
        m = load_mesh(path)
        k = UNIT_SCALE.get(units, 1.0) * float(scale)
        if k != 1.0:
            m.apply_scale(k)
        _cache.update(key=key, mesh=m, spans={})
        return m


def _spans(m, k):
    """(min, max) of every face along axis k, computed once per mesh."""
    with _lock:
        got = _cache['spans'].get(k) if _cache['mesh'] is m else None
        if got is None:
            z = m.vertices[:, k][m.faces]
            got = (z.min(axis=1), z.max(axis=1))
            if _cache['mesh'] is m:
                _cache['spans'][k] = got
        return got


def half_extent(path, axis, units='mm', scale=1.0):
    """Half the part's side along `axis`, mm, in the converted frame."""
    m = _mesh(path, units, scale)
    return float(m.extents['xyz'.index(axis)]) / 2.0


def trace_stack(path, axis, offsets, tol=0.08, units='mm', scale=1.0, join=2.5, outline=False,
                trim=0.0, budget_s=None):
    """One traced section per offset (mm from the bounding-box centre
    along `axis`, 'x'|'y'|'z'), the mesh loaded once and every plane cut
    with just the faces that span it. Each entry is the section_preview
    dict plus 'axis', 'at' (absolute mm), 'offset', 'centre' and 'extent'
    along the axis; a plane no face crosses gives an entry with no curves
    and 'empty': True, so the numbering k/N stays honest. With `budget_s`
    the wall clock is checked between planes and a TimeoutError names how
    many were done."""
    m = _mesh(path, units, scale)
    k = 'xyz'.index(axis)
    V, F = m.vertices, m.faces
    zmin, zmax = _spans(m, k)
    c = m.bounding_box.centroid
    ext = float(m.extents[k])
    t0 = time.monotonic()
    out = []
    for i, off in enumerate(offsets):
        if budget_s is not None and i and time.monotonic() - t0 > budget_s:
            raise TimeoutError(f'{i} of {len(offsets)} slices fitted in {budget_s:.0f} s; raise the '
                               'spacing, narrow the range or tick Outline only')
        at = float(c[k] + off)
        origin = np.zeros(3)
        origin[k] = at
        normal = np.zeros(3)
        normal[k] = 1.0
        mask = (zmin <= at) & (zmax >= at)
        if not mask.any():
            sec = {'origin': origin.tolist(), 'normal': normal.tolist(), 'loops': [], 'open': [],
                   'polylines': [], 'stats': dict(_ZERO_STATS), 'empty': True}
        else:
            sec = section_preview(V, F[mask], origin, normal, tol=tol, join_mm=join,
                                  closed_only=outline, trim_mm=trim)
        sec.update(axis=axis, at=at, offset=float(off), centre=float(c[k]), extent=ext,
                   bbox_centre=np.asarray(c).tolist())
        out.append(sec)
    return out


def trace(path, axis, offset, tol=0.08, units='mm', scale=1.0, join=2.5, outline=False, trim=0.0):
    """Section of the mesh at `offset` mm from its bounding-box centre
    along `axis` ('x'|'y'|'z'), in the converted (mm) frame. Open chains
    stay open (a leaky mesh), but free ends within `join` mm of each other
    are joined first; `outline` drops the open chains altogether. Returns
    the section_preview dict plus 'at' (absolute mm coordinate), 'centre'
    and 'extent' along the axis, so the UI can place its slider."""
    return trace_stack(path, axis, [offset], tol=tol, units=units, scale=scale, join=join,
                       outline=outline, trim=trim)[0]
