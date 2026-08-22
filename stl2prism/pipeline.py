"""Validate rebuilt solid against source mesh + CLI entry point."""
import argparse
import sys
import numpy as np
import trimesh

from .mesh_prep import UNIT_SCALE

# Deterministic surface sampling: the verdict for a given file must not
# depend on the random state, or parts near a gate flip mode between runs.
SAMPLE_SEED = 20240817


def tessellate_solid(solid, tolerance=0.02, angular=0.1):
    """Triangulate a CadQuery Workplane/Shape in memory (no STL round trip)."""
    import cadquery as cq
    shape = solid.val() if hasattr(solid, 'val') else solid
    if not isinstance(shape, cq.Shape):
        shape = cq.Shape.cast(shape)
    verts, tris = shape.tessellate(tolerance, angular)
    V = np.array([(v.x, v.y, v.z) for v in verts], dtype=float)
    F = np.array(tris, dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(V, F, process=False)


def sample_points(mesh, n_samples=None, include_vertices=True, max_points=200000):
    """Points on `mesh` for deviation measurement.

    Returns (pts, n_uniform): the first n_uniform rows are a seeded,
    area-proportional surface sample (used for percentiles — they measure
    surface *area*); the rest are the mesh vertices (features live at
    vertices; used for the max only, since they cluster on curved faces
    and would bias a percentile)."""
    area = float(mesh.area)
    if n_samples is None:
        n_samples = int(np.clip(area / 2.0, 5000, 60000))
    pts = mesh.sample(int(n_samples), seed=SAMPLE_SEED)
    n_uniform = len(pts)
    if include_vertices and len(mesh.vertices):
        V = mesh.vertices
        if len(V) > max_points:
            idx = np.linspace(0, len(V) - 1, max_points).astype(np.int64)
            V = V[idx]
        pts = np.vstack([pts, V])
    return pts, n_uniform


def validate(solid, mesh, n_samples=None, cyls=None, hole_band=0.15,
             symmetric=True):
    """Measure the rebuilt solid against the source mesh.

    Forward deviation: points on the source mesh (all vertices + a seeded
    sample) to the rebuilt surface — catches missing/misplaced material.
    Reverse deviation: points on the rebuilt surface to the source mesh —
    catches material the rebuild *added* (a filled pocket, a bulging arc)
    that forward sampling cannot see. The reverse direction is only
    meaningful when the source is closed; a leaky mesh is measured forward
    only and flagged as unverified.

    Deviation is reported globally and, separately, restricted to points
    lying on cylindrical bores. A millimetre of error on a flat outer wall is
    cosmetic; the same error on a bore changes the hole size and the part
    stops fitting, so the two cannot share one budget.

    hole_band is deliberately tight: it only has to admit sample points on
    the bore wall itself (within the circle-fit residual of radius r), while
    excluding chamfer and edge points a fraction of a radius away — those
    belong to the global budget. A rebuild radius error larger than the band
    is still caught, because the band selects points by the *mesh* fit and
    the deviation is measured against the *rebuilt* wall.
    """
    rb = tessellate_solid(solid)
    pts, n_uni = sample_points(mesh, n_samples)
    _, dist, _ = trimesh.proximity.closest_point(rb, pts)
    uni = dist[:n_uni]
    closed = bool(mesh.is_watertight)
    vol_mesh = mesh.volume if closed else float('nan')
    shape = solid.val() if hasattr(solid, 'val') else solid
    if not hasattr(shape, 'Volume'):            # raw TopoDS_Shape
        import cadquery as cq
        shape = cq.Shape.cast(shape)
    vol_solid = float(shape.Volume())
    worst = int(np.argmax(dist))
    out = {
        'dev_max': float(dist.max()),
        'dev_p95': float(np.percentile(uni, 95)),
        'dev_mean': float(uni.mean()),
        'dev_max_xyz': [round(float(v), 2) for v in pts[worst]],
        'rev_dev_max': float('nan'),
        'rev_dev_p95': float('nan'),
        'vol_mesh': vol_mesh,
        'vol_solid': vol_solid,
        'vol_err_pct': abs(vol_solid - vol_mesh) / vol_mesh * 100
                       if vol_mesh == vol_mesh and vol_mesh > 0 else float('nan'),
        'vol_verified': bool(vol_mesh == vol_mesh and vol_mesh > 0),
        'symmetric': False,
        'hole_dev_max': float('nan'),
        'hole_dev_p95': float('nan'),
        'holes_checked': 0,
        'n_samples': int(len(pts)),
    }
    if symmetric and closed:
        rpts, _ = sample_points(rb, max(2000, n_uni // 2), include_vertices=False)
        _, rdist, _ = trimesh.proximity.closest_point(mesh, rpts)
        out['rev_dev_max'] = float(rdist.max())
        out['rev_dev_p95'] = float(np.percentile(rdist, 95))
        out['symmetric'] = True
    on_hole = np.zeros(len(pts), bool)
    for c in (cyls or []):
        axis, (bx, by) = c['axis'], c['basis']
        c3 = c['center2'][0] * bx + c['center2'][1] * by
        rel = pts - c3
        h = rel @ axis
        radial = np.linalg.norm(rel - np.outer(h, axis), axis=1)
        # Inset axially rather than extend: points at the bore mouths sit on
        # chamfers and edge breaks, whose deviation belongs to the global
        # budget, not the hole-size one.
        on_hole |= ((np.abs(radial - c['r']) < hole_band) &
                    (h >= c['h0'] + hole_band) & (h <= c['h1'] - hole_band))
    if on_hole.any():
        out['hole_dev_max'] = float(dist[on_hole].max())
        out['hole_dev_p95'] = float(np.percentile(dist[on_hole], 95))
        out['holes_checked'] = len(cyls or [])
    return out


def gate_values(metrics):
    """The numbers the acceptance gate compares: worst of both directions."""
    p95 = metrics['dev_p95']
    mx = metrics['dev_max']
    if metrics.get('symmetric'):
        p95 = max(p95, metrics['rev_dev_p95'])
        mx = max(mx, metrics['rev_dev_max'])
    return p95, mx


def run(in_path, out_path, tol=0.08, accept_p95=0.25, accept_vol_pct=2.0,
        accept_max=0.26, accept_hole_max=0.10,
        force_prismatic=False, verbose=True, units='mm', reduce_tol=0.05,
        write_script=True, face_groups=True):
    """Convert one mesh file to one STEP file.

    Every connected body is converted on its own — prismatic where it passes
    the gate, else face-group (one analytic face per fitted region), else
    faceted — and all of them are written into a single STEP as separate
    solids. Internal cavities are subtracted from their body. A single-body
    file returns {'mode': 'prismatic'|'facegroup'|'faceted', 'metrics':
    {...}}; a multi-body file adds a per-body list and reports mode 'mixed'
    when the bodies disagree.
    """
    from .mesh_prep import load_and_prep_bodies
    from .rebuild import write_step

    bodies, is_scan, n_dropped = load_and_prep_bodies(
        in_path, verbose=verbose, units=units)
    gates = dict(tol=tol, accept_p95=accept_p95, accept_max=accept_max,
                 accept_hole_max=accept_hole_max, accept_vol_pct=accept_vol_pct,
                 reduce_tol=reduce_tol, face_groups=face_groups)

    if len(bodies) == 1:
        shape, mode, metrics = _convert_group(
            bodies[0], is_scan, force_prismatic, verbose, **gates)
        write_step([shape], out_path, names=[_body_name(in_path, 1, 1)])
        if verbose:
            print(f"[out] {mode} solid -> {out_path}")
        script = _write_script([metrics.pop('build', {'mode': mode})], out_path,
                               write_script, verbose)
        return {'mode': mode, 'metrics': metrics,
                'n_bodies': 1, 'n_written': 1, 'n_dropped': n_dropped,
                'is_scan': bool(is_scan), 'script': script}

    per_body, shapes = [], []
    for i, body in enumerate(bodies):
        tag = f"[body {i + 1}/{len(bodies)}]"
        if verbose:
            print(f"{tag} converting {len(body.mesh.faces)} faces"
                  + (f" (+{len(body.voids)} void(s))" if body.voids else ""))
        entry = {'index': i, 'faces': int(len(body.mesh.faces)),
                 'watertight': bool(body.mesh.is_watertight),
                 'voids': len(body.voids),
                 'mode': None, 'metrics': None, 'error': None}
        try:
            shape, mode, metrics = _convert_group(
                body, is_scan, force_prismatic, verbose, **gates)
            shapes.append(shape)
            entry.update(mode=mode, metrics=metrics)
            if verbose:
                print(f"{tag} -> {mode}")
        except Exception as e:
            # One bad body must not cost the other 26: record it, move on.
            entry['error'] = f'{type(e).__name__}: {e}'
            if verbose:
                print(f"{tag} failed ({entry['error']}); body left out")
        per_body.append(entry)

    if not shapes:
        raise RuntimeError(
            f"none of the {len(bodies)} bodies could be converted; "
            f"see the per-body log lines above")
    names = [_body_name(in_path, b['index'] + 1, len(bodies)) for b in per_body if b['mode']]
    write_step(shapes, out_path, names=names)
    builds = []
    for b in per_body:
        if b['metrics'] is not None:
            builds.append(b['metrics'].pop('build', {'mode': b['mode']}))
    script = _write_script(builds, out_path, write_script, verbose)

    n_pr = sum(1 for b in per_body if b['mode'] == 'prismatic')
    n_fg = sum(1 for b in per_body if b['mode'] == 'facegroup')
    n_fa = sum(1 for b in per_body if b['mode'] == 'faceted')
    modes = {b['mode'] for b in per_body if b['mode']}
    mode = modes.pop() if len(modes) == 1 else 'mixed'
    if verbose:
        failed = len(per_body) - n_pr - n_fg - n_fa
        print(f"[out] {len(shapes)} solids ({n_pr} prismatic, {n_fg} face-group, "
              f"{n_fa} faceted"
              + (f", {failed} failed" if failed else "")
              + (f", {n_dropped} sliver(s) dropped" if n_dropped else "")
              + f") -> {out_path}")
    return {'mode': mode, 'metrics': _aggregate(per_body), 'bodies': per_body,
            'n_bodies': len(bodies), 'n_written': len(shapes),
            'n_dropped': n_dropped, 'is_scan': bool(is_scan), 'script': script}


def _body_name(in_path, i, n):
    import os
    stem = os.path.splitext(os.path.basename(in_path))[0]
    return stem if n == 1 else f'{stem}_body{i}'


def _write_script(builds, out_path, write_script, verbose):
    """Write the CadQuery script next to the STEP (same stem, .py)."""
    if not write_script:
        return None
    if not any(b.get('mode') == 'prismatic' and 'slabs' in b for b in builds):
        # A script with no recognised bodies would be an empty program that
        # crashes on its first line; better no file than a broken one.
        if verbose:
            print('[out] no prismatic bodies; no script written')
        return None
    try:
        from .script_export import emit_script
        import os
        py_path = os.path.splitext(out_path)[0] + '.py'
        text = emit_script(builds, os.path.basename(out_path))
        with open(py_path, 'w') as f:
            f.write(text)
        if verbose:
            print(f"[out] CadQuery script -> {py_path}")
        try:
            from .fusion_export import emit_fusion_script
            fpath = os.path.splitext(out_path)[0] + '_fusion.py'
            with open(fpath, 'w') as f:
                f.write(emit_fusion_script(builds))
            if verbose:
                print(f"[out] Fusion 360 script -> {fpath}")
        except Exception as e:
            if verbose:
                print(f"[out] Fusion script export failed ({type(e).__name__}: {e})")
        return py_path
    except Exception as e:
        if verbose:
            print(f"[out] script export failed ({type(e).__name__}: {e})")
        return None


def _aggregate(per_body):
    """Worst-case fidelity across the prismatic bodies plus totals, so the
    top-level metrics still answer 'how good is the file' at a glance."""
    pr = [b['metrics'] for b in per_body if b['mode'] == 'prismatic']
    fg = [b['metrics'] for b in per_body if b['mode'] == 'facegroup']
    fa = [b['metrics'] for b in per_body if b['mode'] == 'faceted']

    def worst(key):
        vals = [m[key] for m in pr + fg if m.get(key) == m.get(key)]  # drop NaN
        return max(vals) if vals else float('nan')
    return {
        'n_prismatic': len(pr), 'n_facegroup': len(fg), 'n_faceted': len(fa),
        'n_failed': sum(1 for b in per_body if b['error']),
        'dev_p95': worst('dev_p95'), 'dev_max': worst('dev_max'),
        'hole_dev_p95': worst('hole_dev_p95'),
        'vol_err_pct': float(max([worst('vol_err_pct')] +
                                 [m['vol_err_pct'] for m in fa
                                  if m.get('vol_err_pct') == m.get('vol_err_pct')])),
        'faces_out': sum(m['faces_out'] for m in fa + fg),
    }


def _convert_group(body, is_scan, force_prismatic, verbose, **gates):
    """Convert a Body (outer shell + voids) into one TopoDS_Shape.

    The outer shell and each void are converted independently (each with its
    own gate and fallback); void solids are then subtracted. Metrics are the
    outer shell's, with the volume error recomputed for the hollow result.
    """
    shape, mode, metrics = _convert_body(
        body.mesh, is_scan, force_prismatic, verbose, **gates)
    if not body.voids:
        return shape, mode, metrics
    import cadquery as cq
    outer = cq.Shape.cast(shape)
    void_modes = []
    void_builds = []
    for k, v in enumerate(body.voids):
        if verbose:
            print(f"[void {k + 1}/{len(body.voids)}] converting {len(v.faces)} faces")
        vshape, vmode, vmet = _convert_body(v, is_scan, force_prismatic, verbose, **gates)
        void_modes.append(vmode)
        void_builds.append(vmet.get('build', {'mode': vmode}))
        outer = outer.cut(cq.Shape.cast(vshape), tol=1e-4)
    metrics = dict(metrics)
    metrics['voids'] = len(body.voids)
    metrics['void_modes'] = void_modes
    if 'build' in metrics:
        metrics['build'] = dict(metrics['build'], voids=void_builds)
    vol_mesh = body.volume()
    vol_solid = float(outer.Volume())
    metrics['vol_mesh'] = vol_mesh
    metrics['vol_solid'] = vol_solid
    metrics['vol_err_pct'] = (abs(vol_solid - vol_mesh) / vol_mesh * 100
                              if vol_mesh == vol_mesh and vol_mesh > 0
                              else float('nan'))
    metrics['vol_verified'] = bool(vol_mesh == vol_mesh and vol_mesh > 0)
    if verbose:
        print(f"[void] {len(body.voids)} cavity(ies) subtracted; hollow volume "
              f"{vol_solid:.0f}mm^3, err {metrics['vol_err_pct']:.2f}%")
    return outer.wrapped, mode, metrics


def _passes(metrics, accept_p95, accept_max, accept_hole_max, accept_vol_pct):
    """The acceptance gate every route is held to. Returns (ok, reasons)."""
    g_p95, g_max = gate_values(metrics)
    # Gate bores on p95, not max: a wrong radius shifts every wall sample by
    # the same amount, so p95 catches it just as surely, while a single
    # edge/chamfer outlier cannot fail a good hole.
    hole_p95 = metrics['hole_dev_p95']
    vol = metrics['vol_err_pct']
    why = []
    if g_p95 > accept_p95:
        why.append(f"p95 {g_p95:.3f} > {accept_p95}")
    if g_max > accept_max:
        why.append(f"max {g_max:.3f} > {accept_max} at {metrics['dev_max_xyz']}")
    if hole_p95 == hole_p95 and hole_p95 > accept_hole_max:
        why.append(f"bore p95 {hole_p95:.3f} > {accept_hole_max}")
    if vol == vol and vol > accept_vol_pct:
        why.append(f"volume {vol:.2f}% > {accept_vol_pct}%")
    return not why, why


def _log_check(metrics, verbose, tag='[check]'):
    if not verbose:
        return
    print(f"{tag} p95 dev {metrics['dev_p95']:.3f}mm, "
          f"max {metrics['dev_max']:.3f}mm at {metrics['dev_max_xyz']}"
          + (f"; reverse p95 {metrics['rev_dev_p95']:.3f}, "
             f"max {metrics['rev_dev_max']:.3f}"
             if metrics['symmetric'] else "; reverse n/a (open mesh)")
          + (f", volume err {metrics['vol_err_pct']:.2f}%"
             if metrics['vol_verified'] else ", volume unverified"))
    if metrics['holes_checked']:
        print(f"{tag} bore dev p95 {metrics['hole_dev_p95']:.3f}mm, "
              f"max {metrics['hole_dev_max']:.3f}mm "
              f"over {metrics['holes_checked']} bore(s)")


def _convert_body(mesh, is_scan, force_prismatic, verbose, tol, accept_p95,
                  accept_max, accept_hole_max, accept_vol_pct, reduce_tol=0.05,
                  face_groups=True):
    """Convert one closed body. Returns (TopoDS_Shape, mode, metrics).

    Route ladder, every rung held to the same gate:
      1. prismatic (extrusion engine) — the only route that yields
         sketch+extrude structure for the scripts, so it goes first;
      2. face-group engine — one analytic face per fitted region;
      3. hybrid — the prismatic solid with its failing regions patched by
         exact facets;
      4. faceted — planar-merged, tolerance-reduced triangles (own volume
         gate; raises if even that cannot represent the body).
    Any exception on a rung falls through to the next.
    """
    from .extrusion import dominant_axis, score_axis, _axis_basis
    from .rebuild import build_solid
    gates = dict(accept_p95=accept_p95, accept_max=accept_max,
                 accept_hole_max=accept_hole_max, accept_vol_pct=accept_vol_pct)

    # Before any axis work: scoring an axis means cross-sectioning the mesh
    # several times per candidate, which is wasted on organic geometry.
    if is_scan and not force_prismatic:
        if verbose:
            print('[out] scan input: skipping prismatic attempt '
                  '(use --force-prismatic to override)')
        return _faceted_body(mesh, verbose, reduce_tol=reduce_tol)

    # --- 1. prismatic -------------------------------------------------------
    prism = None            # (solid, metrics, holes) kept for the hybrid rung
    try:
        cands = dominant_axis(mesh)
        best = None
        for frac, ax in cands:
            sc, levels, slabs = score_axis(mesh, ax)
            # perpendicular-face area is a strong prior for the extrusion
            # direction (the 'base faces'); use it to break near-ties
            rank = sc * (1.0 + 0.5 * frac)
            if verbose:
                print(f"[axis] candidate {np.round(ax,3)} area {frac*100:.0f}% "
                      f"-> constancy score {sc:.2f} ({len(slabs)} slabs)")
            if best is None or rank > best[0] + 1e-9:
                best = (rank, ax, levels, slabs, sc)
        _, axis, levels, slabs, score = best
        if verbose:
            print(f"[axis] selected {np.round(axis,3)} "
                  f"(constant-volume score {score:.2f})")
            print(f"[slabs] levels along axis: "
                  f"{[round(float(l), 2) for l in levels]}")
        nonconst = [s for s in slabs if not s['constant']]
        if verbose and nonconst:
            print(f"[slabs] warning: {len(nonconst)} slab(s) have varying "
                  f"cross-section; prismatic fit may be poor there")

        solid, rep = build_solid(slabs, axis, tol=tol, verbose=verbose, mesh=mesh)
        from .features import find_cross_cylinders, subtract_cylinders
        # Concave regions are holes; convex ones are bosses/fillets, which
        # must be neither subtracted (that would carve away material) nor
        # held to the hole tolerance. Of the holes, only cross-axis ones get
        # subtracted — axis-parallel holes are already rings in the extruded
        # profile, but their fitted radii deserve the same tight gate.
        all_cyls = find_cross_cylinders(mesh, axis, exclude_parallel=False)
        holes = [c for c in all_cyls if c['concave']]
        cyls = [c for c in holes if not c['parallel']]
        from .features import find_cross_cones, subtract_cones
        cones = [c for c in find_cross_cones(mesh, axis) if c['concave']]
        if cyls or cones:
            from .rebuild import finish_solid
            if cyls:
                solid = subtract_cylinders(solid, cyls, mesh=mesh, verbose=verbose)
            if cones:
                solid = subtract_cones(solid, cones, mesh=mesh, verbose=verbose)
            solid = finish_solid(solid, verbose=verbose)
        metrics = validate(solid, mesh, cyls=holes)
        _log_check(metrics, verbose)
        ok, why = _passes(metrics, **gates)
        if verbose and not ok:
            print(f"[check] rejected: {'; '.join(why)}")
        metrics['build'] = {'axis': np.asarray(axis, float).tolist(),
                            'xdir': _axis_basis(axis)[:3, 0].tolist(),
                            'slabs': rep, 'cross_cyls': cyls, 'cones': cones,
                            'mode': 'prismatic'}
        if ok:
            return solid.val().wrapped, 'prismatic', metrics
        prism = (solid, metrics, holes)
    except Exception as e:
        if verbose:
            print(f"[out] prismatic rebuild failed ({type(e).__name__}: {e})")

    # --- 2. face groups -----------------------------------------------------
    if face_groups:
        try:
            r = _facegroup_body(mesh, verbose, tol, gates)
            if r is not None:
                return r
        except Exception as e:
            if verbose:
                print(f"[fgroup] failed ({type(e).__name__}: {e})")

    # --- 3. hybrid: the prismatic solid, locally patched -----------------------
    # Not all-or-nothing: patch the regions that fail with the exact
    # faceted geometry and re-check (Fusion keeps the converted face
    # groups too). Small local misfits — a countersink, a taper, a
    # fillet the profile fitter cannot express — no longer cost the
    # whole body its clean faces.
    if prism is not None:
        try:
            solid, metrics, holes = prism
            from .hybrid import try_hybrid
            patched, info = try_hybrid(solid, mesh, metrics, accept_max, tol,
                                       verbose=verbose)
            if patched is not None:
                m2 = validate(patched, mesh, cyls=holes)
                ok2, _ = _passes(m2, **gates)
                p95b, maxb = gate_values(m2)
                if verbose:
                    print(f"[patch] after patching: p95 {p95b:.3f}, max {maxb:.3f}, "
                          f"volume err {m2['vol_err_pct']:.2f}% -> "
                          f"{'accepted' if ok2 else 'still rejected'}")
                if ok2:
                    m2['patched'] = True
                    m2['patches'] = info['patches']
                    m2['patch_boxes'] = info.get('boxes', [])
                    m2['bad_frac'] = info['bad_frac']
                    m2['build'] = dict(metrics['build'], note='patched: the script '
                                       'rebuilds the unpatched extrusion structure')
                    return patched.val().wrapped, 'prismatic', m2
        except Exception as e:
            if verbose:
                print(f"[patch] failed ({type(e).__name__}: {e})")
        if verbose:
            print("[out] prismatic fit rejected by tolerance check; "
                  "falling back to faceted")

    # --- 4. faceted -----------------------------------------------------------
    return _faceted_body(mesh, verbose, reduce_tol=reduce_tol)


def _facegroup_body(mesh, verbose, tol, gates):
    """Face-group engine for one body, held to the same gate as prismatic.
    Returns (shape, 'facegroup', metrics) or None when it steps aside."""
    from . import facegroups
    if not mesh.is_watertight:
        # the engine only accepts a closed sewn shell; a leaky body cannot
        # give one, and trying costs up to a minute of fits per body
        if verbose:
            print('[fgroup] open mesh: skipped (needs a closed shell)')
        return None
    try:
        shape, stats = facegroups.convert(mesh, tol=tol, verbose=verbose)
    except facegroups.FaceGroupError as e:
        if verbose:
            print(f"[fgroup] no closed solid ({e})")
        return None
    if shape is None:
        return None
    n_an = stats['faces_out'] - stats.get('triangle_faces', 0)
    if n_an == 0 or stats.get('unfitted_faces', 0) > 0.5 * len(mesh.faces):
        # (nearly) everything stayed triangles: the faceted route with its
        # tolerance-driven reduce does that job better
        if verbose:
            print(f"[fgroup] only {n_an} analytic face(s); "
                  f"{stats.get('unfitted_faces', 0)} of {len(mesh.faces)} "
                  f"triangles unfitted — leaving the body to the next route")
        return None
    metrics = validate(shape, mesh)
    _log_check(metrics, verbose)
    ok, why = _passes(metrics, **gates)
    if not ok:
        if verbose:
            print(f"[check] face-group solid rejected: {'; '.join(why)}")
        return None
    metrics['fgroup'] = {k: v for k, v in stats.items()
                         if k not in ('fallbacks',)}
    metrics['fgroup']['fallbacks'] = [list(f) for f in stats.get('fallbacks', [])][:20]
    metrics['faces_out'] = stats['faces_out']
    return shape, 'facegroup', metrics


def _faceted_body(mesh, verbose, accept_vol_pct=5.0, reduce_tol=0.05):
    """Faceted solid for one body, checked against the mesh it came from.

    Curved regions are first decimated within `reduce_tol` (Fusion's
    'Reduce by tolerance'); coplanar triangles become single planar faces.
    The prismatic path has a deviation/volume gate; without an equivalent here
    a fragmentary export reports success exactly as loudly as a good one.
    """
    from .rebuild import faceted_solid, reduce_mesh
    src = mesh
    info = {'reduced': False}
    if reduce_tol and reduce_tol > 0:
        src, info = reduce_mesh(mesh, reduce_tol, verbose=verbose)
    shape, stats = faceted_solid(src, verbose=verbose)
    stats['reduce'] = info
    stats['faces_in'] = int(len(mesh.faces))
    vol_mesh = mesh.volume if mesh.is_watertight else float('nan')
    vol_err = (abs(stats['volume'] - vol_mesh) / vol_mesh * 100
               if vol_mesh == vol_mesh and vol_mesh > 0 else float('nan'))
    stats['vol_err_pct'] = vol_err
    stats['vol_verified'] = bool(vol_mesh == vol_mesh and vol_mesh > 0)
    if verbose:
        print(f"[check] faceted {stats['faces_out']} faces from "
              f"{stats['faces_in']}, volume {stats['volume']:.0f}mm^3"
              + (f", volume err {vol_err:.2f}%" if vol_err == vol_err
                 else ", volume unverified"))
    if vol_err == vol_err and vol_err > accept_vol_pct:
        raise RuntimeError(
            f"faceted solid volume differs from the mesh by {vol_err:.1f}% "
            f"(limit {accept_vol_pct}%); refusing to report success")
    return shape, 'faceted', stats


def main():
    ap = argparse.ArgumentParser(
        prog='stl2prism',
        description='Convert an STL or OBJ mesh into a prismatic STEP solid '
                    'via extrusion-structure recognition, with faceted '
                    'fallback.')
    ap.add_argument('input', help='input mesh (.stl or .obj)')
    ap.add_argument('output', nargs='?', default=None)
    ap.add_argument('--tol', type=float, default=0.08,
                    help='profile fit tolerance in mm (default 0.08)')
    ap.add_argument('--accept-p95', type=float, default=0.25,
                    help='max p95 surface deviation to accept prismatic result')
    ap.add_argument('--accept-max', type=float, default=0.26,
                    help='max single-point surface deviation, mm (default 0.26)')
    ap.add_argument('--accept-hole-max', type=float, default=0.10,
                    help='max deviation on cylindrical bores, mm (default 0.10)')
    ap.add_argument('--accept-vol-pct', type=float, default=2.0,
                    help='max volume error in percent (default 2.0)')
    ap.add_argument('--reduce-tol', type=float, default=0.05,
                    help='faceted output: decimate curved regions within this '
                         'deviation in mm (0 disables; default 0.05)')
    ap.add_argument('--force-prismatic', action='store_true',
                    help='attempt prismatic fit even for scan-like input')
    ap.add_argument('--no-face-groups', action='store_true',
                    help='skip the face-group engine (one analytic face per '
                         'fitted region) between the prismatic and faceted routes')
    ap.add_argument('--units', choices=sorted(UNIT_SCALE), default='mm',
                    help='unit the input file is in; STL/OBJ carry none, '
                         'and the tool works in mm (default mm)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    out = args.output or args.input.rsplit('.', 1)[0] + '.step'
    try:
        r = run(args.input, out, tol=args.tol, accept_p95=args.accept_p95,
                accept_max=args.accept_max,
                accept_hole_max=args.accept_hole_max,
                accept_vol_pct=args.accept_vol_pct,
                force_prismatic=args.force_prismatic, verbose=not args.quiet,
                units=args.units, reduce_tol=args.reduce_tol,
                face_groups=not args.no_face_groups)
    except Exception as e:
        # A crash must not look like a success to a calling script.
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if r['mode'] else 1)


if __name__ == '__main__':
    main()
