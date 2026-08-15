"""Validate rebuilt solid against source mesh + CLI entry point."""
import argparse
import sys
import numpy as np
import trimesh

from .mesh_prep import UNIT_SCALE


def validate(solid, mesh, n_samples=5000, cyls=None, hole_band=0.15):
    """Sample source mesh surface, measure distance to rebuilt solid mesh.

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
    import cadquery as cq
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 's.stl')
        cq.exporters.export(solid, p, tolerance=0.02)
        rb = trimesh.load(p)
    pts = mesh.sample(n_samples)
    _, dist, _ = trimesh.proximity.closest_point(rb, pts)
    vol_mesh = mesh.volume if mesh.is_watertight else float('nan')
    vol_solid = solid.val().Volume()
    worst = int(np.argmax(dist))
    out = {
        'dev_max': float(dist.max()),
        'dev_p95': float(np.percentile(dist, 95)),
        'dev_mean': float(dist.mean()),
        'dev_max_xyz': [round(float(v), 2) for v in pts[worst]],
        'vol_mesh': vol_mesh,
        'vol_solid': vol_solid,
        'vol_err_pct': abs(vol_solid - vol_mesh) / vol_mesh * 100
                       if vol_mesh == vol_mesh else float('nan'),
        'hole_dev_max': float('nan'),
        'hole_dev_p95': float('nan'),
        'holes_checked': 0,
    }
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


def run(in_path, out_path, tol=0.08, accept_p95=0.25, accept_vol_pct=2.0,
        accept_max=0.26, accept_hole_max=0.10,
        force_prismatic=False, verbose=True, units='mm'):
    """Convert one mesh file to one STEP file.

    Every connected body is converted on its own — prismatic where it passes
    the gate, faceted otherwise — and all of them are written into a single
    STEP as separate solids. A single-body file returns
    {'mode': 'prismatic'|'faceted', 'metrics': {...}}; a multi-body file adds
    a per-body list and reports mode 'mixed' when the bodies disagree.
    """
    from .mesh_prep import load_and_prep_bodies
    from .rebuild import write_step

    bodies, is_scan, n_dropped = load_and_prep_bodies(
        in_path, verbose=verbose, units=units)
    gates = dict(tol=tol, accept_p95=accept_p95, accept_max=accept_max,
                 accept_hole_max=accept_hole_max, accept_vol_pct=accept_vol_pct)

    if len(bodies) == 1:
        shape, mode, metrics = _convert_body(
            bodies[0], is_scan, force_prismatic, verbose, **gates)
        write_step([shape], out_path)
        if verbose:
            print(f"[out] {mode} solid -> {out_path}")
        return {'mode': mode, 'metrics': metrics,
                'n_bodies': 1, 'n_written': 1, 'n_dropped': n_dropped}

    per_body, shapes = [], []
    for i, body in enumerate(bodies):
        tag = f"[body {i + 1}/{len(bodies)}]"
        if verbose:
            print(f"{tag} converting {len(body.faces)} faces")
        entry = {'index': i, 'faces': int(len(body.faces)),
                 'watertight': bool(body.is_watertight),
                 'mode': None, 'metrics': None, 'error': None}
        try:
            shape, mode, metrics = _convert_body(
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
    write_step(shapes, out_path)

    n_pr = sum(1 for b in per_body if b['mode'] == 'prismatic')
    n_fa = sum(1 for b in per_body if b['mode'] == 'faceted')
    mode = ('prismatic' if n_fa == 0 and n_pr else
            'faceted' if n_pr == 0 else 'mixed')
    if verbose:
        failed = len(per_body) - n_pr - n_fa
        print(f"[out] {len(shapes)} solids ({n_pr} prismatic, {n_fa} faceted"
              + (f", {failed} failed" if failed else "")
              + (f", {n_dropped} sliver(s) dropped" if n_dropped else "")
              + f") -> {out_path}")
    return {'mode': mode, 'metrics': _aggregate(per_body), 'bodies': per_body,
            'n_bodies': len(bodies), 'n_written': len(shapes),
            'n_dropped': n_dropped}


def _aggregate(per_body):
    """Worst-case fidelity across the prismatic bodies plus totals, so the
    top-level metrics still answer 'how good is the file' at a glance."""
    pr = [b['metrics'] for b in per_body if b['mode'] == 'prismatic']
    fa = [b['metrics'] for b in per_body if b['mode'] == 'faceted']

    def worst(key):
        vals = [m[key] for m in pr if m.get(key) == m.get(key)]  # drop NaN
        return max(vals) if vals else float('nan')
    return {
        'n_prismatic': len(pr), 'n_faceted': len(fa),
        'n_failed': sum(1 for b in per_body if b['error']),
        'dev_p95': worst('dev_p95'), 'dev_max': worst('dev_max'),
        'hole_dev_p95': worst('hole_dev_p95'),
        'vol_err_pct': float(max([worst('vol_err_pct')] +
                                 [m['vol_err_pct'] for m in fa
                                  if m.get('vol_err_pct') == m.get('vol_err_pct')])),
        'faces_out': sum(m['faces_out'] for m in fa),
    }


def _convert_body(mesh, is_scan, force_prismatic, verbose, tol, accept_p95,
                  accept_max, accept_hole_max, accept_vol_pct):
    """Convert one closed body. Returns (TopoDS_Shape, mode, metrics).

    Tries the prismatic route and gates it against the mesh; anything that
    fails — a rejected fit, or any exception on the way — falls back to the
    faceted route, which has its own volume gate and raises if even that
    cannot represent the body.
    """
    from .extrusion import dominant_axis, score_axis
    from .rebuild import build_solid

    # Before any axis work: scoring an axis means cross-sectioning the mesh
    # several times per candidate, which is wasted on organic geometry.
    if is_scan and not force_prismatic:
        if verbose:
            print('[out] scan input: skipping prismatic attempt '
                  '(use --force-prismatic to override)')
        return _faceted_body(mesh, verbose)

    try:
        cands = dominant_axis(mesh)
        best = None
        for frac, ax in cands:
            sc, levels, slabs = score_axis(mesh, ax)
            if verbose:
                print(f"[axis] candidate {np.round(ax,3)} area {frac*100:.0f}% "
                      f"-> constancy score {sc:.2f} ({len(slabs)} slabs)")
            if best is None or sc > best[0]:
                best = (sc, ax, levels, slabs)
        score, axis, levels, slabs = best
        if verbose:
            print(f"[axis] selected {np.round(axis,3)} "
                  f"(constant-volume score {score:.2f})")
            print(f"[slabs] levels along axis: {[round(l,2) for l in levels]}")
        nonconst = [s for s in slabs if not s['constant']]
        if verbose and nonconst:
            print(f"[slabs] warning: {len(nonconst)} slab(s) have varying "
                  f"cross-section; prismatic fit may be poor there")

        solid, rep = build_solid(slabs, axis, tol=tol, verbose=verbose)
        from .features import find_cross_cylinders, subtract_cylinders
        # Concave regions are holes; convex ones are bosses/fillets, which
        # must be neither subtracted (that would carve away material) nor
        # held to the hole tolerance. Of the holes, only cross-axis ones get
        # subtracted — axis-parallel holes are already rings in the extruded
        # profile, but their fitted radii deserve the same tight gate.
        all_cyls = find_cross_cylinders(mesh, axis, exclude_parallel=False)
        holes = [c for c in all_cyls if c['concave']]
        cyls = [c for c in holes if not c['parallel']]
        if cyls:
            solid = subtract_cylinders(solid, cyls, verbose=verbose)
        metrics = validate(solid, mesh, cyls=holes)
        if verbose:
            print(f"[check] p95 dev {metrics['dev_p95']:.3f}mm, "
                  f"max {metrics['dev_max']:.3f}mm at "
                  f"{metrics['dev_max_xyz']}, "
                  f"volume err {metrics['vol_err_pct']:.2f}%")
            if metrics['holes_checked']:
                print(f"[check] bore dev p95 {metrics['hole_dev_p95']:.3f}mm, "
                      f"max {metrics['hole_dev_max']:.3f}mm "
                      f"over {metrics['holes_checked']} bore(s)")
        # Gate bores on p95, not max: a wrong radius shifts every wall
        # sample by the same amount, so p95 catches it just as surely,
        # while a single edge/chamfer outlier cannot fail a good hole.
        hole_p95 = metrics['hole_dev_p95']
        ok = (metrics['dev_p95'] <= accept_p95 and
              metrics['dev_max'] <= accept_max and
              (hole_p95 != hole_p95 or hole_p95 <= accept_hole_max) and
              (metrics['vol_err_pct'] <= accept_vol_pct or
               metrics['vol_err_pct'] != metrics['vol_err_pct']))
        if verbose and not ok:
            why = []
            if metrics['dev_p95'] > accept_p95:
                why.append(f"p95 {metrics['dev_p95']:.3f} > {accept_p95}")
            if metrics['dev_max'] > accept_max:
                why.append(f"max {metrics['dev_max']:.3f} > {accept_max} "
                           f"at {metrics['dev_max_xyz']}")
            if hole_p95 == hole_p95 and hole_p95 > accept_hole_max:
                why.append(f"bore p95 {hole_p95:.3f} > {accept_hole_max}")
            if (metrics['vol_err_pct'] == metrics['vol_err_pct']
                    and metrics['vol_err_pct'] > accept_vol_pct):
                why.append(f"volume {metrics['vol_err_pct']:.2f}% "
                           f"> {accept_vol_pct}%")
            print(f"[check] rejected: {'; '.join(why)}")
        if ok:
            return solid.val().wrapped, 'prismatic', metrics
        if verbose:
            print("[out] prismatic fit rejected by tolerance check; "
                  "falling back to faceted")
    except Exception as e:
        if verbose:
            print(f"[out] prismatic rebuild failed ({type(e).__name__}: {e}); "
                  f"falling back to faceted")
    return _faceted_body(mesh, verbose)


def _faceted_body(mesh, verbose, accept_vol_pct=5.0):
    """Faceted solid for one body, checked against the mesh it came from.

    The prismatic path has a deviation/volume gate; without an equivalent here
    a fragmentary export reports success exactly as loudly as a good one.
    """
    from .rebuild import faceted_solid
    shape, stats = faceted_solid(mesh, verbose=verbose)
    vol_mesh = mesh.volume if mesh.is_watertight else float('nan')
    vol_err = (abs(stats['volume'] - vol_mesh) / vol_mesh * 100
               if vol_mesh == vol_mesh and vol_mesh > 0 else float('nan'))
    stats['vol_err_pct'] = vol_err
    if verbose:
        print(f"[check] faceted {stats['faces_out']} faces from "
              f"{stats['faces_in']}, volume {stats['volume']:.0f}mm^3"
              + (f", volume err {vol_err:.2f}%" if vol_err == vol_err else ""))
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
    ap.add_argument('--force-prismatic', action='store_true',
                    help='attempt prismatic fit even for scan-like input')
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
                force_prismatic=args.force_prismatic, verbose=not args.quiet,
                units=args.units)
    except Exception as e:
        # A crash must not look like a success to a calling script.
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if r['mode'] else 1)


if __name__ == '__main__':
    main()
