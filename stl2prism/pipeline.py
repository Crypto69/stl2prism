"""Validate rebuilt solid against source mesh + CLI entry point."""
import argparse
import sys
import numpy as np
import trimesh


def validate(solid, mesh, n_samples=5000):
    """Sample source mesh surface, measure distance to rebuilt solid mesh."""
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
    return {
        'dev_max': float(dist.max()),
        'dev_p95': float(np.percentile(dist, 95)),
        'dev_mean': float(dist.mean()),
        'vol_mesh': vol_mesh,
        'vol_solid': vol_solid,
        'vol_err_pct': abs(vol_solid - vol_mesh) / vol_mesh * 100
                       if vol_mesh == vol_mesh else float('nan'),
    }


def run(in_path, out_path, tol=0.08, accept_p95=0.25, accept_vol_pct=2.0,
        force_prismatic=False, verbose=True):
    from .mesh_prep import load_and_prep
    from .extrusion import dominant_axis
    from .rebuild import build_solid, export_step, faceted_fallback

    mesh, is_scan = load_and_prep(in_path, verbose=verbose)
    from .extrusion import score_axis
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

    result = {'mode': None, 'metrics': None}
    if is_scan and not force_prismatic:
        if verbose:
            print('[out] scan input: skipping prismatic attempt (use --force-prismatic to override)')
        faceted_fallback(mesh, out_path)
        result.update(mode='faceted')
        if verbose:
            print(f'[out] faceted solid -> {out_path}')
        return result
    try:
        solid, rep = build_solid(slabs, axis, tol=tol, verbose=verbose)
        from .features import find_cross_cylinders, subtract_cylinders
        cyls = find_cross_cylinders(mesh, axis)
        if cyls:
            solid = subtract_cylinders(solid, cyls, verbose=verbose)
        metrics = validate(solid, mesh)
        if verbose:
            print(f"[check] p95 dev {metrics['dev_p95']:.3f}mm, "
                  f"max {metrics['dev_max']:.3f}mm, "
                  f"volume err {metrics['vol_err_pct']:.2f}%")
        ok = (metrics['dev_p95'] <= accept_p95 and
              (metrics['vol_err_pct'] <= accept_vol_pct or
               metrics['vol_err_pct'] != metrics['vol_err_pct']))
        if ok:
            export_step(solid, out_path)
            result.update(mode='prismatic', metrics=metrics)
            if verbose:
                print(f"[out] prismatic solid -> {out_path}")
            return result
        if verbose:
            print("[out] prismatic fit rejected by tolerance check; "
                  "falling back to faceted")
    except Exception as e:
        if verbose:
            print(f"[out] prismatic rebuild failed ({type(e).__name__}: {e}); "
                  f"falling back to faceted")
    faceted_fallback(mesh, out_path)
    result.update(mode='faceted')
    if verbose:
        print(f"[out] faceted solid -> {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(
        prog='stl2prism',
        description='Convert an STL mesh into a prismatic STEP solid via '
                    'extrusion-structure recognition, with faceted fallback.')
    ap.add_argument('input')
    ap.add_argument('output', nargs='?', default=None)
    ap.add_argument('--tol', type=float, default=0.08,
                    help='profile fit tolerance in mm (default 0.08)')
    ap.add_argument('--accept-p95', type=float, default=0.25,
                    help='max p95 surface deviation to accept prismatic result')
    ap.add_argument('--force-prismatic', action='store_true',
                    help='attempt prismatic fit even for scan-like input')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()
    out = args.output or args.input.rsplit('.', 1)[0] + '.step'
    r = run(args.input, out, tol=args.tol, accept_p95=args.accept_p95,
            force_prismatic=args.force_prismatic, verbose=not args.quiet)
    sys.exit(0 if r['mode'] else 1)


if __name__ == '__main__':
    main()
