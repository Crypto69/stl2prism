"""Subprocess entry point: run one conversion, write result JSON.

Runs in its own process so an OpenCascade crash or OOM kills the worker,
not the API server, and so the pipeline's verbose stdout can be captured
per-job by simple file redirection.
"""
import json
import sys
import traceback


def main():
    in_path, out_path, params_path, result_path = sys.argv[1:5]
    with open(params_path) as f:
        params = json.load(f)

    from stl2prism import run
    from .analysis import sanitize, step_stats, body_list

    # The UI picks shells by their index in /bodies (every connected shell
    # of the raw mesh). Preparation folds cavities into their parent and
    # drops slivers, so those indices do not address the prepared list:
    # translate them to the shell keys pipeline._pick_bodies matches on.
    picked = params.get('bodies')
    if picked:
        shells = body_list(in_path)['bodies']
        picked = [tuple(shells[i]['key']) for i in picked if 0 <= i < len(shells)]

    result = {'ok': False, 'error': None, 'mode': None,
              'metrics': None, 'output_stats': None, 'params': params}
    try:
        r = run(in_path, out_path,
                tol=params['tol'],
                accept_p95=params['accept_p95'],
                accept_max=params['accept_max'],
                accept_hole_max=params['accept_hole_max'],
                accept_vol_pct=params['accept_vol_pct'],
                force_prismatic=params['force_prismatic'],
                units=params.get('units', 'mm'),
                scale=params.get('scale', 1.0),
                reduce_tol=params.get('reduce_tol', 0.05),
                face_groups=params.get('face_groups', True),
                bodies=picked or None,
                method=params.get('method', 'auto'),
                slice_mm=params.get('slice_mm', 0.2),
                slice_axis=params.get('slice_axis', 'auto'),
                loft_ruled=params.get('loft_ruled', False),
                verbose=True)
        # Pass the whole pipeline result through (mode, metrics, and for
        # multi-body files the per-body list and counts).
        result.update(r, ok=True, output_stats=step_stats(out_path))
        result['has_script'] = bool(r.get('script'))
        result.pop('script', None)     # server path; the API serves it by job id
        result['has_bfill_script'] = bool(r.get('bfill_script'))
        result.pop('bfill_script', None)
        # the Fusion script exists for prismatic bodies (next to the
        # CadQuery script) and for sliced lofts (on its own)
        result['has_fusion_script'] = bool(r.get('fusion_script'))
        result.pop('fusion_script', None)
    except Exception as e:
        traceback.print_exc()
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        sys.stdout.flush()
        with open(result_path, 'w') as f:
            json.dump(sanitize(result), f)
    sys.exit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
