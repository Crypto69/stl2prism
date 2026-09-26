"""Subprocess entry point: run one conversion, write result JSON.

Runs in its own process so an OpenCascade crash or OOM kills the worker,
not the API server, and so the pipeline's verbose stdout can be captured
per-job by simple file redirection.

Whatever goes wrong short of the process being killed, result.json is
written with a readable 'error': the UI shows that sentence, and the
traceback goes to the log for whoever wants it.
"""
import json
import os
import sys
import traceback


def _prepare_streams():
    """Line-buffered stdout, so the job log stays live when started
    without -u. A worker without a stdout at all (a windowed Windows
    build started by hand) gets one pointed at nowhere rather than
    crashing on its first print."""
    import io
    import os
    for name in ('stdout', 'stderr'):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, 'w', encoding='utf-8'))
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError, io.UnsupportedOperation):
        pass


def _write_result(result_path, result):
    from .analysis import sanitize
    try:
        with open(result_path, 'w') as f:
            json.dump(sanitize(result), f)
    except Exception:
        # the server then reports "no result"; this traceback says why
        print('could not write the result file:', file=sys.stderr)
        traceback.print_exc()
        return False
    return True


def _pick_shells(in_path, picked):
    """The UI picks shells by their index in /bodies (every connected
    shell of the raw mesh). Preparation folds cavities into their parent
    and drops slivers, so those indices do not address the prepared list:
    translate them to the shell keys pipeline._pick_bodies matches on."""
    from .analysis import body_list
    shells = body_list(in_path)['bodies']
    return [tuple(shells[i]['key']) for i in picked if 0 <= i < len(shells)]


def main(argv=None):
    _prepare_streams()
    args = argv[:4] if argv is not None else sys.argv[1:5]
    if len(args) != 4:
        print(f'worker: expected 4 arguments (input, output, params, result), got {len(args)}',
              file=sys.stderr)
        sys.exit(2)
    in_path, out_path, params_path, result_path = args

    from .errors import describe

    result = {'ok': False, 'error': None, 'failure': None, 'mode': None,
              'metrics': None, 'output_stats': None, 'params': None}
    try:
        try:
            with open(params_path) as f:
                params = json.load(f)
        except Exception as e:
            raise RuntimeError(f'the conversion settings could not be read ({describe(e)})') from e
        result['params'] = params

        if params.get('tool') == 'blueprint':
            # Blueprint: no mesh; the recipe in params is built by CadQuery
            # (STEP + preview + Fusion script) next to out_path
            _blueprint(params, out_path, result)
            return                # through finally: result.json written, exit code 0

        picked = params.get('bodies')
        if picked:
            try:
                picked = _pick_shells(in_path, picked)
            except Exception as e:
                raise RuntimeError(
                    f'the chosen bodies could not be found in the mesh ({describe(e)})') from e

        from stl_to_solid import run
        from .analysis import step_stats

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
                slice_join=params.get('slice_join', 2.5),
                slice_trim=params.get('slice_trim', 0.0),
                slice_from=params.get('slice_from'),
                slice_range_mm=params.get('slice_range_mm', 0.0),
                slice_range_dir=params.get('slice_range_dir', '-'),
                verbose=True)
        # Pass the whole pipeline result through (mode, metrics, and for
        # multi-body files the per-body list and counts).
        try:
            stats = step_stats(out_path)
        except Exception as e:
            # the STEP is written; a failed read-back is a note, not a failure
            traceback.print_exc()
            print(f'note: could not measure the written STEP ({describe(e)})')
            stats = {'file_size': None, 'faces': 0, 'solids': None, 'solids_claimed': None,
                     'closed': None, 'surface_types': {}, 'unmeasured': True}
        result.update(r, ok=True, output_stats=stats)
        result['has_script'] = bool(r.get('script'))
        result.pop('script', None)     # server path; the API serves it by job id
        result['has_bfill_script'] = bool(r.get('bfill_script'))
        result.pop('bfill_script', None)
        # the Fusion script exists for prismatic bodies (next to the
        # CadQuery script) and for sliced lofts (on its own)
        result['has_fusion_script'] = bool(r.get('fusion_script'))
        result.pop('fusion_script', None)
    except MemoryError as e:
        traceback.print_exc()
        result['error'] = 'The conversion ' + describe(e) + '.'
        result['failure'] = 'oom'
    except KeyboardInterrupt:
        result['error'] = 'Conversion cancelled.'
        result['failure'] = 'cancelled'
    except BaseException as e:            # noqa: BLE001 - every failure must leave a result
        traceback.print_exc()
        text = describe(e)
        result['error'] = text if text[:1].isupper() else 'The conversion failed: ' + text
        result['failure'] = type(e).__name__
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        _write_result(result_path, result)
    sys.exit(0 if result['ok'] else 1)


def _blueprint(params, out_path, result):
    from stl_to_solid.blueprint.compile import compile_recipe
    from stl_to_solid.blueprint.build_cq import BlueprintError
    from .analysis import step_stats
    result['mode'] = 'blueprint'
    out_dir = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0]
    read = params.get('read') or {}
    if read.get('model'):
        u = read.get('usage') or {}
        print('read with %s (%s): %s in / %s out tokens in %s s%s' % (
            read['model'], read.get('provider', '?'), u.get('input_tokens', '?'),
            u.get('output_tokens', '?'), read.get('seconds', '?'),
            ', repaired once' if read.get('repaired') else ''))
        for w in (read.get('validation') or {}).get('warnings') or []:
            print('warning:', w)
    print('blueprint: building %d feature(s)' % len((params.get('recipe') or {}).get('features') or []))
    try:
        r = compile_recipe(params['recipe'], out_dir, title=params.get('title') or 'Blueprint', stem=stem)
    except BlueprintError as e:
        if e.report is not None:
            result['validation'] = {'errors': list(e.report.errors), 'warnings': list(e.report.warnings)}
        raise
    try:
        stats = step_stats(out_path)
    except Exception as e:
        traceback.print_exc()
        print(f'note: could not measure the written STEP ({describe(e)})')
        stats = {'file_size': None, 'faces': 0, 'solids': None, 'solids_claimed': None,
                 'closed': None, 'surface_types': {}, 'unmeasured': True}
    result.update(r, ok=True, output_stats=stats)
    for w in r.get('warnings', []):
        print('warning:', w)
    print('blueprint: %d solid(s), %.1f mm^3, bbox %s' % (
        r['solids'], r['volume_mm3'], ' x '.join('%.2f' % v for v in r['bbox']['size'])))


if __name__ == '__main__':
    main()
