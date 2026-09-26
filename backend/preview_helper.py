"""The Blueprint preview helper: a long-lived child process with CadQuery
already imported, so a live preview of an edited recipe takes well under
a second instead of the several seconds a fresh worker spends importing
the geometry kernel. One JSON request per line on stdin
({"recipe": {...}, "out_dir": "...", "stem": "live"}), one JSON answer per
line on stdout. It is its own process so a geometry-kernel crash on an
odd shape kills only the helper, which backend.preview restarts.

Started as `python -m backend.preview_helper`; a frozen build starts it
with a `--preview-helper` flag its entry point dispatches (like `--worker`).
"""
import json
import sys
import traceback


def main():
    # warm up: the imports are the slow part
    try:
        import cadquery  # noqa: F401
        from stl_to_solid.blueprint.compile import quick_preview
        from stl_to_solid.blueprint.build_cq import BlueprintError
        from .errors import describe
    except Exception:
        sys.stdout.write(json.dumps({'ok': False, 'error': 'the preview helper could not start: '
                                     + traceback.format_exc().strip().splitlines()[-1]}) + '\n')
        sys.stdout.flush()
        return 1
    sys.stdout.write(json.dumps({'ok': True, 'ready': True}) + '\n')
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            out = quick_preview(req['recipe'], req['out_dir'], req.get('stem', 'live'))
        except BlueprintError as e:
            out = {'ok': False, 'error': str(e), 'kind': e.kind,
                   'validation': ({'errors': list(e.report.errors), 'warnings': list(e.report.warnings)}
                                  if e.report is not None else None)}
        except Exception as e:      # noqa: BLE001 - the caller must always get an answer
            traceback.print_exc(file=sys.stderr)
            out = {'ok': False, 'error': 'the preview failed: ' + describe(e), 'kind': 'build'}
        sys.stdout.write(json.dumps(out) + '\n')
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
