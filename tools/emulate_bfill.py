"""Emulate a generated Fusion Boundary Fill script in OpenCascade.

Thin CLI over stl2prism.bfill_check (the pipeline runs the same check for
its Boundary Fill outlook). Builds the oversized tool surfaces from the
script's BODIES table exactly as the script would (its SKIP list included),
computes the cells, keeps the material ones and reports probe coverage and
the enclosed volume; optionally fuses the kept cells and writes them as a
STEP. A development check only — Fusion's kernel is the real judge. Usage:

    python tools/emulate_bfill.py <stem>_fusion_bfill.py [fused.step]
"""
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stl2prism.bfill_check import parse_script, check_body, outlook  # noqa: E402
from stl2prism.rebuild import _faces, _volume  # noqa: E402


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    with open(argv[0]) as f:
        text = f.read()
    ns = parse_script(text)
    results = []
    for bd in ns['BODIES']:
        t0 = time.time()
        res = check_body(bd, ns)
        results.append(res)
        dt = time.time() - t0
        extra = ''.join(f", {res[k]} tool(s) {w}" for k, w in
                        (('tools_failed', 'could not be built'), ('tools_skipped', 'skipped (SKIP)'))
                        if res.get(k))
        if res.get('error'):
            print(f"{res['name']}: {res['error']} ({res['tool_faces']} tool faces{extra}, {dt:.1f}s)")
            continue
        print(f"{res['name']}: {res['tool_faces']} tool faces{extra} -> {res['cells']} cells "
              f"in {dt:.1f}s")
        if res['unenclosed']:
            print('  unenclosed probe points by region:',
                  dict(Counter(res['unenclosed'])))
        print(f"  kept {res['kept_cells']} cells ({res['by_centre']} by an interior point, "
              f"{res['by_face']} by a face probe"
              + (", closest-volume fallback" if res['fallback'] else "") + "); "
              f"largest cells {[round(v, 4) for v in res['cell_volumes'][:5]]}")
        print(f"  probe points outside every kept cell: {len(res['unenclosed'])}/{res['n_probes']}"
              f"  (mesh {res['mesh_volume']:.4f} cm^3)")
        print(f"  kept volume {res['kept_volume']:.4f} cm^3 "
              f"({res['enclosed_pct']:.2f}% of the mesh)")
        kept = res['kept']
        if kept and len(kept) < 80:
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
            from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
            from OCP.BRepCheck import BRepCheck_Analyzer
            t1 = time.time()
            fused = kept[0]
            for c in kept[1:]:
                fused = BRepAlgoAPI_Fuse(fused, c).Shape()
            u = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
            u.Build()
            fused = u.Shape()
            vol = _volume(fused)
            print(f"  union: {len(_faces(fused))} faces, {vol:.4f} cm^3 "
                  f"(err {100 * (vol - res['mesh_volume']) / res['mesh_volume']:+.2f}%), "
                  f"valid={BRepCheck_Analyzer(fused).IsValid()} ({time.time() - t1:.1f}s)")
            if len(argv) > 1:
                import cadquery as cq
                cq.exporters.export(cq.Workplane().newObject([cq.Shape.cast(fused)]),
                                    argv[1])
                print('  wrote', argv[1])
    v = outlook(results)
    print(f"outlook: {'OK' if v['ok'] else 'NOT CHECKED' if v['ok'] is None else 'LIKELY TO FAIL'}"
          f" ({v['reason']})")
    return 0


if __name__ == '__main__':
    sys.exit(main())
