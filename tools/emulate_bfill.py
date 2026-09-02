"""Emulate a generated Fusion Boundary Fill script in OpenCascade.

Thin CLI over stl2prism.bfill_check (the pipeline runs the same check for
its Boundary Fill outlook). Builds the oversized tool surfaces from the
script's BODIES table, computes the cells, keeps the material ones and
reports probe coverage and the enclosed volume; optionally fuses the kept
cells and writes them as a STEP. A development check only — Fusion's
kernel is the real judge. Usage:

    python tools/emulate_bfill.py <stem>_fusion_bfill.py [fused.step]
"""
import sys
import time
from collections import Counter

sys.path.insert(0, __file__.rsplit('/', 2)[0])
from stl2prism.bfill_check import check_script, faces_of, volume_of  # noqa: E402


def main():
    text = open(sys.argv[1]).read()
    t0 = time.time()
    results, _ns = check_script(text)
    for res in results:
        print(f"{res['name']}: {res['tool_faces']} tool faces -> {res['cells']} cells "
              f"in {time.time() - t0:.1f}s")
        if res['unenclosed']:
            print('  unenclosed probe points by region:',
                  dict(Counter(res['unenclosed'])))
        print(f"  kept {res['kept_cells']} cells ({res['by_centre']} by centre of mass); "
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
            fused = kept[0]
            for c in kept[1:]:
                fused = BRepAlgoAPI_Fuse(fused, c).Shape()
            u = ShapeUpgrade_UnifySameDomain(fused, True, True, True)
            u.Build()
            fused = u.Shape()
            print(f"  union: {len(faces_of(fused))} faces, {volume_of(fused):.4f} cm^3 "
                  f"(err {100 * (volume_of(fused) - res['mesh_volume']) / res['mesh_volume']:+.2f}%), "
                  f"valid={BRepCheck_Analyzer(fused).IsValid()}")
            if len(sys.argv) > 2:
                import cadquery as cq
                cq.exporters.export(cq.Workplane().newObject([cq.Shape.cast(fused)]),
                                    sys.argv[2])
                print('  wrote', sys.argv[2])


main()
