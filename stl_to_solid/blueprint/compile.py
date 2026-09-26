"""compile_recipe: one recipe in, the deliverables out — output.step,
preview.stl (for the web viewer), output_fusion.py (the parametric Fusion
script), recipe.json (the normalized recipe that was built) — and a
summary dict the worker puts in result.json."""
import json
import os

from .build_cq import Build, BlueprintError, build
from .fusion_blueprint import emit_fusion_blueprint_script
from .schema import normalize
from .validate import validate

PREVIEW_TOL = 0.02
PREVIEW_ANG = 0.1


def compile_recipe(recipe, out_dir, title=None, stem='output'):
    """Validate, build, write. Raises BlueprintError (with .report when
    the recipe was refused). Returns the summary."""
    r = normalize(recipe)
    rep = validate(r)
    if not rep.ok:
        n = len(rep.errors)
        raise BlueprintError(
            f'The recipe is not buildable ({n} problem{"s" if n != 1 else ""}): '
            + '; '.join(rep.errors[:4]) + ('; …' if n > 4 else '')
            + '. Fix the values in the table and rebuild.', kind='recipe', report=rep)
    b = build(rep.resolved)
    os.makedirs(out_dir, exist_ok=True)
    step_path = os.path.join(out_dir, f'{stem}.step')
    _write_step(b, step_path, r['name'])
    preview = os.path.join(out_dir, 'preview.stl')
    _write_preview(b, preview)
    script = os.path.join(out_dir, f'{stem}_fusion.py')
    with open(script, 'w') as f:
        f.write(emit_fusion_blueprint_script(r, title=title or r['name']))
    with open(os.path.join(out_dir, 'recipe.json'), 'w') as f:
        json.dump(r, f, indent=1)
    want = [rep.resolved['overall'].get(k) for k in ('w', 'd', 'h')]
    got = b.bbox['size']
    dev = [((got[i] - want[i]) / want[i] * 100.0) if want[i] else None for i in range(3)]
    return {
        'mode': 'blueprint', 'recipe': r,
        'warnings': list(rep.warnings) + list(b.warnings), 'errors': [],
        'bbox': b.bbox, 'volume_mm3': b.volume_mm3, 'solids': b.n_solids,
        'overall_check': {'expected': want, 'got': got, 'dev_pct': dev},
        'per_feature': b.per_feature,
        'has_fusion_script': True, 'has_step': True, 'preview': 'preview.stl',
        'features': len(r['features']), 'params': len(r['params']),
    }


def _write_step(b: Build, path, name):
    from ..rebuild import write_step
    solids = b.solid.solids().vals()
    shapes = [s.wrapped for s in solids] if solids else [b.solid.val().wrapped]
    names = [name] * len(shapes)
    write_step(shapes, path, names=names)


def _write_preview(b: Build, path):
    from ..pipeline import tessellate_solid
    tessellate_solid(b.solid, PREVIEW_TOL, PREVIEW_ANG).export(path)
