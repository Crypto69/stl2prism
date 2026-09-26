"""A real read of samples/sg-90-drawing1.jpg. Runs only when asked
(STLTOSOLID_LIVE_LLM=1) and a key is at hand: the environment variable
for the provider, or a one-line file under .secrets/ (git-excluded). The
key is never printed. Costs a few cents per run."""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, 'samples', 'sg-90-drawing1.jpg')
PROVIDERS = {
    'anthropic': ('STLTOSOLID_ANTHROPIC_API_KEY', 'anthropic.key', None),
    'openai': ('STLTOSOLID_OPENAI_API_KEY', 'openai.key', None),
    'deepseek': ('STLTOSOLID_DEEPSEEK_API_KEY', 'deepseek.key', 'https://api.deepseek.com'),
}

pytestmark = pytest.mark.skipif(os.environ.get('STLTOSOLID_LIVE_LLM') != '1',
                                reason='set STLTOSOLID_LIVE_LLM=1 to spend a few cents on a real read')


def _key(provider):
    env, fname, _ = PROVIDERS[provider]
    k = os.environ.get(env)
    if k:
        return k
    p = os.path.join(ROOT, '.secrets', fname)
    if os.path.exists(p):
        with open(p) as f:
            return f.read().strip()
    return None


@pytest.mark.parametrize('provider', list(PROVIDERS))
def test_live_read_sg90(provider, tmp_path):
    key = _key(provider)
    if not key:
        pytest.skip(f'no key for {provider}')
    if not os.path.exists(SAMPLE):
        pytest.skip('sample drawing missing')
    pytest.importorskip('anthropic' if provider == 'anthropic' else 'openai')
    from backend.blueprint import PRESETS
    from stl_to_solid.blueprint.read_drawing import read_drawing
    from stl_to_solid.blueprint.compile import compile_recipe
    with open(SAMPLE, 'rb') as f:
        img = f.read()
    out = read_drawing(img, provider, key, model=PRESETS[provider]['default_model'],
                       base_url=PROVIDERS[provider][2])
    rec = out['recipe']
    print(f"\n[{provider}] {out['model']}: {out['usage']} in {out['seconds']} s, repaired={out['repaired']}")
    print('params:', [(p['name'], p['value']) for p in rec['params']])
    print('features:', [(f['id'], f['op'], f['plane']) for f in rec['features']])
    print('checks:', out['validation'])
    with open(tmp_path / f'{provider}_recipe.json', 'w') as f:
        json.dump(rec, f, indent=1)
    assert out['validation']['errors'] == [], out['validation']['errors']
    r = compile_recipe(rec, str(tmp_path / provider))
    w, d, h = r['bbox']['size']
    print('built:', r['solids'], 'solid(s)', f'{w:.1f} x {d:.1f} x {h:.1f} mm', f"{r['volume_mm3']:.0f} mm^3")
    assert r['solids'] == 1
    assert abs(d - 11.8) / 11.8 < 0.05, d
    assert abs(h - 29.9) / 29.9 < 0.06 or abs(h - 22.7) / 22.7 < 0.05, h
    assert abs(w - 32.2) / 32.2 < 0.06 or abs(w - 22.5) / 22.5 < 0.05, w
