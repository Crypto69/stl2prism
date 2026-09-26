"""Read a drawing image into a recipe: prepare the picture, ask the
provider for the JSON, validate it, and give the model one round to fix
what the checks found. The result carries the recipe even when checks
still fail, so the user can fix the numbers in the table."""
import io
import json
import re
import time

from .prompts import REPAIR_PROMPT, SYSTEM_PROMPT, USER_PROMPT
from .providers import ReadError, make_provider
from .schema import SCHEMA, normalize
from .validate import validate

MAX_SIDE = 1568           # px: what the vision models resample to anyway
MIN_SIDE = 200
MAX_INPUT = 20 * 1024 * 1024


def prepare_image(data):
    """(bytes, media_type): rotated by EXIF, RGB, the longest side at most
    MAX_SIDE px, JPEG (q92) when the source was a JPEG else PNG (line art
    stays crisp)."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    if len(data) > MAX_INPUT:
        raise ReadError(f'The image is larger than {MAX_INPUT // (1 << 20)} MB.', 'image')
    try:
        im = Image.open(io.BytesIO(data))
        fmt = (im.format or '').upper()
        im = ImageOps.exif_transpose(im)
        w, h = im.size
    except UnidentifiedImageError:
        raise ReadError('The file is not an image the server can read (JPEG, PNG or WebP).', 'image')
    except OSError as e:
        raise ReadError(f'The image could not be read ({e}).', 'image')
    if min(w, h) < MIN_SIDE:
        raise ReadError(f'The image is only {w} x {h} pixels; the labels would be unreadable.', 'image')
    if im.mode not in ('RGB', 'L'):
        bg = Image.new('RGB', im.size, 'white')
        try:
            bg.paste(im.convert('RGBA'), mask=im.convert('RGBA').split()[-1])
            im = bg
        except Exception:
            im = im.convert('RGB')
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / float(max(w, h))
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    out = io.BytesIO()
    if fmt == 'JPEG':
        im.convert('RGB').save(out, 'JPEG', quality=92)
        return out.getvalue(), 'image/jpeg'
    im.save(out, 'PNG', optimize=True)
    return out.getvalue(), 'image/png'


_FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.IGNORECASE)


def parse_recipe_text(text):
    """The JSON object in a model answer (fences and chatter tolerated)."""
    t = _FENCE.sub('', text or '').strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    a, b = t.find('{'), t.rfind('}')
    if a >= 0 and b > a:
        try:
            return json.loads(t[a:b + 1])
        except ValueError:
            pass
    raise ReadError('The model\'s answer was not a JSON recipe.', 'parse')


def read_drawing(image_bytes, provider, api_key, model=None, base_url=None, hints='',
                 timeout=300.0, repair=True, client=None):
    """-> {'recipe', 'raw_text', 'usage': {input_tokens, output_tokens, calls},
    'model', 'provider', 'repaired', 'validation': {errors, warnings},
    'seconds'}. `client` (a Provider) is injectable for tests."""
    t0 = time.time()
    img, media = prepare_image(image_bytes)
    p = client or make_provider(provider, api_key, model, base_url, timeout)
    text = USER_PROMPT + (f'\n\nNotes from the user: {hints.strip()}' if hints and hints.strip() else '')
    usage = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}

    def ask(history=()):
        out, u, m = p.complete(SYSTEM_PROMPT, img, media, text, SCHEMA, history)
        usage['calls'] += 1
        for k in ('input_tokens', 'output_tokens'):
            if u.get(k) is not None:
                usage[k] += int(u[k])
        return out, m

    raw, used_model = ask()
    recipe = normalize(parse_recipe_text(raw))
    rep = validate(recipe)
    repaired = False
    if rep.errors and repair:
        errs = '\n'.join(f'- {e}' for e in rep.errors[:12])
        raw2, used_model = ask(history=(('assistant', raw), ('user', REPAIR_PROMPT.format(errors=errs))))
        try:
            recipe2 = normalize(parse_recipe_text(raw2))
        except ReadError:
            recipe2 = None
        if recipe2 is not None:
            rep2 = validate(recipe2)
            # keep the better of the two
            if len(rep2.errors) <= len(rep.errors):
                recipe, rep, raw, repaired = recipe2, rep2, raw2, True
    return {'recipe': recipe, 'raw_text': raw, 'usage': usage, 'model': used_model,
            'provider': provider, 'repaired': repaired,
            'validation': {'errors': list(rep.errors), 'warnings': list(rep.warnings)},
            'seconds': round(time.time() - t0, 1)}
