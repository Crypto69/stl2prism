"""Blueprint's server side: the drawing upload (an image, not a mesh),
which vision providers can read it and where their keys come from, and
the glue between the reader (stl_to_solid.blueprint.read_drawing) and
the job registry. The geometry lives in stl_to_solid.blueprint.

Keys: the browser sends the user's own key per request (header
X-Api-Key) and it is used for that one call and dropped; it is never
written to disk or logged. With no header, a server-side key from the
environment is used when set (an opt-in for a private install: on a
shared NAS anyone on the LAN could spend it).
"""
import os

IMAGE_EXTS = ('jpg', 'jpeg', 'png', 'webp')
IMAGE_MEDIA = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}
MAX_IMAGE = int(os.environ.get('STLTOSOLID_MAX_IMAGE', 20 * 1024 * 1024))
READ_TIMEOUT_S = float(os.environ.get('STLTOSOLID_BLUEPRINT_TIMEOUT', 300))

# The vision providers the panel offers. `env` is the server-side key's
# variable; `base_url` None means the SDK's default endpoint; 'custom'
# takes both the URL and the model from the request (an OpenAI-compatible
# server such as Ollama or OpenRouter — Ollama with a vision model keeps
# the drawing on the user's own machine).
PRESETS = {
    'anthropic': {'label': 'Anthropic (Claude)', 'default_model': 'claude-opus-5', 'base_url': None,
                  'env': 'STLTOSOLID_ANTHROPIC_API_KEY', 'sdk': 'anthropic', 'needs_key': True,
                  'note': ''},
    'openai': {'label': 'OpenAI', 'default_model': 'gpt-5', 'base_url': None,
               'env': 'STLTOSOLID_OPENAI_API_KEY', 'sdk': 'openai', 'needs_key': True,
               'note': 'Model names change; edit the model if the API says it does not exist.'},
    'deepseek': {'label': 'DeepSeek', 'default_model': 'deepseek-chat', 'base_url': 'https://api.deepseek.com',
                 'env': 'STLTOSOLID_DEEPSEEK_API_KEY', 'sdk': 'openai', 'needs_key': True,
                 'note': "DeepSeek's hosted API may not accept images; if it refuses, the server's "
                         'own message is shown here.'},
    'custom': {'label': 'Custom (OpenAI-compatible URL)', 'default_model': '', 'base_url': '',
               'env': 'STLTOSOLID_CUSTOM_API_KEY', 'sdk': 'openai', 'needs_key': False,
               'note': 'Any OpenAI-compatible server: base URL and model are yours to type. Ollama '
                       '(http://localhost:11434/v1) with a vision model keeps the drawing local.'},
}
PROVIDER_ORDER = ('anthropic', 'openai', 'deepseek', 'custom')

NO_KEY_MESSAGE = ('No API key for {label}: paste one in the Blueprint panel (it stays in your '
                  'browser), or set {env} on the server.')


def server_key(provider):
    p = PRESETS.get(provider)
    if not p:
        return None
    return os.environ.get(p['env']) or None


def resolve_key(provider, header_key):
    """The key to use: the request's, else the server's, else None."""
    k = (header_key or '').strip()
    if k:
        return k
    return server_key(provider)


def config():
    """What the panel needs to offer the providers."""
    return {
        'providers': [{
            'key': k, 'label': p['label'], 'default_model': p['default_model'],
            'base_url': p['base_url'], 'server_key': server_key(k) is not None,
            'needs_key': p['needs_key'], 'note': p['note'], 'env': p['env'],
        } for k, p in ((k, PRESETS[k]) for k in PROVIDER_ORDER)],
        'max_image_mb': MAX_IMAGE // (1 << 20),
    }


def sdk_available(provider):
    """None when the provider's SDK imports, else the pip hint."""
    sdk = PRESETS[provider]['sdk']
    try:
        __import__(sdk)
    except ImportError:
        return f'The server is missing the {sdk} package: pip install {sdk}'
    return None


def verify_image(path):
    """(width, height) of a real image, else ValueError with a sentence.
    PIL's verify() leaves the file object unusable, so it is reopened."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
    except UnidentifiedImageError:
        raise ValueError('the file is not an image the server can read (JPEG, PNG or WebP)')
    except OSError as e:
        raise ValueError(f'the image could not be read ({e})')
    if w < 64 or h < 64:
        raise ValueError(f'the image is only {w} x {h} pixels; the labels would be unreadable')
    return w, h


# OpenAI-compatible servers only give model ids; these are the families that
# take images in a chat completion, and the words that mark models that do not
_OPENAI_VISION_PREFIXES = ('gpt-4o', 'gpt-4.1', 'gpt-4.5', 'gpt-5', 'o1', 'o3', 'o4', 'chatgpt-')
_OPENAI_SKIP = ('audio', 'realtime', 'transcribe', 'tts', 'search', 'image', 'embedding', 'moderation',
                'codex', 'computer-use', 'instruct', 'preview-2024', 'whisper', 'dall-e', 'babbage', 'davinci')


def list_models(provider, api_key, base_url=None, timeout=30.0):
    """[{'id', 'label', 'created'}], newest first, for the panel's dropdown:
    Anthropic filtered to models that take images (the API says so); an
    OpenAI-compatible server filtered by family names for OpenAI itself and
    left whole for DeepSeek / a custom server (they say nothing about
    images). Raises ReadError with a sentence."""
    from stl_to_solid.blueprint.providers import ReadError, _short
    preset = PRESETS[provider]
    url = (base_url or '').strip() or preset['base_url'] or None
    if provider == 'anthropic':
        import anthropic
        try:
            client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=1)
            out = []
            for m in client.models.list():
                caps = getattr(m, 'capabilities', None)
                img = getattr(getattr(caps, 'image_input', None), 'supported', True)
                if not img:
                    continue
                created = getattr(m, 'created_at', None)
                out.append({'id': m.id, 'label': getattr(m, 'display_name', None) or m.id,
                            'created': created.isoformat() if hasattr(created, 'isoformat') else str(created or '')})
            out.sort(key=lambda x: x['created'], reverse=True)
            return out
        except anthropic.AuthenticationError:
            raise ReadError('Anthropic refused the API key. Check the key in the Blueprint panel.', 'key')
        except anthropic.APIStatusError as e:
            raise ReadError(f'Anthropic answered {e.status_code}: {_short(e.message)}', 'api')
        except anthropic.APIConnectionError:
            raise ReadError('Could not reach the Anthropic API from the server (network or timeout).', 'network')
    import openai
    where = f'the server at {url}' if url else 'OpenAI'
    try:
        client = openai.OpenAI(api_key=api_key or 'none', base_url=url, timeout=timeout, max_retries=1)
        models = list(client.models.list())
    except openai.AuthenticationError:
        raise ReadError(f'{where} refused the API key. Check the key in the Blueprint panel.', 'key')
    except openai.APIStatusError as e:
        raise ReadError(f'{where} answered {e.status_code}: {_short(getattr(e, "message", str(e)))}', 'api')
    except openai.APIConnectionError:
        raise ReadError(f'Could not reach {where} from the server (network or timeout).', 'network')
    out = []
    for m in models:
        mid = getattr(m, 'id', None)
        if not mid:
            continue
        if provider == 'openai':
            if not mid.startswith(_OPENAI_VISION_PREFIXES) or any(w in mid for w in _OPENAI_SKIP):
                continue
        out.append({'id': mid, 'label': mid, 'created': int(getattr(m, 'created', 0) or 0)})
    out.sort(key=lambda x: (x['created'], x['id']), reverse=True)
    return out
