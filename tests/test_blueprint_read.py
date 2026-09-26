"""The reader without a network: image preparation, answer parsing, the
repair round against a scripted provider, and the provider factory."""
import io
import json
import os

import pytest

pytest.importorskip('PIL')

from stl_to_solid.blueprint.providers import Provider, ReadError, make_provider
from stl_to_solid.blueprint.read_drawing import parse_recipe_text, prepare_image, read_drawing

FIX = os.path.join(os.path.dirname(__file__), 'fixtures', 'sg90_recipe.json')


def _png_bytes(size=(2400, 1200), mode='RGB'):
    from PIL import Image
    b = io.BytesIO()
    Image.new(mode, size, 'white').save(b, 'PNG')
    return b.getvalue()


def test_prepare_image_downscales_and_keeps_png():
    from PIL import Image
    data, media = prepare_image(_png_bytes())
    assert media == 'image/png'
    im = Image.open(io.BytesIO(data))
    assert max(im.size) == 1568
    data, media = prepare_image(_png_bytes((400, 300), 'RGBA'))
    assert Image.open(io.BytesIO(data)).mode == 'RGB'
    with pytest.raises(ReadError):
        prepare_image(_png_bytes((100, 100)))
    with pytest.raises(ReadError):
        prepare_image(b'not an image at all')


def test_prepare_image_jpeg_stays_jpeg():
    from PIL import Image
    b = io.BytesIO()
    Image.new('RGB', (800, 600), 'white').save(b, 'JPEG')
    _, media = prepare_image(b.getvalue())
    assert media == 'image/jpeg'


def test_parse_recipe_text():
    assert parse_recipe_text('{"a": 1}') == {'a': 1}
    assert parse_recipe_text('```json\n{"a": 1}\n```') == {'a': 1}
    assert parse_recipe_text('Here it is:\n{"a": {"b": [1]}}\nDone.') == {'a': {'b': [1]}}
    with pytest.raises(ReadError):
        parse_recipe_text('no json here')


class Scripted(Provider):
    """Answers in order; records what it was asked."""
    def __init__(self, answers):
        super().__init__('k', 'm')
        self.answers = list(answers)
        self.calls = []

    def complete(self, system, image_bytes, media_type, text, schema, history=()):
        self.calls.append({'system': system, 'text': text, 'history': list(history), 'media': media_type,
                           'schema': schema})
        return self.answers.pop(0), {'input_tokens': 100, 'output_tokens': 50}, 'scripted-1'


def test_read_drawing_good_answer_no_repair():
    with open(FIX) as f:
        rec = f.read()
    p = Scripted([rec])
    out = read_drawing(_png_bytes((800, 600)), 'anthropic', 'k', client=p, hints='the top view has the boss')
    assert out['validation']['errors'] == [] and not out['repaired']
    assert out['usage'] == {'input_tokens': 100, 'output_tokens': 50, 'calls': 1}
    assert out['model'] == 'scripted-1' and out['provider'] == 'anthropic'
    assert out['recipe']['params'][0]['name'] == 'body_w'
    assert 'the top view has the boss' in p.calls[0]['text']
    assert 'Origin = the minimum corner' in p.calls[0]['system']
    assert p.calls[0]['schema']['type'] == 'object'


def test_read_drawing_repairs_once():
    with open(FIX) as f:
        good = json.load(f)
    bad = json.loads(json.dumps(good))
    bad['overall']['w'] = 50
    p = Scripted([json.dumps(bad), json.dumps(good)])
    out = read_drawing(_png_bytes((800, 600)), 'openai', 'k', client=p)
    assert out['repaired'] and out['validation']['errors'] == []
    assert out['usage']['calls'] == 2
    hist = p.calls[1]['history']
    assert hist[0][0] == 'assistant' and hist[1][0] == 'user'
    assert 'along X' in hist[1][1] and 'corrected recipe' in hist[1][1]


def test_read_drawing_keeps_the_better_answer_when_repair_is_worse():
    with open(FIX) as f:
        good = json.load(f)
    bad = json.loads(json.dumps(good))
    bad['overall']['w'] = 50
    worse = json.loads(json.dumps(bad))
    worse['features'][0]['op'] = 'cut'
    p = Scripted([json.dumps(bad), json.dumps(worse)])
    out = read_drawing(_png_bytes((800, 600)), 'openai', 'k', client=p, repair=True)
    assert out['recipe']['overall']['w'] == 50 and len(out['validation']['errors']) == 1


def test_read_drawing_unparseable_answer():
    p = Scripted(['I cannot see the image.'])
    with pytest.raises(ReadError):
        read_drawing(_png_bytes((800, 600)), 'openai', 'k', client=p, repair=False)


def test_make_provider():
    assert make_provider('anthropic', 'k', 'claude-opus-5').name == 'anthropic'
    p = make_provider('deepseek', 'k', 'deepseek-chat', 'https://api.deepseek.com')
    assert p.name == 'openai' and p.base_url == 'https://api.deepseek.com'
    with pytest.raises(ReadError):
        make_provider('gemini', 'k', 'x')
