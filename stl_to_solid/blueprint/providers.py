"""The vision providers behind Blueprint, one small adapter each:
Anthropic through its SDK (structured output by JSON schema), and every
OpenAI-compatible chat endpoint through the openai SDK (OpenAI itself,
DeepSeek, Ollama, OpenRouter... by base URL), which tries strict JSON
schema, then json_object, then plain text, since compatible servers
differ in what they accept.

Each adapter's complete() takes the same neutral request and returns
(text, usage, model). Failures come back as ReadError with a sentence
for the user; the key is used for the one client and never repeated in
any message.
"""
import base64


class ReadError(RuntimeError):
    """A sentence for the user (errors.describe returns it verbatim)."""
    readable = True

    def __init__(self, message, kind='read'):
        super().__init__(message)
        self.kind = kind


# the model's thinking counts against this too, so it is generous; the
# request streams so the HTTP connection is not held open for one answer
MAX_TOKENS = 64000


class Provider:
    name = 'base'

    def __init__(self, api_key, model, base_url=None, timeout=300.0):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def complete(self, system, image_bytes, media_type, text, schema, history=(), effort='high'):
        """`history`: (role, text) pairs after the first user turn, for the
        repair round. `effort`: 'low' | 'medium' | 'high', how hard the model
        thinks (Anthropic output_config.effort; OpenAI reasoning_effort,
        dropped for a server that rejects it). Returns
        (text, {'input_tokens', 'output_tokens'}, model)."""
        raise NotImplementedError


EFFORTS = ('low', 'medium', 'high')


class AnthropicProvider(Provider):
    name = 'anthropic'

    def complete(self, system, image_bytes, media_type, text, schema, history=(), effort='high'):
        import anthropic
        effort = effort if effort in EFFORTS else 'high'
        client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout, max_retries=2)
        b64 = base64.standard_b64encode(image_bytes).decode('ascii')
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type, 'data': b64}},
            {'type': 'text', 'text': text}]}]
        messages += [{'role': r, 'content': t} for r, t in history]
        kwargs = dict(model=self.model, max_tokens=MAX_TOKENS,
                      system=[{'type': 'text', 'text': system, 'cache_control': {'type': 'ephemeral'}}],
                      messages=messages)
        try:
            try:
                with client.messages.stream(
                        output_config={'effort': effort,
                                       'format': {'type': 'json_schema', 'schema': schema}},
                        **kwargs) as stream:
                    resp = stream.get_final_message()
            except TypeError:
                # an older SDK without output_config: the prompt alone asks for JSON
                with client.messages.stream(**kwargs) as stream:
                    resp = stream.get_final_message()
        except anthropic.AuthenticationError:
            raise ReadError('Anthropic refused the API key. Check the key in the Blueprint panel.', 'key')
        except anthropic.PermissionDeniedError as e:
            raise ReadError(f'Anthropic refused the request: {e.message}', 'key')
        except anthropic.NotFoundError:
            raise ReadError(f'Anthropic has no model called {self.model!r}. Edit the model name.', 'model')
        except anthropic.RateLimitError:
            raise ReadError('Anthropic is rate-limiting this key; wait a minute and read again.', 'rate')
        except anthropic.APIStatusError as e:
            raise ReadError(f'Anthropic answered {e.status_code}: {_short(e.message)}', 'api')
        except anthropic.APIConnectionError:
            raise ReadError('Could not reach the Anthropic API from the server (network or timeout).', 'network')
        stop = getattr(resp, 'stop_reason', None)
        if stop == 'refusal':
            raise ReadError('The model declined to read this image.', 'refusal')
        u = getattr(resp, 'usage', None)
        if stop == 'max_tokens':
            raise ReadError('The answer was cut off after %s output tokens; try a drawing with fewer '
                            'features.' % getattr(u, 'output_tokens', '?'), 'truncated')
        out = ''.join(b.text for b in resp.content if getattr(b, 'type', '') == 'text')
        usage = {'input_tokens': getattr(u, 'input_tokens', None),
                 'output_tokens': getattr(u, 'output_tokens', None)}
        return out, usage, getattr(resp, 'model', self.model)


class OpenAICompatibleProvider(Provider):
    name = 'openai'

    def complete(self, system, image_bytes, media_type, text, schema, history=(), effort='high'):
        import openai
        effort = effort if effort in EFFORTS else 'high'
        client = openai.OpenAI(api_key=self.api_key or 'none', base_url=self.base_url,
                               timeout=self.timeout, max_retries=2)
        b64 = base64.standard_b64encode(image_bytes).decode('ascii')
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{b64}'}},
                        {'type': 'text', 'text': text}]}]
        messages += [{'role': r, 'content': t} for r, t in history]
        formats = [
            {'type': 'json_schema', 'json_schema': {'name': 'recipe', 'schema': schema, 'strict': True}},
            {'type': 'json_object'},
            None,
        ]
        resp = None
        # the OpenAI reasoning knob; a compatible server that does not know it
        # gets the same request without it
        send_effort = True
        attempts = list(formats)
        while attempts:
            fmt = attempts[0]
            kwargs = dict(model=self.model, messages=messages)
            if fmt is not None:
                kwargs['response_format'] = fmt
            if send_effort:
                kwargs['reasoning_effort'] = effort
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except openai.AuthenticationError:
                raise ReadError(f'{_where(self)} refused the API key. Check the key in the Blueprint panel.', 'key')
            except openai.PermissionDeniedError as e:
                raise ReadError(f'{_where(self)} refused the request: {_short(_msg(e))}', 'key')
            except openai.NotFoundError as e:
                raise ReadError(f'{_where(self)} has no model called {self.model!r} ({_short(_msg(e))}). '
                                'Edit the model name.', 'model')
            except openai.RateLimitError:
                raise ReadError(f'{_where(self)} is rate-limiting this key; wait a minute and read again.', 'rate')
            except openai.BadRequestError as e:
                msg = _msg(e)
                low = msg.lower()
                if send_effort and ('reasoning' in low or 'effort' in low):
                    send_effort = False               # same format, without the knob
                    continue
                # a server that does not know this response_format: try the next
                if fmt is not None and any(k in low for k in
                                           ('response_format', 'json_schema', 'json_object', 'schema',
                                            'strict', 'unsupported', 'not supported', 'invalid_request')):
                    attempts.pop(0)
                    continue
                raise ReadError(f'{_where(self)} rejected the request: {_short(msg)}', 'api')
            except openai.APIStatusError as e:
                raise ReadError(f'{_where(self)} answered {e.status_code}: {_short(_msg(e))}', 'api')
            except openai.APIConnectionError:
                raise ReadError(f'Could not reach {_where(self)} from the server (network or timeout).', 'network')
        if resp is None:
            raise ReadError(f'{_where(self)} accepted none of the JSON output modes.', 'api')
        choice = resp.choices[0] if getattr(resp, 'choices', None) else None
        if choice is None:
            raise ReadError(f'{_where(self)} returned no answer.', 'api')
        msg = choice.message
        if getattr(msg, 'refusal', None):
            raise ReadError('The model declined to read this image: ' + _short(msg.refusal), 'refusal')
        if getattr(choice, 'finish_reason', None) == 'length':
            raise ReadError('The answer was cut off; try a drawing with fewer features.', 'truncated')
        out = msg.content or ''
        if isinstance(out, list):            # some servers return content parts
            out = ''.join(p.get('text', '') if isinstance(p, dict) else getattr(p, 'text', '') for p in out)
        u = getattr(resp, 'usage', None)
        usage = {'input_tokens': getattr(u, 'prompt_tokens', None),
                 'output_tokens': getattr(u, 'completion_tokens', None)}
        return out, usage, getattr(resp, 'model', None) or self.model


def _msg(e):
    m = getattr(e, 'message', None) or str(e)
    body = getattr(e, 'body', None)
    if isinstance(body, dict):
        err = body.get('error')
        if isinstance(err, dict) and err.get('message'):
            m = err['message']
    return str(m)


def _short(m, n=300):
    m = ' '.join(str(m).split())
    return m if len(m) <= n else m[:n - 1] + '…'


def _where(p):
    return f'the server at {p.base_url}' if p.base_url else 'OpenAI'


def make_provider(provider, api_key, model, base_url=None, timeout=300.0):
    if provider == 'anthropic':
        return AnthropicProvider(api_key, model, None, timeout)
    if provider in ('openai', 'deepseek', 'custom'):
        return OpenAICompatibleProvider(api_key, model, base_url, timeout)
    raise ReadError(f'unknown provider {provider!r}', 'api')
