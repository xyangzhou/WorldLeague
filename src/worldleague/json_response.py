"""Extract an unambiguous response object.

Used by worldleague scene --help.
"""
import json
import re

PARSER_VERSION = 'embedded_json_object_v1'


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _container_end(text, start):
    """Skip a malformed container whole, never salvage one of its nested objects."""
    stack, quoted, escaped = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c in '{[':
            stack.append(c)
        elif c in '}]':
            if not stack or (stack.pop(), c) not in (('{', '}'), ('[', ']')):
                return i + 1
            if not stack:
                return i + 1
    return len(text)


def extract_json_object(raw, *, response_keys):
    """Return one distinct relevant object plus wrapper provenance; never repair JSON."""
    if not isinstance(raw, str):
        raise ValueError('Response must be text.')
    decoder = json.JSONDecoder(object_pairs_hook=_unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value.')))
    candidates = {}
    i = 0
    while i < len(raw):
        if raw[i] not in '{[':
            i += 1
            continue
        start = i
        try:
            value, i = decoder.raw_decode(raw, i)
        except ValueError:
            i = _container_end(raw, start)
            continue
        if isinstance(value, dict) and set(value).intersection(response_keys):
            key = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
            candidates[key] = (value, start, i)
    if len(candidates) != 1:
        raise ValueError('Expected one unambiguous response JSON object.')
    value, start, end = next(iter(candidates.values()))
    if raw.strip() == raw[start:end]:
        wrapper = 'none'
    elif re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n.*?\r?\n```\s*', raw, re.DOTALL | re.IGNORECASE):
        wrapper = 'single_markdown_fence'
    else:
        wrapper = 'embedded_json'
    return value, wrapper
