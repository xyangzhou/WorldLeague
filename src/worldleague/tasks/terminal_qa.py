"""Deterministic terminal QA and separate operation scoring.

Used by worldleague scene --help.
"""
from copy import deepcopy
import re
from worldleague.json_response import extract_json_object, PARSER_VERSION

OPERATION_VERSION = 'worldleague_physical_v3'
ANSWER_PARSER_VERSION = 'terminal_qa_' + PARSER_VERSION
SEMANTIC_KINDS = {'answer_requirement', 'semantic_answer'}
SEMANTIC_IDS = {'appearance_report', 'source_relation_answer', 'change_answer', 'history_answer'}


def public_questions(spec):
    return [{k: deepcopy(q[k]) for k in ('id', 'question', 'schema', 'time')}
            for q in spec['questions']]


def at(value, path):
    for key in path:
        value = value[key]
    if value is None:
        raise ValueError('Reference evidence is unavailable.')
    return value


def reference(rule, native):
    """Read only frozen measurements, never the model answer or task success."""
    kind = rule['kind']
    if kind == 'constant':
        return deepcopy(rule['value'])
    if kind == 'any_of':
        return {'any_of': deepcopy(rule['values'])}
    if kind in ('all', 'any'):
        decisive = kind == 'any'
        missing = False
        for child in rule['rules']:
            try:
                value = reference(child, native)
            except (KeyError, IndexError, TypeError, ValueError):
                missing = True
                continue
            if type(value) is not bool:
                missing = True
            elif value is decisive:
                return decisive
        if missing:
            raise ValueError('Missing Boolean evidence for the QA reference.')
        return not decisive
    if (native.get('measurement_error') or native.get('measurement_errors')
            or native.get('unresolved_measurement_errors')):
        from .tray_evidence import count_reference_available
        if not count_reference_available(rule, native):
            raise ValueError('Native QA reference measurements are unavailable.')
    value = at(native, rule['path'])
    if kind == 'value':
        return value
    if kind == 'length':
        return len(value)
    if kind == 'count_true':
        return sum(all(at(row, path) is True for path in rule['fields']) for row in value.values())
    if kind == 'yes_no':
        if type(value) is not bool:
            raise ValueError('Expected measured boolean.')
        return 'Yes' if value else 'No'
    if kind == 'ramp_fastest':
        successful = {key: row['arrival_s'] for key, row in value.items()
            if row.get('valid') and row.get('arrival_s') is not None}
        if not successful:
            return ['none']
        best = min(successful.values())
        return sorted(rule['labels'][key] for key, seconds in successful.items()
                      if seconds <= best + rule['tie_tolerance_s'])
    raise ValueError('Unknown QA reference rule: ' + kind)


def canonical(value):
    if isinstance(value, str):
        return re.sub(r'\s+', ' ', value.strip().casefold().replace('_', ' '))
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def matches(answer, expected, question):
    import jsonschema
    try:
        jsonschema.validate(answer, question['schema'])
    except jsonschema.ValidationError:
        return False
    options = expected.get('any_of') if isinstance(expected, dict) else [expected]
    def normalize(v):
        v = canonical(v)
        return sorted(v) if question.get('unordered') and isinstance(v, list) else v
    return any(normalize(answer) == normalize(v) for v in options)


def grade(spec, raw, native):
    wrapper = 'none'
    try:
        payload, wrapper = extract_json_object(raw, response_keys={'answers'})
        answers = payload['answers'] if isinstance(payload, dict) and set(payload) == {'answers'} else {}
        if (not isinstance(answers, dict)
                or set(answers) - {question['id'] for question in spec['questions']}):
            answers = {}
    except (ValueError, TypeError, KeyError):
        answers = {}
    items = []
    for q in spec['questions']:
        try:
            expected = reference(q['reference'], native)
            correct = q['id'] in answers and matches(answers[q['id']], expected, q)
            items.append({'id': q['id'], 'time': q.get('time'), 'validity': 'complete', 'correct': correct,
                          'answer': answers.get(q['id']), 'reference': expected})
        except (KeyError, IndexError, ValueError, TypeError) as error:
            items.append({'id': q['id'], 'time': q.get('time'), 'validity': 'unavailable', 'correct': None,
                          'error': str(error), 'answer': answers.get(q['id'])})
    complete = all(item['validity'] == 'complete' for item in items)
    return {'items': items, 'validity': 'complete' if complete else 'unavailable',
            'answer_parser_version': ANSWER_PARSER_VERSION, 'response_wrapper': wrapper,
            'fraction': sum(item['correct'] is True for item in items)/len(items) if complete else None}


def operation_progress(evaluation, *, require_text_confirmation=True):
    from .completion import progress
    native = dict(evaluation['native_result'])
    validity = evaluation['progress']['evaluation_validity']
    items = native.get('completion_items')
    if items is not None:
        physical = []
        missing_semantic = False
        for item in items:
            semantic = (item.get('kind') in SEMANTIC_KINDS
                        or item['id'].split(':')[-1] in SEMANTIC_IDS
                        or item['id'].startswith('answer:'))
            if semantic:
                missing_semantic |= item.get('evaluation_validity', 'valid') not in ('valid', 'complete')
            else:
                physical.append(item)
        native['completion_items'] = physical
        if not physical:
            validity = 'unavailable'
        elif missing_semantic and validity == 'partial' and native.get('evaluation_validity', 'complete') in ('complete', 'partial'):
            # Item-level validity is recomputed below using only physical items.
            validity = 'complete'
    native['evaluation_validity'] = validity
    if require_text_confirmation and native.get('physical_branch') == 'no_change' and not native.get('model_declared_done'):
        native['completion_items'] = [dict(item, completed=False, credit=0, terminal_confirmation='active_DONE_required')
                                      for item in native.get('completion_items') or []]
    result = progress(native)
    result['scoring_version'] = OPERATION_VERSION
    if 'final_released' in native:
        released = native['final_released']
        result['terminal_release_satisfied'] = released
        if released is False or (released is None and result.get('constraints_satisfied') is True):
            result['constraints_satisfied'] = released
    success = None
    if result['score_percent'] is not None:
        success = (result['completed_goals'] == result['total_goals'] and result['constraints_satisfied'])
        # Some reducers latch verified targets but separately require empty hands at termination.
        if success is True and native.get('evaluation_validity', 'complete') == 'complete' and native.get('physical_success') is False:
            success = False
    result['task_success'] = success
    return result
