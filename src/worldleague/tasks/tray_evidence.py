"""Resolve independent tray predicates without inventing contact evidence.

Used by worldleague scene --help.
"""
from copy import deepcopy

VERSION = 'tray_contact_dependency_review_v1'
SCHEMAS = {'wia_spatial_tray_final_state_progress_v1', 'wia_spatial_tray_current_tray_progress_v2'}
FIELDS = {'wholly_inside_usable_xy', 'actual_nonpenetrating_deck_contact', 'on_top_deck',
          'bottom_at_deck', 'no_robot_support', 'dynamic_collidable', 'stationary',
          'asset_scale_preserved', 'original_identity_preserved'}


def isolated_contact_errors(native):
    """Only native deck-contact uncertainty may be excluded from other predicates."""
    checks = native.get('terminal_checks', {})
    errors = native.get('measurement_errors', [])
    allowed = {key + ':native_deck_contact_unavailable' for key in checks}
    return bool(native.get('schema') in SCHEMAS
        and native.get('manipulation_revision') == 'final_state_v1'
        and not native.get('measurement_error') and errors and set(errors) <= allowed
        and not native.get('initial_measurement_errors') and not native.get('constraint_violations')
        and not native.get('sample_gap_exceeded') and len(checks) in (2, 3)
        and all(set(row) == FIELDS and all(type(v) is bool for v in row.values())
                for row in checks.values()))


def count_reference_available(rule, native):
    """Permit only the authored on-tray count whose own inputs remain valid."""
    return (rule.get('kind') == 'count_true' and rule.get('path') == ['terminal_checks']
        and rule.get('fields') == [['on_top_deck'], ['wholly_inside_usable_xy']]
        and isolated_contact_errors({**native, 'measurement_errors':
            native.get('measurement_errors') or native.get('unresolved_measurement_errors', [])}))


def resolve_known_failure(native):
    """Certify false conjunctions; never infer positive contact or historical dwell."""
    result = deepcopy(native)
    if not isolated_contact_errors(result):
        return result
    relation = result.get('declared_relation', {})
    # The legacy shape reducer omits 'available'; failed geometry is instead
    # reported as declared_relation_unavailable, excluded by the gate above.
    relation_available = relation.get('available') is True or (
        'available' not in relation and relation.get('shape') in
        ('equilateral', 'isosceles', 'random triangle', 'collinear')
        and relation.get('plane') == 'world_xy')
    if not relation_available or type(relation.get('satisfied')) is not bool:
        return result
    witnesses = {key: [field for field, value in row.items()
                       if field not in ('stationary', 'actual_nonpenetrating_deck_contact') and value is False]
                 for key, row in result['terminal_checks'].items()}
    witnesses = {key: fields for key, fields in witnesses.items() if fields}
    if not witnesses:
        return result
    items = result.get('completion_items', [])
    if [i['id'] for i in items] != ['targets_on_tray', 'declared_relation', 'stable_released_tray_placement']:
        return result
    # The missing contact cannot make all-target placement true when a required
    # independently measured predicate is false. Stable placement requires it too.
    if items[0].get('completed') or items[2].get('completed'):
        return result
    values = [False, relation['satisfied'], False]
    result['unresolved_measurement_errors'] = result['measurement_errors']
    result['measurement_errors'] = []
    result['evidence_resolution'] = {'version': VERSION, 'known_false_witnesses': witnesses,
        'scope': 'operation_conjunction_only; contact remains unknown'}
    for item, value in zip(items, values):
        item.update(completed=value, evaluation_validity='valid')
    result.update(evaluation_validity='complete', available=True, task_success=False, physical_success=False,
                  completed_goals=sum(values), completion_score=sum(values)/3,
                  score_percent=100*sum(values)/3, terminal_valid=False)
    return result
