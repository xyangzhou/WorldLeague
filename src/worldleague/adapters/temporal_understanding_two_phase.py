"""Upgrade original Unobserved Change snapshots into a continuous two-phase task.

Used by worldleague scene --help.
"""
from __future__ import annotations


import json

from worldleague.adapters import temporal_understanding_unobserved_change as legacy


WARNING_CODES = ('external_source_phase_transition', 'native_temporal_execution_unverified')


def physical_target_keys(parameters):
    """Return only declared movable targets, including compound event obligations."""
    operations = parameters.get('physical_operations')
    return [row['physical_target_key'] for row in operations] if operations is not None else [
        parameters['physical_target_key']] if parameters.get('physical_target_key') else []


class TemporalInitializationError(RuntimeError):
    """Fail initialization with structured provenance instead of hiding source lids."""
    def __init__(self, stage, reason, parameters, **evidence):
        self.details = {'schema': 'unobserved_two_phase_initialization_failure_v1',
            'accepted': False, 'failure_stage': stage, 'failure_reason': reason,
            'source_identity': parameters.get('source_identity'), 'temporal_episode': True,
            'phase': 'phase_1', **evidence}
        super().__init__(json.dumps(self.details, sort_keys=True))


def configure_scene(config, parameters):
    """Configure fixed collidable source contents before native construction only."""
    if parameters.get('phase') != 'image1' or parameters.get('temporal_episode') is not True:
        raise ValueError('A continuous temporal episode must initialize the original first phase.')
    evidence = legacy.configure_scene(config, parameters)
    if parameters.get('survey_revision') == 'free_survey_10_v1':
        config['scene']['load_room_instances'] = [parameters['room']]
    names = {parameters['bindings'][key]['name'] for phase in legacy.PHASES
             for key in parameters['phase_keys'][phase]}
    for obj in config['objects']:
        if obj.get('name') in names:
            obj.update(fixed_base=True, visual_only=False)
    if parameters.get('task_edit2_physical'):
        from .temporal_understanding_agent_observation import apparatus_objects
        for target_key in physical_target_keys(parameters):
            if target_key not in parameters['phase_keys']['image2']:
                continue
            target_name = parameters['bindings'][target_key]['name']
            next(row for row in config['objects'] if row['name'] == target_name)['fixed_base'] = False
        additions = apparatus_objects(parameters['apparatus'])
        names = {row['name'] for row in config['objects']}
        if names.intersection(row['name'] for row in additions):
            raise ValueError('Temporal apparatus cannot overwrite source objects.')
        config['objects'].extend(additions)
    return {**evidence, 'temporal_episode': True, 'phase': 'phase_1', 'source_phase': 'image1',
        'inactive_phase_policy': 'Original distant preload; inactive actors stay outside Agent observations.',
        'warning_codes': list(WARNING_CODES)}


def initialize_scene(session, parameters):
    """Restore original containers once, open supported lids, then install phases."""
    if parameters.get('phase') != 'image1' or parameters.get('temporal_episode') is not True:
        raise TemporalInitializationError('phase_configuration', 'original_phase_1_required', parameters)
    try:
        evidence = legacy.initialize_scene(session, parameters)
    except Exception as error:
        raise TemporalInitializationError('source_initialization', 'legacy_source_initialization_failed', parameters,
            exception_summary=f'{type(error).__name__}: {error}') from error
    from omnigibson.object_states import Open
    opened = []
    for key in parameters['container_keys']:
        obj = session.env.scene.object_registry('name', parameters['bindings'][key]['name'])
        try:
            supported = Open in obj.states
            if supported:
                state = obj.states[Open]
                # The new survey keeps every lid fully open: a random partially
                # open flap can satisfy Open while still hiding all contents.
                requested = (state.set_value(True, fully=True)
                    if parameters.get('survey_revision') == 'free_survey_10_v1'
                    else state.set_value(True))
                is_open = bool(state.get_value())
                if requested is False or not is_open:
                    raise ValueError('Native Open did not reach the requested first-phase state.')
                for joint in state.relevant_joints_info[1]:
                    for link in obj.links.values():
                        if link.prim_path == joint.body1:
                            link.visible = True
            else:
                is_open = None
            opened.append({'binding': key, 'open_supported': supported, 'open': is_open})
        except Exception as error:
            raise TemporalInitializationError('phase_1_opening', 'original_lid_open_initialization_failed', parameters,
                container_binding=key, exception_summary=f'{type(error).__name__}: {error}') from error
    if parameters.get('task_edit2_physical'):
        for key in physical_target_keys(parameters):
            if key not in parameters['phase_keys']['image2']:
                continue
            target = session.env.scene.object_registry('name', parameters['bindings'][key]['name'])
            if target.fixed_base or not callable(getattr(target, 'disable_gravity', None)):
                raise TemporalInitializationError('dynamic_preload', 'native_dynamic_gravity_adapter_unavailable', parameters)
            target.disable_gravity()
            target.keep_still()
    try:
        from worldleague.native.esi_temporal_phases import install_temporal_phases
        installed = install_temporal_phases(session, parameters)
    except Exception as error:
        raise TemporalInitializationError('temporal_installation', 'continuous_phase_installation_failed', parameters,
            exception_summary=f'{type(error).__name__}: {error}') from error
    return {**evidence, 'temporal_episode': True, 'phase': 'phase_1', 'source_phase': 'image1',
        'initial_open_states': opened, 'temporal_phase_installation': installed,
        'human_framing_names': [parameters['bindings'][key]['name'] for key in
            parameters['container_keys'] + parameters['phase_keys']['image1']],
        'historical_agent_observations_created': False, 'warning_codes': list(WARNING_CODES)}


def create_evaluator(session, controller, config, output):
    """Delegate the continuous observation/history scorer to its shared owner."""
    from worldleague.native.esi_temporal_phases import create_evaluator as native_evaluator
    return native_evaluator(session, controller, config, output)


def on_interaction_initialized(session, evidence):
    """Rebase both source phases after accepted placement, before Agent images."""
    controller = getattr(session, 'temporal_phase_controller', None)
    if controller is None:
        raise ValueError('Temporal placement requires the initialized continuous controller.')
    if evidence.get('accepted') is not True:
        failures = [{'bindings': row.get('bindings'),
            'reason': row.get('evidence', {}).get('reason'),
            'failure_stage': row.get('evidence', {}).get('failure_stage'),
            'last_attempt': row.get('evidence', {}).get('attempts', [])[-1:],
            'support_search': row.get('evidence', {}).get('support_search'),
            'initial_support_contacts': row.get('evidence', {}).get('initial_support_contacts'),
            'height_screen_before': row.get('evidence', {}).get('height_screen_before'),
            'layout_policy': row.get('evidence', {}).get('layout_policy', evidence.get('layout_policy')),
            'spawn_grid': {key: row.get('evidence', {}).get('spawn_grid', {})[key]
                           for key in ('requested_source_region', 'measured_source_region')
                           if key in row.get('evidence', {}).get('spawn_grid', {})}}
            for row in evidence.get('per_group', []) if not row.get('evidence', {}).get('accepted')]
        raise ValueError('Temporal support placement failed: ' + json.dumps({
            'reason': evidence.get('reason'), 'groups': failures}, default=str))
    return controller.rebase_interaction(evidence)
