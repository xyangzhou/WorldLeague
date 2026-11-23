"""Convert Partial Occlusion into identification and stable destination placement.

Run PartialOcclusionConverter.convert_all(sources, ConversionContext(batch_root))
with the behavior environment and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=2.
Native hooks use the shared task runtime; they only read measured state. This
module never submits capture jobs, changes source data, or performs robot actions.
"""
from __future__ import annotations


import json
from pathlib import Path

import numpy as np


def _array(value):
    from worldleague.tasks.interaction_setup import array
    return array(value)


def _normalize_answer(value):
    """Normalize category spelling without guessing a semantic synonym."""
    return ' '.join(str(value or '').lower().replace('_', ' ').split())


def _placement_measurements(session, controller, config, target, destination):
    """Measure the named destination directly, never an adapted initial support."""
    values = {'measurement_errors': {}}

    def read(key, callback):
        try:
            values[key] = callback()
        except (KeyError, ValueError, AttributeError, RuntimeError, TypeError, ImportError) as error:
            values[key] = None
            values['measurement_errors'][key] = f'{type(error).__name__}: {error}'

    def grasp():
        state = controller.debug_gripper_state()
        for key in ('holding', 'constraint_active', 'release_pending'):
            if type(state.get(key)) is not bool:
                raise ValueError(f'Native {key} telemetry unavailable.')
        if 'object_name' not in state:
            raise ValueError('Native bound-object identity telemetry unavailable.')
        return state

    def relation():
        from omnigibson.object_states import Inside, OnTop, Touching
        if config['parameters']['placement_relation'] == 'inside':
            # Native Inside may use spatial sampling or bounding-box approximations.
            # Its conjunction with contact is measured here, while independent
            # interior geometry and fit qualification remains a release gate.
            return bool(target.states[Inside].get_value(destination)
                        and target.states[Touching].get_value(destination))
        return bool(target.states[OnTop].get_value(destination)
                    and target.states[Touching].get_value(destination))

    def withdrawn():
        from worldleague.tasks.interaction_setup import _body_under, _contact_parts
        return not any(_body_under(body, session.robot.prim_path)
                       for contact in target.contact_list() for body in _contact_parts(contact))

    read('gripper', grasp)
    read('destination_relation', relation)
    read('linear_speed_m_s', lambda: float(np.linalg.norm(_array(target.get_linear_velocity()))))
    read('angular_speed_rad_s', lambda: float(np.linalg.norm(_array(target.get_angular_velocity()))))
    read('robot_withdrawn', withdrawn)
    return values


class IdentificationPlacementEvaluator:
    """Keep semantic reporting independent of native selection and released placement."""

    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.target, self.destination = [session.env.scene.object_registry('name', config['bindings'][key])
                                        for key in ('target', 'destination')]
        if self.target is None or self.destination is None:
            raise ValueError('Original target and named destination must be bound natively.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.active = False

    def measure(self):
        return _placement_measurements(self.session, self.controller, self.config, self.target, self.destination)

    def begin_episode(self, observation):
        self.active = True
        self.start_step = observation.action_step
        self.answer = None
        self.answer_history = []
        self.selected = False
        self.selection_history_complete = True
        self.stable_steps = 0
        self.samples = 0
        self.last = self.measure()
        return self.record(observation, action=None)

    def report_answer(self, answer, observation):
        if not self.active:
            raise RuntimeError('Category reports require an active episode.')
        # The public transport sends the category text. A later explicit correction
        # replaces the current answer while history remains reviewable.
        self.answer = _normalize_answer(answer)
        self.answer_history.append({'answer': str(answer), 'normalized_answer': self.answer,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step})

    def _update(self, *, count_step=False):
        measured = self.measure()
        gripper = measured.get('gripper')
        if gripper is None:
            self.selection_history_complete = False
        elif count_step and (gripper['holding'] is True and gripper['constraint_active'] is True
                             and gripper['object_name'] == self.target.name):
            self.selected = True
        released = None if gripper is None else bool(gripper['holding'] is False
            and gripper['constraint_active'] is False and gripper['release_pending'] is False
            and gripper['object_name'] is None)
        physical_available = all(measured.get(key) is not None for key in (
            'destination_relation', 'linear_speed_m_s', 'angular_speed_rad_s', 'robot_withdrawn')) and released is not None
        stable = bool(physical_available and released and measured['robot_withdrawn']
            and measured['destination_relation']
            and measured['linear_speed_m_s'] <= self.parameters['maximum_release_speed_m_s']
            and measured['angular_speed_rad_s'] <= self.parameters['maximum_release_angular_speed_rad_s'])
        if not stable:
            self.stable_steps = 0
        elif count_step:
            self.stable_steps += 1
        if count_step:
            self.samples += 1
        self.last = measured
        return physical_available, stable

    def sample_step(self):
        if self.active:
            self._update(count_step=True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        available, stable = self._update()
        identity = self.answer is not None and self.answer == self.parameters['expected_identity']
        items = [
            {'id': 'identity', 'kind': 'semantic_answer', 'completed': bool(identity),
             'evaluation_validity': 'valid'},
            {'id': 'correct_bound_selection', 'kind': 'checkpoint', 'completed': self.selected,
             'evaluation_validity': 'valid' if self.selected or self.selection_history_complete else 'unavailable'},
            {'id': 'stable_placement', 'kind': 'terminal_condition',
             'completed': bool(stable and self.stable_steps >= self.parameters['release_stability_steps']),
             'evaluation_validity': 'valid' if available else 'unavailable'}]
        valid = all(item['evaluation_validity'] == 'valid' for item in items)
        result = {'schema': 'private_identification_placement_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'action': action,
            'task_success': bool(valid and all(item['completed'] for item in items)),
            'evaluation_validity': 'complete' if valid else 'partial', 'completion_items': items,
            'answer_history': self.answer_history.copy(), 'release_stable_steps': self.stable_steps,
            'physics_samples': self.samples, 'measurements': self.last, 'configuration': self.config}
        from worldleague.tasks.completion import progress
        result.update(progress(result))
        from worldleague.authoring.io import write_json
        write_json(self.output / 'state.json', result)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(result, allow_nan=False) + '\n')
        return result
