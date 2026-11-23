"""Convert original three-object scenes into guarded, physically measured stacks.

Run CPU batching from the repository root with the behavior environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
The campaign owns native captures; this module never starts GPU work or actions.
"""
from __future__ import annotations



from worldleague.tasks.background_policy import legal_background_grasp

from itertools import permutations
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from worldleague.authoring.io import write_json
from worldleague.tasks.interaction_setup import touching_task_support, on_top_task_support, support_contact_paths


MODULE = 'worldleague.adapters.physical_dynamics_stacking_stability'
TASK_REVISION = 'task_edit_20261018_v1'
OPERATION = 'build_released_three_object_stack'
FINAL_STATE_REVISION = 'final_state_v1'
OBJECT_KEYS = ('object_0', 'object_1', 'object_2')
MINIMUM_LIFT_M = .05
STABILITY_SECONDS = 1.
MINIMUM_STABILITY_SAMPLES = 20
MAXIMUM_SAMPLE_GAP_S = .1
MAXIMUM_LINEAR_SPEED = .02
MAXIMUM_ANGULAR_SPEED = .1
MAXIMUM_DRIFT_M = .01
MAXIMUM_ROTATION_RAD = math.radians(3.)
MINIMUM_UP_DOT = .9


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite native or source measurement.')
    return value


def _native_config(session, config):
    parameters = config['parameters']
    if parameters.get('manipulation_revision') not in (None, FINAL_STATE_REVISION):
        raise ValueError('Unsupported tower manipulation revision.')
    return parameters


def _pose(obj):
    position, quaternion = [_array(value) for value in obj.get_position_orientation()]
    low, high = [_array(value) for value in obj.aabb]
    return {'name': obj.name, 'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(), 'scale': _array(obj.scale).tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'up_dot': float(Rotation.from_quat(quaternion).as_matrix()[2, 2]),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}


def _external_contacts(obj):
    return [[str(body) for body in (contact.body0, contact.body1) if not (
        str(body) == obj.prim_path or str(body).startswith(obj.prim_path + '/'))]
        for contact in obj.contact_list()]


def _belongs(body, obj):
    return body == obj.prim_path or body.startswith(obj.prim_path + '/')


def _gripper_state(session, controller):
    state = dict(controller.debug_gripper_state())
    robot = session.robot
    state['all_arm_held_names'] = {arm: obj.name if obj is not None else None
                                  for arm, obj in robot._ag_obj_in_hand.items()}
    state['all_arm_constraints'] = {arm: value is not None for arm, value in robot._ag_obj_constraints.items()}
    state['all_arm_release_pending'] = {arm: value is not None for arm, value in robot._ag_release_counter.items()}
    state['all_arms_free'] = bool(all(value is None for value in state['all_arm_held_names'].values()) and
                                 not any(state['all_arm_constraints'].values()) and
                                 not any(state['all_arm_release_pending'].values()))
    return state


def _native_time():
    import omnigibson as og
    now, dt = float(og.sim.current_time), float(og.sim.get_sim_step_dt())
    if not math.isfinite(now) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('Native clock and sampling dt are unavailable.')
    return now, dt


class StackOperationEvaluator:
    """Read real lift/release histories and independent consecutive stack evidence."""
    def __init__(self, session, controller, config, output):
        self.parameters = _native_config(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.bound = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.bound.values()):
            raise ValueError('Original stack binding is missing.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.initial, self.last = 0, None, None

    def measure(self):
        from omnigibson.object_states import OnTop, Touching
        rows = {key: _pose(self.bound[key]) for key in OBJECT_KEYS}
        on_top, touching = {}, {}
        for upper in OBJECT_KEYS:
            obj = self.bound[upper]
            rows[upper]['touching_original_floor'] = bool(obj.states[Touching].get_value(self.bound['floor']))
            rows[upper]['on_top_original_floor'] = bool(obj.states[OnTop].get_value(self.bound['floor']))
            rows[upper]['external_contacts'] = [bodies for bodies in _external_contacts(obj) if bodies]
            for lower in (*OBJECT_KEYS, 'task_support'):
                if upper == lower:
                    continue
                pair = (upper, lower)
                if lower == 'task_support':
                    on_top[pair] = on_top_task_support(self.session, obj, self.bound['floor'])
                    touching[pair] = touching_task_support(self.session, obj, self.bound['floor'])
                else:
                    on_top[pair] = bool(obj.states[OnTop].get_value(self.bound[lower]))
                    touching[pair] = bool(obj.states[Touching].get_value(self.bound[lower]))
        return rows, on_top, touching, _gripper_state(self.session, self.controller), _native_time()

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = self.measure()[0]

        self.previous_held, self.previous_on_top, self.previous_touching = None, {}, {}
        self.current_lifted, self.lifted, self.released = False, set(), {}
        self.release_events, self.wrong_object_grasped = [], False
        self.trace, self.trace_order, self.last_time = [], None, None
        self.record(observation, action=None)

    def _terminal_orders(self, rows, on_top, touching, gripper):
        if not gripper['all_arms_free'] or gripper.get('gripper_open') is not True:
            return []
        if any(row['visual_only'] or row['fixed_base'] or row['up_dot'] < MINIMUM_UP_DOT or
               row['linear_speed_m_s'] > MAXIMUM_LINEAR_SPEED or row['angular_speed_rad_s'] > MAXIMUM_ANGULAR_SPEED
               for row in rows.values()):
            return []
        orders = []
        for bottom, middle, top in permutations(OBJECT_KEYS):
            pairs = ((bottom, 'task_support'), (middle, bottom), (top, middle))
            if not all(on_top[pair] and touching[pair] for pair in pairs):
                continue
            if touching[(middle, 'task_support')] or touching[(top, 'task_support')]:
                continue
            if (getattr(self, 'parameters', {}).get('manipulation_revision') != FINAL_STATE_REVISION and
                    not all(key in self.lifted and self.released.get(key) == lower
                       for key, lower in ((middle, bottom), (top, middle)))):
                continue
            # No robot, furniture, wall, floor support of an upper object, or
            # unrelated body may prop up the completed tower.
            allowed = {bottom: ('task_support', middle), middle: (bottom, top), top: (middle,)}
            support_paths = support_contact_paths(self.session, self.bound[bottom].name) or (
                self.bound['floor'].prim_path,)
            if any(not any((any(body == path or body.startswith(path + '/') for path in support_paths)
                           if other == 'task_support' else _belongs(body, self.bound[other]))
                           for other in allowed[key])
                   for key in OBJECT_KEYS for bodies in rows[key]['external_contacts'] for body in bodies):
                continue
            orders.append((bottom, middle, top))
        return orders

    def _update_final(self, *, advance):
        """Inspect final native support; only controller callbacks add dwell samples."""
        try:
            rows, on_top, touching, gripper, (now, dt) = self.measure()
        except Exception as error:
            self.trace, self.trace_order = [], None
            if self.last is None:
                raise
            self.last = {**self.last, 'measurement_error': f'{type(error).__name__}: {error}',
                'measured_order': None, 'stability_duration_s': 0., 'stability_samples': 0,
                'criteria': {**self.last['criteria'], 'native_measurement_available': False,
                    'native_contact_order_upright_stationary_unheld': False,
                    'continuous_native_stability_window': False}, 'task_success': False}
            return self.last
        if self.last_time is not None and now < self.last_time - 1e-9:
            raise ValueError('Native simulation clock moved backwards within the episode.')
        orders = self._terminal_orders(rows, on_top, touching, gripper)
        order = orders[0] if len(orders) == 1 else None
        gap = now - self.last_time if self.last_time is not None else 0.
        restart = order != self.trace_order or gap > MAXIMUM_SAMPLE_GAP_S + 1e-7
        if self.trace and not restart:
            first = self.trace[0]['objects']
            restart = any(np.linalg.norm(np.asarray(row['position']) - first[key]['position']) > MAXIMUM_DRIFT_M or
                (Rotation.from_quat(row['quaternion_xyzw']).inv() * Rotation.from_quat(
                    first[key]['quaternion_xyzw'])).magnitude() > MAXIMUM_ROTATION_RAD
                for key, row in rows.items())
        if order is None or restart:
            self.trace, self.trace_order = [], None
        if advance and (self.last_time is None or now > self.last_time + 1e-9):
            if order is not None:
                self.trace_order = order
                if not self.trace or self.trace[-1]['time_s'] - self.trace[0]['time_s'] < STABILITY_SECONDS - 1e-7:
                    self.trace.append({'time_s': now, 'objects': rows})
            self.last_time = now
        duration = self.trace[-1]['time_s'] - self.trace[0]['time_s'] if self.trace else 0.
        criteria = {
            'native_measurement_available': True,
            'native_contact_order_upright_stationary_unheld': order is not None,
            'continuous_native_stability_window': duration >= STABILITY_SECONDS - 1e-7 and
                len(self.trace) >= MINIMUM_STABILITY_SAMPLES,
            'original_scales_preserved': all(np.max(np.abs(np.asarray(rows[key]['scale']) -
                self.parameters['original_objects'][key]['scale'])) <= 1e-5 for key in OBJECT_KEYS)}
        categories = [self.parameters['original_objects'][key]['category'] for key in order] if order else None
        self.last = {'native_objects': rows, 'native_gripper': gripper, 'time_s': now, 'sim_step_dt_s': dt,
            'native_relations': [{'upper': upper, 'lower': lower, 'on_top': value,
                'touching': touching[(upper, lower)]} for (upper, lower), value in on_top.items()],
            'measured_order': list(order) if order else None, 'measurement_error': None,
            'stability_duration_s': duration, 'stability_samples': len(self.trace), 'criteria': criteria,
            'auxiliary_original_order_match': categories in self.parameters['legacy_accepted_category_orders']
                if categories else None, 'task_success': bool(all(criteria.values()))}
        return self.last

    def _update(self, *, advance=True):
        if self.initial is None:
            return None
        if self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION:
            return self._update_final(advance=advance)
        rows, on_top, touching, gripper, (now, dt) = self.measure()
        # record() and the control callback can see the same native step.
        if self.last_time is not None and now <= self.last_time + 1e-9:
            if now < self.last_time - 1e-9:
                raise ValueError('Native simulation clock moved backwards within the episode.')
            return self.last
        held_name = gripper.get('object_name') if (gripper.get('holding') is True and
                    gripper.get('constraint_active') is True and gripper.get('release_pending') is False) else None
        held = next((key for key in OBJECT_KEYS if self.bound[key].name == held_name), None)
        if held_name is not None and held is None and not legal_background_grasp(self.session, held_name):
            self.wrong_object_grasped = True
        if held != self.previous_held:
            if self.previous_held is not None:
                released = self.previous_held
                supported_by = [lower for lower in OBJECT_KEYS if lower != released and
                                self.previous_on_top.get((released, lower)) and
                                self.previous_touching.get((released, lower))]
                if self.current_lifted and len(supported_by) == 1:
                    self.lifted.add(released)
                    self.released[released] = supported_by[0]
                    self.release_events.append({'time_s': now, 'upper': released, 'lower': supported_by[0],
                                                'native_held_lift_observed': True})
                else:
                    self.released.pop(released, None)
            self.current_lifted = False
            if held is not None:
                self.released.pop(held, None)
        if held is not None:
            self.current_lifted = bool(self.current_lifted or
                (rows[held]['aabb_min'][2] - self.initial[held]['aabb_min'][2] >= MINIMUM_LIFT_M and
                 not touching[(held, 'task_support')]))
        orders = self._terminal_orders(rows, on_top, touching, gripper)
        order = orders[0] if len(orders) == 1 else None
        gap = now - self.last_time if self.last_time is not None else 0.
        if order is None:
            self.trace, self.trace_order = [], None
        else:
            restart = order != self.trace_order or gap > MAXIMUM_SAMPLE_GAP_S + 1e-7
            if self.trace and not restart:
                first = self.trace[0]['objects']
                restart = any(np.linalg.norm(np.asarray(row['position']) - first[key]['position']) > MAXIMUM_DRIFT_M or
                    (Rotation.from_quat(row['quaternion_xyzw']).inv() * Rotation.from_quat(
                        first[key]['quaternion_xyzw'])).magnitude() > MAXIMUM_ROTATION_RAD
                    for key, row in rows.items())
            if restart:
                self.trace = []
            self.trace_order = order
            # Keep one full terminal window, continuing to check current state.
            if not self.trace or self.trace[-1]['time_s'] - self.trace[0]['time_s'] < STABILITY_SECONDS - 1e-7:
                self.trace.append({'time_s': now, 'objects': rows})
        duration = self.trace[-1]['time_s'] - self.trace[0]['time_s'] if self.trace else 0.
        criteria = {
                    'two_upper_real_lift_and_supported_release_events': order is not None,
                    'native_contact_order_upright_stationary_unheld': order is not None,
                    'continuous_native_stability_window': duration >= STABILITY_SECONDS - 1e-7 and
                        len(self.trace) >= MINIMUM_STABILITY_SAMPLES,
                    'no_wrong_object_grasp': not self.wrong_object_grasped,
                    'original_scales_preserved': all(np.max(np.abs(np.asarray(rows[key]['scale']) -
                        self.parameters['original_objects'][key]['scale'])) <= 1e-5 for key in OBJECT_KEYS)}
        categories = [self.parameters['original_objects'][key]['category'] for key in order] if order else None
        self.last = {'native_objects': rows, 'native_gripper': gripper, 'time_s': now, 'sim_step_dt_s': dt,
                     'native_relations': [{'upper': upper, 'lower': lower, 'on_top': value,
                         'touching': touching[(upper, lower)]} for (upper, lower), value in on_top.items()],
                     'measured_order': list(order) if order else None, 'release_events': list(self.release_events),
                     'stability_duration_s': duration, 'stability_samples': len(self.trace), 'criteria': criteria,
                     'auxiliary_original_order_match': categories in self.parameters['legacy_accepted_category_orders']
                         if categories else None, 'task_success': bool(all(criteria.values()))}
        self.previous_held, self.previous_on_top, self.previous_touching = held, on_top, touching
        self.last_time = now
        return self.last

    def sample_step(self):
        self._update()

    def record(self, observation, *, action):
        final_state = self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION
        value = self._update(advance=not final_state)
        if value is None:
            return None
        record = {**value, 'schema': 'private_real_stack_v1', 'episode': self.episode,
                  'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
                  'action': action, 'configuration': self.config, 'initial_native_objects': self.initial,
                  'stability_trace': self.trace}
        available = value.get('measurement_error') is None
        released = len({event['upper'] for event in self.release_events}) >= 2
        record.update(task_revision=TASK_REVISION, evaluation_validity='complete' if available else 'unavailable',
            completion_items=[{'id': 'two_supported_upper_releases', 'kind': 'checkpoint', 'completed': released,
                'evaluation_validity': 'valid' if available else 'unavailable'},
                {'id': 'withdrawn_stable_tower', 'kind': 'terminal_condition', 'completed': value['task_success'],
                 'evaluation_validity': 'valid' if available else 'unavailable'}])
        if final_state:
            available = available and value['criteria']['native_measurement_available']
            record.update(schema='private_final_stack_v1',
                task_revision=self.parameters.get('task_revision', FINAL_STATE_REVISION),
                manipulation_revision=FINAL_STATE_REVISION,
                measured_capability='released_final_tower',
                evaluation_validity='complete' if available else 'unavailable',
                completion_items=[{'id': 'withdrawn_stable_tower', 'kind': 'terminal_condition',
                    'completed': value['task_success'],
                    'evaluation_validity': 'valid' if available else 'unavailable'}])
        from worldleague.tasks.completion import progress
        record.update(progress(record))
        write_json(self.output / 'state.json', record)
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        return record


def create_evaluator(session, controller, config, output):
    return StackOperationEvaluator(session, controller, config, output)
