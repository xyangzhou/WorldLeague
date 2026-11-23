"""Convert Agent Observation into measured, embodied observation patrols.

Used by worldleague scene --help.
"""
from __future__ import annotations



import json
import math
from pathlib import Path

import numpy as np

from worldleague.tasks.background_policy import legal_background_contact
from worldleague.authoring.io import write_json
from worldleague.tasks.interaction_setup import (
    array as native_array, reference_position, support_contact_paths)

MODULE = 'worldleague.adapters.temporal_understanding_agent_observation'
OPERATION = 'observe_objects_and_return'
RETRIEVAL_OPERATION = 'observe_retrieve_and_return'
DEPARTURE_M = 0.5
RETURN_M = 0.15
MINIMUM_TRAVEL_M = 0.8
STABILITY_STEPS = 15
POSE_TOLERANCE_M = 0.05
OBJECT_DRIFT_M = 0.03


def _completion_result(value, items, *, available=True):
    """Expose required physical obligations without averaging safety predicates."""
    completion_items = [{'id': key, 'completed': bool(done),
        'evaluation_validity': 'valid' if available else 'unavailable',
        'kind': kind, 'evidence': evidence} for key, done, kind, evidence in items]
    completed = sum(item['completed'] for item in completion_items)
    total = len(completion_items)
    score = completed / total if total and available else None
    value.update(scoring_version='task_completion_v2',
        evaluation_validity='complete' if available else 'unavailable',
        completion_items=completion_items, completed_goals=completed, total_goals=total,
        completion_score=score, score_percent=100. * score if score is not None else None)
    return value


def _vector(value, size):
    return isinstance(value, (tuple, list)) and len(value) == size and all(
        isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def _array(value):
    return native_array(value)


def _yaw(quaternion):
    q = _array(quaternion)
    if q.shape != (4,) or not np.linalg.norm(q):
        raise ValueError('Missing native or source orientation.')
    x, y, z, w = q / np.linalg.norm(q)
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def initialize_scene(session, parameters):
    """Restore the saved actor start once, after endpoint-only source processing.

    The runtime may already bind the retained R1, but observations and guarded
    controls must not have started. No callbacks, actor replay or runtime pose
    setters are installed. The original external observer camera stays untouched.
    """
    robot = session.env.robots[0]
    payload = session.payload
    expected = [*payload['start_xy'], float(payload['robot_start_z'])]
    quaternion = payload['step_log'][0]['robot_quat_xyzw']
    import torch
    before = [_array(v).tolist() for v in robot.get_position_orientation()]
    robot.set_position_orientation(position=torch.tensor(expected, dtype=torch.float32),
                                   orientation=torch.tensor(quaternion, dtype=torch.float32))
    # Flush stale endpoint contacts and use the original builder's 30-step
    # initialization settling convention. This is not task execution/evidence.
    for _ in range(30):
        session.pipeline.og.sim.step()
    session._agent_observation_start_restored = True
    return {
            'before_position_quaternion': before,
            'after_position_quaternion': [_array(v).tolist() for v in robot.get_position_orientation()],
            'settle_steps': 30, 'original_observer_camera_preserved': True,
            'actor_replay_restored': False, 'physical_success_evidence': False}





def _within(body, root):
    return body == root or body.startswith(root + '/')


def _native_contacts(obj, floor_paths, session=None):
    floor, blocked = [], []
    for contact in obj.contact_list():
        if session is not None and legal_background_contact(session, str(contact.body0), str(contact.body1)):
            continue
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not _within(body, obj.prim_path)]
        if not external:
            continue
        row = {'body0': bodies[0], 'body1': bodies[1]}
        for key in ('position', 'normal'):
            value = _array(getattr(contact, key)) if hasattr(contact, key) else np.empty(0)
            row[key] = value.tolist() if value.shape == (3,) else None
        is_floor = [any(_within(body, path) for path in floor_paths) or
                    body.endswith('/groundPlane') or '/groundPlane/' in body for body in external]
        if all(is_floor):
            floor.append(row)
        else:
            blocked.append(row)
    supported = bool(floor) and all(row['position'] is not None and row['normal'] is not None and
                                  abs(row['normal'][2]) >= .5 for row in floor)
    return {'floor_supported': supported, 'floor_contacts': floor, 'nonfloor_contacts': blocked}


def _native_state(session, config):
    """Read measured bodies, original target geometry and floor contacts only."""
    p = config['parameters']
    robot, scene = session.robot, session.env.scene
    if type(robot).__name__ != 'R1' or robot.visual_only:
        raise ValueError('An actual collidable installed R1 is required.')
    # Installed HolonomicBaseRobot deliberately has fixed_base=True. Its
    # articulated base_link moves; rejecting fixed_base would reject every R1.
    position, quaternion = (_array(v) for v in robot.get_position_orientation())
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError('Native base pose unavailable.')
    _yaw(quaternion)
    q = quaternion / np.linalg.norm(quaternion)
    up_dot = float(1 - 2 * (q[0]**2 + q[1]**2))
    scene_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floor_paths = [obj.prim_path for obj in scene_objects if obj.category in ('floor', 'floors')]
    if not floor_paths:
        raise ValueError('Native floor identity unavailable.')
    contacts = _native_contacts(robot, floor_paths, session)
    measured_objects = {}
    for key, reference in p['source_objects'].items():
        obj = scene.object_registry('name', config['bindings'][key])
        if obj is None or obj.name != reference['name'] or obj.category != reference['category'] or obj.model != reference['model']:
            raise ValueError('Native original object binding mismatch: ' + key)
        point, _ = (_array(v) for v in obj.get_position_orientation())
        low, high = (_array(v) for v in obj.aabb)
        if any(v.shape != (3,) for v in (point, low, high)) or np.any(high <= low):
            raise ValueError('Native target geometry unavailable: ' + key)
        support = _native_contacts(obj, floor_paths)
        measured_objects[key] = {'name': obj.name, 'position': point.tolist(),
            'centre': ((low + high) / 2).tolist(), 'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
            'visual_only': bool(obj.visual_only), 'floor_supported': support['floor_supported'],
            'source_position_error_m': float(np.linalg.norm(point - reference['position'])),
            'linear_speed_m_s': float(np.linalg.norm(_array(obj.get_linear_velocity()))),
            'angular_speed_rad_s': float(np.linalg.norm(_array(obj.get_angular_velocity())))}
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(), 'up_dot': up_dot,
        'linear_speed_m_s': float(np.linalg.norm(_array(robot.get_linear_velocity()))),
        'angular_speed_rad_s': float(np.linalg.norm(_array(robot.get_angular_velocity()))),
        'objects': measured_objects, **contacts}





class ObservationPatrolEvaluator:
    """Measure actual visits and a subsequent safe return, without moving bodies."""
    def __init__(self, session, controller, config, output):
        self.parameters = config['parameters']
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial = None
        self.episode = 0

    def _time(self):
        return float(self.session.pipeline.og.sim.current_time)

    def _measure(self):
        return _native_state(self.session, self.config)

    def _record_details(self, value):
        return value

    def begin_episode(self, observation):
        measured = self._measure()
        self.episode += 1
        self.initial = measured
        self.start = np.array(measured['position'])
        self.previous = self.start.copy()
        self.previous_yaw = _yaw(measured['quaternion_xyzw'])
        self.previous_time = self._time()
        self.samples, self.travel_m, self.maximum_departure_m = 0, 0., 0.
        self.visits, self.faults = {}, set()
        self.visit_streaks = {key: 0 for key in measured['objects']}
        self.sample_motion_stopped = True
        self.stable_steps = 0
        return self.record(observation, action=None)

    def _safe(self, measured):
        preserved = all(math.dist(obj['position'], self.initial['objects'][key]['position']) <= OBJECT_DRIFT_M and
                        obj['floor_supported'] and not obj['visual_only'] and obj['linear_speed_m_s'] <= .02 and
                        obj['angular_speed_rad_s'] <= .1 for key, obj in measured['objects'].items())
        return bool(measured['floor_supported'] and not measured['nonfloor_contacts'] and
                    measured['up_dot'] >= .95 and preserved)

    def _terminal(self, measured):
        return {'all_original_objects_visited': len(self.visits) == len(self.initial['objects']),
            'actual_departure': self.maximum_departure_m >= DEPARTURE_M,
            'actual_travel': self.travel_m >= MINIMUM_TRAVEL_M,
            'returned_to_measured_start': math.dist(measured['position'][:2], self.start[:2]) <= RETURN_M,
            'measured_control_steps': self.samples > 0, 'continuous_safe_episode': not self.faults,
            'currently_safe': self._safe(measured),
            'stopped': self.sample_motion_stopped and measured['linear_speed_m_s'] <= .02 and measured['angular_speed_rad_s'] <= .1}

    def sample_step(self):
        if self.initial is None:
            return
        measured = self._measure()
        point, now = np.asarray(measured['position']), self._time()
        dt = now - self.previous_time
        displacement = float(np.linalg.norm(point - self.previous))
        yaw = _yaw(measured['quaternion_xyzw'])
        yaw_step = abs(math.atan2(math.sin(yaw - self.previous_yaw), math.cos(yaw - self.previous_yaw)))
        # Bound measured per-step motion by the existing controller contract.
        limit = self.controller.config.max_translation_speed_m_s * self.controller.dt + self.controller.config.translation_tolerance_m
        yaw_limit = math.radians(self.controller.config.max_rotation_speed_deg_s * self.controller.dt +
                                 self.controller.config.rotation_tolerance_deg)
        if dt <= 0 or dt > self.controller.dt * 1.5 or displacement > limit or yaw_step > yaw_limit:
            self.faults.add('unsampled_or_discontinuous_native_motion')
        self.sample_motion_stopped = bool(dt > 0 and displacement / dt <= .02 and yaw_step / dt <= .1)
        if not self._safe(measured):
            self.faults.add('native_contact_support_upright_or_object_preservation_failed')
        self.samples += 1
        self.travel_m += float(np.linalg.norm(point[:2] - self.previous[:2]))
        self.maximum_departure_m = max(self.maximum_departure_m, float(np.linalg.norm(point[:2] - self.start[:2])))
        if not self.faults and self.travel_m >= .1:
            for key, obj in measured['objects'].items():
                distance = math.dist(point[:2], obj['centre'][:2])
                self.visit_streaks[key] = self.visit_streaks[key] + 1 if distance <= self.parameters['visit_radius_m'] else 0
                if self.visit_streaks[key] >= 3 and key not in self.visits:
                    self.visits[key] = {'sample': self.samples, 'simulation_time_s': now,
                        'native_distance_m': distance, 'base_position': point.tolist(), 'object_centre': obj['centre']}
        criteria = self._terminal(measured)
        self.stable_steps = self.stable_steps + 1 if all(criteria.values()) else 0
        self.previous, self.previous_time = point.copy(), now
        self.previous_yaw = yaw
        with (self.output / 'physical_trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps({'episode': self.episode, 'sample': self.samples, 'simulation_time_s': now,
                'native_state': measured, 'faults': sorted(self.faults)}, allow_nan=False) + '\n')

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        measured = self._measure()
        yaw_error = _yaw(measured['quaternion_xyzw']) - self.previous_yaw
        if math.dist(measured['position'], self.previous) > 1e-6 or abs(
                math.atan2(math.sin(yaw_error), math.cos(yaw_error))) > 1e-6:
            self.faults.add('native_motion_outside_control_step_sampling')
        if not self._safe(measured):
            self.faults.add('native_contact_support_upright_or_object_preservation_failed')
        criteria = self._terminal(measured)
        if not all(criteria.values()):
            self.stable_steps = 0
        value = {'schema': 'private_agent_observation_patrol_v1', 'episode': self.episode,
            'task_success': bool(all(criteria.values()) and self.stable_steps >= STABILITY_STEPS),
            'criteria': criteria, 'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'native_state': measured, 'visits': self.visits,
            'physical_travel_m': self.travel_m, 'maximum_departure_m': self.maximum_departure_m,
            'control_samples': self.samples, 'stable_steps': self.stable_steps,
            'required_stability_steps': STABILITY_STEPS, 'faults': sorted(self.faults),
            'configuration': self.config}
        items = [(f'{key}:geometric_visit', key in self.visits, 'historical_checkpoint', self.visits.get(key))
                 for key in sorted(self.initial['objects'])]
        if self.config.get('operation') == RETRIEVAL_OPERATION:
            items.append(('retrieve_at_least_one', bool(self.retrieved), 'historical_checkpoint', self.retrieved))
        items.append(('return_and_park', value['task_success'], 'terminal_condition',
            {'stable_steps': self.stable_steps, 'physical_travel_m': self.travel_m}))
        _completion_result(value, items)
        value['scored_capabilities'] = ['geometric_object_visits', 'safe_return'] + (
            ['native_grasp_and_lift'] if self.config.get('operation') == RETRIEVAL_OPERATION else [])
        value['visual_discovery_verified'] = False
        value = self._record_details(value)
        write_json(self.output / 'state.json', value)
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def _retrieval_state(session, config):
    """Read actual original identities, grasp constraints and accepted supports."""
    from worldleague.adapters.specular_reflection_reflection_authoring import _contacts, _native_grasp, _under
    measured = _native_state(session, config)
    robot, scene = session.robot, session.env.scene
    arm = config['parameters']['working_arm']
    objects = {key: scene.object_registry('name', name) for key, name in config['bindings'].items()}
    scene_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floors = [obj.prim_path for obj in scene_objects if obj.category in ('floor', 'floors')]
    gripper_paths = [link.prim_path for link in list(robot.finger_links[arm]) + [robot.eef_links[arm]]]
    prop_paths = [obj.prim_path for obj in objects.values()]
    floor_contacts, forbidden, _ = _contacts(robot, floors,
        allowed_contact=lambda owned, external: all(any(_under(body, path) for path in gripper_paths) for body in owned)
            and all(any(_under(body, path) for path in prop_paths) for body in external)
            or all(legal_background_contact(session, a, b) for a in owned for b in external))
    measured.update(native_grasp=_native_grasp(robot, arm), floor_contacts=floor_contacts,
        floor_supported=bool(floor_contacts and all(abs(row['normal'][2]) >= .5 for row in floor_contacts)),
        nonfloor_contacts=forbidden)
    for key, obj in objects.items():
        support_paths = list(support_contact_paths(session, obj.name)) or floors
        contacts, blocked, gripper = _contacts(obj, support_paths,
            allowed_contact=lambda owned, external: all(any(_under(body, path) for path in gripper_paths) for body in external))
        reference = reference_position(session, obj.name, config['parameters']['source_objects'][key]['position'], root=True)
        measured['objects'][key].update(fixed_base=bool(obj.fixed_base),
            scale=_array(obj.scale).tolist(), support_supported=bool(contacts and all(abs(row['normal'][2]) >= .5 for row in contacts)),
            support_contacts=contacts, forbidden_contacts=blocked, gripper_contacts=gripper,
            accepted_reference_position=reference.tolist(),
            accepted_reference_error_m=float(np.linalg.norm(_array(measured['objects'][key]['position']) - reference)))
    return measured


class ObservationRetrievalEvaluator(ObservationPatrolEvaluator):
    """Keep real patrol evidence and require a physically grasped original prop."""
    def _measure(self):
        return _retrieval_state(self.session, self.config)

    def begin_episode(self, observation):
        measured = self._measure()
        self.grasped, self.retrieved, self.lift_streaks = set(), {}, {key: 0 for key in measured['objects']}
        self.object_previous = {key: _array(obj['position']) for key, obj in measured['objects'].items()}
        self.object_previous_speed = {key: obj['linear_speed_m_s'] for key, obj in measured['objects'].items()}
        self.object_time = self._time()
        return super().begin_episode(observation)

    def _held_key(self, measured):
        grasp = measured['native_grasp']
        if grasp['holding'] is not True or not grasp['constraint_active']:
            return None
        return next((key for key, obj in measured['objects'].items() if obj['name'] == grasp['object_name']), None)

    def _safe(self, measured):
        held = self._held_key(measured)
        preserved = True
        for key, obj in measured['objects'].items():
            original = self.initial['objects'][key]
            moved_legitimately = key == held or key in self.grasped
            preserved &= bool(not obj['visual_only'] and not obj['fixed_base'] and not obj['forbidden_contacts']
                and obj['scale'] == original['scale']
                and (moved_legitimately or math.dist(obj['position'], original['position']) <= OBJECT_DRIFT_M)
                and (key == held or (obj['support_supported'] and obj['linear_speed_m_s'] <= .02 and obj['angular_speed_rad_s'] <= .1)))
        return bool(measured['floor_supported'] and not measured['nonfloor_contacts'] and measured['up_dot'] >= .95 and preserved)

    def _terminal(self, measured):
        criteria = super()._terminal(measured)
        criteria['encountered_original_prop_grasped_and_lifted'] = bool(self.retrieved)
        held = self._held_key(measured)
        criteria['held_prop_stable_if_any'] = held is None or (
            measured['objects'][held]['linear_speed_m_s'] <= .02 and measured['objects'][held]['angular_speed_rad_s'] <= .1)
        return criteria

    def sample_step(self):
        if self.initial is None:
            return
        measured, now = self._measure(), self._time()
        elapsed, held = now - self.object_time, self._held_key(measured)
        for key, obj in measured['objects'].items():
            distance = float(np.linalg.norm(_array(obj['position']) - self.object_previous[key]))
            limit = max(obj['linear_speed_m_s'], self.object_previous_speed[key]) * max(elapsed, 0.) + OBJECT_DRIFT_M
            if elapsed <= 0 or distance > limit:
                self.faults.add('unsampled_or_discontinuous_original_prop_motion')
            # A native held identity is accepted only at this actual control
            # sample. It cannot retroactively authorize earlier moved props.
            if key == held and not self.faults:
                self.grasped.add(key)
            initial_low = self.initial['objects'][key]['aabb_min'][2]
            lifted = bool(key == held and not obj['support_contacts'] and
                obj['aabb_min'][2] - initial_low >= .05 - 1e-9 and
                obj['linear_speed_m_s'] <= .02 and obj['angular_speed_rad_s'] <= .1)
            self.lift_streaks[key] = self.lift_streaks[key] + 1 if lifted and not self.faults else 0
        super().sample_step()
        for key, count in self.lift_streaks.items():
            if count >= 3 and key in self.visits and not self.faults:
                self.retrieved.setdefault(key, {'sample': self.samples, 'simulation_time_s': now,
                    'native_name': measured['objects'][key]['name'],
                    'lowest_point_lift_m': measured['objects'][key]['aabb_min'][2] - self.initial['objects'][key]['aabb_min'][2],
                    'native_grasp': measured['native_grasp']})
        self.object_previous = {key: _array(obj['position']) for key, obj in measured['objects'].items()}
        self.object_previous_speed = {key: obj['linear_speed_m_s'] for key, obj in measured['objects'].items()}
        self.object_time = now

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        measured = self._measure()
        if any(np.linalg.norm(_array(obj['position']) - self.object_previous[key]) > 1e-6
                for key, obj in measured['objects'].items()):
            self.faults.add('original_prop_motion_outside_control_step_sampling')
        return super().record(observation, action=action)

    def _record_details(self, value):
        value.update(schema='private_agent_observation_retrieval_v1', retrieved=self.retrieved,
            physically_grasped_original_bindings=sorted(self.grasped), native_visit_count=len(self.visits),
            native_retrieval_count=len(self.retrieved), bag_or_containment_implemented=False)
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == 'history_multi_retrieval_v1':
        from .wia_history_collection import HistoryCollectionEvaluator
        return HistoryCollectionEvaluator(session, controller, config, output)
    if config.get('operation') == 'survey_history_retrieve_place':
        return HistoryRetrievalEvaluator(session, controller, config, output)
    if config.get('operation') == RETRIEVAL_OPERATION:
        return ObservationRetrievalEvaluator(session, controller, config, output)
    return ObservationPatrolEvaluator(session, controller, config, output)


def apparatus_objects(plan):
    """Translate declared table/tray proposals into existing primitive constructors."""
    objects = []
    for item in plan.get('primitives', []):
        objects.append({'type': 'PrimitiveObject', 'name': item['name'],
            'category': item.get('category', 'breakfast_table'), 'primitive_type': 'Cube',
            'size': 1., 'scale': item['dimensions_m'], 'position': item['position'],
            'orientation': [0., 0., 0., 1.], 'fixed_base': True, 'visual_only': False,
            'rgba': item.get('rgba', [.45, .35, .22, 1.])})
    objects.extend(plan.get('dataset_objects', []))
    return objects


def configure_scene(config, parameters):
    """Create only the explicit versioned apparatus, before native construction."""
    from worldleague.tasks.temporal_policy import TASK_EDIT2_TEMPORAL_REVISION
    if parameters.get('task_revision') != TASK_EDIT2_TEMPORAL_REVISION:
        raise ValueError('History apparatus requires the explicit TaskEdit2 revision.')
    plan = parameters['apparatus']
    if config['scene']['scene_model'] != parameters['scene_model']:
        raise ValueError('Temporal apparatus belongs to a different world.')
    names = {row['name'] for row in config.get('objects', [])}
    additions = apparatus_objects(plan)
    if names.intersection(row['name'] for row in additions):
        raise ValueError('Temporal apparatus cannot overwrite existing source assets.')
    config.setdefault('objects', []).extend(additions)
    for name, position in plan.get('source_target_positions', {}).items():
        obj = next((x for x in config['objects'] if x['name'] == name), None)
        if obj is None:
            raise ValueError('A history target is absent from the source constructor.')
        obj.update(position=position, fixed_base=False, visual_only=False)
    return {'task_revision': parameters['task_revision'], 'apparatus_proposal': plan,
        'native_support_and_access_verified': False, 'source_assets_preserved': True}


def native_temporal_placement(session, executor, config, target_key, destination_key):
    """Read native grasp, contact, destination and current robot support directly."""
    from .partial_occlusion import _placement_measurements
    scene = session.env.scene
    target = scene.object_registry('name', config['bindings'][target_key])
    destination = scene.object_registry('name', config['bindings'][destination_key])
    if target is None or destination is None:
        raise ValueError('Native target or destination binding unavailable.')
    measured = _placement_measurements(session, executor, config, target, destination)
    measured['target_dynamic_collidable'] = not target.fixed_base and not target.visual_only
    return measured


class NativePlacementProgress:
    """Count real control samples; repeated score reads cannot create a dwell."""
    def __init__(self, parameters):
        self.parameters = parameters
        self.selected = False
        self.selection_available = True
        self.stable_steps = 0
        self.last = None

    def update(self, measured, native_target_name, *, count_step=False):
        self.last = measured
        grasp = measured.get('gripper')
        if grasp is None:
            self.selection_available = False
        elif count_step and grasp.get('holding') is True and grasp.get('constraint_active') is True:
            if grasp.get('object_name') == native_target_name:
                self.selected = True
        released = isinstance(grasp, dict) and all(grasp.get(key) is False for key in
            ('holding', 'constraint_active', 'release_pending')) and grasp.get('object_name') is None
        available = all(measured.get(key) is not None for key in ('destination_relation', 'linear_speed_m_s',
            'angular_speed_rad_s', 'robot_withdrawn', 'target_dynamic_collidable')) and grasp is not None
        stable = bool(available and released and measured['destination_relation'] and measured['robot_withdrawn']
            and measured['target_dynamic_collidable']
            and measured['linear_speed_m_s'] <= self.parameters['maximum_release_speed_m_s']
            and measured['angular_speed_rad_s'] <= self.parameters['maximum_release_angular_speed_rad_s'])
        if not stable:
            self.stable_steps = 0
        elif count_step:
            self.stable_steps += 1
        return available, bool(stable and self.stable_steps >= self.parameters['release_stability_steps'])


def native_protected_inventory(session, bindings, keys):
    """Capture exact native poses/scales for task items and their apparatus."""
    result = {}
    for key in keys:
        obj = session.env.scene.object_registry('name', bindings[key])
        if obj is None:
            raise ValueError('Protected temporal object unavailable: ' + key)
        result[key] = {'name': obj.name, 'category': obj.category, 'model': getattr(obj, 'model', None),
            'position': _array(obj.get_position_orientation()[0]).tolist(),
            'orientation': _array(obj.get_position_orientation()[1]).tolist(), 'scale': _array(obj.scale).tolist()}
    return result


def protected_inventory_preserved(before, after, *, except_keys=()):
    for key, row in before.items():
        if key in except_keys:
            continue
        current = after.get(key)
        if current is None or any(current[field] != row[field] for field in ('name', 'category', 'model', 'scale')):
            return False
        a, b = np.array(row['orientation']), np.array(current['orientation'])
        if (np.linalg.norm(np.array(row['position']) - current['position']) > .03
                or min(np.linalg.norm(a - b), np.linalg.norm(a + b)) > .05):
            return False
    return True


class HistoryRetrievalEvaluator:
    """Freeze actual unique encounters before selecting the seeded rank question."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters, self.output = config['parameters'], Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history, self.seen, self.ties = [], set(), []
        self.frozen, self.selected_key, self.question_rank = None, None, None
        self.answer, self.active, self.visual_available = None, False, False
        self.placement = NativePlacementProgress(self.parameters)
        self.constraint_violation = False
        self.protection_available = True
        self.last_boundary = None

    def _protected(self):
        return native_protected_inventory(self.session, self.config['bindings'], self.config['binding_keys'])

    def _placement(self):
        return native_temporal_placement(self.session, self.controller, self.config,
            self.selected_key, self.parameters['destination_key'])

    def begin_episode(self, observation):
        self.initial = self._protected()
        self.active = True
        return self.record(observation)

    def _preserved(self):
        try:
            current = self._protected()
            allowed = [self.selected_key] if self.frozen is not None else []
            if not protected_inventory_preserved(self.initial, current, except_keys=allowed):
                self.constraint_violation = True
        except (ValueError, AttributeError, RuntimeError, KeyError, TypeError):
            self.protection_available = False

    def _observe(self, observation):
        from worldleague.tasks.temporal_policy import informative_temporal_rows
        evidence = informative_temporal_rows(self.session, observation, self.config['bindings'], self.parameters)
        self.visual_available = evidence['available']
        if self.frozen is None and evidence['available']:
            fresh = sorted(set(evidence['objects']).intersection(self.parameters['target_keys']) - self.seen)
            if len(fresh) > 1:
                # All cameras belong to one snapshot; camera iteration order cannot define chronology.
                self.ties.append({'snapshot_id': observation.snapshot_id, 'targets': fresh})
            for key in fresh:
                self.history.append({'key': key, 'snapshot_id': observation.snapshot_id,
                    'action_step': observation.action_step, 'evidence': evidence['objects'][key]})
                self.seen.add(key)
        self.last_boundary = (observation.snapshot_id, observation.action_step)
        return evidence

    def complete_survey(self, observation):
        import random
        if self.frozen is not None:
            raise ValueError('The encounter history has already been frozen.')
        self._observe(observation)
        self._preserved()
        if not self.visual_available or self.seen != set(self.parameters['target_keys']):
            raise ValueError('The survey lacks informative encounters for every target.')
        if self.ties:
            raise ValueError('Simultaneous first encounters make the rank question ambiguous.')
        if self.constraint_violation or not self.protection_available:
            raise ValueError('Survey preservation is not verified.')
        self.frozen = tuple(row['key'] for row in self.history)
        self.question_rank = random.Random(self.parameters['followup_seed']).randrange(1, len(self.frozen) + 1)
        self.selected_key = self.frozen[self.question_rank - 1]
        question = (f'Survey complete. Which target was number {self.question_rank} in your first-observed order? '
            'Submit its public target label in the answer field, retrieve that same target, and place it on '
            + self.parameters['destination_label'] + '.')
        if self.parameters.get('followup_rule') in ('before_reference_encounter', 'after_reference_encounter'):
            before = self.parameters['followup_rule'] == 'before_reference_encounter'
            reference_index = random.Random(self.parameters['followup_seed']).randrange(
                1 if before else 0, len(self.frozen) if before else len(self.frozen) - 1)
            selected_index = reference_index - 1 if before else reference_index + 1
            self.question_rank = selected_index + 1
            self.selected_key = self.frozen[selected_index]
            reference = self.parameters['public_target_labels'][self.frozen[reference_index]]
            question = ('Survey complete. For the next delivery, retrieve the item you first inspected immediately '
                + ('before ' if before else 'after ') + reference + '. Use your actual survey history. '
                'Submit its item label in the answer field, then place that same item on '
                + self.parameters['destination_label'] + '.')
        events = getattr(self.session, 'public_task_events', None)
        if events is None:
            events = []
            self.session.public_task_events = events
        if not isinstance(events, list):
            raise ValueError('Public follow-up event transport must be a list.')
        events.append({'event_id': 'history_followup_' + str(len(events) + 1),
            'kind': 'history_followup', 'text': question,
            'action_step': observation.action_step, 'snapshot_id': observation.snapshot_id})
        self.followup_public_text = question
        return {'survey_complete': True, 'followup_question': question}

    def report_answer(self, answer, observation):
        if str(answer).strip() == 'SURVEY_COMPLETE':
            return self.complete_survey(observation)
        if self.frozen is None:
            raise ValueError('The rank question is unavailable before survey completion.')
        self.answer = str(answer).strip()

    def sample_step(self):
        if not self.active:
            return
        self._preserved()
        if self.frozen is None:
            try:
                grasp = self.controller.debug_gripper_state()
                if grasp.get('holding') is True or grasp.get('constraint_active') is True:
                    self.constraint_violation = True
                elif grasp.get('holding') is not False or grasp.get('constraint_active') is not False:
                    self.protection_available = False
            except (ValueError, AttributeError, RuntimeError, KeyError, TypeError):
                self.protection_available = False
        if self.frozen is not None:
            try:
                measured = self._placement()
            except (ValueError, AttributeError, RuntimeError, KeyError, TypeError) as error:
                measured = {'measurement_error': str(error)}
            self.placement.update(measured, self.config['bindings'][self.selected_key], count_step=True)

    def record(self, observation, *, action=None):
        if not self.active:
            return None
        visual = self._observe(observation)
        self._preserved()
        physical_available, placed = False, False
        if self.frozen is not None:
            try:
                measured = self._placement()
            except (ValueError, AttributeError, RuntimeError, KeyError, TypeError) as error:
                measured = {'measurement_error': str(error)}
            physical_available, placed = self.placement.update(measured, self.config['bindings'][self.selected_key])
        correct = self.frozen is not None and self.answer == self.parameters['public_target_labels'][self.selected_key]
        items = [{'id': 'survey', 'completed': self.seen == set(self.parameters['target_keys']) and not self.ties,
                  'kind': 'checkpoint', 'evaluation_validity': 'valid' if self.visual_available else 'unavailable'},
            {'id': 'history_answer', 'completed': bool(correct), 'kind': 'semantic_answer', 'evaluation_validity': 'valid'},
            {'id': 'correct_history_bound_selection', 'completed': self.placement.selected,
             'kind': 'checkpoint', 'evaluation_validity': 'valid' if self.placement.selection_available else 'unavailable'},
            {'id': 'stable_destination', 'completed': placed, 'kind': 'terminal_condition',
             'evaluation_validity': 'valid' if physical_available else 'unavailable'}]
        available = all(item['evaluation_validity'] == 'valid' for item in items) and self.protection_available
        result = {'schema': 'task_edit2_history_retrieval_v1', 'completion_items': items,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'evaluation_validity': 'complete' if available else 'partial',
            'task_success': bool(available and all(item['completed'] for item in items) and not self.constraint_violation),
            'constraints_satisfied': not self.constraint_violation if self.protection_available else None,
            'encounter_history': self.history, 'ambiguous_first_encounters': self.ties,
            'frozen_history': self.frozen, 'followup_rank': self.question_rank,
            'history_selected_binding': self.selected_key, 'reported_answer': self.answer,
            'visual_evidence': visual, 'release_stable_steps': self.placement.stable_steps,
            'native_placement': self.placement.last}
        from worldleague.tasks.completion import progress
        result.update(progress(result))
        write_json(self.output / 'state.json', result)
        return result
