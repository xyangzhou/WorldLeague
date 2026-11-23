"""Convert Connectivity into native room exploration without changing its map.

Run CategoryConverter().convert_all(sources, ConversionContext(output_root))
for CPU conversion. The owned operation hooks run only during native episodes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldleague.tasks.background_policy import legal_background_contact

from worldleague.adapters.cognitive_mapping_visit_reports import NativeVisitReportEvaluator, native_array, settled_native_state


from worldleague.adapters.cognitive_mapping_visit_reports import REVISED_MAPPING_OPERATIONS, legacy_mapping_config, SourceAnswerMappingEvaluator

REVISED_OPERATION = REVISED_MAPPING_OPERATIONS['Connectivity']
EXPLORATION_OPERATION = 'region_exploration_arrivals'
LEGACY_OPERATIONS = ('region_route_arrival', 'region_round_trip')
OPERATIONS = (EXPLORATION_OPERATION, *LEGACY_OPERATIONS)


def _array(value):
    return native_array(value)





def measure_native_state(session, pose=None, *, parameters, include_safety=True):
    """Read actual robot/chassis, original room segmentation and native contacts."""
    robot, scene = session.robot, session.env.scene
    if pose is None:
        position, quaternion = robot.get_position_orientation()
    else:
        position, quaternion = pose['position'], pose['quaternion_xyzw']
    position, quaternion = _array(position), _array(quaternion)
    if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(
            quaternion).all() or not np.linalg.norm(quaternion):
        raise ValueError('Native robot pose is invalid.')
    segmentation = scene.seg_map
    room_names = segmentation.room_ins_name_to_ins_id
    heights = getattr(segmentation, 'floor_heights', None)
    floor_idx = parameters['floor_idx']
    layer_known = (isinstance(heights, (list, tuple, np.ndarray)) and len(heights) > floor_idx
        and np.isfinite(heights[floor_idx]))
    floor_z = float(heights[floor_idx]) if layer_known else None
    root_region = segmentation.get_room_instance_by_point(position[:2])
    low, high = (_array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high[:2] <= low[:2]):
        raise ValueError('Native physical chassis footprint is unavailable.')
    footprint_regions = [segmentation.get_room_instance_by_point(np.array([x, y]))
        for x in (low[0], (low[0] + high[0]) / 2, high[0])
        for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    floor_paths = [obj.prim_path for obj in objects if getattr(obj, 'category', None) in ('floor', 'floors')]
    contacts, nonfloor, floor_contacts = [], [], []
    for contact in robot.contact_list() if include_safety else ():
        bodies = [str(contact.body0), str(contact.body1)]
        external = [body for body in bodies if not (body == robot.prim_path or body.startswith(robot.prim_path + '/'))]
        if not external:
            continue
        contact_position = _array(contact.position) if hasattr(contact, 'position') else np.empty(0)
        contact_normal = _array(contact.normal) if hasattr(contact, 'normal') else np.empty(0)
        row = {'body0': bodies[0], 'body1': bodies[1],
            'position': contact_position.tolist() if contact_position.shape == (3,) and np.isfinite(contact_position).all() else None,
            'normal': contact_normal.tolist() if contact_normal.shape == (3,) and np.isfinite(contact_normal).all() else None}
        contacts.append(row)
        is_floor = [bool('groundPlane' in body or any(body == root or body.startswith(root + '/')
            for root in floor_paths)) for body in external]
        if any(is_floor):
            floor_contacts.append(row)
        if not all(is_floor) and not legal_background_contact(session, *bodies):
            nonfloor.append(row)
    region = root_region if root_region is not None and all(room == root_region for room in footprint_regions) else None
    floor_contact = bool(layer_known and floor_contacts and all(row['position'] is not None
        and row['normal'] is not None and abs(row['position'][2] - floor_z) <= parameters[
            'floor_contact_height_tolerance_m'] and abs(row['normal'][2]) >= parameters[
            'minimum_floor_contact_vertical_normal'] for row in floor_contacts))
    measured = {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'root_region': root_region, 'chassis_region': region, 'footprint_regions': footprint_regions,
        'known_regions': sorted(room_names), 'floor_contact': floor_contact,
        'floor_layer_known': bool(layer_known), 'segmentation_floor_idx': floor_idx,
        'segmentation_floor_z_m': floor_z, 'floor_contact_present': bool(floor_contacts),
        'native_floor_contacts': floor_contacts,
        'nonfloor_contacts': nonfloor, 'contacts': contacts,
        # Unlabelled doorway cells do not invalidate measured floor/contact safety.
        # Only a known region containing the sampled chassis footprint earns a visit.
        'state_valid': bool(floor_contact and not nonfloor),
        'visit_eligible': bool(region in room_names)}
    if not include_safety:
        for key in ('floor_contact', 'floor_contact_present', 'native_floor_contacts', 'nonfloor_contacts', 'contacts'):
            measured.pop(key)
        measured['state_valid'] = bool(layer_known and region in room_names)
    return measured


class LegacyConnectivityEvaluator:
    """Track physical ordered visits; source graphs and answers cannot score success."""
    def __init__(self, session, controller, config, output):
        parameters = config['parameters']
        self.session, self.controller, self.config = session, controller, config
        self.parameters = parameters
        self.required = [parameters['source_region'], parameters['target_region']]
        if config['operation'] == 'region_round_trip':
            self.required.append(parameters['source_region'])
        self.stability_steps = int(parameters['stability_steps'])
        # Preserve the actual controller's movement bound; no new speed limit is imposed.
        self.maximum_sample_displacement = (controller.config.max_translation_speed_m_s * controller.dt
            + controller.config.translation_tolerance_m)
        if not math.isfinite(self.maximum_sample_displacement) or self.maximum_sample_displacement <= 0:
            raise ValueError('Native controller sampling bounds are unavailable.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _observe(self, *, count_step):
        measured = measure_native_state(self.session, self.controller.pose(), parameters=self.parameters)
        point = np.asarray(measured['position'][:2])
        if self.previous is not None:
            displacement = float(np.linalg.norm(point - self.previous))
            self.travel_m += displacement
            if displacement > self.maximum_sample_displacement:
                self.continuous = False
        self.previous = point
        if not measured['state_valid']:
            self.contact_safe = False
        eligible = (self.stage < len(self.required) and measured['state_valid']
            and measured['visit_eligible'] and self.continuous and self.contact_safe
            and measured['chassis_region'] == self.required[self.stage])
        if not eligible:
            # Observation-only boundaries also interrupt consecutive visit credit.
            self.consecutive = 0
        if count_step:
            self.sample_count += 1
            if eligible:
                self.consecutive += 1
                if self.consecutive >= self.stability_steps:
                    self.visits.append({'region': self.required[self.stage], 'sample': self.sample_count,
                        'position': measured['position']})
                    self.stage += 1
                    self.consecutive = 0
        self.last = measured
        return measured

    def begin_episode(self, observation):
        self.episode += 1
        self.active = True
        self.stage, self.consecutive, self.sample_count = 0, 0, 0
        self.visits, self.previous = [], None
        self.continuous, self.contact_safe, self.travel_m = True, True, 0.0
        self._observe(count_step=False)
        return self.record(observation, action=None)

    def sample_step(self):
        if self.active:
            self._observe(count_step=True)

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = self._observe(count_step=False)
        checks = {'ordered_native_visits': self.stage == len(self.required),
            'inside_terminal_region': measured['chassis_region'] == self.required[-1],
            'native_floor_contact': measured['floor_contact'], 'contact_safe_trajectory': self.contact_safe,
            'native_floor_layer_known': measured['floor_layer_known'],
            'continuous_sampled_motion': self.continuous, 'sampled_real_displacement': self.travel_m > 0,
            'native_control_samples': self.sample_count >= self.stability_steps * len(self.required)}
        value = {'schema': 'private_connectivity_operation_v1', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'operation': self.config['operation'], 'task_success': bool(all(checks.values())),
            'checks': checks, 'native_state': measured, 'ordered_visits': self.visits,
            'required_regions': self.required, 'sample_count': self.sample_count,
            'sampled_travel_m': self.travel_m, 'maximum_sample_displacement_m': self.maximum_sample_displacement,
            'configuration': self.config}
        (self.output / 'state.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    if config.get('operation') == REVISED_OPERATION:
        native = create_evaluator(session, controller, legacy_mapping_config(config), output)
        return SourceAnswerMappingEvaluator(session, controller, config, output, native)
    evaluator = ConnectivityEvaluator if config['operation'] == EXPLORATION_OPERATION else LegacyConnectivityEvaluator
    return evaluator(session, controller, config, output)


class ConnectivityEvaluator(NativeVisitReportEvaluator):
    """Unordered native room set plus explicit same-boundary Agent reports."""

    def _measure_native(self):
        measured = measure_native_state(self.session, parameters=self.parameters, include_safety=False)
        return settled_native_state(self.session, measured)
