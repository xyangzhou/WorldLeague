"""Preserve Regional Boundary scenes and measure native room visits and reports.

Run: python src/worldleague/runtime/commands/run_esi_category_campaign.py --help
CategoryConverter.convert_all performs CPU generation. Native qualification and
episode hooks read the actual robot, anchors and private room map without writes.
"""
from __future__ import annotations

from worldleague.resources import ESI_ROOT

import ast
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np

from worldleague.tasks.background_policy import legal_background_contact

from worldleague.adapters.cognitive_mapping_connectivity import measure_native_state as measure_room_state
from worldleague.adapters.cognitive_mapping_visit_reports import NativeVisitReportEvaluator, native_array, settled_native_state

from worldleague.adapters.cognitive_mapping_visit_reports import REVISED_MAPPING_OPERATIONS, legacy_mapping_config, SourceAnswerMappingEvaluator

REVISED_OPERATION = REVISED_MAPPING_OPERATIONS['Regional Boundary']
EXPLORATION_OPERATION = 'anchor_region_exploration'
GENERATOR = ESI_ROOT / 'src/dataset_generation/task_cognitive_map/batch_cognitivemap_region.py'


def _array(value):
    return native_array(value)


@lru_cache(maxsize=1)
def _geometry():
    """Reuse exact pure generator formulas without its global simulator changes."""
    names = {'_normalize_bbox_xyxy', '_bboxes_intersect_xy', '_expand_bbox_until_wall_touch',
             '_expand_bbox_xyxy', '_room_bbox_center_xy', '_distance_xy', '_distance_point_to_bbox_xy'}
    parsed = ast.parse(GENERATOR.read_text())
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError('Original private room-distance formulas are unavailable.')
    scope = {'np': np, 'math': math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(GENERATOR), 'exec'), scope)
    return scope


def _query_region(scene, position):
    import torch
    seg = getattr(scene, 'seg_map', None)
    if seg is None or not callable(getattr(seg, 'get_room_instance_by_point', None)):
        raise ValueError('native_room_segmentation_unavailable')
    value = seg.get_room_instance_by_point(torch.tensor(position[:2], dtype=torch.float32))
    if value is None:
        return None
    return str(value)


def _anchor_state(obj):
    position, quaternion = obj.get_position_orientation()
    low, high = (_array(value).tolist() for value in obj.aabb)
    position, quaternion = _array(position).tolist(), _array(quaternion).tolist()
    if not all(math.isfinite(value) for row in (position, quaternion, low, high) for value in row):
        raise ValueError('nonfinite_native_anchor_geometry')
    return {'name': obj.name, 'position': position, 'quaternion_xyzw': quaternion,
            'aabb_min': low, 'aabb_max': high, 'centre_xy': [(a + b) / 2 for a, b in zip(low[:2], high[:2])]}


def _floor_contact_evidence(session):
    """Require measured external contacts on actual floor objects in map layer 0.

    Installed EntityPrim.contact_list aggregates RigidPrim.contact_list records,
    whose body paths are decoded native rigid bodies. No contact-reporting flag
    is enabled here, and an unavailable/empty report never establishes support.
    """
    scene, robot = session.env.scene, session.robot
    evidence = {'segmentation_floor_idx': 0, 'segmentation_floor_z_m': None,
                'native_floor_layer_known': False, 'native_floor_object_paths': [],
                'native_floor_contacts': [], 'nonfloor_contacts': [],
                'native_floor_contact': False, 'no_nonfloor_robot_contact': True,
                'native_contacts_available': False, 'contact_support_reason': None}
    try:
        heights = _array(scene.seg_map.floor_heights).reshape(-1)
        if not len(heights) or not math.isfinite(float(heights[0])):
            raise ValueError('native_segmentation_floor_0_height_unavailable')
        floor = float(heights[0])
        evidence.update(segmentation_floor_z_m=floor, native_floor_layer_known=True)
        objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
        floor_paths = [str(obj.prim_path) for obj in objects
                       if getattr(obj, 'category', None) in ('floor', 'floors')
                       and getattr(obj, 'prim_path', None)]
        evidence['native_floor_object_paths'] = floor_paths
        if not floor_paths:
            raise ValueError('native_scene_floor_objects_unavailable')
        robot_path = str(robot.prim_path)
        if not robot_path or not callable(getattr(robot, 'contact_list', None)):
            raise ValueError('native_robot_contact_measurement_unavailable')
        contacts = robot.contact_list()
        if contacts is None:
            raise ValueError('native_robot_contact_measurement_unavailable')
        evidence['native_contacts_available'] = True
        owns = lambda body, root: body == root or body.startswith(root + '/')
        for contact in contacts:
            bodies = [str(contact.body0), str(contact.body1)]
            robot_bodies = [owns(body, robot_path) for body in bodies]
            if all(robot_bodies):
                continue  # Self-contact is not an external support/collision.
            external = [body for body, owned in zip(bodies, robot_bodies) if not owned]
            position, normal = getattr(contact, 'position', None), getattr(contact, 'normal', None)
            position = _array(position) if position is not None else np.empty(0)
            normal = _array(normal) if normal is not None else np.empty(0)
            row = {'body0': bodies[0], 'body1': bodies[1],
                   'position': position.tolist() if position.shape == (3,) and np.isfinite(position).all() else None,
                   'normal': normal.tolist() if normal.shape == (3,) and np.isfinite(normal).all() else None}
            is_floor = bool(any(robot_bodies) and external and all(
                any(owns(body, root) for root in floor_paths) for body in external))
            if is_floor:
                evidence['native_floor_contacts'].append(row)
            elif not legal_background_contact(session, *bodies):
                evidence['nonfloor_contacts'].append(row)
        evidence['no_nonfloor_robot_contact'] = not evidence['nonfloor_contacts']
        floor_contacts = evidence['native_floor_contacts']
        evidence['native_floor_contact'] = bool(floor_contacts and all(
            row['position'] is not None and row['normal'] is not None
            and abs(row['position'][2] - floor) <= .005 and abs(row['normal'][2]) >= .5
            for row in floor_contacts))
        if not evidence['no_nonfloor_robot_contact']:
            evidence['contact_support_reason'] = 'native_robot_nonfloor_external_contact'
        elif not evidence['native_floor_contact']:
            evidence['contact_support_reason'] = 'native_robot_floor_0_contact_not_verified'
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, IndexError, AssertionError) as error:
        evidence['contact_support_reason'] = f'native_floor_contact_measurement_unavailable:{error}'
        evidence['native_floor_contact'] = False
    return evidence


def _robot_support_evidence(session, pose):
    evidence = _floor_contact_evidence(session)
    quaternion = _array(pose['quaternion_xyzw'])
    norm = float(np.linalg.norm(quaternion))
    evidence.update(robot_upright=False, robot_floor_aabb_screen=False,
                    native_robot_bottom_m=None, robot_supported_upright=False)
    if quaternion.shape != (4,) or norm == 0 or not np.isfinite(quaternion).all():
        return evidence
    x, y, z, w = quaternion / norm
    evidence['robot_upright'] = bool(1 - 2 * (x*x + y*y) >= .9)
    low, high = (_array(value) for value in session.robot.aabb)
    if low.shape == high.shape == (3,) and np.isfinite([low, high]).all():
        evidence['native_robot_bottom_m'] = float(low[2])
        if evidence['native_floor_layer_known']:
            evidence['robot_floor_aabb_screen'] = abs(float(low[2]) - evidence['segmentation_floor_z_m']) <= .15
    evidence['robot_supported_upright'] = bool(evidence['robot_upright']
        and evidence['robot_floor_aabb_screen'] and evidence['native_floor_contact']
        and evidence['no_nonfloor_robot_contact'])
    return evidence


def _native_region_distances(scene, anchor, parameters):
    """Reconstruct the original metric from actual room masks and wall geometry."""
    import torch
    seg, geometry = scene.seg_map, _geometry()
    mapping, room_map = seg.room_ins_id_to_ins_name, _array(seg.room_ins_map)
    raw_objects = scene.objects
    objects = raw_objects.values() if isinstance(raw_objects, dict) else raw_objects
    walls = []
    for obj in objects:
        name, category = str(obj.name).lower(), str(obj.category).lower()
        if 'wall' in name and ('walls' in name or category in ('walls', 'wall')):
            low, high = (_array(value) for value in obj.aabb)
            walls.append(tuple(geometry['_normalize_bbox_xyxy']([*low[:2], *high[:2]])))
    measured = {}
    for room_id, name in mapping.items():
        if name not in parameters['candidate_regions']:
            continue
        pixels = np.argwhere(room_map == int(room_id))
        if not len(pixels):
            continue
        low, high = pixels.min(axis=0), pixels.max(axis=0)
        corners = torch.tensor([[low[0], low[1]], [low[0], high[1]],
                                [high[0], low[1]], [high[0], high[1]]], dtype=torch.float32)
        world = _array(seg.map_to_world(corners))
        bbox = tuple([*world.min(axis=0), *world.max(axis=0)])
        if walls:
            expanded = geometry['_expand_bbox_until_wall_touch'](bbox, walls,
                parameters['expansion_ratio'], parameters['expansion_min_m'])
        else:
            expanded = geometry['_expand_bbox_xyxy'](bbox, parameters['expansion_ratio'], parameters['expansion_min_m'])
        measured[str(name)] = {
            'bbox_world_xy': list(bbox), 'expanded_bbox_world_xy': list(expanded),
            'center_xy': list(geometry['_room_bbox_center_xy'](bbox)),
            'bbox_distance': geometry['_distance_point_to_bbox_xy'](anchor['centre_xy'], expanded),
            'center_distance': geometry['_distance_xy'](anchor['centre_xy'], geometry['_room_bbox_center_xy'](bbox))}
    if set(measured) != set(parameters['candidate_regions']):
        raise ValueError('native_candidate_room_masks_unavailable')
    return measured


def _binding_objects(session, config):
    bound = {key: session.env.scene.object_registry('name', config['bindings'][key])
             for key in config['binding_keys']}
    if any(obj is None for obj in bound.values()):
        raise ValueError('native_anchor_binding_missing')
    return bound


def _relation_evidence(session, config, bound):
    scene, p = session.env.scene, config['parameters']
    states = [_anchor_state(bound[key]) for key in config['binding_keys']]
    regions = [_query_region(scene, state['centre_xy']) for state in states]
    result = {'native_anchor_states': states, 'native_anchor_seg_regions': regions}
    if config['operation'] == 'enter_closest_region' or (config['operation'] == EXPLORATION_OPERATION
            and p['source_task_type'] == 'object_closer_region'):
        measured = _native_region_distances(scene, states[0], p)
        ranked = sorted(measured, key=lambda name: (measured[name]['bbox_distance'], measured[name]['center_distance'], name))
        result.update(native_region_distances=measured, native_selected_region=ranked[0])
        tolerance = p['distance_reproduction_tolerance_m']
        reproduced = all(name in measured and all(abs(float(record[key]) - measured[name][key]) <= tolerance
            for key in ('bbox_distance', 'center_distance')) for name, record in p['recorded_distances'].items())
        result.update(recorded_metric_reproduced=reproduced,
                      region_relation_consistent=reproduced and ranked[0] == p['target_regions'][0])
    elif p.get('actual_capability') == 'observed_anchor_region_relation':
        # Use the source generator's assignment, retaining segmentation for comparison.
        settings = {**p, 'candidate_regions': list(scene.seg_map.room_ins_id_to_ins_name.values()),
                    'expansion_ratio': p.get('expansion_ratio', .35),
                    'expansion_min_m': p.get('expansion_min_m', .75)}
        assignments = []
        for key, state, segmented in zip(config['binding_keys'], states, regions):
            geometry = _native_region_distances(scene, state, settings)
            containing = sorted((name for name, row in geometry.items() if row['bbox_distance'] == 0),
                                key=lambda name: (geometry[name]['center_distance'], name))
            in_rooms = list(getattr(bound[key], 'in_rooms', ()) or ())
            assigned = containing[0] if containing else segmented or next((r for r in in_rooms if r), None)
            assignments.append({'binding': key, 'assigned_region': assigned,
                'segmentation_region': segmented, 'in_rooms': in_rooms, 'containing_regions': containing,
                'method': 'expanded_bbox' if containing else 'seg_query' if segmented else 'in_rooms',
                'region_geometry': geometry})
        result.update(native_anchor_assignment=assignments,
            assignment_policy='source_expanded_bounds_then_segmentation_then_in_rooms',
            assignment_parameters={k: settings[k] for k in ('expansion_ratio', 'expansion_min_m')},
            region_relation_consistent=[r['assigned_region'] for r in assignments] == p['target_regions'])
    else:
        result['region_relation_consistent'] = regions == p['target_regions']
    return result


def _native_door_evidence(session):
    """Read installed door state; joint ranges never certify opening or passage."""
    evidence = {'schema': 'private_regional_door_inspection_v1',
                'inventory_available': False, 'doors': [],
                'door_opening_verified': False, 'doorway_clearance_verified': False,
                'route_feasibility_verified': False}
    try:
        objects = session.env.scene.objects
        objects = list(objects.values()) if isinstance(objects, dict) else list(objects)
        evidence['inventory_available'] = True
        doors = [obj for obj in objects if getattr(obj, 'category', None) in ('door', 'sliding_door')]
        for obj in sorted(doors, key=lambda item: str(item.name)):
            row = {'name': str(obj.name), 'category': str(obj.category),
                   'joint_state_available': False}
            evidence['doors'].append(row)
            try:
                row.update(prim_path=str(obj.prim_path), fixed_base=bool(obj.fixed_base),
                           visual_only=bool(obj.visual_only), native_joint_count=int(obj.n_joints))
                if not obj.n_joints:
                    row['reason'] = 'native_door_has_no_articulated_joints'
                    continue
                positions, lower, upper = (_array(value).reshape(-1) for value in (
                    obj.get_joint_positions(), obj.joint_lower_limits, obj.joint_upper_limits))
                if not (positions.shape == lower.shape == upper.shape and positions.size
                        and np.isfinite([positions, lower, upper]).all() and np.all(lower <= upper)):
                    raise ValueError('invalid_native_door_joint_arrays')
                row.update(joint_state_available=True, joint_positions=positions.tolist(),
                           joint_lower_limits=lower.tolist(), joint_upper_limits=upper.tolist(),
                           joint_motion_range_observed=bool(np.any(upper - lower > 1e-6)))
            except (ValueError, RuntimeError, AttributeError, KeyError, TypeError, AssertionError) as error:
                row['reason'] = 'native_door_joint_measurement_unavailable:' + str(error)
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        evidence['inventory_available'] = False
        evidence['reason'] = 'native_door_inventory_unavailable:' + str(error)
    return evidence


class LegacyRegionalOperationEvaluator:
    """Measure real native base motion; source answers never satisfy a predicate."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.bound = _binding_objects(session, config)
        self.episode, self.initial = 0, None

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = self.controller.pose()
        self.initial_region = _query_region(self.session.env.scene, self.initial['position'])
        self.initial_anchors = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        relation = _relation_evidence(self.session, self.config, self.bound)
        self.relation_valid = relation['region_relation_consistent']
        self.relation_evidence = relation
        self.last = self.initial['position']
        self.path_m, self.excursion_m, self.stage, self.quiet_samples = 0., 0., 0, 0
        self.survey_position = None
        self.outside_selected = self.initial_region != self.config['parameters']['target_regions'][0]
        self.discontinuous, self.anchors_preserved = False, True
        self.contact_safe_trajectory = True
        self.record(observation, action=None)

    def sample_step(self):
        if self.initial is None:
            return
        p, scene = self.config['parameters'], self.session.env.scene
        pose = self.controller.pose()
        position = pose['position']
        if not np.isfinite(position).all():
            raise ValueError('Nonfinite native robot trajectory.')
        delta = math.dist(position, self.last)
        self.discontinuous |= delta > p['maximum_step_displacement_m']
        self.path_m += math.dist(position[:2], self.last[:2])
        self.excursion_m = max(self.excursion_m, math.dist(position[:2], self.initial['position'][:2]))
        self.last = list(position)
        region = _query_region(scene, position)
        support = _robot_support_evidence(self.session, pose)
        robot_supported = support['robot_supported_upright']
        self.contact_safe_trajectory &= robot_supported
        anchors = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        for initial, current in zip(self.initial_anchors, anchors):
            quaternion_change = 1 - abs(float(np.dot(initial['quaternion_xyzw'], current['quaternion_xyzw'])) /
                max(float(np.linalg.norm(initial['quaternion_xyzw']) * np.linalg.norm(current['quaternion_xyzw'])), 1e-12))
            preserved = (math.dist(initial['position'], current['position']) <= p['anchor_preservation_tolerance_m']
                         and math.dist(initial['aabb_min'], current['aabb_min']) <= p['anchor_preservation_tolerance_m']
                         and math.dist(initial['aabb_max'], current['aabb_max']) <= p['anchor_preservation_tolerance_m']
                         and quaternion_change <= 1e-4)
            self.anchors_preserved &= preserved
        if self.config['operation'] == 'ordered_anchor_survey_return':
            if self.stage < 2:
                index = self.stage
                anchor = self.initial_anchors[index]
                distance = math.sqrt(sum(max(anchor['aabb_min'][axis] - position[axis],
                    position[axis] - anchor['aabb_max'][axis], 0.) ** 2 for axis in (0, 1)))
                separated = index == 0 or math.dist(position[:2], self.survey_position) >= p['survey_separation_m']
                if (robot_supported and region == p['target_regions'][index]
                        and distance <= p['survey_distance_m'] and separated):
                    self.survey_position = list(position[:2])
                    self.stage += 1
            endpoint = robot_supported and self.stage == 2 and region == self.initial_region and math.dist(
                position[:2], self.initial['position'][:2]) <= p['return_distance_m']
        else:
            self.outside_selected |= robot_supported and region is not None and region != p['target_regions'][0]
            endpoint = robot_supported and self.outside_selected and region == p['target_regions'][0]
        speed = float(np.linalg.norm(_array(self.session.robot.get_linear_velocity())))
        angular_speed = float(np.linalg.norm(_array(self.session.robot.get_angular_velocity())))
        quiet = (endpoint and robot_supported and speed <= p['maximum_stop_speed_m_s']
                 and angular_speed <= p.get('maximum_stop_angular_speed_rad_s', .10))
        # Repeated decision-boundary reads cannot count as additional physics steps.
        if getattr(self, '_in_record', False):
            if not quiet:
                self.quiet_samples = 0
        else:
            self.quiet_samples = self.quiet_samples + 1 if quiet else 0
        self.measured = {'base': pose, 'native_region': region, 'anchor_states': anchors,
                         'speed_m_s': speed, 'angular_speed_rad_s': angular_speed,
                         'endpoint_reached': bool(endpoint),
                         'robot_supported_upright': robot_supported, 'native_support': support}

    def record(self, observation, *, action=None):
        if self.initial is None:
            return None
        self._in_record = True
        try:
            self.sample_step()
        finally:
            self._in_record = False
        p = self.config['parameters']
        criteria = {'native_source_relation_preserved': bool(self.relation_valid),
                    'anchors_unchanged': bool(self.anchors_preserved),
                    'continuous_actual_base_trajectory': not self.discontinuous,
                    'native_floor_0_supported_trajectory': bool(self.contact_safe_trajectory),
                    'minimum_route_completed': self.path_m >= p['minimum_route_m'],
                    'minimum_excursion_completed': self.excursion_m >= p['minimum_excursion_m'],
                    'native_endpoint_reached': self.measured['endpoint_reached'],
                    'native_robot_supported_upright': self.measured['robot_supported_upright'],
                    'native_stop_measured': self.quiet_samples >= p['stop_samples']}
        record = {'schema': 'private_regional_operation_v1', 'episode': self.episode,
                  'snapshot_id': getattr(observation, 'snapshot_id', None),
                  'action_step': getattr(observation, 'action_step', None), 'action': action,
                  'task_success': all(criteria.values()), 'criteria': criteria,
                  'operation': self.config['operation'], 'path_m': self.path_m,
                  'excursion_m': self.excursion_m, 'survey_stage': self.stage,
                  'quiet_physics_samples': self.quiet_samples, 'initial_base': self.initial,
                  'initial_region': self.initial_region, 'initial_relation': self.relation_evidence,
                  'configuration': self.config, **self.measured}
        temporary = self.output / 'state.json.tmp'
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.output / 'state.json')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        return record


def create_evaluator(session, controller, config, output):
    if config.get('operation') == REVISED_OPERATION:
        native = create_evaluator(session, controller, legacy_mapping_config(config), output)
        return SourceAnswerMappingEvaluator(session, controller, config, output, native)
    evaluator = RegionalOperationEvaluator if config['operation'] == EXPLORATION_OPERATION else LegacyRegionalOperationEvaluator
    return evaluator(session, controller, config, output)


class RegionalOperationEvaluator(NativeVisitReportEvaluator):
    """Unordered anchor-defined native room visits with explicit Agent reports."""
    def __init__(self, session, controller, config, output):
        self.bound = _binding_objects(session, config)
        super().__init__(session, controller, config, output)

    def _measure_native(self):
        measured = measure_room_state(self.session, parameters=self.parameters, include_safety=False)
        return settled_native_state(self.session, measured, minimum_up_dot=.9)

    def _begin_baseline(self):
        self.initial_anchors = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        self.relation_evidence = _relation_evidence(self.session, self.config, self.bound)


    def _additional_checks(self, measured):
        unchanged = True
        current = [_anchor_state(self.bound[key]) for key in self.config['binding_keys']]
        tolerance = self.parameters['anchor_preservation_tolerance_m']
        for initial, anchor in zip(self.initial_anchors, current):
            initial_q, current_q = (np.asarray(value) for value in (
                initial['quaternion_xyzw'], anchor['quaternion_xyzw']))
            denominator = float(np.linalg.norm(initial_q) * np.linalg.norm(current_q))
            orientation_error = 1 - abs(float(np.dot(initial_q, current_q))) / max(denominator, 1e-12)
            unchanged &= (all(math.dist(initial[key], anchor[key]) <= tolerance
                for key in ('position', 'aabb_min', 'aabb_max')) and orientation_error <= 1e-4)
        return {'native_source_relation_preserved': bool(self.relation_evidence['region_relation_consistent']),
            'anchors_unchanged': bool(unchanged)}
