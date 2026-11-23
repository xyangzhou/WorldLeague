"""Measure ordered native room pickups and physical right-hand tray contents.

Used by worldleague scene --help.
"""
from __future__ import annotations

from worldleague.resources import ESI_ROOT

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from worldleague.adapters import hidden_object_collection as collection

MODULE = 'worldleague.adapters.wia_room_collection'
REVISION = 'wia_room_collection_v1'
ORDERED_OPERATION = 'visit_rooms_and_collect_on_right_hand_tray'
SEARCH_OPERATION = 'search_and_collect_on_right_hand_tray'


def inventory(layout):
    return layout.get('native_inventory', []) + layout.get('added_inventory', [])


def configure_scene(config, parameters):
    """Retain native scene objects and add only explicitly declared apparatus."""
    layout = parameters['layout']
    config['scene'].update(deepcopy(layout.get('scene_config', {})))
    replaced_source_objects = []
    if layout.get('source_objects_policy') == 'replace_authored_source_objects':
        replaced_source_objects = [row.get('name') for row in config.get('objects', [])]
        config['objects'] = []
    objects = config.setdefault('objects', [])
    existing_names = {row.get('name') for row in objects}
    authored = []
    for row in layout.get('added_inventory', []):
        authored.append(_added_object_spec(row))
    authored.extend(deepcopy(layout.get('support_objects', [])))
    plate = layout['plate']
    authored.append(dict(type='DatasetObject', name=plate['name'], category=plate['category'],
                         model=plate['model'], position=[100., 100., 1.], fixed_base=False, visual_only=False))
    authored.append(dict(type='PrimitiveObject', name='collection_initial_support', primitive_type='Cube',
                         scale=[.25, .36, .04], position=[100., 100., .8], fixed_base=True, visual_only=False))
    if len({row['name'] for row in authored}) != len(authored) or existing_names.intersection(row['name'] for row in authored):
        raise ValueError('Added collection apparatus must not overwrite existing objects.')
    objects.extend(authored)
    import yaml
    from worldleague.runtime.sim.esi_omnigibson import _import_pipeline
    pipeline = _import_pipeline(ESI_ROOT)
    robot = yaml.safe_load((Path(pipeline.og.example_config_path) / 'r1_primitives.yaml').read_text())['robots'][0]
    robot.update(deepcopy(layout['robot_start']), name='robot', grasping_mode='assisted', disable_grasp_handling=False)
    robot['reset_joint_pos'][6:10] = [.6981317007977318, -1.3962634015954636, -.6981317007977318, 0.]
    config['robots'] = [robot]
    return {'revision': REVISION, 'native_objects_retained': True,
            'native_inventory_spawned': False, 'added_inventory': [row['name'] for row in layout.get('added_inventory', [])],
            'source_objects_policy': layout.get('source_objects_policy', 'retain'),
            'replaced_authored_source_objects': replaced_source_objects,
            'native_layout_qualified': False}


def _added_object_spec(row):
    return dict(type='DatasetObject', fixed_base=False, visual_only=False,
                **{key: deepcopy(row[key]) for key in
                   ('name', 'category', 'model', 'position', 'orientation', 'scale', 'bounding_box') if key in row})


def postprocess_joint_world(session, parameters):
    return {'joint_world_postprocess': True, 'revision': REVISION, 'native_objects_retained': True,
            'source_mechanism_reused': False, 'physical_success_verified': False}


def _object(session, row):
    obj = session.env.scene.object_registry('name', row['name'])
    if (obj is None or any(getattr(obj, key, None) != row[key] for key in ('name', 'category', 'model'))
            or obj.fixed_base or obj.visual_only):
        raise ValueError('Exact movable collidable inventory unavailable: ' + row['name'])
    return obj


def _region(session, position):
    return session.env.scene.seg_map.get_room_instance_by_point(collection.array(position)[:2])


def _support_bounds(obj):
    bounds = np.asarray([collection.array(value) for value in obj.aabb])
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError('Collection support measurement requires finite ordered native AABBs: ' + obj.name)
    return bounds[0], bounds[1]


def _support_surface(obj, support, mode, stage):
    low, high = _support_bounds(support)
    if mode == 'aabb_top':
        return {'surface_z_m': float(high[2])}
    from worldleague.adapters.active_layout import _support_ray_surface
    xy = np.mean(_support_bounds(obj), axis=0)[:2].tolist()
    return _support_ray_surface(obj, support, [*xy, float(high[2]) + .10],
                                [*xy, float(low[2]) - .10], stage)











def _compact_failure_diagnostic(value, list_limit, string_limit):
    """Keep diagnostic field names while bounding oversized native detail."""
    if isinstance(value, dict):
        return {key: _compact_failure_diagnostic(item, list_limit, string_limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_failure_diagnostic(item, list_limit, string_limit) for item in value[:list_limit]]
    return value[:string_limit] if isinstance(value, str) else value


def initialize_scene(session, parameters):
    """Align only declared native relocations, then initialize the held tray."""
    if not getattr(session, '_category_initialization_active', False):
        raise RuntimeError('Collection setup is restricted to explicit initialization.')
    preview_only = parameters.get('unqualified_tray_preview') is True
    rows = inventory(parameters['layout'])
    positions = parameters.get('initial_object_positions')
    relocations, alignments, edge_placements = [], [], []
    added_names = {row['name'] for row in parameters['layout'].get('added_inventory', [])}
    for row in rows:
        obj = _object(session, row)
        pose = row.get('relocation')
        if pose and positions is None:
            if set(pose) != {'position', 'orientation'}:
                raise ValueError('Relocation requires only explicit position and orientation.')
            obj.set_position_orientation(position=pose['position'], orientation=pose['orientation'])
            obj.keep_still()
            relocations.append(row['name'])
    for row in rows:
        if not row.get('support'):
            continue
        obj = _object(session, row)
        support = session.env.scene.object_registry('name', row['support'])
        if support is None:
            raise ValueError('Declared collection support is missing: ' + row['support'])
        if positions is not None:
            position = positions[row['name']]
            before_position, before_quaternion = (collection.array(value) for value in obj.get_position_orientation())
            # The original constructor and physics may already have laid a box
            # flat. Restoring its authored quaternion at a settled-height XYZ
            # can penetrate the support. Preserve that native settled attitude.
            obj.set_position_orientation(position=position)
            obj.keep_still()
            alignments.append({'target': obj.name, 'support': support.name, 'mode': 'frozen_parent_native_reset_v1',
                               'reference_position_m': position,
                               'pre_restore_position_m': before_position.tolist(),
                               'preserved_native_orientation_xyzw': before_quaternion.tolist()})
            continue
        if row['name'] not in added_names and not row.get('relocation') and not getattr(session, '_release_edge_policy', False):
            continue
        if row.get('pickup_xy_policy') == 'preserve_authored_v1':
            # Concave furniture's outer AABB edge can be behind a partition.
            # Generic preload physics may already have displaced an added object.
            # Restore its declared pose before measuring the local support surface.
            previous_position, previous_orientation = (
                collection.array(value) for value in obj.get_position_orientation())
            obj.set_position_orientation(position=row['position'], orientation=row['orientation'])
            obj.keep_still()
            edge_placements.append({'object': obj.name, 'support': support.name,
                'policy': row['pickup_xy_policy'], 'authored_xy_m': row['position'][:2],
                'pre_alignment_position_m': previous_position.tolist(),
                'pre_alignment_orientation_xyzw': previous_orientation.tolist(),
                'authored_pose_restored_before_support_check': True,
                'physical_grasp_verified': False, 'route_connectivity_verified': False})
        elif row.get('pickup_edge_margin_m') is not None:
            from worldleague.tasks.pickup_edges import place_pickup_near_edge
            edge_placements.append(place_pickup_near_edge(session, obj, support, row['pickup_edge_margin_m'], row['room'], row.get('pickup_approach_xy_m')))
            if not hasattr(session, 'pickup_edge_placements'):
                session.pickup_edge_placements = []
            session.pickup_edge_placements.append(dict(edge_placements[-1], status='placed'))
        elif getattr(session, '_release_edge_policy', False):
            from worldleague.tasks.pickup_edges import apply_edge_policy
            edge_placements.extend(apply_edge_policy(session, [obj], supports={obj.name: support.name}))
        mode = row.get('support_alignment_mode', row.get('support_surface_mode', 'aabb_top'))
        surface = _support_surface(obj, support, mode, 'initial_alignment')
        position, quaternion = (collection.array(value) for value in obj.get_position_orientation())
        position[2] += surface['surface_z_m'] - float(_support_bounds(obj)[0][2]) + .003
        obj.set_position_orientation(position=position.tolist(), orientation=quaternion.tolist())
        obj.keep_still()
        alignments.append({'target': obj.name, 'support': support.name, 'mode': mode})
    if parameters.get('protocol') == 'single_arm_mobile_tray_v1':
        from worldleague.adapters.mobile_tray_collection import initialize_apparatus as initialize_collection_tray
    else:
        from worldleague.runtime.sim.esi_collection_setup import initialize_collection_tray
    receipt = initialize_collection_tray(session, parameters)
    result = {**receipt, 'explicit_relocations': relocations, 'support_alignments': alignments,
              'pickup_edge_placements': edge_placements}
    session.wia_room_collection_initialization = result
    session.category_interaction_setup = result
    return result


def _state(session, parameters):
    state = collection._state(session, parameters)
    state['native_region'] = _region(session, session.robot.get_position_orientation()[0])
    # Requiring the chassis footprint inside one segmented room avoids doorway credit.
    robot = session.robot
    low, high = (collection.array(value) for value in robot.links[robot.base_footprint_link_name].aabb)
    footprint = [_region(session, [x, y]) for x in (low[0], (low[0] + high[0]) / 2, high[0])
                 for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    state['room_occupancy_valid'] = bool(state['native_region'] is not None
                                        and all(room == state['native_region'] for room in footprint))
    state['object_rooms'] = {row['name']: _region(session, _object(session, row).get_position_orientation()[0])
                             for row in parameters['layout'].get('native_inventory', []) + parameters['layout'].get('added_inventory', [])}
    return state


class RoomCollectionEvaluator:
    """Keep room-route progress separate from valid pickups and tray contents."""
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config['parameters']
        self.rows = inventory(self.parameters['layout'])
        self.targets = {row['name']: row for row in self.rows if row['role'] == 'target'}
        self.route = self.parameters['layout'].get('route', [])
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.measure_state = _state
        self.reduce_result = collection.collection_result
        self.holding_field = 'left_holding'
        self.begin_episode(None)

    def begin_episode(self, observation):
        self.route_stage, self.acquisition_stage = 0, 0
        self.picked, self.wrong_pickups, self.previous_collected = set(), set(), set()
        self.visits, self.acquisitions, self.spills, self.faults = [], [], [], []
        self.previous_held = None
        self.clock_regression = None
        self.last_time = self.stable_since = self.state = None

    def _result(self, duration, dwell):
        result = self.reduce_result(self.state, list(self.targets), self.parameters.get('distractors', []),
                                              self.picked, stable_seconds=duration, required_dwell=dwell)
        result['checks'].update(declared_route_completed=self.route_stage == len(self.route),
                                ordered_room_acquisitions_completed=self.acquisition_stage == len(self.route),
                                native_measurements_valid=not self.faults)
        result['task_success'] = all(result['checks'].values())
        return result

    def sample_step(self):
        state = self.measure_state(self.session, self.parameters)
        now = float(state['native_time_s'])
        if self.last_time is not None and now <= self.last_time:
            if now < self.last_time:
                if 'native_clock_regressed' not in self.faults:
                    self.faults.append('native_clock_regressed')
                self.clock_regression = {'previous_time_s': self.last_time, 'current_time_s': now}
                self.state = state
                self.stable_since = None
            return
        room, valid = state['native_region'], state['room_occupancy_valid']
        if (not self.visits or self.visits[-1]['room'] != room
                or self.visits[-1]['eligible'] != valid):
            self.visits.append({'room': room, 'eligible': valid, 'native_time_s': now})
        if valid and self.route_stage < len(self.route) and room == self.route[self.route_stage]:
            self.route_stage += 1
        held = state[self.holding_field]
        if held != self.previous_held and held in self.targets:
            target = self.targets[held]
            room_ok = valid and room == target['room'] and state['object_rooms'].get(held) == target['room']
            order_ok = not self.route or (self.acquisition_stage < len(self.route)
                                         and target['room'] == self.route[self.acquisition_stage])
            accepted = bool(room_ok and order_ok)
            self.acquisitions.append({'target': held, 'room': room, 'object_room': state['object_rooms'].get(held),
                                      'in_designated_room': bool(room_ok), 'correct_order': bool(order_ok),
                                      'accepted': accepted, 'native_time_s': now})
            if accepted:
                self.picked.add(held)
                if self.route and all(name in self.picked for name, row in self.targets.items() if row['room'] == target['room']):
                    self.acquisition_stage += 1
        if held in self.parameters.get('distractors', []):
            self.wrong_pickups.add(held)
        present = set(state['supported_objects']).intersection(self.targets)
        self.spills.extend({'target': name, 'native_time_s': now} for name in sorted(self.previous_collected - present))
        self.previous_collected, self.previous_held = present, held
        self.state, self.last_time = state, now
        if self._result(0., 0.)['task_success']:
            if self.stable_since is None:
                self.stable_since = now
        else:
            self.stable_since = None

    def record(self, observation, *, action=None):
        self.sample_step()
        duration = 0. if self.stable_since is None else self.last_time - self.stable_since
        result = self._result(duration, self.parameters['terminal_dwell_s'])
        validity = 'unavailable' if self.faults else 'complete'
        result.update(schema='native_wia_room_collection_v1', snapshot_id=observation.snapshot_id,
                      action_step=observation.action_step, evaluation_validity=validity,
                      terminal_conditions_satisfied=result['task_success'], route=list(self.route),
                      route_stage=self.route_stage, acquisition_stage=self.acquisition_stage,
                      native_room_visits=self.visits, acquisitions=self.acquisitions,
                      validly_acquired_targets=sorted(self.picked), wrong_pickups=sorted(self.wrong_pickups),
                      spills=self.spills, native_state=self.state, configuration=self.config,
                      faults=list(self.faults), clock_regression=getattr(self, 'clock_regression', None),
                      completion_items=[{'id': 'collect:' + name, 'kind': 'current_physical_inventory',
                                         'completed': validity == 'complete' and name in result['collected_targets'],
                                         'evaluation_validity': 'valid' if validity == 'complete' else validity}
                                        for name in self.targets])
        (self.output / 'latest.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return result


def create_evaluator(session, controller, config, output):
    if config['operation'] not in (ORDERED_OPERATION, SEARCH_OPERATION) or config['parameters']['revision'] != REVISION:
        raise ValueError('Unsupported room collection contract.')
    return RoomCollectionEvaluator(session, controller, config, output)
