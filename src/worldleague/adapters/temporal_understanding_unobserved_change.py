"""Restore saved Unobserved Change snapshots and add measured robot operations.

Run from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
CPU conversion never imports the simulator. Initialization restores source
assets only once; qualification and evaluation only read native measurements.
"""
from __future__ import annotations


from worldleague.resources import ESI_ROOT

from functools import lru_cache
import importlib
import math
from pathlib import Path
import sys

import numpy as np

from worldleague.authoring.io import write_json

MODULE = 'worldleague.adapters.temporal_understanding_unobserved_change'
PHASES = ('image1', 'image2')
OPERATIONS = ('phase_inventory_return', 'phase_inventory_extract', 'box_inspection_route')
MINIMUM_LIFT_M = .05
EXTRACTION_CLEARANCE_M = .01
STABILITY_STEPS = 15


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Source/native measurements must be finite.')
    return result


def _vector(value, size):
    value = _array(value)
    if value.shape != (size,):
        raise ValueError('Source/native vector dimensions are unresolved.')
    return value


def _angle(a, b):
    a, b = _vector(a, 4), _vector(b, 4)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        raise ValueError('Source/native quaternion is unresolved.')
    return math.degrees(2 * math.acos(min(1., abs(float(np.dot(a, b) / norm)))))


@lru_cache(maxsize=1)
def _replay():
    """Reuse the original CPU-only parser/config helper, not its GPU renderer."""
    source_root = str(ESI_ROOT / 'src')
    added = source_root not in sys.path
    if added:
        sys.path.insert(0, source_root)
    try:
        return importlib.import_module('passive_gt.unobserved_regeneration')
    finally:
        if added:
            sys.path.remove(source_root)


def configure_scene(config, parameters):
    """Reuse exact replay preload configs, with no mid-episode object placement."""
    if config['scene']['scene_model'] != parameters['scene_model'] or parameters['phase'] not in PHASES:
        raise ValueError('Source snapshot restoration belongs to a different scene/phase.')
    if not config.get('robots') or config['robots'][0].get('type') != 'R1':
        raise ValueError('The audited active pipeline must supply its installed R1 configuration.')
    plan = _replay().build_unobserved_replay_plan(parameters['replay_payload'], validate_selector=False)
    objects = _replay()._preloaded_object_configs(plan)
    active_names = {parameters['bindings'][key]['name'] for key in parameters['phase_keys'][parameters['phase']]}
    prior = {item.get('name') for item in config.setdefault('objects', [])}
    for item in objects:
        if item['name'] in prior:
            raise ValueError('Source restoration cannot overwrite an existing object.')
        key = next(key for key, binding in parameters['bindings'].items() if binding['name'] == item['name'])
        if item['name'] in active_names:
            item['position'] = parameters['source_objects'][key]['position']
        config['objects'].append(item)
    return {'phase': parameters['phase'],
            'preloaded_objects': objects,
            'mid_episode_pose_changes': False,
            'inactive_phase_policy': 'Original replay fixed visual-only offstage preload; never an Agent observation.'}


def initialize_scene(session, parameters):
    """Restore visible closed source lids once and enable private sensor evidence."""
    from omnigibson.object_states import Open
    if hasattr(session, '_unobserved_change_initialization'):
        raise ValueError('Original snapshot restoration runs only once at initialization.')
    evidence = []
    for key in parameters['container_keys']:
        binding = parameters['bindings'][key]
        obj = session.env.scene.object_registry('name', binding['name'])
        if obj is None or obj.category != binding['category'] or obj.model != binding['model']:
            raise ValueError('An exact original container could not be restored.')
        links = []
        if Open in obj.states:
            state = obj.states[Open]
            state.set_value(False)
            for joint in state.relevant_joints_info[1]:
                for link in obj.links.values():
                    if link.prim_path == joint.body1:
                        link.visible = True
                        links.append(link.prim_path)
            closed = not bool(state.get_value())
            if not closed:
                raise ValueError('Original container failed initialization-only closing.')
        else:
            closed = None
        evidence.append({'key': key, 'name': obj.name, 'closed': closed, 'restored_lid_links': links,
                         'fixed_base': bool(obj.fixed_base), 'visual_only': bool(obj.visual_only)})
    sensors = []
    # Extra segmentation stays private. The Agent still receives its ordinary
    # RGB cameras, without names, source models, masks, poses or scorer state.
    for name, sensor in session.env.robots[0].sensors.items():
        if 'rgb' in sensor.modalities:
            if 'seg_instance' not in sensor.modalities:
                sensor.add_modality('seg_instance')
            sensors.append(name)
    session._unobserved_change_initialization = {'phase': parameters['phase'], 'containers': evidence}
    return {'containers': evidence,
            'private_segmentation_sensors': sensors, 'phase': parameters['phase'],
            'human_framing_names': [parameters['bindings'][key]['name'] for key in
                parameters['container_keys'] + parameters['phase_keys'][parameters['phase']]],
            'original_reference_cameras_preserved': True, 'historical_agent_observations_created': False,
            'warning_codes': ['assumption_review_required', 'independent_phase_snapshots_only']}


def _snapshot(obj, *, dynamic=False):
    position, quaternion = obj.get_position_orientation()
    low, high = [_array(v) for v in obj.aabb]
    row = {'name': obj.name, 'category': obj.category, 'model': obj.model,
           'position': _array(position).tolist(), 'quaternion_xyzw': _array(quaternion).tolist(),
           'scale': _array(obj.scale).tolist(),
           'bbox': [low.tolist(), high.tolist()], 'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base)}
    if dynamic:
        row.update(linear_speed_m_s=float(np.linalg.norm(_array(obj.get_linear_velocity()))),
                   angular_speed_rad_s=float(np.linalg.norm(_array(obj.get_angular_velocity()))))
    return row


def _stationary(row):
    return row['linear_speed_m_s'] <= .02 and row['angular_speed_rad_s'] <= .1


def _near(row, original, tolerance=.04):
    return math.dist(row['position'], original['position']) <= tolerance and _angle(
        row['quaternion_xyzw'], original['quaternion_xyzw']) <= 10.


def _configuration(session, config):
    p = config['parameters']
    keys = p['container_keys'] + p['phase_keys']['image1'] + p['phase_keys']['image2'] + ['floor']
    if (len(set(keys)) != len(keys) or set(config['binding_keys']) != set(keys) or set(config['bindings']) != set(keys)
            or len(set(config['bindings'].values())) != len(keys)):
        raise ValueError('Complete distinct original snapshot bindings are required.')
    phase = session._unobserved_change_initialization['phase']
    if phase not in PHASES:
        raise ValueError('The original source snapshot was not initialized.')
    return p, phase


def _robot_state(session, controller, floor):
    pose = controller.pose()
    low, high = [_array(v) for v in session.robot.links[session.robot.base_footprint_link_name].aabb]
    regions = [session.env.scene.seg_map.get_room_instance_by_point(np.array([x, y]))
               for x in (low[0], (low[0] + high[0]) / 2, high[0])
               for y in (low[1], (low[1] + high[1]) / 2, high[1])]
    scene_objects = session.env.scene.objects
    scene_objects = scene_objects.values() if isinstance(scene_objects, dict) else scene_objects
    floor_paths = [obj.prim_path for obj in scene_objects if obj.category in ('floor', 'floors')]
    contacts, obstacles = [], []
    for contact in session.robot.contact_list():
        for raw in (contact.body0, contact.body1):
            body = str(raw)
            if body == session.robot.prim_path or body.startswith(session.robot.prim_path + '/'):
                continue
            contacts.append(body)
            if not any(body == path or body.startswith(path + '/') for path in floor_paths):
                obstacles.append(body)
    return {**pose, 'in_original_room': all(r == session.room_name for r in regions),
            'supported': any(body == floor.prim_path or body.startswith(floor.prim_path + '/') for body in contacts),
            'nonfloor_contacts': obstacles}


def _measure(session, controller, config, objects):
    from omnigibson.object_states import Inside, Touching
    p, phase = _configuration(session, config)
    containers = {key: _snapshot(objects[key]) for key in p['container_keys']}
    active = {}
    for key in p['phase_keys'][phase]:
        row = _snapshot(objects[key], dynamic=True)
        row['inside'] = bool(objects[key].states[Inside].get_value(objects[p['relations'][key]]))
        row['touching_container'] = bool(objects[key].states[Touching].get_value(objects[p['relations'][key]]))
        active[key] = row
    inactive = {key: {'position': _array(objects[key].get_position_orientation()[0]).tolist(),
                      'fixed_base': bool(objects[key].fixed_base), 'visual_only': bool(objects[key].visual_only)}
                for other in PHASES if other != phase for key in p['phase_keys'][other]}
    return {'phase': phase, 'containers': containers, 'active_contents': active, 'inactive_contents': inactive,
            'robot': _robot_state(session, controller, objects['floor']), 'gripper': controller.debug_gripper_state()}


def _source_checks(config, measured, objects, *, check_initial_dimensions=True):
    p, active, robot = config['parameters'], measured['active_contents'], measured['robot']
    source = p['source_objects']
    checks = {'original_floor_collidable': not objects['floor'].visual_only,
            'original_native_identities': all(all(getattr(objects[key], field) == value for field, value in identity.items())
                                              for key, identity in p['identities'].items()),
            'original_containers_fixed_collidable': all(row['fixed_base'] and not row['visual_only']
                                                       for row in measured['containers'].values()),
            'original_container_root_layout': all(_near(row, source[key], .03)
                for key, row in measured['containers'].items()),
            'active_contents_movable_collidable': all(not row['fixed_base'] and not row['visual_only'] for row in active.values()),
            'inactive_contents_offstage_fixed_visual_only': all(row['fixed_base'] and row['visual_only'] and
                row['position'][0] > 900. and row['position'][1] > 900. for row in measured['inactive_contents'].values()),
            'robot_in_original_room': robot['in_original_room'], 'robot_supported': robot['supported']}
    if check_initial_dimensions:
        checks.update(original_closed_container_dimensions=all(
            np.max(np.abs(_array(row['bbox']) - _array(source[key]['bbox']))) <= .04
            for key, row in measured['containers'].items()),
            active_contents_original_dimensions=all(np.max(np.abs((_array(row['bbox'])[1] - _array(row['bbox'])[0]) -
                _array(source[key]['bbox_size_m']))) <= .03 for key, row in active.items()))
    return checks


class SnapshotOperationEvaluator:
    """Credit real extraction/return or a distinct native inspection route."""
    def __init__(self, session, controller, config, output):
        self.parameters, self.phase = _configuration(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        if any(obj is None for obj in self.objects.values()):
            raise ValueError('An original native snapshot object is missing.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.initial, self.episode = None, 0

    def begin_episode(self, observation):
        self.episode += 1
        self.initial = _measure(self.session, self.controller, self.config, self.objects)
        self.history_valid, self.wrong_grasp = True, False
        self.lifted, self.completed, self.visits = set(), set(), {}
        self.stable_steps, self.last_action_step = 0, observation.action_step
        self.initial_action_step = observation.action_step
        self.record(observation, action=None)

    def _update(self, count_step=False):
        if self.initial is None:
            return None
        measured = _measure(self.session, self.controller, self.config, self.objects)
        contents, gripper = measured['active_contents'], measured['gripper']
        checks = _source_checks(self.config, measured, self.objects, check_initial_dimensions=False)
        # Articulation and grasp rotation legitimately change world AABBs. Use
        # closed initial dimensions plus unchanged native scale/root evidence.
        checks['native_scales_unchanged'] = all(np.allclose(row['scale'], self.initial[group][key]['scale'], atol=1e-6, rtol=0.)
            for group in ('containers', 'active_contents') for key, row in measured[group].items())
        self.history_valid = self.history_valid and measured['phase'] == self.phase and all(checks.values())
        names = {row['name']: key for key, row in contents.items()}
        held = names.get(gripper.get('object_name')) if (gripper.get('holding') is True and
                                                        gripper.get('constraint_active') is True) else None
        if gripper.get('object_name') is not None and gripper['object_name'] not in names:
            self.wrong_grasp = True
        lifts = {key: row['bbox'][0][2] - self.initial['active_contents'][key]['bbox'][0][2]
                 for key, row in contents.items()}
        extracted = False
        if held:
            row = contents[held]
            container = self.initial['containers'][self.parameters['relations'][held]]
            extracted = (not row['inside'] and lifts[held] >= MINIMUM_LIFT_M and
                         row['bbox'][0][2] >= container['bbox'][1][2] + EXTRACTION_CLEARANCE_M)
            if extracted:
                self.lifted.add(held)
        empty_open = (gripper.get('object_name') is None and gripper.get('constraint_active') is False and
                      gripper.get('gripper_open') is True)
        all_returned = all(row['inside'] and row['touching_container'] and _stationary(row)
                           and _near(row, self.initial['active_contents'][key])
                           for key, row in contents.items())
        if empty_open:
            for key in self.lifted:
                row = contents[key]
                if row['inside'] and row['touching_container'] and _stationary(row) and _near(row, self.initial['active_contents'][key]):
                    self.completed.add(key)
        operation = self.config['operation']
        route_required = operation == 'box_inspection_route' or not contents
        route_complete = any(len(visits) >= 2 for visits in self.visits.values())
        at_start = math.dist(measured['robot']['position'][:2], self.initial['robot']['position'][:2]) <= .2
        other_contents_preserved = all(row['inside'] and row['touching_container'] and _stationary(row) and
            _near(row, self.initial['active_contents'][key]) for key, row in contents.items() if key != held)
        operated = (route_complete and at_start and empty_open and all_returned if route_required else
                    bool(held and extracted and other_contents_preserved) if operation == 'phase_inventory_extract' else
                    bool(self.completed) and empty_open and all_returned)
        # Navigation must be contact-free. Legitimate target/lid contacts during
        # manipulation are handled by the existing guarded controls.
        navigation_clear = not measured['robot']['nonfloor_contacts'] if route_required else True
        terminal = (self.history_valid and not self.wrong_grasp and navigation_clear and
                    operated and self.last_action_step > self.initial_action_step)
        if count_step:
            self.stable_steps = self.stable_steps + 1 if terminal else 0
        elif not terminal:
            self.stable_steps = 0
        return {'native_state': measured, 'held_binding': held, 'native_lift_m': lifts,
                'extracted_bindings': sorted(self.lifted), 'returned_bindings': sorted(self.completed),
                'inspection_visits': self.visits, 'criteria': {
                    'source_native_history_valid': self.history_valid, 'no_wrong_grasp': not self.wrong_grasp,
                    'physical_operation_complete': bool(operated), 'guarded_action_history_present':
                        self.last_action_step > self.initial_action_step, 'navigation_clear': navigation_clear},
                'stable_steps': self.stable_steps, 'required_stability_steps': STABILITY_STEPS,
                'task_success': bool(terminal and self.stable_steps >= STABILITY_STEPS)}

    def _record_visible_visits(self, observation, measured):
        """Read existing private segmentation only for RGB delivered to the Agent."""
        if observation.action_step <= self.initial_action_step:
            return
        robot = measured['robot']
        if not robot['in_original_room'] or not robot['supported'] or robot['nonfloor_contacts']:
            return
        xy = _array(robot['position'])[:2]
        if math.dist(xy, self.initial['robot']['position'][:2]) < .5:
            return
        visible = set()
        for view, name in self.session.robot_camera_registry.items():
            delivered = observation.rgb_images().get(view)
            if delivered is None:
                continue
            sensor = self.session.robot.sensors[name]
            native, info = sensor.get_obs()
            if 'seg_instance' not in native or not np.array_equal(_array(native['rgb'])[..., :3], delivered):
                continue
            labels = info.get('seg_instance', {})
            for key, row in measured['containers'].items():
                ids = [int(label) for label, object_name in labels.items() if object_name == row['name']]
                mask = np.isin(_array(native['seg_instance']), ids)
                ys, xs = np.nonzero(mask)
                if len(xs) >= 64 and min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) >= 8:
                    visible.add(key)
        for key in sorted(visible):
            center = _array(measured['containers'][key]['bbox']).mean(axis=0)[:2]
            if not .7 <= math.dist(xy, center) <= 1.8:
                continue
            entries = self.visits.setdefault(key, [])
            if len(entries) >= 2:
                continue
            if entries:
                previous = _array(entries[0]['xy'])
                a, b = previous - center, xy - center
                angle = math.degrees(math.acos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.)))
                if math.dist(previous, xy) < .65 or angle < 60.:
                    continue
            entries.append({'xy': xy.tolist(), 'snapshot_id': observation.snapshot_id,
                            'action_step': observation.action_step, 'native_rgb_segmentation_match': True})

    def sample_step(self):
        self._update(count_step=True)

    def record(self, observation, *, action):
        if self.initial is None:
            return None
        if action is not None:
            self.last_action_step = max(self.last_action_step, observation.action_step)
        measured = _measure(self.session, self.controller, self.config, self.objects)
        if self.config['operation'] == 'box_inspection_route' or not measured['active_contents']:
            self._record_visible_visits(observation, measured)
        value = self._update()
        value.update(schema='private_unobserved_snapshot_operation_v1', episode=self.episode,
                     snapshot_id=observation.snapshot_id, action_step=observation.action_step,
                     action=action, configuration=self.config, initial_native_state=self.initial)
        write_json(self.output / 'state.json', value)
        import json
        with (self.output / 'trajectory.jsonl').open('a') as handle:
            handle.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    return SnapshotOperationEvaluator(session, controller, config, output)
