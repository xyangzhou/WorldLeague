"""Preserve Action Order assets and measure an ordered physical stack.

Run the complete category through ActionOrderConverter().convert_all(sources,
ConversionContext(output_root)); the shared campaign runner owns GPU capture.
"""
from __future__ import annotations


import copy
import math
import json
from pathlib import Path

import numpy as np


MODULE = 'worldleague.adapters.action_order'
OPERATION = 'ordered_grasp_lift_stack'
TASK_REVISION = 'task_edit_20261018_v1'
INITIALIZATION = 'measured_rigid_row_startup_v2'
SUPPORT_TRANSFER_REVISION = 'support_transfer_v1'
FINAL_STATE_REVISION = 'final_state_v1'


def configure_scene(config, parameters):
    """Check the original construction; do not replace assets or robot physics."""
    if parameters.get('operation') != INITIALIZATION:
        raise ValueError('Unsupported Action Order initialization.')
    robots = config.get('robots', [])
    if len(robots) != 1 or robots[0].get('type') != 'R1':
        raise ValueError('Action Order requires its original installed R1.')
    objects = {obj['name']: obj for obj in config.get('objects', [])}
    for binding in parameters['bindings'].values():
        obj = objects.get(binding['name'])
        if obj is None or any(obj.get(key) != binding[key] for key in ('category', 'model')):
            raise ValueError('Initialization must retain every exact original asset.')
    return {'original_assets_and_robot_configuration_preserved': True,
        'native_initialization_pending': True, 'task_actions': 0}


def _row_yaws(group, supports):
    """Measure the original row and propose only existing support-axis rotations."""
    from scipy.spatial.transform import Rotation

    points = np.stack([_array(obj.get_position_orientation()[0], (3,)) for obj in group.values()])
    pivot = points.mean(axis=0)
    _, singular, axes = np.linalg.svd(points[:, :2] - pivot[:2], full_matrices=False)
    if singular[0] <= 1e-6:
        raise ValueError('The original native row has no measurable planar axis.')
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    heading = math.atan2(axis[1], axis[0])
    choices = [{'yaw_rad': 0., 'support_axis_from': None}]
    for obj in supports:
        _, quaternion, extent, _ = obj.get_base_aligned_bbox()
        extent = _array(extent, (3,))
        rotation = Rotation.from_quat(_array(quaternion, (4,))).as_matrix()
        for index in sorted((0, 1), key=lambda i: (-extent[i], i)):
            direction = rotation[:2, index]
            if np.linalg.norm(direction) <= 1e-6:
                continue
            yaw = (math.atan2(direction[1], direction[0]) - heading + math.pi / 2) % math.pi - math.pi / 2
            if not any(abs(yaw - row['yaw_rad']) < math.radians(1.) for row in choices):
                choices.append({'yaw_rad': yaw, 'support_axis_from': obj.name})
    return pivot, choices[:9]


def _safe_diagnostic_spawn(session, group, policy):
    """Search a native floor start even when no tabletop can fit the row."""
    from worldleague.tasks import interaction_setup as interaction
    from omnigibson.utils.usd_utils import PoseAPI

    scene, sim, robot = session.env.scene, session.pipeline.og.sim, session.env.robots[0]
    names = {obj.name for obj in group.values()}
    world, floor_z, grid = interaction._navigation_cells(session, robot, policy['source_region'], names)
    floors = [obj.prim_path for obj in interaction._objects(scene) if obj.category in ('floor', 'floors')]
    if not floors:
        raise ValueError('Native floor bodies are required for an independent robot start.')
    target = np.mean([_array(obj.get_position_orientation()[0], (3,)) for obj in group.values()], axis=0)
    order = np.argsort(np.linalg.norm(world - target[:2], axis=1), kind='stable')[:32]
    baseline = sim.dump_state(serialized=False)
    protected = {obj.name: (obj, tuple(interaction.array(value).copy() for value in obj.get_position_orientation()))
        for obj in interaction._objects(scene) if obj is not robot and not obj.fixed_base and not obj.visual_only}
    evidence = {'accepted': False, 'spawn_grid': grid, 'attempts': [], 'robot_actions': 0}
    try:
        for index in order:
            sim.load_state(baseline, serialized=False)
            PoseAPI.invalidate()
            interaction._place_robot(session, robot, world[index], target, floor_z)
            checks = interaction._robot_contact_check(robot, floors, floor_z)
            checks['original_objects_and_background_preserved'] = all(
                interaction._pose_unchanged(obj, pose, np.zeros(3), .005) for obj, pose in protected.values())
            attempt = {'robot_position_m': interaction.array(robot.get_position_orientation()[0]).tolist(),
                'robot_quaternion_xyzw': interaction.array(robot.get_position_orientation()[1]).tolist(),
                'checks': checks}
            evidence['attempts'].append(attempt)
            if all(checks[key] for key in ('floor_supported', 'collision_free', 'original_objects_and_background_preserved')):
                evidence.update(accepted=True, robot_spawn=attempt)
                return evidence
    finally:
        if not evidence['accepted']:
            sim.load_state(baseline, serialized=False)
            PoseAPI.invalidate()
        session.invalidate_camera_cache()
    evidence['reason'] = 'No measured collision-free same-room diagnostic stance passed the bounded search.'
    return evidence


def _initialization_record(evidence):
    """Retain private diagnostics when capture skips task qualification saving."""
    print('[action-order-initialization] ' + json.dumps(evidence, allow_nan=False), flush=True)
    return evidence


def initialize_scene(session, parameters):
    """Restore interaction access once, before controls or evaluator history exist.

    Reuse the native support/stance initializer for every rotated proposal. A
    rejected proposal restores its full baseline, including source camera pose.
    The later shared patch independently screens the resulting final layout.
    """
    from scipy.spatial.transform import Rotation
    from worldleague.tasks import interaction_setup as interaction
    from worldleague.tasks.schema import Binding
    from omnigibson.utils.usd_utils import PoseAPI
    import torch

    if hasattr(session, '_action_order_initialization'):
        raise ValueError('Action Order restoration is initialization-only and runs once.')
    evidence = {'schema': INITIALIZATION, 'accepted': False, 'robot_actions': 0, 'physical_grasp_verified': False,
        'attempts': [], 'warning_codes': ['assumption_review_required', 'native_stack_execution_unverified']}
    if parameters.get('initialization_review_contract') is not None:
        evidence.update(initialization_review_contract=copy.deepcopy(parameters['initialization_review_contract']),
            candidate_inspection_verified=False,
            candidate_inspection_status='Pending native sensor coverage and human review; footprint clearance alone is insufficient.')
    session._action_order_initialization = evidence
    scene, sim = session.env.scene, session.pipeline.og.sim
    group = {}
    for key, binding in parameters['bindings'].items():
        obj = scene.object_registry('name', binding['name'])
        if obj is None or any(getattr(obj, field) != binding[field] for field in ('category', 'model')):
            raise ValueError('Every original row asset must resolve exactly.')
        if obj.visual_only or obj.fixed_base:
            raise ValueError('Original row physics must be enabled by declared object_physics patches.')
        group[key] = obj
    policy = parameters['interaction_policy']
    native_bindings = {key: Binding.model_validate(binding) for key, binding in parameters['bindings'].items()}
    if policy['relocation_policy'] == 'independent_groups':
        # The following shared patch owns the single final initialization. No
        # provisional placement may overwrite its original reference geometry.
        evidence.update(deferred_to_shared_initializer=True,
            placement_groups=copy.deepcopy(policy['placement_groups']),
            source_camera_policy='preserve',
            reason='Independent candidates require measured accessible supports; final initialization is pending.')
        return _initialization_record(evidence)
    try:
        evidence['independent_robot_spawn'] = _safe_diagnostic_spawn(session, group, policy)
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        evidence['independent_robot_spawn'] = {'accepted': False, 'reason': f'{type(error).__name__}: {error}'}
    # A rejected support search returns to this safe stance, rather than the
    # off-map source robot preset. Object and camera baselines remain original.
    baseline = sim.dump_state(serialized=False)
    poses = {obj.name: tuple(interaction.array(value).copy() for value in obj.get_position_orientation())
        for obj in group.values()}
    scales = {obj.name: interaction.array(obj.scale).copy() for obj in group.values()}
    camera = (interaction.array(session.scene_camera_position).copy(),
        interaction.array(session.scene_camera_quaternion).copy())
    viewer = copy.deepcopy(getattr(session, 'postprocess', {}).get('viewer_camera'))
    framing = session.scene_camera_framing

    def restore():
        sim.load_state(baseline, serialized=False)
        PoseAPI.invalidate()
        session.scene_camera_position, session.scene_camera_quaternion = (value.copy() for value in camera)
        session.scene_camera_framing = framing
        if viewer is not None:
            session.postprocess['viewer_camera'] = copy.deepcopy(viewer)
        session.invalidate_camera_cache()

    try:
        supports = [obj for obj in interaction._objects(scene) if obj.name not in poses and not obj.visual_only
            and any(word in obj.category for word in ('table', 'counter', 'desk', 'bench', 'shelf', 'cabinet'))
            and policy['source_region'] in (getattr(obj, 'in_rooms', ()) or ())]
        supports.sort(key=lambda obj: (float(np.linalg.norm(np.mean([interaction.array(value) for value in obj.aabb], axis=0)[:2]
            - np.mean([pose[0][:2] for pose in poses.values()], axis=0))), obj.name))
        pivot, choices = _row_yaws(group, supports)
        evidence.update(rotation_pivot_m=pivot.tolist(), rotation_proposals=choices,
            original_geometry={name: {'position_m': pose[0].tolist(), 'quaternion_xyzw': pose[1].tolist(),
                'scale': scales[name].tolist()} for name, pose in poses.items()})
        for proposal in choices:
            restore()
            rotation = Rotation.from_euler('z', proposal['yaw_rad'])
            for obj in group.values():
                position, quaternion = poses[obj.name]
                obj.set_position_orientation(position=torch.as_tensor(rotation.apply(position - pivot) + pivot, dtype=torch.float32),
                    orientation=torch.as_tensor((rotation * Rotation.from_quat(quaternion)).as_quat(), dtype=torch.float32))
                obj.keep_still()
            session.scene_camera_position = rotation.apply(camera[0] - pivot) + pivot
            session.scene_camera_quaternion = (rotation * Rotation.from_quat(camera[1])).as_quat()
            if viewer is not None and viewer.get('position') is not None:
                session.postprocess['viewer_camera']['position'] = (rotation.apply(np.asarray(viewer['position']) - pivot) + pivot).tolist()
                for key in ('quaternion_xyzw', 'orientation'):
                    if viewer.get(key) is not None:
                        session.postprocess['viewer_camera'][key] = (rotation * Rotation.from_quat(viewer[key])).as_quat().tolist()
            result = interaction.initialize_interaction_trial(session, native_bindings, policy)
            evidence['attempts'].append({'proposal': proposal, 'accepted': result['accepted'],
                'reason': result.get('reason'), 'support_search': result.get('support_search', []),
                'placement_attempts': result.get('attempts', [])})
            if not result['accepted']:
                continue
            if not all(np.array_equal(interaction.array(obj.scale), scales[obj.name]) for obj in group.values()):
                raise ValueError('Initialization changed an original asset scale.')
            delta = interaction.array(result.get('group_translation_m', [0., 0., 0.]))
            if not all(np.linalg.norm(interaction.array(obj.get_position_orientation()[0])
                    - rotation.apply(poses[obj.name][0] - pivot) - pivot - delta) <= .015
                    and (Rotation.from_quat(interaction.array(obj.get_position_orientation()[1])).inv()
                        * rotation * Rotation.from_quat(poses[obj.name][1])).magnitude() <= .035
                    for obj in group.values()):
                raise ValueError('Initialization did not preserve the common original row transform.')
            if (not np.allclose(interaction.array(session.scene_camera_position),
                    rotation.apply(camera[0] - pivot) + pivot + delta, atol=1e-6, rtol=0.) or
                    (Rotation.from_quat(interaction.array(session.scene_camera_quaternion)).inv()
                        * rotation * Rotation.from_quat(camera[1])).magnitude() > 1e-6):
                raise ValueError('The original reference camera did not follow the exact row transform.')
            evidence.update(accepted=True, native_support_and_stance=result,
                transform={'pivot_m': pivot.tolist(), 'yaw_rad': proposal['yaw_rad'], 'translation_m': delta.tolist()},
                source_camera_before={'position_m': camera[0].tolist(), 'quaternion_xyzw': camera[1].tolist()},
                source_camera_after={'position_m': interaction.array(session.scene_camera_position).tolist(),
                    'quaternion_xyzw': interaction.array(session.scene_camera_quaternion).tolist()},
                source_camera_transform_matches_objects=True, object_scales_unchanged=True)
            session.scene_camera_framing = 'action_order_common_rigid_reference'
            return _initialization_record(evidence)
        evidence['reason'] = 'No identity or measured support-axis rigid row proposal passed native initialization.'
        evidence['warning_codes'].append('native_rigid_row_initialization_failed')
    except (ValueError, RuntimeError, AttributeError, KeyError, TypeError) as error:
        evidence.update(reason=f'{type(error).__name__}: {error}')
        evidence['warning_codes'].append('native_rigid_row_initialization_failed')
    finally:
        if not evidence['accepted']:
            restore()
    evidence['diagnostic_robot_start_retained'] = evidence['independent_robot_spawn']['accepted']
    return _initialization_record(evidence)


def _array(value, shape):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError('Finite native geometry/velocity readback is required.')
    return result


def _under(path, root):
    return path == root or path.startswith(root + '/')


def _bind_objects(session, config):
    return {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}


def _native_hands(robot):
    """Require native evidence for every arm, including attachment release windows."""
    arms = list(robot.arm_names)
    if not arms or len(set(arms)) != len(arms):
        raise ValueError('Native arm inventory is unavailable.')
    rows = {}
    for arm in arms:
        held = robot._ag_obj_in_hand[arm]
        reading = int(robot.is_grasping(arm))
        if reading not in (-1, 1):
            raise ValueError('Native grasp state is unknown.')
        row = {'object_name': held.name if held is not None else None,
            'holding': reading == 1, 'constraint_active': robot._ag_obj_constraints[arm] is not None,
            'release_pending': robot._ag_release_counter[arm] is not None}
        row['released'] = (held is None and reading == -1 and not row['constraint_active']
            and not row['release_pending'])
        rows[arm] = row
    return rows


def _native_speeds(obj):
    """Fixed kinematic supports have no dynamic velocity API in OmniGibson 3.7."""
    if getattr(obj, 'fixed_base', False) and getattr(obj, 'kinematic_only', False):
        return 0., 0.
    return (float(np.linalg.norm(_array(obj.get_linear_velocity(), (3,)))),
            float(np.linalg.norm(_array(obj.get_angular_velocity(), (3,)))))


def _native_measure(session, objects, p, controller=None):
    """Only native readback; construction labels, observations and QA are unused."""
    from omnigibson.object_states import OnTop, Touching

    robot, scene = session.robot, session.env.scene
    hands = _native_hands(robot)
    scene_objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    external = [obj for obj in scene_objects if obj not in objects.values() and obj is not robot
        and not obj.visual_only]
    native = {}
    for key, obj in objects.items():
        position, quaternion = obj.get_position_orientation()
        position, quaternion = _array(position, (3,)), _array(quaternion, (4,))
        low, high = (_array(value, (3,)) for value in obj.aabb)
        if not np.linalg.norm(quaternion) or np.any(high <= low):
            raise ValueError('Native object geometry is unavailable.')
        linear, angular = _native_speeds(obj)
        contacts = [{'body0': str(row.body0), 'body1': str(row.body1)} for row in obj.contact_list()]
        if any(not row['body0'] or not row['body1'] for row in contacts):
            raise ValueError('Native contact body identity is unavailable.')
        def contact_with(other):
            return any((_under(row['body0'], obj.prim_path) and _under(row['body1'], other.prim_path)) or
                (_under(row['body1'], obj.prim_path) and _under(row['body0'], other.prim_path))
                for row in contacts)
        relations = {}
        for other_key, other in objects.items():
            if other is not obj:
                relations[other_key] = {'on_top': bool(obj.states[OnTop].get_value(other)),
                    'touching': bool(obj.states[Touching].get_value(other)),
                    'native_contact': contact_with(other)}
        supports = []
        for other in external:
            # Native contacts identify possible original scene supports, not a guessed floor height.
            if contact_with(other) and bool(obj.states[OnTop].get_value(other)) and bool(
                    obj.states[Touching].get_value(other)):
                support_linear, support_angular = _native_speeds(other)
                supports.append({'name': other.name, 'linear_speed_m_s': support_linear,
                    'angular_speed_rad_s': support_angular,
                    'velocity_source': ('native_fixed_kinematic' if getattr(other, 'fixed_base', False)
                        and getattr(other, 'kinematic_only', False) else 'native_dynamic_velocity'),
                    'stable': support_linear <= p['maximum_linear_speed_m_s'] and
                        support_angular <= p['maximum_angular_speed_rad_s']})
        native[key] = {'name': obj.name, 'position': position.tolist(), 'bottom_z_m': float(low[2]),
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular,
            'stable': linear <= p['maximum_linear_speed_m_s'] and angular <= p['maximum_angular_speed_rad_s'],
            'robot_contact': bool(obj.states[Touching].get_value(robot)) or contact_with(robot),
            'relations': relations, 'external_supports': supports,
            'native_contacts': contacts}
    detached = all(row['released'] for row in hands.values())
    released = detached
    gripper = None
    acquired_key = None
    if controller is not None:
        arm = controller.working_arm
        gripper = controller.debug_gripper_state()
        hand = hands[arm]
        required = {'object_name', 'holding', 'constraint_active', 'release_pending', 'gripper_open'}
        if not required <= gripper.keys() or gripper['holding'] not in (True, False):
            raise ValueError('Native working gripper readback is unavailable.')
        if any(gripper[key] != hand[key] for key in ('object_name', 'holding', 'constraint_active', 'release_pending')):
            raise ValueError('Native grasp and controller attachment readings disagree.')
        released = released and gripper['gripper_open'] is True
        if (hand['holding'] and hand['constraint_active'] and not hand['release_pending'] and
                all(row['released'] for other_arm, row in hands.items() if other_arm != arm)):
            acquired_key = next((key for key, obj in objects.items() if obj.name == hand['object_name']), None)
    return {'objects': native, 'hands': hands, 'all_detached': bool(detached), 'all_released': bool(released),
        'gripper': gripper, 'acquired_key': acquired_key}


def _supported(measured, upper, lower):
    relation = measured['objects'][upper]['relations'][lower]
    return all(relation.values())


def _external_support(measured, key):
    return any(row['stable'] for row in measured['objects'][key]['external_supports'])


def _only_stack_support(measured, upper, lower):
    """Reject upper members propped by scene furniture or another candidate."""
    return (_supported(measured, upper, lower) and
        not measured['objects'][upper]['external_supports'] and
        not any(_supported(measured, upper, other) for other in measured['objects']
            if other not in (upper, lower)))


class OrderedStackEvaluator:
    """Measure ordered native acquisition, lift, release and stable support history."""
    def __init__(self, session, controller, config, output):
        self.parameters, self.objects = config['parameters'], _bind_objects(session, config)
        self.session, self.controller, self.config = session, controller, config
        self.dt = float(controller.dt)
        native_dt = 1.0 / float(session.env.env_config['action_frequency'])
        if not math.isfinite(self.dt) or self.dt <= 0 or not math.isclose(self.dt, native_dt, rel_tol=1e-9):
            raise ValueError('Native controller clock and environment frequency must agree.')
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.episode, self.active = 0, False

    def _read(self):
        try:
            value = _native_measure(self.session, self.objects, self.parameters, self.controller)
            # Final-state scoring can recover from a transient read failure.
            # Historical manipulation scoring must retain gaps in its evidence.
            if self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION:
                self.measurement_error = None
            self.last = value
            return value
        except Exception as error:
            self.measurement_error = f'{type(error).__name__}: {error}'
            self.final_since = None
            return None

    def _elapsed(self, start):
        return 0.0 if start is None else (self.sample_count - start) * self.dt

    def _base_structural(self, measured, base):
        return _only_stack_support(measured, 'middle', base) and _external_support(measured, base)

    def _invalidate_credits(self, measured):
        """Every measured boundary can revoke credit; only sample_step adds time."""
        middle_name = measured['objects']['middle']['name']
        middle_reacquired = any(row['object_name'] == middle_name and
            (row['holding'] or row['constraint_active'] or row['release_pending'])
            for row in measured['hands'].values())
        if self.middle_placement is not None and (middle_reacquired or not self._base_structural(
                measured, self.middle_placement['base'])):
            self.events.append({'event': 'middle_placement_revoked', 'step': self.sample_count,
                'reason': 'measured_support_order_interrupted'})
            self.middle_placement = None
            self.base_since = {key: None for key in self.parameters['base_keys']}
        for base in self.parameters['base_keys']:
            if not (self._base_structural(measured, base) and
                    self._stable_clear(measured, ['middle', base]) and not any(
                        _supported(measured, top, 'middle') for top in self.parameters['top_keys'])):
                self.base_since[base] = None

    def _stable_clear(self, measured, keys):
        return measured['all_released'] and all(measured['objects'][key]['stable'] and not
            measured['objects'][key]['robot_contact'] for key in keys)

    def _lift_released(self, key):
        history = self.histories.get(key)
        return bool(history and history['release_step'] is not None and history['peak_lift_m'] >=
            self.parameters['minimum_lift_m'])

    def _manipulation_released(self, key):
        if self.parameters.get('manipulation_revision') != SUPPORT_TRANSFER_REVISION:
            return self._lift_released(key)
        history = self.histories.get(key)
        return bool(history and history['unsupported_while_acquired']
            and history['acquired_step'] is not None and history['separation_step'] is not None
            and history['release_step'] is not None
            and history['release_step'] > history['separation_step'] >= history['acquired_step'])

    def _final_pair(self, measured):
        if self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION:
            for base in self.parameters['base_keys']:
                if not self._base_structural(measured, base):
                    continue
                if not self.parameters['top_keys']:
                    if self._stable_clear(measured, ['middle', base]):
                        return base, None
                for top in self.parameters['top_keys']:
                    if (_only_stack_support(measured, top, 'middle') and
                            self._stable_clear(measured, ['middle', base, top])):
                        return base, top
            return None
        if self.middle_placement is None or not self._manipulation_released('middle'):
            return None
        base = self.middle_placement['base']
        if not self._base_structural(measured, base):
            return None
        if not self.parameters['top_keys']:
            return (base, None) if self._stable_clear(measured, ['middle', base]) else None
        for top in self.parameters['top_keys']:
            history = self.histories.get(top)
            if (self._manipulation_released(top) and history['acquired_step'] > self.middle_placement['step'] and
                    _only_stack_support(measured, top, 'middle') and
                    self._stable_clear(measured, ['middle', base, top])):
                return base, top
        return None

    def begin_episode(self, observation):
        self.episode += 1
        self.active = True
        self.sample_count, self.measurement_error = 0, None
        self.histories, self.events = {}, []
        self.previous_acquired = None
        self.middle_placement, self.final_since, self.final_pair = None, None, None
        self.ordered_top_seen = False
        self.base_since = {key: None for key in self.parameters['base_keys']}
        self.last = None
        initial = self._read()
        self.initial = initial
        self.initial_eligible = bool(initial and initial['all_detached'] and
            all(row['stable'] and not row['robot_contact'] and _external_support(initial, key)
                for key, row in initial['objects'].items()) and
            not any(_supported(initial, 'middle', base) and (not self.parameters['top_keys'] or
                any(_supported(initial, top, 'middle') for top in self.parameters['top_keys']))
                for base in self.parameters['base_keys']))
        self.initial_difficulty_eligible = self.initial_eligible
        if self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION:
            self.initial_eligible = initial is not None
        return self.record(observation, action=None)

    def sample_step(self):
        if not self.active:
            return
        self.sample_count += 1
        measured = self._read()
        if measured is None or self.measurement_error or not self.initial_eligible:
            return
        if self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION:
            pair = self._final_pair(measured)
            if pair is None or pair != self.final_pair:
                self.final_since = self.sample_count if pair else None
            self.final_pair = pair
            return
        self._invalidate_credits(measured)
        acquired = measured['acquired_key']
        movable = ['middle', *self.parameters['top_keys']]
        if acquired in movable and acquired != self.previous_acquired:
            current, initial = measured['objects'][acquired], self.initial['objects'][acquired]
            self.histories[acquired] = {'acquired_step': self.sample_count,
                'baseline_bottom_z_m': max(initial['bottom_z_m'], current['bottom_z_m']),
                'baseline_root_z_m': max(initial['position'][2], current['position'][2]),
                'peak_lift_m': 0.0, 'release_step': None}
            if self.parameters.get('manipulation_revision') == SUPPORT_TRANSFER_REVISION:
                self.histories[acquired].update(unsupported_while_acquired=False, separation_step=None)
            self.events.append({'event': 'native_acquired_grasp', 'object': acquired, 'step': self.sample_count})
            if acquired == 'middle':
                self.middle_placement = None
                self.base_since = {key: None for key in self.parameters['base_keys']}
        if acquired in self.histories:
            history, current = self.histories[acquired], measured['objects'][acquired]
            lift = min(current['bottom_z_m'] - history['baseline_bottom_z_m'],
                current['position'][2] - history['baseline_root_z_m'])
            history['peak_lift_m'] = max(history['peak_lift_m'], lift)
            if (self.parameters.get('manipulation_revision') == SUPPORT_TRANSFER_REVISION
                    and not history['unsupported_while_acquired']
                    and not current['external_supports']
                    and not any(_supported(measured, acquired, other)
                        for other in measured['objects'] if other != acquired)):
                history.update(unsupported_while_acquired=True, separation_step=self.sample_count)
                self.events.append({'event': 'native_acquired_support_separation',
                    'object': acquired, 'step': self.sample_count})
        if measured['all_released']:
            for key, history in self.histories.items():
                if history['release_step'] is None:
                    history['release_step'] = self.sample_count
                    self.events.append({'event': 'native_release', 'object': key, 'step': self.sample_count})
        self.previous_acquired = acquired
        if self.middle_placement is None:
            no_top_on_middle = not any(_supported(measured, top, 'middle') for top in self.parameters['top_keys'])
            for base in self.parameters['base_keys']:
                eligible = (self._manipulation_released('middle') and no_top_on_middle and
                    self._base_structural(measured, base) and self._stable_clear(measured, ['middle', base]))
                self.base_since[base] = (self.sample_count if self.base_since[base] is None else
                    self.base_since[base]) if eligible else None
                if eligible and self._elapsed(self.base_since[base]) + 1e-12 >= self.parameters['minimum_stability_duration_s']:
                    self.middle_placement = {'base': base, 'step': self.sample_count,
                        'stable_duration_s': self._elapsed(self.base_since[base])}
                    self.events.append({'event': 'middle_supported_placement', **self.middle_placement})
                    break
        pair = self._final_pair(measured)
        if pair and pair[1] is not None:
            self.ordered_top_seen = True
        if pair is None or pair != self.final_pair:
            self.final_since = self.sample_count if pair else None
        self.final_pair = pair

    def record(self, observation, *, action):
        if not self.active:
            return None
        measured = self._read()
        final_state = self.parameters.get('manipulation_revision') == FINAL_STATE_REVISION
        if measured is not None and not final_state:
            self._invalidate_credits(measured)
        pair = self._final_pair(measured) if measured is not None else None
        if pair != self.final_pair or pair is None:
            # Observation boundaries can revoke stability but cannot advance its clock.
            self.final_since, self.final_pair = None, None
        support_transfer = self.parameters.get('manipulation_revision') == SUPPORT_TRANSFER_REVISION
        manipulation_check = 'middle_grasp_support_transfer_release' if support_transfer else 'middle_grasp_lift_release'
        checks = {('initial_native_scene_measured' if final_state else 'initial_incomplete_supported_scene'): self.initial_eligible,
            'native_measurement_available': self.measurement_error is None}
        if not final_state:
            checks.update({manipulation_check: self._manipulation_released('middle'),
                'ordered_middle_placement': self.middle_placement is not None})
        checks.update({'selected_native_support_order': pair is not None,
            'released_and_stable_duration': self.final_since is not None and
                self._elapsed(self.final_since) + 1e-12 >= self.parameters['minimum_stability_duration_s']})
        if self.parameters['top_keys'] and not final_state:
            checks['top_grasp_after_middle_placement'] = bool(pair and pair[1] and
                self.histories[pair[1]]['acquired_step'] > self.middle_placement['step'])
        value = {'schema': 'private_ordered_stack_support_transfer_v1' if support_transfer else
                'private_ordered_stack_operation_v2', 'episode': self.episode,
            'snapshot_id': observation.snapshot_id, 'action_step': observation.action_step,
            'action': action, 'operation': OPERATION, 'task_success': bool(all(checks.values())),
            'checks': checks, 'selected_base': pair[0] if pair else None,
            'selected_top': pair[1] if pair else None, 'histories': self.histories,
            'middle_placement': self.middle_placement, 'events': self.events,
            'native_state': measured, 'measurement_error': self.measurement_error,
            'initial_difficulty_eligible': self.initial_difficulty_eligible,
            'controller_steps': self.sample_count, 'controller_dt_s': self.dt,
            'stable_controller_duration_s': self._elapsed(self.final_since),
            'stability_clock': 'consecutive native controller callbacks after env.step; not physics substeps',
            'configuration': self.config}
        middle_seen = any(event.get('event') == 'middle_supported_placement' for event in self.events)
        items = [{'id': 'middle_stable_placement', 'kind': 'checkpoint', 'completed': middle_seen},
            {'id': 'released_final_stack', 'kind': 'terminal_condition', 'completed': bool(checks['released_and_stable_duration'] and pair)}]
        if self.parameters['top_keys']:
            items.insert(1, {'id': 'top_after_middle', 'kind': 'checkpoint',
                'completed': self.ordered_top_seen})
        available = self.measurement_error is None and self.initial_eligible
        for item in items:
            item['evaluation_validity'] = 'valid' if available else 'unavailable'
        value.update(completion_items=items, evaluation_validity='complete' if available else 'unavailable',
            task_revision=self.parameters.get('task_revision') if support_transfer else TASK_REVISION,
            measured_capability='supplied_order_execution')
        if support_transfer:
            value.update(manipulation_revision=SUPPORT_TRANSFER_REVISION,
                manipulation_check='native_acquired_support_separation_then_release')
        if final_state:
            for key in ('histories', 'middle_placement', 'events'):
                value.pop(key)
            value.update(schema='private_ordered_stack_final_state_v1',
                task_revision=self.parameters.get('task_revision', FINAL_STATE_REVISION),
                manipulation_revision=FINAL_STATE_REVISION, measured_capability='released_final_support_order',
                manipulation_check='native_final_support_order_without_manipulation_history',
                completion_items=[{'id': 'released_final_stack', 'kind': 'terminal_condition',
                    'completed': value['task_success'],
                    'evaluation_validity': 'valid' if available else 'unavailable'}])
        from worldleague.tasks.completion import progress
        value.update(progress(value))
        (self.output / 'state.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        return value


def create_evaluator(session, controller, config, output):
    """Create a read-only native evaluator through the standard task episode hooks."""
    return OrderedStackEvaluator(session, controller, config, output)
