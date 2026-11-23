"""Configure and measure source-aligned reflection embodiment without moving optics.

Used by worldleague scene --help.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from worldleague.tasks.interaction_setup import array

ACTION_PROFILE_VERSION = 'reflection_sensor_profile_v3'


def _revised_profile(profile):
    return profile.get('version') == ACTION_PROFILE_VERSION


def _alignment_identity(session, profile):
    return {'reset_id': session.native_reset_id, 'profile': deepcopy(profile)}


def _alignment_completed(session, profile):
    receipt = getattr(session, 'reflection_alignment_initialization', {})
    return receipt.get('completed') is True and all(receipt.get(key) == value
        for key, value in _alignment_identity(session, profile).items())


def profile_from_definition(definition):
    runtime = getattr(definition, 'runtime', None)
    if runtime is not None and getattr(runtime, 'reflection_sensor_profile', None):
        return deepcopy(runtime.reflection_sensor_profile)
    private = getattr(definition, 'private', None)
    parameters = getattr(getattr(private, 'evaluation', None), 'parameters', {})
    return deepcopy(parameters.get('reflection_sensor_profile'))


def configure_embodiment(session, config):
    adapter = getattr(session, 'scene_adapter', None)
    loaded = getattr(session, 'loaded_task', None) or getattr(adapter, 'task', None)
    profile = profile_from_definition(loaded.definition) if loaded is not None else None
    if not profile:
        return
    from scipy.spatial.transform import Rotation
    scale = profile.get('robot_uniform_scale') or 1.
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Reflection scaling requires a positive uniform scalar.')
    robots = config.get('robots', [])
    if len(robots) != 1 or robots[0]['type'] != 'R1':
        raise ValueError('Reflection requires the original mounted R1 robot.')
    forward = Rotation.from_quat(profile['quaternion_xyzw']).apply([0., 0., -1.])
    yaw = math.atan2(forward[1], forward[0])
    floor = profile['source_floor_reference']['proposed_z_m']
    position = [*profile['position_m'][:2], floor]
    robots[0].update(scale=float(scale), position=position,
        orientation=[0., 0., math.sin(yaw / 2), math.cos(yaw / 2)], pose_frame='world')
    # Explicit reflection exception changes intrinsics, never camera mounting.
    session.head_pitch_deg = 0.
    session.reflection_sensor_profile = profile


def align_initial_robot(session):
    """Align whole robot yaw/XY once, retaining its floor-level root and joints."""
    profile = getattr(session, 'reflection_sensor_profile', None)
    if profile is None:
        return
    if _revised_profile(profile):
        # Shared interaction setup can route here after the explicit hook.
        # Return the exact same attempt; never realign or overwrite its receipt.
        if _alignment_completed(session, profile):
            return session.reflection_robot_alignment
        if not getattr(session, '_reflection_initialization_hook_active', False):
            raise RuntimeError('Revised reflection alignment requires its explicit initialization hook.')
    from scipy.spatial.transform import Rotation
    from omnigibson.utils.usd_utils import PoseAPI
    robot = session.env.robots[0]
    head = next(sensor for name, sensor in robot.sensors.items() if 'eyes' in name or 'head' in name)
    target = array(profile['position_m'])
    wanted = Rotation.from_quat(profile['quaternion_xyzw']).apply([0., 0., -1.])
    initial = {'robot_pose': [array(v).tolist() for v in robot.get_position_orientation()],
        'head_mount': [array(v).tolist() for v in head.get_position_orientation(frame='parent')]}
    # R1's fixed native eye mount looks down. Correct its pitch using the real
    # third torso joint, whose installed R1 axis is negative world Y at the
    # neutral posture. This uses its existing position controller, not a camera
    # transform or a joint-position setter. Final measured pose remains a gate.
    from worldleague.runtime.sim.esi_base import R1BaseController, BaseControlConfig
    session.robot = robot
    hold = R1BaseController(session, BaseControlConfig.load())
    dofs = array(robot.controllers['trunk'].dof_idx).astype(int)
    lower, upper = [array(value)[dofs] for value in robot.control_limits['position']]
    joint_trace = []
    for _ in range(2):
        current_forward = Rotation.from_quat(array(head.get_position_orientation()[1])).apply([0., 0., -1.])
        elevation_error = math.asin(float(np.clip(wanted[2], -1, 1))) - math.asin(float(np.clip(current_forward[2], -1, 1)))
        if abs(elevation_error) < math.radians(.1):
            break
        q = array(robot.get_joint_positions())[dofs]
        target_q = q.copy()
        target_q[2] += elevation_error
        if np.any(target_q <= lower + .02) or np.any(target_q >= upper - .02):
            raise ValueError('Source head pitch requires a torso posture outside native joint limits.')
        for index in range(150):
            q = array(robot.get_joint_positions())[dofs]
            error = target_q - q
            hold.hold_action[hold.indices['trunk']] = q + np.clip(error, -.4 * hold.dt, .4 * hold.dt)
            hold.step_base_debug()
            joint_trace.append({'joints_rad': q.tolist(), 'target_rad': target_q.tolist()})
            if np.max(np.abs(error)) < .001:
                break
        hold.hold_action[hold.indices['trunk']] = array(robot.get_joint_positions())[dofs]
        hold.stop()
    initial['native_torso_alignment'] = {'controller': 'JointController', 'control_trace': joint_trace,
        'joint_limits_rad': [lower.tolist(), upper.tolist()], 'camera_mount_changed': False}
    for _ in range(2):
        hp, hq = (array(v) for v in head.get_position_orientation())
        rp, rq = (array(v) for v in robot.get_position_orientation())
        actual = Rotation.from_quat(hq).apply([0., 0., -1.])
        delta = math.atan2(wanted[1], wanted[0]) - math.atan2(actual[1], actual[0])
        rotation = Rotation.from_euler('z', delta)
        rp[:2] += target[:2] - hp[:2]
        robot.set_position_orientation(position=rp.tolist(), orientation=(rotation * Rotation.from_quat(rq)).as_quat().tolist())
        PoseAPI.invalidate()
    # Do not correct Z by floating the base or rotate the camera to hide pitch error.
    head.focal_length = float(head.horizontal_aperture) / (2. * math.tan(math.radians(profile['horizontal_fov_deg']) / 2.))
    session.reflection_robot_alignment = initial
    return initial
