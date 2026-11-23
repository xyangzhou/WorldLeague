"""XR-1 RoboCasa365 measurements, temporal history and executable action mapping.

Used by worldleague continuous --help.
"""
from collections import deque

import numpy as np


CAMERA_ROLES = ("agentview_left", "agentview_right", "wrist")


def axis_angle(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quaternion = quaternion / norm
    if quaternion[3] < 0:
        quaternion = -quaternion
    sine = np.linalg.norm(quaternion[:3])
    if sine < 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2 * np.arctan2(sine, np.clip(quaternion[3], -1., 1.))
    return (quaternion[:3] / sine * angle).astype(np.float32)


def state14(measurements):
    result = np.concatenate((
        np.asarray(measurements["state.end_effector_position_relative"], dtype=np.float32),
        axis_angle(measurements["state.end_effector_rotation_relative"]),
        np.asarray(measurements["state.gripper_qpos"], dtype=np.float32),
        np.asarray(measurements["state.base_position"], dtype=np.float32),
        axis_angle(measurements["state.base_rotation"]))).astype(np.float32)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError("XR-1 RoboCasa365 requires 14 finite native state values.")
    return result


class ObservationHistory:
    """Sample actual native control boundaries, repeating only initial unavailable history."""

    def __init__(self, length, interval_steps, *, allow_sparse=False):
        self.length, self.interval_steps = length, interval_steps
        self.allow_sparse = allow_sparse
        self.samples = deque(maxlen=(length - 1) * interval_steps + 1)

    def append(self, observation):
        if self.samples:
            gap = observation["control_step"] - self.samples[-1]["control_step"]
            if gap <= 0 or (not self.allow_sparse and gap != 1):
                raise ValueError("XR-1 history requires increasing consecutive native control boundaries unless sparse capture is selected.")
        self.samples.append({"state": np.asarray(observation["state"], dtype=np.float32).copy(),
                             "images": {key: np.asarray(observation["images"][key]).copy() for key in CAMERA_ROLES},
                             "control_step": observation["control_step"],
                             "simulation_time_s": observation["simulation_time_s"]})

    def sample(self, prompt):
        if not self.samples:
            raise ValueError("XR-1 history cannot be sampled before reset.")
        items = list(self.samples)
        last = items[-1]["control_step"]
        steps = [max(0, last - (self.length - 1 - i) * self.interval_steps) for i in range(self.length)]
        by_step = {item["control_step"]: item for item in items}
        missing = set(steps) - by_step.keys()
        if missing:
            raise ValueError(f"XR-1 history is missing required native control samples: {sorted(missing)}")
        selected = [by_step[step] for step in steps]
        return {"state": np.stack([item["state"] for item in selected]),
                "images": {role: np.stack([item["images"][role] for item in selected]) for role in CAMERA_ROLES},
                "prompt": prompt, "history_control_steps": steps,
                "history_simulation_times_s": [item["simulation_time_s"] for item in selected]}


def required_history_captures(start, chunk_steps, execution_horizon, budget, length, interval):
    """Schedule only current/future replan history frames inside this chunk.

    Look ahead one history window so a short or odd final chunk also retains
    exact prior-control frames instead of repeating an incorrectly timed image.
    """
    end = min(start + chunk_steps, budget)
    boundary, span, required = end, (length - 1) * interval, set()
    while True:
        required.update(step for i in range(length)
                        if start < (step := max(0, boundary - i * interval)) <= end)
        if boundary == budget or boundary - end >= span:
            break
        boundary = min(boundary + execution_horizon, budget)
    return required


def native_measurements(robot):
    """Keep the source grip-site position, hand-body orientation and support-center origin."""
    from scipy.spatial.transform import Rotation

    def values(tensor):
        return tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)

    support_position, support_quaternion = map(values, robot.links["mobilebase0_support"].get_position_orientation())
    grip_position, _ = map(values, robot.links["gripper0_right_eef"].get_position_orientation())
    _, hand_quaternion = map(values, robot.links["robot0_right_hand"].get_position_orientation())
    support_rotation = Rotation.from_quat(support_quaternion)
    finger_indices = values(robot.gripper_control_idx["right"]).astype(int)
    positions = values(robot.get_joint_positions())
    return {"state.end_effector_position_relative": support_rotation.inv().apply(grip_position - support_position),
            "state.end_effector_rotation_relative": (support_rotation.inv() * Rotation.from_quat(hand_quaternion)).as_quat(),
            "state.gripper_qpos": positions[finger_indices], "state.base_position": support_position,
            "state.base_rotation": support_quaternion}


class PandaOmronActionAdapter:
    """Map 12 model channels to native controller groups and hybrid controller state."""

    policy_action_dim = 12

    def __init__(self, robot, source_physics):
        self.robot = robot
        self.indices = {name: np.asarray(value.detach().cpu() if hasattr(value, "detach") else value, dtype=int)
                        for name, value in robot.controller_action_idx.items()}
        self.gripper_progress = np.zeros(2, dtype=np.float64)
        self.gripper_speed = .2  # Pinned PandaGripper.speed, not a physical joint speed.
        actuators = {item["joint"]: item for item in source_physics["actuators"]}
        base_names = robot.base_joint_names
        self.base_limits = np.array([actuators[name]["ctrlrange"] for name in base_names])
        self.gripper_limits = np.array([actuators[name]["ctrlrange"] for name in robot.finger_joint_names["right"]])
        joints = robot.get_joint_positions()
        self.initial_yaw = float(joints[robot.base_control_idx[2]])

    def convert(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("XR-1 requires a finite executable 12-dimensional action.")
        command = np.zeros(self.robot.action_dim, dtype=np.float32)
        command[self.indices["arm_right"]] = action[:6]
        command[self.indices["trunk"]] = action[10]
        close = bool(action[6] >= .5)
        self.gripper_progress = np.clip(self.gripper_progress + np.array([-1., 1.]) * self.gripper_speed * (1 if close else -1), -1., 1.)
        low, high = self.gripper_limits[:, 0], self.gripper_limits[:, 1]
        command[self.indices["gripper_right"]] = low + (self.gripper_progress + 1) * .5 * (high - low)
        # Match the pinned MobileBaseJointVelocityController's inverse-yaw mapping.
        yaw = float(self.robot.get_joint_positions()[self.robot.base_control_idx[2]]) - self.initial_yaw
        x, y, turn = action[7:10]
        base = np.array([x * np.cos(yaw) + y * np.sin(yaw), -x * np.sin(yaw) + y * np.cos(yaw), turn])
        command[self.indices["base"]] = np.clip(base, self.base_limits[:, 0], self.base_limits[:, 1])
        mode = bool(action[11] >= .5)
        self.robot.controllers["arm_right"].base_mode = mode
        return command, {"control_mode": "base" if mode else "arm", "gripper_close": close,
                         "gripper_progress": self.gripper_progress.tolist(), "native_command": command.tolist()}
