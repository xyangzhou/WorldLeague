"""Read native robot poses and control tolerances. Used by worldleague continuous and scene."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from worldleague.native.utils import _array, _main_thread
from worldleague.resources import CONFIG_ROOT


@dataclass(frozen=True)
class BaseControlConfig:
    translation_m: float
    rotation_deg: float
    translation_tolerance_m: float
    rotation_tolerance_deg: float
    max_translation_speed_m_s: float
    max_rotation_speed_deg_s: float
    timeout_sim_s: float
    timeout_wall_s: float | None
    settle_steps: int
    stop_steps: int
    progress_timeout_sim_s: float = 1.5
    progress_min_translation_m: float = 0.001
    progress_min_rotation_deg: float = 0.1
    recovery_settle_steps: int = 6
    recovery_max_steps: int = 30
    recovery_max_joint_step_rad: float = 0.001
    recovery_max_finger_step_m: float = 0.0005
    recovery_max_base_step_m: float = 0.0005
    recovery_max_base_yaw_step_deg: float = 0.05

    def __post_init__(self):
        object.__setattr__(self, 'timeout_wall_s', None)

    @classmethod
    def load(cls, path: str | Path = CONFIG_ROOT / "esi_base.json") -> BaseControlConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
            # Historical launch files may still contain these inert keys.
            for key in ("inactive_joint_tolerance", "right_arm_hold_check"):
                values.pop(key, None)
            return cls(**values)


def base_pose(robot) -> dict:
    _main_thread()
    position, quaternion = (np.asarray(_array(v), dtype=float).copy()
                            for v in robot.get_position_orientation())
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or np.linalg.norm(quaternion) == 0):
        raise RuntimeError("R1 base pose must contain finite position and a valid xyzw quaternion.")
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return {"position": position.tolist(), "quaternion_xyzw": quaternion.tolist(), "yaw_rad": yaw}


@dataclass(frozen=True)
class GripperControlConfig:
    grasp_mode: str = "assisted"
    joint_tolerance_m: float = 0.001
    minimum_settle_sim_s: float = 0.4
    settle_steps: int = 6
    timeout_sim_s: float = 120.0
    timeout_wall_s: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'timeout_wall_s', None)

    @classmethod
    def load(cls, path=CONFIG_ROOT / "esi_gripper.json"):
        return cls(**json.loads(Path(path).read_text()))


def ee_pose(robot, working_arm) -> dict:
    """Debug pose: actual base-relative control frame and world EE pose."""
    _main_thread()
    control = robot.get_control_dict()
    pos = np.asarray(_array(control[f"eef_{working_arm}_pos_relative"]), dtype=float)
    quat = np.asarray(_array(control[f"eef_{working_arm}_quat_relative"]), dtype=float)
    world_pos, world_quat = (_array(v) for v in robot.get_eef_pose(working_arm))
    if (pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(pos).all()
            or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0
            or not np.isfinite(world_pos).all() or not np.isfinite(world_quat).all()):
        raise RuntimeError("Working-arm EE pose must be finite with a valid quaternion.")
    return {"base_position": pos.tolist(), "base_quaternion_xyzw": quat.tolist(),
            "world_position": world_pos.tolist(), "world_quaternion_xyzw": world_quat.tolist()}
