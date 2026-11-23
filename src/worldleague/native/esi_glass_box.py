"""Private hinged/sliding-box checks; use run_esi_robot.py --task-file FILE.

Used by worldleague scene --help.
"""
from __future__ import annotations

import numpy as np

from .utils import _array
from .esi_task_evaluation import GraspLiftEvaluator


def hinge_angle_deg(box, name):
    return float(np.rad2deg(_array(box.joints[name].get_state()[0]).reshape(-1)[0]))


class OpenExtractLiftEvaluator(GraspLiftEvaluator):
    def __init__(self, session, controller, config, output):
        basic = {key: config[key] for key in ("scene", "room", "target", "minimum_lift_m")}
        super().__init__(session, controller, dict(basic, type="grasp_lift"), output)
        self.config = dict(config)
        self.box = session.env.scene.object_registry("name", config["container"])
        self.opened = False
        self.initially_closed = False
        self.max_angle_deg = 0.

    def begin_episode(self, observation):
        self.opened = False
        self.max_angle_deg = 0.
        self.initially_closed = abs(hinge_angle_deg(self.box, self.config["hinge_joint"])) < 3.
        super().begin_episode(observation)

    def additional_evidence(self, target):
        self.sample_step()
        angle = hinge_angle_deg(self.box, self.config["hinge_joint"])
        rim = float(_array(self.box.get_position_orientation()[0])[2] + self.config["rim_height_m"])
        extracted = target["aabb_min"][2] >= rim + self.config["extraction_clearance_m"]
        return {"door_initially_closed": self.initially_closed, "door_opened": self.opened,
                "apple_above_box_rim": bool(extracted)}, {
                    "door_angle_deg": angle, "maximum_door_angle_deg": self.max_angle_deg,
                    "box_rim_world_z": rim}

    def sample_step(self):
        """Read-only physics sampling captures opening between action boundaries."""
        if self.initial is None:
            return
        angle = hinge_angle_deg(self.box, self.config["hinge_joint"])
        self.max_angle_deg = max(self.max_angle_deg, angle)
        self.opened = self.opened or angle >= self.config["minimum_open_angle_deg"]


def sliding_displacement_m(box, name):
    """Native prismatic joint positions are linear distances in metres."""
    return float(_array(box.joints[name].get_state()[0]).reshape(-1)[0])


class SlideExtractLiftEvaluator(OpenExtractLiftEvaluator):
    """Reuse held-extraction scoring, replacing the angular gate with travel."""

    def begin_episode(self, observation):
        self.opened = False
        self.max_displacement_m = 0.
        self.initially_closed = abs(sliding_displacement_m(self.box, self.config["slide_joint"])) < .003
        GraspLiftEvaluator.begin_episode(self, observation)

    def additional_evidence(self, target):
        self.sample_step()
        displacement = sliding_displacement_m(self.box, self.config["slide_joint"])
        rim = float(_array(self.box.get_position_orientation()[0])[2] + self.config["rim_height_m"])
        extracted = target["aabb_min"][2] >= rim + self.config["extraction_clearance_m"]
        return {"door_initially_closed": self.initially_closed, "door_opened": self.opened,
                "apple_above_box_rim": bool(extracted)}, {
                    "door_slide_displacement_m": displacement,
                    "maximum_door_slide_displacement_m": self.max_displacement_m,
                    "box_rim_world_z": rim}

    def sample_step(self):
        if self.initial is None:
            return
        displacement = sliding_displacement_m(self.box, self.config["slide_joint"])
        self.max_displacement_m = max(self.max_displacement_m, displacement)
        self.opened = self.opened or displacement >= self.config["minimum_open_displacement_m"]
