"""Native right-hand plate collection evidence for hidden-object episodes.

Used by worldleague scene --help.
"""
from __future__ import annotations


from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

MODULE = "worldleague.adapters.hidden_object_collection"
REVISION = "hidden_collection_20261020_v1"
OPERATION = "find_and_collect_on_right_hand_plate"


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def collection_result(state, targets, distractors, picked, *, stable_seconds, required_dwell):
    """Reduce current physical inventory, never cumulative visits or sightings."""
    present = set(state["supported_objects"])
    held = set(state["held_objects"])
    correct = sorted(present.intersection(targets).difference(held).intersection(picked))
    wrong = sorted(present.intersection(distractors))
    complete = len(correct) == len(targets)
    conditions = {
        "right_hand_holds_plate": state["right_holds_plate"],
        "plate_level": state["plate_level"],
        "plate_stable": state["plate_stable"],
        "left_hand_released": state["left_empty"],
        "all_targets_collected": complete,
        "no_distractors_on_plate": not wrong,
        "stable_terminal_dwell": stable_seconds >= required_dwell,
    }
    return {"task_success": all(conditions.values()), "checks": conditions,
        "collected_targets": correct, "distractors_on_plate": wrong,
        "collection_recall": len(correct) / len(targets),
        "stable_seconds": stable_seconds}


def _under(path, root):
    return str(path) == root or str(path).startswith(root + "/")


def measure_inventory(session, parameters, holding):
    """Measure free-body support/contact in the tray frame for either protocol."""
    robot = session.robot
    from worldleague.environments.mobile_tray import apparatus_object
    plate = apparatus_object(robot, session.env.scene, parameters["plate"]["name"])
    position, quaternion = (array(v) for v in plate.get_position_orientation())
    rotation = Rotation.from_quat(quaternion)
    linear, angular = array(plate.get_linear_velocity()), array(plate.get_angular_velocity())
    inner = parameters["plate"]["usable_bounds_local_m"]
    lower, upper = array(inner[0]), array(inner[1])
    supported, details = [], {}
    for name in parameters["targets"] + parameters.get("distractors", []):
        obj = session.env.scene.object_registry("name", name)
        if obj is None or obj.visual_only or obj.fixed_base:
            raise ValueError("Collection requires the exact movable collidable inventory.")
        points = [array(link.collision_boundary_points_world) for link in obj.links.values()
                  if link.collision_boundary_points_world is not None]
        if not points:
            raise ValueError("Native target collision boundary unavailable.")
        local = rotation.inv().apply(np.concatenate(points) - position)
        lo, hi = local.min(0), local.max(0)
        contact = (bool(plate.native_contacts_with(obj)) if hasattr(plate, "native_contacts_with") else
                   any(_under(getattr(c, side), plate.prim_path)
                       for c in obj.contact_list() for side in ("body0", "body1")))
        on_deck = (np.all(lo[:2] >= lower[:2] - .002)
                   and np.all(hi[:2] <= upper[:2] + .002)
                   and lower[2] - .006 <= lo[2] <= lower[2] + .015
                   and hi[2] <= upper[2])
        obj_position = array(obj.get_position_orientation()[0])
        relative_speed = float(np.linalg.norm(array(obj.get_linear_velocity())
            - linear - np.cross(angular, obj_position - position)))
        stable = relative_speed <= parameters["maximum_relative_speed_m_s"]
        ok = bool(contact and on_deck and stable and name not in holding.values())
        if ok:
            supported.append(name)
        details[name] = {"contact": bool(contact), "inside_usable_deck": bool(on_deck),
            "relative_speed_m_s": relative_speed, "supported": ok,
            "bounds_in_plate_frame": [lo.tolist(), hi.tolist()]}
    up = float(rotation.apply([0., 0., 1.])[2])
    return {"supported_objects": supported, "held_objects": [x for x in holding.values() if x],
        "plate_level": bool(up >= np.cos(np.deg2rad(parameters["maximum_plate_tilt_deg"]))),
        "plate_stable": bool(np.linalg.norm(linear) <= parameters["maximum_plate_speed_m_s"]
                             and np.linalg.norm(angular) <= parameters["maximum_plate_angular_speed_rad_s"]),
        "plate_pose": [position.tolist(), quaternion.tolist()], "target_details": details,
        "native_time_s": float(session.pipeline.og.sim.current_time)}


def _state(session, parameters):
    robot = session.robot
    hand = getattr(robot, "_ag_obj_in_hand", {})
    holding = {arm: getattr(hand.get(arm), "name", None) for arm in ("left", "right")}
    state = measure_inventory(session, parameters, holding)
    state.update(left_empty=holding["left"] is None and int(robot.is_grasping("left")) == -1,
                 left_holding=holding["left"],
                 right_holds_plate=bool(holding["right"] == parameters["plate"]["name"]
                                         and int(robot.is_grasping("right")) == 1))
    return state


class CollectionEvaluator:
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.parameters = config["parameters"]
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.picked = set()
        self.wrong_pickups = set()
        self.previous_collected = set()
        self.spills = []
        self.last_time = None
        self.stable_since = None
        self.state = None
        self.measure_state = _state
        self.reduce_result = collection_result
        self.holding_field = "left_holding"

    def begin_episode(self, observation):
        self.picked.clear()
        self.wrong_pickups.clear()
        self.spills.clear()
        self.previous_collected.clear()
        self.last_time = self.stable_since = None
        self.sample_step()

    def sample_step(self):
        state = self.measure_state(self.session, self.parameters)
        now = state["native_time_s"]
        if self.last_time is not None and now <= self.last_time:
            return
        held = state[self.holding_field]
        if held in self.parameters["targets"]:
            self.picked.add(held)
        if held in self.parameters.get("distractors", []):
            self.wrong_pickups.add(held)
        present = set(state["supported_objects"]).intersection(self.parameters["targets"])
        for name in self.previous_collected - present:
            self.spills.append({"target": name, "native_time_s": now})
        self.previous_collected = present
        candidate = self.reduce_result(state, self.parameters["targets"],
            self.parameters.get("distractors", []), self.picked,
            stable_seconds=0., required_dwell=0.)
        if candidate["task_success"]:
            if self.stable_since is None:
                self.stable_since = now
        else:
            self.stable_since = None
        self.last_time, self.state = now, state

    def record(self, observation, *, action=None):
        self.sample_step()
        duration = 0. if self.stable_since is None else self.last_time - self.stable_since
        result = self.reduce_result(self.state, self.parameters["targets"],
            self.parameters.get("distractors", []), self.picked, stable_seconds=duration,
            required_dwell=self.parameters["terminal_dwell_s"])
        result.update(schema="native_hidden_collection_v1", snapshot_id=observation.snapshot_id,
            action_step=observation.action_step, evaluation_validity="complete",
            terminal_conditions_satisfied=result["task_success"],
            completion_items=[{"id": "collect:" + name, "kind": "current_physical_inventory",
                "completed": name in result["collected_targets"], "evaluation_validity": "valid"}
                for name in self.parameters["targets"]],
            wrong_pickups=sorted(self.wrong_pickups), spills=self.spills,
            native_state=self.state, configuration=self.config)
        (self.output / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
