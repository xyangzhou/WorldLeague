"""Separate single-arm mobile-tray collection protocol and physical evaluator.

Used by worldleague scene --help.
"""

import numpy as np

from worldleague.adapters import hidden_object_collection as collection
from worldleague.adapters import wia_room_collection as room
from worldleague.environments.mobile_tray import PROTOCOL, array, evaluator_grasp, measure_mount, mount_tray, apparatus_object


MODULE = "worldleague.adapters.mobile_tray_collection"
OPERATIONS = {"visit_rooms_and_collect_on_mobile_tray", "search_and_collect_on_mobile_tray"}
PARENTS = {collection.MODULE: collection, room.MODULE: room}


def parent(parameters):
    if parameters.get("protocol") != PROTOCOL or parameters.get("parent_module") not in PARENTS:
        raise ValueError("Use an explicit supported mobile-tray parent protocol.")
    return PARENTS[parameters["parent_module"]]


def configure_scene(config, parameters):
    result = parent(parameters).configure_scene(config, parameters)
    config["objects"] = [obj for obj in config["objects"] if obj["name"] != "collection_initial_support"]
    return {**result, "protocol": PROTOCOL, "support_mode": "native_chassis_mount"}


def postprocess_joint_world(session, parameters):
    return {**parent(parameters).postprocess_joint_world(session, parameters), "protocol": PROTOCOL}


def initialize_apparatus(session, parameters):
    if not getattr(session, "_category_initialization_active", False):
        raise RuntimeError("Mobile-tray mounting is restricted to explicit initialization.")
    config = session.embodiment_config
    support = config["eval"].get("mobile_tray_support")
    if support is None or config["eval"]["arm_names"] != list(session.robot.arm_names):
        raise ValueError("The selected robot must explicitly declare the native single-arm mount.")
    mount = mount_tray(session.robot, session.env, support["config"], parameters["plate"]["name"])
    robot = session.robot
    action = np.zeros(robot.action_dim, dtype=np.float32)
    control = robot.get_control_dict()
    for name, controller in robot.controllers.items():
        action[array(robot.controller_action_idx[name]).astype(int)] = array(controller.compute_no_op_action(control))
    import torch
    for _ in range(30):
        session.step_control(torch.as_tensor(action))
    measurement = measure_mount(robot, session.env, session.pipeline.og.sim)
    grasp = evaluator_grasp(robot, session.env.scene, mount["working_arm"],
                            native_time_s=float(session.pipeline.og.sim.current_time))
    result = {"accepted": measurement["tray_mount_valid"] and grasp["empty"],
              "protocol": PROTOCOL, "support_mode": "native_chassis_mount", "stage": "complete",
              "mount": mount, "measurement": measurement, "working_arm_grasp": grasp,
              "native_layout_qualified": False, "manual_cargo_attachment": False}
    session.hidden_collection_initialization = result
    return result


def initialize_scene(session, parameters):
    return parent(parameters).initialize_scene(session, parameters)


def native_state(session, parameters):
    robot = session.robot
    mount = measure_mount(robot, session.env, session.pipeline.og.sim)
    arm = mount["working_arm"]
    grasp = evaluator_grasp(robot, session.env.scene, arm,
                            native_time_s=float(session.pipeline.og.sim.current_time))
    state = collection.measure_inventory(session, parameters, {arm: grasp["holding"]})
    state.update(protocol=PROTOCOL, tray_mount_valid=mount["tray_mount_valid"],
                 working_arm=arm, working_arm_empty=grasp["empty"], working_arm_holding=grasp["holding"],
                 grasp_measurement=grasp, mount_measurement=mount)
    if parent(parameters) is room:
        state["native_region"] = room._region(session, robot.get_position_orientation()[0])
        # Include the real carrying apparatus footprint when crediting room entry.
        bodies = [robot.links[robot.base_footprint_link_name]] + [
            apparatus_object(robot, session.env.scene, row["name"]) for row in robot.mobile_tray_mount["apparatus"]]
        footprint = []
        for body in bodies:
            low, high = map(array, body.aabb)
            footprint.extend(room._region(session, [x, y])
                for x in (low[0], (low[0] + high[0]) / 2, high[0])
                for y in (low[1], (low[1] + high[1]) / 2, high[1]))
        state["room_occupancy_valid"] = state["native_region"] is not None and all(
            value == state["native_region"] for value in footprint)
        state["object_rooms"] = {row["name"]: room._region(session, room._object(session, row).get_position_orientation()[0])
            for row in parameters["layout"].get("native_inventory", []) + parameters["layout"].get("added_inventory", [])}
    return state


def collection_result(state, targets, distractors, picked, *, stable_seconds, required_dwell):
    # Reuse the parent's inventory/recovery calculation while explicitly replacing
    # the two embodiment-dependent predicates. No held-tray claim escapes this reducer.
    result = collection.collection_result({**state, "right_holds_plate": state["tray_mount_valid"],
        "left_empty": state["working_arm_empty"]}, targets, distractors, picked,
        stable_seconds=stable_seconds, required_dwell=required_dwell)
    checks = result["checks"]
    checks["tray_mount_valid"] = checks.pop("right_hand_holds_plate")
    checks["manipulation_arm_released"] = checks.pop("left_hand_released")
    result.update(protocol=PROTOCOL, support_mode="native_chassis_mount")
    return result


def create_evaluator(session, controller, config, output):
    if config["operation"] not in OPERATIONS:
        raise ValueError("Use a separately declared mobile-tray collection operation.")
    source = parent(config["parameters"])
    evaluator_type = room.RoomCollectionEvaluator if source is room else collection.CollectionEvaluator
    evaluator = evaluator_type(session, controller,
        {**config, "operation": config["parameters"]["parent_operation"]}, output)
    evaluator.config = config
    evaluator.measure_state = native_state
    evaluator.reduce_result = collection_result
    evaluator.holding_field = "working_arm_holding"
    return evaluator
