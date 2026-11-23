"""Native physical carrying support for explicitly selected single-arm variants.

Used by worldleague continuous --help.
"""
import numpy as np
from scipy.spatial.transform import Rotation


PROTOCOL = "single_arm_mobile_tray_v1"


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def evaluator_grasp(robot, scene, arm, *, native_time_s=None):
    """Identify physical grasps from closing effort, opposing contacts and relative motion.

    This evaluator-only accessor never creates an attachment or supplies object
    identities to the policy. The native generic JointController reports UNKNOWN.
    """
    if arm not in robot.arm_names:
        raise ValueError("The selected working arm is unavailable.")
    if robot.grasping_mode != "physical":
        held = getattr(robot._ag_obj_in_hand.get(arm), "name", None)
        return {"holding": held if int(robot.is_grasping(arm)) == 1 else None,
                "empty": held is None and int(robot.is_grasping(arm)) == -1,
                "measurement": "native_assisted_grasp_state"}
    groups = robot.physical_grasp_contact_link_names[arm]
    paths = [[robot.links[name].prim_path for name in group] for group in groups]
    controller = robot.controllers[f"gripper_{arm}"]
    goal = controller._goal
    indices = array(robot.gripper_control_idx[arm]).astype(int)
    current = array(robot.get_joint_positions())[indices]
    target = current if goal is None else array(goal["target"])
    closing = bool(target[0] < current[0] - .0005 and target[1] > current[1] + .0005)
    candidates, contacts, speeds, constrained = [], {}, {}, []
    samples = getattr(robot, "_wia_evaluator_grasp_samples", {})
    robot._wia_evaluator_grasp_samples = samples
    for obj in scene.objects:
        if obj is robot or obj.visual_only or obj.fixed_base:
            continue
        touched = set()
        for contact in obj.contact_list():
            bodies = [str(contact.body0), str(contact.body1)]
            for side, group in enumerate(paths):
                if any(body == root or body.startswith(root + "/") for body in bodies for root in group):
                    touched.add(side)
        if touched:
            contacts[obj.name] = sorted(touched)
        if len(touched) != 2:
            samples.pop((arm, obj.name), None)
        if len(touched) == 2:
            speed = float(np.linalg.norm(array(obj.get_linear_velocity()) - array(robot.eef_links[arm].get_linear_velocity())))
            if native_time_s is not None:
                eef_position, eef_orientation = map(array, robot.eef_links[arm].get_position_orientation())
                relative = Rotation.from_quat(eef_orientation).inv().apply(array(obj.get_position_orientation()[0]) - eef_position)
                key = (arm, obj.name)
                previous = samples.get(key)
                speed = None
                if previous is not None:
                    dt = native_time_s - previous["time"]
                    speed = previous["speed"] if dt == 0 else float(np.linalg.norm(relative - previous["position"]) / dt) if dt > 0 else None
                samples[key] = {"time": native_time_s, "position": relative, "speed": speed}
            speeds[obj.name] = speed
            constrained.append(obj.name)
            if closing:
                if speed is not None and speed <= .05:
                    candidates.append(obj.name)
    return {"holding": candidates[0] if len(candidates) == 1 else None,
            "empty": not constrained, "ambiguous": len(constrained) > 1, "closing": closing,
            "contacts": contacts, "relative_speeds_m_s": speeds,
            "relative_speed_method": "eef_frame_native_pose_difference" if native_time_s is not None else "native_velocity",
            "measurement": "opposing_native_contacts_closing_and_relative_speed"}


def support_objects(spec):
    if spec.get("attachment_mode") == "chassis_compound_body_v1":
        return []  # The support geometry is already part of the robot asset.
    return [{"type": "PrimitiveObject", "primitive_type": "Cube", "name": part["name"],
             "size": 1., "scale": part["dimensions_m"], "fixed_base": False, "visual_only": False,
             "position": [100. + i, 100., 2.], "rgba": [.2, .25, .3, 1.]}
            for i, part in enumerate(spec["parts"])]


def mount_tray(robot, environment, spec, tray_name):
    """Install the declared apparatus once, with excluded native fixed constraints."""
    import torch
    from omnigibson.utils.usd_utils import create_joint

    if len(robot.arm_names) != 1 or spec["protocol"] != PROTOCOL:
        raise ValueError("Mounted collection requires one explicitly declared manipulation arm.")
    if hasattr(robot, "mobile_tray_mount"):
        raise RuntimeError("A mobile tray can only be initialized once per native robot.")
    if spec.get("attachment_mode") == "chassis_compound_body_v1":
        from worldleague.environments.compound_tray import initialize_compound_frames
        return initialize_compound_frames(robot, environment, spec, tray_name)
    support = robot.links[spec["support_link"]]
    position, quaternion = map(array, support.get_position_orientation())
    rotation = Rotation.from_quat(quaternion)
    records = []

    def attach(obj, local_position, mass, label):
        if obj is None or obj.fixed_base or obj.visual_only:
            raise ValueError("Mounted apparatus must be a native dynamic collidable body.")
        if obj.n_dof:
            raise ValueError("Mounted apparatus must have no movable internal joints.")
        if mass is not None:
            obj.root_link.mass = float(mass)
        obj.set_position_orientation(position=(position + rotation.apply(local_position)).tolist(),
                                     orientation=quaternion.tolist())
        obj.keep_still()
        joint = create_joint(prim_path=f"{obj.prim_path}/mobile_mount", joint_type="FixedJoint",
                             body0=support.prim_path, body1=obj.root_link.prim_path,
                             exclude_from_articulation=True,
                             joint_frame_in_parent_frame_pos=torch.tensor(local_position, dtype=torch.float32),
                             joint_frame_in_parent_frame_quat=torch.tensor([0., 0., 0., 1.]),
                             joint_frame_in_child_frame_pos=torch.zeros(3),
                             joint_frame_in_child_frame_quat=torch.tensor([0., 0., 0., 1.]),
                             break_force=spec["break_force_n"], break_torque=spec["break_torque_nm"])
        records.append({"name": obj.name, "joint_path": str(joint.GetPath()),
                        "position_support_m": list(local_position), "orientation_support_xyzw": [0., 0., 0., 1.],
                        "link_masses_kg": {name: float(link.mass) for name, link in obj.links.items()},
                        "role": label})

    for part in spec["parts"]:
        attach(environment.scene.object_registry("name", part["name"]), part["position_support_m"],
               part["mass_kg"], "support")
    tray = environment.scene.object_registry("name", tray_name)
    # Measure the actual tray collision extent; do not guess its asset-origin offset.
    origin, orientation = map(array, tray.get_position_orientation())
    points = [array(link.collision_boundary_points_world) for link in tray.links.values()
              if link.collision_boundary_points_world is not None]
    if not points:
        raise ValueError("Native tray collision geometry is required for mounting.")
    local = Rotation.from_quat(orientation).inv().apply(np.concatenate(points) - origin)
    low, high = local.min(0), local.max(0)
    tray_position = np.asarray(spec["tray_center_support_xy_m"] + [spec["deck_top_support_z_m"] - low[2] + .001])
    attach(tray, tray_position.tolist(), spec["tray_mass_kg"], "tray")
    robot.mobile_tray_mount = {"protocol": PROTOCOL, "support_link": spec["support_link"],
                              "working_arm": robot.arm_names[0], "apparatus": records,
                              "tray_name": tray_name, "tray_collision_bounds_local_m": [low.tolist(), high.tolist()],
                              "specification": spec, "cargo_attachment": False}
    return robot.mobile_tray_mount


def measure_mount(robot, environment, simulator):
    """Read constraint identity and relative poses without changing physics state."""
    receipt = robot.mobile_tray_mount
    if receipt.get("attachment_mode") == "chassis_compound_body_v1":
        from worldleague.environments.compound_tray import measure_compound
        return measure_compound(robot, simulator)
    spec = receipt["specification"]
    support = robot.links[receipt["support_link"]]
    origin, quaternion = map(array, support.get_position_orientation())
    rotation = Rotation.from_quat(quaternion)
    measured = []
    for part in receipt["apparatus"]:
        obj = environment.scene.object_registry("name", part["name"])
        position, orientation = map(array, obj.get_position_orientation())
        local_position = rotation.inv().apply(position - origin)
        local_rotation = rotation.inv() * Rotation.from_quat(orientation)
        joint = simulator.stage.GetPrimAtPath(part["joint_path"])
        enabled = bool(joint and joint.GetAttribute("physics:jointEnabled").Get())
        parent_ok = bool(joint and [str(p) for p in joint.GetRelationship("physics:body0").GetTargets()] == [support.prim_path])
        child_ok = bool(joint and [str(p) for p in joint.GetRelationship("physics:body1").GetTargets()] == [obj.root_link.prim_path])
        error = float(np.linalg.norm(local_position - part["position_support_m"]))
        angle = float(local_rotation.magnitude())
        measured.append({"name": obj.name, "joint_enabled": enabled, "parent_matches": parent_ok,
                         "child_matches": child_ok, "relative_position_error_m": error,
                         "relative_rotation_error_rad": angle,
                         "valid": enabled and parent_ok and child_ok and error <= spec["position_tolerance_m"]
                                  and angle <= spec["rotation_tolerance_rad"]})
    return {"tray_mount_valid": all(row["valid"] for row in measured), "apparatus": measured,
            "protocol": PROTOCOL, "working_arm": receipt["working_arm"]}


def apparatus_object(robot, scene, name):
    """Resolve an explicit rigid component or the original free-body apparatus."""
    frames = getattr(robot, "compound_tray_frames", {})
    return frames[name] if name in frames else scene.object_registry("name", name)
