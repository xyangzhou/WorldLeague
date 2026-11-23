"""PandaOmron registration for the installed OmniGibson 3.7 API.

Used by worldleague continuous --help.
"""
from copy import deepcopy
from functools import cached_property
import json
from pathlib import Path

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.controllers.joint_controller import JointController
from omnigibson.controllers.osc_controller import OperationalSpaceController
from omnigibson.robots.articulated_trunk_robot import ArticulatedTrunkRobot
from omnigibson.robots.locomotion_robot import LocomotionRobot
from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.utils.python_utils import classproperty
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

from worldleague.resources import ROOT
from worldleague.environments.source_collision_filters import source_topology_collision_pairs


class PandaOmronOSCController(OperationalSpaceController):
    """Keep the source's desired-goal accumulation during mobile control mode."""

    base_mode = False

    def compute_control(self, goal_dict, control_dict):
        return super().compute_control(goal_dict, control_dict) - (
            control_dict["panda_passive_damping"][self.dof_idx] * control_dict["joint_velocity"][self.dof_idx])

    def _update_goal(self, command, control_dict):
        result = super()._update_goal(command, control_dict)
        if self.base_mode and self._goal is not None:
            result["target_pos"] = self._goal["target_pos"] + command[:3]
            result["target_ori_mat"] = cb.T.quat2mat(cb.T.axisangle2quat(command[3:6])) @ self._goal["target_ori_mat"]
        return result


class PandaOmronTorsoController(JointController):
    """Use native mass-scaled joint PD plus the source's bias compensation."""

    def compute_control(self, goal_dict, control_dict):
        return super().compute_control(goal_dict, control_dict) - (
            control_dict["panda_passive_damping"][self.dof_idx] * control_dict["joint_velocity"][self.dof_idx])


class PandaOmron(ArticulatedTrunkRobot, LocomotionRobot):
    """Complete source assembly, with three physical planar base degrees of freedom."""

    def __init__(self, name, asset_path=None, source_manifest=None, compound_tray_manifest=None, relative_prim_path=None,
                 scale=None, visible=True, fixed_base=True, visual_only=False, self_collisions=True,
                 link_physics_materials=None, load_config=None, abilities=None, control_freq=None,
                 controller_config=None, action_type="continuous", action_normalize=False,
                 reset_joint_pos=None, obs_modalities=("proprio", "rgb"), include_sensor_names=None,
                 exclude_sensor_names=None, proprio_obs="default", sensor_config=None,
                 grasping_mode="physical", disable_grasp_handling=False,
                 finger_static_friction=None, finger_dynamic_friction=None, **kwargs):
        self._asset_path = str(Path(asset_path or ROOT / "assets/robots/pandaomron/native/pandaomron.usda").resolve())
        manifest_path = Path(source_manifest or ROOT / "assets/robots/pandaomron/source/manifest.json")
        self._source_manifest = json.loads(manifest_path.read_text())
        self._source_physics = json.loads(manifest_path.with_name("physics.json").read_text())
        self._source_mjcf_path = manifest_path.with_name("pandaomron.xml")
        self._compound_tray = (json.loads(Path(compound_tray_manifest).read_text())
                               if compound_tray_manifest is not None else None)
        self._world_base_fixed_joint_prim = None
        self.native_controller_errors = []
        self.native_pose_rejections = []
        # This SDK filters constructor keywords by signature; **kwargs alone
        # would silently discard the configured controller and sensor settings.
        super().__init__(name=name, relative_prim_path=relative_prim_path, scale=scale,
                         visible=visible, fixed_base=fixed_base, visual_only=visual_only,
                         self_collisions=self_collisions, link_physics_materials=link_physics_materials,
                         load_config=load_config, abilities=abilities, control_freq=control_freq,
                         controller_config=controller_config, action_type=action_type,
                         action_normalize=action_normalize, reset_joint_pos=reset_joint_pos,
                         obs_modalities=obs_modalities, include_sensor_names=include_sensor_names,
                         exclude_sensor_names=exclude_sensor_names, proprio_obs=proprio_obs,
                         sensor_config=sensor_config, grasping_mode=grasping_mode,
                         disable_grasp_handling=disable_grasp_handling,
                         finger_static_friction=finger_static_friction,
                         finger_dynamic_friction=finger_dynamic_friction, **kwargs)

    @classproperty
    def arm_names(cls):
        return ["right"]

    @property
    def model_name(self):
        return "pandaomron"

    @property
    def usd_path(self):
        return self._asset_path

    @property
    def urdf_path(self):
        return None

    @cached_property
    def disabled_collision_pairs(self):
        # The importer retains separate links for bodies that MuJoCo treats as
        # one welded body. Restore those filters and source parent-weld filters.
        return source_topology_collision_pairs(self._source_mjcf_path, self._links)

    @property
    def _raw_controller_order(self):
        return ["base", "trunk", "arm_right", "gripper_right"]

    @property
    def _default_controllers(self):
        return {"base": "JointController", "trunk": "PandaOmronTorsoController",
                "arm_right": "PandaOmronOSCController", "gripper_right": "JointController"}

    def _create_discrete_action_space(self):
        raise ValueError("PandaOmron requires its declared continuous controller interface.")

    @cached_property
    def base_joint_names(self):
        return ["mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw"]

    @cached_property
    def base_footprint_link_name(self):
        return "mobilebase0_wheeled_base"

    @cached_property
    def floor_touching_base_link_names(self):
        return ["mobilebase0_wheeled_base"]

    @cached_property
    def trunk_joint_names(self):
        return ["mobilebase0_joint_torso_height"]

    @cached_property
    def trunk_link_names(self):
        return ["mobilebase0_support"]

    @cached_property
    def arm_joint_names(self):
        return {"right": [f"robot0_joint{i}" for i in range(1, 8)]}

    @cached_property
    def arm_link_names(self):
        return {"right": [f"robot0_link{i}" for i in range(8)]}

    @cached_property
    def eef_link_names(self):
        return {"right": "gripper0_right_eef"}

    @cached_property
    def gripper_link_names(self):
        return {"right": ["robot0_right_hand", "gripper0_right_right_gripper", "gripper0_right_eef",
                          "gripper0_right_finger_joint1_tip", "gripper0_right_finger_joint2_tip"]}

    @cached_property
    def finger_joint_names(self):
        return {"right": ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]}

    @property
    def native_open_finger_positions(self):
        # The source uses opposite signed joint coordinates for the two fingers.
        actuators = {item['joint']: item for item in self._source_physics['actuators']}
        first, second = self.finger_joint_names['right']
        return {'right': [actuators[first]['ctrlrange'][1], actuators[second]['ctrlrange'][0]]}

    @cached_property
    def finger_link_names(self):
        return {"right": ["gripper0_right_leftfinger", "gripper0_right_rightfinger"]}

    @property
    def physical_grasp_contact_link_names(self):
        return {"right": [["gripper0_right_leftfinger", "gripper0_right_finger_joint1_tip"],
                          ["gripper0_right_rightfinger", "gripper0_right_finger_joint2_tip"]]}

    @property
    def _default_joint_pos(self):
        reset = self._source_manifest["reset_joint_positions"]
        return th.tensor([reset.get(name, 0.) for name in self.joints], dtype=th.float32)

    @property
    def _default_arm_osc_controller_configs(self):
        configs = super()._default_arm_osc_controller_configs
        configs["right"]["name"] = "PandaOmronOSCController"
        return configs

    @property
    def _default_trunk_joint_controller_config(self):
        config = super()._default_trunk_joint_controller_config
        config["name"] = "PandaOmronTorsoController"
        return config

    def _post_load(self):
        super()._post_load()
        # ControllableObject disables link gravity by default in this SDK.
        # This embodiment uses source-style physical gravity and explicit bias
        # compensation, so retaining that default would double-compensate.
        for link in self.links.values():
            link.enable_gravity()
        self._world_base_fixed_joint_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(f"{self.prim_path}/rootJoint")
        # The installed rigid-link loader recomputes COM from collision volume.
        # Restore compiled source values after that loader has completed.
        for value in self._source_manifest["bodies"]:
            if value["mass_kg"] <= 0:
                continue
            prim = self.links[value["name"]].prim
            mass = lazy.pxr.UsdPhysics.MassAPI.Apply(prim)
            mass.GetMassAttr().Set(value["mass_kg"])
            mass.GetCenterOfMassAttr().Set(lazy.pxr.Gf.Vec3f(*value["inertial_position"]))
            mass.GetDiagonalInertiaAttr().Set(lazy.pxr.Gf.Vec3f(*value["inertia_kg_m2"]))
            q = value["inertial_quaternion_wxyz"]
            mass.GetPrincipalAxesAttr().Set(lazy.pxr.Gf.Quatf(q[0], lazy.pxr.Gf.Vec3f(*q[1:])))
        if self._compound_tray is not None:
            # The SDK recomputes geometric COM during link loading; restore the
            # combined native mass tensor after its source-only restoration.
            value = self._compound_tray["combined_mass_properties"]
            prim = self.links[self._compound_tray["support_link"]].prim
            mass = lazy.pxr.UsdPhysics.MassAPI.Apply(prim)
            mass.GetMassAttr().Set(value["mass_kg"])
            mass.GetCenterOfMassAttr().Set(lazy.pxr.Gf.Vec3f(*value["com_m"]))
            mass.GetDiagonalInertiaAttr().Set(lazy.pxr.Gf.Vec3f(*value["principal_inertia_kg_m2"]))
            q = value["principal_quaternion_xyzw"]
            mass.GetPrincipalAxesAttr().Set(lazy.pxr.Gf.Quatf(q[3], lazy.pxr.Gf.Vec3f(*q[:3])))

    def _load_controllers(self):
        super()._load_controllers()
        for name, controller in self.controllers.items():
            original = controller.compute_control
            def checked(*values, _original=original, _name=name, **options):
                try:
                    return _original(*values, **options)
                except Exception as error:
                    self.native_controller_errors.append({"controller": _name, "error": f"{type(error).__name__}: {error}"})
                    raise
            controller.compute_control = checked

    def get_position_orientation(self, frame="world", clone=True):
        if self._links and "mobilebase0_base" in self._links:
            return self.links["mobilebase0_base"].get_position_orientation(frame=frame, clone=clone)
        return super().get_position_orientation(frame=frame, clone=clone)

    def native_world_anchor_measurement(self):
        """Read the actual world fixed-joint frame and root agreement."""
        import numpy as np
        from scipy.spatial.transform import Rotation
        prim = self._world_base_fixed_joint_prim
        if prim is None or not prim.IsValid() or prim.GetTypeName() != "PhysicsFixedJoint":
            raise ValueError("The native world fixed joint is unavailable.")
        body0 = [str(path) for path in prim.GetRelationship("physics:body0").GetTargets()]
        body1 = [str(path) for path in prim.GetRelationship("physics:body1").GetTargets()]
        enabled = prim.GetAttribute("physics:jointEnabled").Get()
        if enabled is None:
            raise ValueError("Native fixed-joint enabled state is unavailable.")
        def quaternion(name):
            value = prim.GetAttribute(name).Get()
            return np.asarray([*value.GetImaginary(), value.GetReal()], dtype=float)
        p0 = np.asarray(prim.GetAttribute("physics:localPos0").Get(), dtype=float)
        p1 = np.asarray(prim.GetAttribute("physics:localPos1").Get(), dtype=float)
        q0, q1 = quaternion("physics:localRot0"), quaternion("physics:localRot1")
        root_p, root_q = (value.detach().cpu().numpy() for value in self.root_link.get_position_orientation())
        expected_p = root_p + Rotation.from_quat(root_q).apply(p1)
        expected_q = (Rotation.from_quat(root_q) * Rotation.from_quat(q1)).as_quat()
        position_error = float(np.linalg.norm(p0 - expected_p))
        orientation_error = float(1 - abs(np.dot(q0 / np.linalg.norm(q0), expected_q)))
        finite = all(np.isfinite(value).all() for value in (p0, p1, q0, q1, root_p, root_q))
        return {"valid": bool(finite and enabled and not body0 and body1 == [self.root_link.prim_path]
                              and position_error <= 1e-5 and orientation_error <= 1e-6),
                "joint_prim_path": str(prim.GetPath()), "joint_enabled": bool(enabled),
                "body0_targets": body0, "body1_targets": body1,
                "world_anchor_position_m": p0.tolist(), "world_anchor_orientation_xyzw": q0.tolist(),
                "root_world_position_m": root_p.tolist(), "root_world_orientation_xyzw": root_q.tolist(),
                "root_local_anchor_position_m": p1.tolist(), "root_local_anchor_orientation_xyzw": q1.tolist(),
                "anchor_position_error_m": position_error, "anchor_orientation_dot_error": orientation_error}

    def _restore_initial_world_anchor(self, saved):
        """Restore the USD anchor omitted by SDK state rollback during reset."""
        import numpy as np
        from worldleague.environments.pandaomron_initialization import initialization_active
        session = getattr(self, "_pandaomron_initialization_session", None)
        if session is None or not initialization_active(session, self):
            raise RuntimeError("Initial anchor restoration is authorized only during reset.")
        current = self.native_world_anchor_measurement()
        for key in ("joint_prim_path", "joint_enabled", "body0_targets", "body1_targets",
                    "root_local_anchor_position_m", "root_local_anchor_orientation_xyzw"):
            if current[key] != saved[key]:
                raise ValueError("Native anchor identity or local frame changed during baseline rollback.")
        if (not np.allclose(current["root_world_position_m"], saved["root_world_position_m"], atol=1e-5, rtol=0)
                or 1 - abs(np.dot(current["root_world_orientation_xyzw"], saved["root_world_orientation_xyzw"])) > 1e-6):
            raise ValueError("Simulator baseline did not restore the saved articulation root pose.")
        prim = self._world_base_fixed_joint_prim
        prim.GetAttribute("physics:localPos0").Set(lazy.pxr.Gf.Vec3f(*saved["world_anchor_position_m"]))
        q = saved["world_anchor_orientation_xyzw"]
        prim.GetAttribute("physics:localRot0").Set(lazy.pxr.Gf.Quatf(q[3], lazy.pxr.Gf.Vec3f(*q[:3])))
        restored = self.native_world_anchor_measurement()
        if not restored["valid"]:
            raise ValueError("Restored baseline root and world anchor disagree.")
        return restored

    def _set_initial_root_height(self, delta_z):
        """Translate the live articulation and its world anchor only inside reset."""
        from worldleague.environments.pandaomron_initialization import initialization_active
        session = getattr(self, "_pandaomron_initialization_session", None)
        if session is None or not initialization_active(session, self):
            raise RuntimeError("Root floor alignment is authorized only during reset initialization.")
        anchor = self.native_world_anchor_measurement()
        if not anchor["valid"]:
            raise ValueError("Native world fixed-joint anchor is not valid and aligned.")
        position, orientation = self.root_link.get_position_orientation()
        old_position, old_orientation = position.clone(), orientation.clone()
        old_joints = self.get_joint_positions().clone()
        position = position.clone()
        position[2] += delta_z
        # The installed SDK routes this full pose to articulation_view.set_world_poses.
        # No joint setter, controller reset, or simulator stop/play is involved.
        super().set_position_orientation(position=position, orientation=orientation, frame="world")
        local_position = th.tensor(anchor["root_local_anchor_position_m"], dtype=th.float32)
        local_orientation = th.tensor(anchor["root_local_anchor_orientation_xyzw"], dtype=th.float32)
        world_position, world_orientation = T.pose_transform(position, orientation, local_position, local_orientation)
        prim = self._world_base_fixed_joint_prim
        prim.GetAttribute("physics:localPos0").Set(lazy.pxr.Gf.Vec3f(*world_position.tolist()))
        prim.GetAttribute("physics:localRot0").Set(lazy.pxr.Gf.Quatf(*world_orientation[[3, 0, 1, 2]].tolist()))
        after = self.native_world_anchor_measurement()
        if not after["valid"]:
            raise ValueError("Relocated articulation and world anchor disagree.")
        measured_position, measured_orientation = self.root_link.get_position_orientation()
        joint_error = float(th.max(th.abs(self.get_joint_positions() - old_joints)))
        orientation_error = float(1 - th.abs(th.dot(old_orientation, measured_orientation)))
        if (joint_error > 1e-6 or orientation_error > 1e-6
                or float(th.max(th.abs(measured_position - position))) > 1e-5):
            raise ValueError("Root floor alignment changed the retained articulation configuration.")
        return {"world_anchor_before": anchor, "world_anchor_after": after,
                "root_world_position_before_m": old_position.tolist(),
                "root_world_position_after_m": measured_position.tolist(),
                "requested_root_translation_z_m": float(delta_z),
                "measured_root_translation_z_m": float(measured_position[2] - old_position[2]),
                "maximum_joint_position_change": joint_error,
                "root_orientation_dot_error": orientation_error}

    def _set_planar_position_orientation(self, position, orientation, preserve_controllers=False):
        position = th.as_tensor(position, dtype=th.float32)
        orientation = th.as_tensor(orientation, dtype=th.float32)
        root_pos, root_quat = self.root_link.get_position_orientation()
        relative_pos, relative_quat = T.relative_pose_transform(position, orientation, root_pos, root_quat)
        pivot = th.tensor([-.21, 0., 0.], dtype=th.float32)
        translation = relative_pos - pivot + T.quat2mat(relative_quat) @ pivot
        yaw = T.quat2euler(relative_quat)[2]
        if abs(float(relative_pos[2])) > .005:
            self.native_pose_rejections.append({
                'requested_frame': 'world', 'requested_world_position_m': position.tolist(),
                'root_world_position_m': root_pos.tolist(),
                'root_world_orientation_xyzw': root_quat.tolist(),
                'requested_root_relative_position_m': relative_pos.tolist(),
                'maximum_vertical_displacement_m': .005})
            raise ValueError("A planar-base runtime pose cannot change the source floor height.")
        states = ({name: deepcopy(controller.dump_state()) for name, controller in self.controllers.items()}
                  if preserve_controllers else None)
        try:
            self.set_joint_positions(th.stack((translation[0], translation[1], yaw)), indices=self.base_control_idx, drive=False)
        finally:
            if states is not None:
                for name, state in states.items():
                    self.controllers[name].load_state(state)

    def set_position_orientation(self, position=None, orientation=None, frame="world"):
        if og.sim.is_playing() and self.initialized:
            old_pos, old_quat = self.get_position_orientation(frame=frame)
            requested_position = position
            position = old_pos if position is None else th.as_tensor(position, dtype=th.float32)
            orientation = old_quat if orientation is None else th.as_tensor(orientation, dtype=th.float32)
            if frame == "scene":
                position, orientation = self.scene.convert_scene_relative_pose_to_world(position, orientation)
            from worldleague.environments.pandaomron_initialization import initialization_active, initialize_requested_pose
            session = getattr(self, "_pandaomron_initialization_session", None)
            if requested_position is not None and session is not None and initialization_active(session, self):
                initialize_requested_pose(session, self, position, orientation)
            else:
                self._set_planar_position_orientation(position, orientation)
        else:
            super().set_position_orientation(position, orientation, frame)
            if self._world_base_fixed_joint_prim is not None:
                pos, quat = self.root_link.get_position_orientation()
                self._world_base_fixed_joint_prim.GetAttribute("physics:localPos0").Set(lazy.pxr.Gf.Vec3f(*pos.tolist()))
                self._world_base_fixed_joint_prim.GetAttribute("physics:localRot0").Set(lazy.pxr.Gf.Quatf(*quat[[3, 0, 1, 2]].tolist()))

    def get_linear_velocity(self):
        return self.links["mobilebase0_base"].get_linear_velocity()

    def get_angular_velocity(self):
        return self.links["mobilebase0_base"].get_angular_velocity()

    def get_control_dict(self):
        fcns = super().get_control_dict()
        damping = {value["name"]: value["damping"] for value in self._source_physics["joints"]}
        fcns["panda_passive_damping"] = lambda: cb.array([damping[name] for name in self.joints])
        # Source OSC origin is the torso support center, not the stationary dummy root.
        fcns["_panda_raw_eef_pose"] = fcns.get_fcn("_eef_right_pos_quat_relative")
        fcns["_panda_raw_jacobian"] = fcns.get_fcn("eef_right_jacobian_relative")
        fcns["_panda_support_pose"] = lambda: ControllableObjectViewAPI.get_link_relative_position_orientation(
            self.articulation_root_path, "mobilebase0_support")
        fcns["_panda_rotation"] = lambda: cb.T.quat2mat(fcns["_panda_support_pose"][1]).T
        fcns["_panda_eef_pose"] = lambda: cb.T.relative_pose_transform(*fcns["_panda_raw_eef_pose"], *fcns["_panda_support_pose"])
        fcns["eef_right_pos_relative"] = lambda: fcns["_panda_eef_pose"][0]
        fcns["eef_right_quat_relative"] = lambda: fcns["_panda_eef_pose"][1]
        fcns["eef_right_jacobian_relative"] = lambda: cb.cat((
            fcns["_panda_rotation"] @ fcns["_panda_raw_jacobian"][:3],
            fcns["_panda_rotation"] @ fcns["_panda_raw_jacobian"][3:]))
        for suffix in ("lin", "ang"):
            eef_velocity = ControllableObjectViewAPI.get_link_relative_linear_velocity if suffix == "lin" else ControllableObjectViewAPI.get_link_relative_angular_velocity
            fcns[f"eef_right_{suffix}_vel_relative"] = lambda f=eef_velocity: fcns["_panda_rotation"] @ f(self.articulation_root_path, "gripper0_right_eef", estimate=True)
            fcns[f"root_rel_{suffix}_vel"] = lambda f=eef_velocity: fcns["_panda_rotation"] @ f(self.articulation_root_path, "mobilebase0_support", estimate=True)
        return fcns
