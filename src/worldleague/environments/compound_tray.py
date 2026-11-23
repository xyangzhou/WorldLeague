"""Chassis-integrated carrying geometry and evaluator-only component frames.

Used by worldleague continuous --help.
"""
import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


MODE = "chassis_compound_body_v1"


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def combine_mass_properties(parts):
    """Combine body-frame COM tensors, using the parallel-axis theorem."""
    total = sum(float(p["mass_kg"]) for p in parts)
    center = sum(float(p["mass_kg"]) * np.asarray(p["com_m"]) for p in parts) / total
    inertia = np.zeros((3, 3))
    for part in parts:
        offset = np.asarray(part["com_m"]) - center
        inertia += np.asarray(part["inertia_com_kg_m2"]) + part["mass_kg"] * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    inertia = (inertia + inertia.T) / 2
    values, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    return {"mass_kg": total, "com_m": center.tolist(), "inertia_com_kg_m2": inertia.tolist(),
            "principal_inertia_kg_m2": values.tolist(),
            "principal_quaternion_xyzw": Rotation.from_matrix(axes).as_quat().tolist()}


def native_mass_properties(link, transform=None):
    """Read inertia about the COM, expressed along native body axes.

    The installed rigid-body tensor view already returns the full body-axis
    tensor, including when get_coms() reports nonidentity principal axes. Applying
    that quaternion again would rotate the tensor twice (native audit v1).
    """
    transform = np.eye(4) if transform is None else transform
    position, orientation = link._rigid_prim_view.get_coms()
    center = array(position).reshape(-1, 3)[0]
    quaternion = array(orientation).reshape(-1, 4)[0][[1, 2, 3, 0]]
    inertia = array(link._rigid_prim_view.get_inertias()).reshape(-1, 3, 3)[0]
    rotation = transform[:3, :3]
    return {"name": link.name, "mass_kg": float(link.mass),
            "com_m": (transform[:3, 3] + transform[:3, :3] @ center).tolist(),
            "inertia_com_kg_m2": (rotation @ inertia @ rotation.T).tolist(),
            "native_com_quaternion_xyzw": quaternion.tolist(), "native_inertia_tensor_kg_m2": inertia.tolist()}


def world_pose(parent, component):
    position, quaternion = map(array, parent.get_position_orientation())
    rotation = Rotation.from_quat(quaternion)
    return (position + rotation.apply(component["position_support_m"]),
            (rotation * Rotation.from_quat(component["orientation_support_xyzw"])).as_quat())


class CompoundContactTracker:
    """Track actual collider identities, including contacts on sleeping cargo."""

    def __init__(self):
        import omni.physx
        from omni.physx.bindings._physx import ContactEventType
        from pxr import PhysicsSchemaTools
        self.active = {}
        self.lost = int(ContactEventType.CONTACT_LOST)
        self.present_types = {int(ContactEventType.CONTACT_FOUND), int(ContactEventType.CONTACT_PERSIST)}
        self.decode = PhysicsSchemaTools.intToSdfPath
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)

    def on_contacts(self, headers, data):
        for header in headers:
            colliders = (str(self.decode(header.collider0)), str(self.decode(header.collider1)))
            key = tuple(sorted(colliders))
            if int(header.type) == self.lost:
                self.active.pop(key, None)
                continue
            if int(header.type) not in self.present_types:
                continue
            self.active[key] = {"colliders": colliders,
                "actors": (str(self.decode(header.actor0)), str(self.decode(header.actor1))),
                "contact_count": int(header.num_contact_data)}

    def contacts_with(self, collider_paths, target_path):
        paths = set(collider_paths)
        def under(path, parent):
            return path == parent or path.startswith(parent + "/")
        return [row for row in self.active.values() if
                any((left in paths and under(right, target_path)) or
                    (right in paths and under(left, target_path))
                    for left, right in (row["colliders"],))]


class CompoundFrame:
    """Logical component of one native body; this is not an independent object."""

    def __init__(self, support, component, tracker, *, category="robot_fixture", model=None):
        self.support, self.component, self.tracker = support, component, tracker
        self.name, self.category, self.model = component["name"], category, model
        self.prim_path = support.prim_path + "/" + component["frame_relative_path"]
        self.collider_paths = [support.prim_path + "/" + p for p in component["collider_relative_paths"]]
        self.fixed_base, self.visual_only, self.n_dof = False, False, 0
        self.links = {self.name: self}
        self.root_link = self

    def get_position_orientation(self):
        return world_pose(self.support, self.component)

    def get_angular_velocity(self):
        return array(self.support.get_angular_velocity())

    def get_linear_velocity(self):
        position, quaternion = map(array, self.support.get_position_orientation())
        center = position + Rotation.from_quat(quaternion).apply(array(self.support.center_of_mass))
        point = self.get_position_orientation()[0]
        return array(self.support.get_linear_velocity()) + np.cross(self.get_angular_velocity(), point - center)

    @property
    def collision_boundary_points_world(self):
        position, quaternion = self.get_position_orientation()
        return position + Rotation.from_quat(quaternion).apply(self.component["collision_points_local_m"])

    @property
    def aabb(self):
        points = self.collision_boundary_points_world
        return points.min(0), points.max(0)

    @property
    def mass(self):
        return self.component["mass_kg"]

    def native_contacts_with(self, obj):
        return self.tracker.contacts_with(self.collider_paths, obj.prim_path)

    def contact_list(self):
        paths = set(self.collider_paths)
        return [SimpleNamespace(body0=row["actors"][0], body1=row["actors"][1],
                                collider0=row["colliders"][0], collider1=row["colliders"][1])
                for row in self.tracker.active.values() if paths.intersection(row["colliders"])]


def configure_compound_objects(config, spec, parameters):
    """Use the same declared tray identity as geometry already in the robot."""
    from worldleague.resources import ROOT
    receipt = json.loads((ROOT / spec["compound_manifest"]).read_text())
    plate = parameters["plate"]
    if any(plate[key] != receipt["tray_identity"][key] for key in ("category", "model")):
        raise ValueError("The compound robot asset contains a different tray identity.")
    config["objects"] = [obj for obj in config.get("objects", []) if obj["name"] != plate["name"]]
    return {"attachment_mode": MODE, "tray_identity": receipt["tray_identity"], "independent_tray_body": False}


def initialize_compound_frames(robot, environment, spec, tray_name):
    """Bind existing compound shapes; do not author joints or set object poses."""
    from worldleague.resources import ROOT
    receipt = json.loads((ROOT / spec["compound_manifest"]).read_text())
    if receipt["schema"] != MODE or receipt["support_link"] != spec["support_link"]:
        raise ValueError("Compound apparatus does not match its declared support.")
    components = {row["name"]: row for row in receipt["components"]}
    if any(environment.scene.object_registry("name", name) is not None for name in components):
        raise RuntimeError("A compound component also exists as a separate scene object.")
    for part in spec["parts"]:
        compiled = components[part["name"]]
        dimensions = np.ptp(np.asarray(compiled["collision_points_local_m"]), axis=0)
        if (not np.allclose(dimensions, part["dimensions_m"], atol=1e-5, rtol=0)
                or not np.allclose(compiled["position_support_m"], part["position_support_m"], atol=1e-6, rtol=0)
                or not np.isclose(compiled["mass_kg"], part["mass_kg"], rtol=1e-5)):
            raise ValueError("Compiled support geometry differs from the selected apparatus specification.")
    tray_low = receipt["tray_collision_bounds_local_m"][0]
    expected = spec["tray_center_support_xy_m"] + [spec["deck_top_support_z_m"] - tray_low[2] + .001]
    if not np.allclose(components[tray_name]["position_support_m"], expected, atol=1e-6, rtol=0):
        raise ValueError("Compiled tray placement differs from its declared mount.")
    support = robot.links[spec["support_link"]]
    measured = native_mass_properties(support)
    expected_mass = receipt["combined_mass_properties"]
    mass_audit = {"measured": measured, "expected": expected_mass,
                  "matches": all(np.allclose(measured[key], expected_mass[key], rtol=2e-4, atol=1e-5)
                                 for key in ("mass_kg", "com_m", "inertia_com_kg_m2"))}
    if not mass_audit["matches"]:
        raise RuntimeError("Native compound mass, center of mass or inertia differs from the compiled asset: "
                           + json.dumps(mass_audit))
    tracker = CompoundContactTracker()
    robot.compound_tray_frames = {}
    for component in receipt["components"]:
        tray = component["role"] == "tray"
        name = tray_name if tray else component["name"]
        if name != component["name"]:
            raise ValueError("Compound tray identity differs from its task declaration.")
        robot.compound_tray_frames[name] = CompoundFrame(
            support, component, tracker, category=receipt["tray_identity"]["category"] if tray else "robot_fixture",
            model=receipt["tray_identity"]["model"] if tray else None)
    robot.mobile_tray_mount = {"protocol": spec["protocol"], "attachment_mode": MODE,
        "support_link": spec["support_link"], "working_arm": robot.arm_names[0],
        "tray_name": tray_name, "tray_collision_bounds_local_m": receipt["tray_collision_bounds_local_m"],
        "apparatus": receipt["components"], "specification": spec, "cargo_attachment": False,
        "compound_mass_properties": receipt["combined_mass_properties"], "mass_audit": mass_audit}
    return robot.mobile_tray_mount


def measure_compound(robot, simulator):
    """Verify physical membership and collision schemas, not a fabricated joint."""
    from pxr import UsdGeom, UsdPhysics
    receipt = robot.mobile_tray_mount
    parent = robot.links[receipt["support_link"]].prim_path
    support = robot.links[receipt["support_link"]]
    expected_mass = receipt["compound_mass_properties"]["mass_kg"]
    mass_ok = bool(np.isclose(float(support.mass), expected_mass, rtol=1e-5, atol=1e-6))
    values = []
    for part in receipt["apparatus"]:
        frame = robot.compound_tray_frames[part["name"]]
        geometry_ok = bool(frame.collider_paths)
        for path in frame.collider_paths:
            prim = simulator.stage.GetPrimAtPath(path)
            enabled = bool(prim and prim.IsActive() and prim.HasAPI(UsdPhysics.CollisionAPI)
                           and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            geometry_ok &= enabled and bool(owner) and str(owner.GetPath()) == parent
        prim = simulator.stage.GetPrimAtPath(frame.prim_path)
        geometry_ok &= bool(prim and not UsdGeom.Xformable(prim).GetResetXformStack())
        matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
        expected = np.asarray(part["position_support_m"])
        position_error = float(np.linalg.norm(np.asarray(matrix.ExtractTranslation()) - expected))
        quaternion = matrix.ExtractRotationQuat()
        rotation_error = float(Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).magnitude())
        values.append({"name": part["name"], "same_rigid_body": bool(geometry_ok),
                       "joint_enabled": None, "parent_matches": bool(geometry_ok), "child_matches": bool(geometry_ok),
                       "relative_position_error_m": position_error, "relative_rotation_error_rad": rotation_error,
                       "valid": bool(geometry_ok and mass_ok and position_error <= receipt["specification"]["position_tolerance_m"]
                                     and rotation_error <= receipt["specification"]["rotation_tolerance_rad"])})
    return {"tray_mount_valid": all(row["valid"] for row in values), "apparatus": values,
            "attachment_mode": MODE, "mass_matches": mass_ok, "measured_chassis_mass_kg": float(support.mass),
            "protocol": receipt["protocol"], "working_arm": receipt["working_arm"]}
