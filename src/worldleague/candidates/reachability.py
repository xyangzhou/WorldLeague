"""Bounded CPU geometric screening, without simulator imports or robot commands.

Run ``python -m worldleague.candidates.reachability INVENTORY_JSON OUTPUT_DIRECTORY``
in the behavior environment. USD fallback calibration additionally needs the
installed omni.usd.libs extension's ``bin`` and ``bin/deps`` in LD_LIBRARY_PATH.
The input quaternion convention is xyzw; distances are metres and joints radians
(prismatic joints use metres). A witness is an open-gripper geometric candidate,
never a claim that closing, carrying, controller tracking, or a grasp will work.
The default continuous mode optimizes torso and working arm together. Preserve
arm-only screening with ``audit_scene(..., options={"torso_search": "fixed_presets"})``.
Destination queries are separate initial-state hypothetical approaches; they
never connect an initial pickup to transport, door opening, or final placement.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import sys

import numpy as np

_BASE_GEOMETRY_AABB_TOLERANCE_M = .002


class AuditUnavailable(RuntimeError):
    """Required measured geometry or a CPU dependency is unavailable."""


@dataclass(frozen=True)
class AuditOptions:
    max_stances: int = 12
    max_ik_evaluations: int = 100
    max_ik_attempts_per_target: int = 192
    torso_search: str = "continuous"
    navigation_mode: str = "circle"
    grid_resolution_m: float = .06
    max_grid_cells: int = 250000
    max_joint_step: float = .06
    max_prismatic_step_m: float = .005
    collision_margin_m: float = .006
    position_tolerance_m: float = .005
    orientation_tolerance_deg: float = 3.
    aperture_m: float = .1
    approach_distance_m: float = .1
    max_stance_target_distance_m: float = 1.2


def _box(value):
    result = np.asarray(value, dtype=float)
    if (result.shape != (2, 3) or not np.isfinite(result).all()
            or np.any(result[0] > result[1])):
        raise AuditUnavailable("A finite, ordered world AABB is required for every actor.")
    return result


def _corners(box):
    return np.asarray(list(product(*zip(box[0], box[1]))))


def _validated_base_vertices(geometry, measured_base_aabb):
    evidence = {"footprint_source": "measured_base_aabb_corners", "base_geometry_validated": False,
        "base_geometry_fallback_reason": "optional_base_geometry_not_supplied",
        "base_geometry_world_aabb_residual_m": None, "base_geometry_reported_world_aabb_residual_m": None,
        "base_geometry_aabb_tolerance_m": _BASE_GEOMETRY_AABB_TOLERANCE_M,
        "base_geometry_enabled_mesh_names": []}
    if geometry is None:
        return None, evidence
    try:
        if not isinstance(geometry, dict) or geometry.get("schema") != "wia_robot_base_geometry_v1":
            raise ValueError("Unsupported or missing native base-geometry schema.")
        if geometry.get("available") is not True or geometry.get("status") != "available":
            raise ValueError(f"Native base geometry is unavailable: {geometry.get('reason', geometry.get('status'))}.")
        reported = float(geometry["max_world_aabb_residual_m"])
        if np.isfinite(reported):
            evidence["base_geometry_reported_world_aabb_residual_m"] = reported
        if not np.isfinite(reported) or not 0 <= reported <= _BASE_GEOMETRY_AABB_TOLERANCE_M:
            raise ValueError("Reported native base AABB residual is not finite and within 2 mm.")
        vertices = np.asarray(geometry["world_vertices"], float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("Native base vertices must be a finite Nx3 array with at least four rows.")
        extrema = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        residual = float(np.max(np.abs(extrema - _box(measured_base_aabb))))
        evidence["base_geometry_world_aabb_residual_m"] = residual
        if residual > _BASE_GEOMETRY_AABB_TOLERANCE_M:
            raise ValueError("Actual native vertex extrema disagree with robot.base_aabb by more than 2 mm.")
        names = geometry.get("enabled_mesh_names", geometry.get("reference_mesh_metadata", {}).get("enabled_mesh_names", []))
        transferred = geometry.get('provenance') == 'rigid_pose_transfer_of_native_reference_base_vertices'
        evidence.update(footprint_source=("validated_reference_base_vertices_rigid_pose_transfer" if transferred else "validated_native_base_collision_vertices"),
            base_geometry_provenance=geometry.get('provenance'),
            target_vertices_newly_measured=not transferred,
            base_geometry_validated=True,
            base_geometry_fallback_reason=None, base_geometry_enabled_mesh_names=deepcopy(names) if isinstance(names, list) else [])
        # Supplied base_local_xy_bounds are intentionally unused. The planner
        # recomputes both full-pose and planar projections from these vertices.
        return vertices.copy(), evidence
    except (KeyError, TypeError, ValueError) as error:
        evidence["base_geometry_fallback_reason"] = f"{type(error).__name__}: {error}"
        return None, evidence


def transform_points(points, transform):
    """Apply a column-vector rigid transform to row-vector xyz points."""
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def joint_limit_margin(values, limits):
    """Minimum native-unit distance to a limit; negative means a violation."""
    margins = [min(value - limits[name][0], limits[name][1] - value)
               for name, value in values.items() if name in limits]
    return float(min(margins)) if margins else None


def _support_bounds(target, objects):
    if target.get("support_aabb") is not None:
        return _box(target["support_aabb"]), "explicit_target_support_aabb"
    support = target.get("support")
    named = [row for row in objects if row["name"] == support]
    if named:
        return _box(named[0]["aabb"]), "named_support_object_aabb"
    bottom = _box(target["aabb"])[0, 2]
    panels = []
    for row in objects:
        if not support or not (row.get("parent_object") == support or row["name"].startswith(support + "::")):
            continue
        box = _box(row["aabb"])
        span = np.ptp(box, axis=0)
        if abs(box[1, 2] - bottom) <= .08 and span[2] <= .5 * min(span[:2]):
            panels.append(box)
    if panels:
        return np.stack((np.min([box[0] for box in panels], axis=0),
                         np.max([box[1] for box in panels], axis=0))), "union_of_horizontal_support_collision_mesh_aabbs"
    return _box(target["aabb"]), "support_unavailable_target_aabb_fallback"


def _usd_calibration(path, pose_matrix):
    try:
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            directories = sorted((Path(sys.prefix) / "lib").glob(
                "python*/site-packages/isaacsim/extscache/omni.usd.libs-*"))
            if directories:
                sys.path.insert(0, str(directories[-1]))
            from pxr import Usd, UsdPhysics
    except ImportError as error:
        raise AuditUnavailable(f"CPU USD reader unavailable: {error}; supply complete joint_calibration "
                               "or start Python with omni.usd.libs bin and bin/deps in LD_LIBRARY_PATH.") from error
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise AuditUnavailable(f"Cannot open supplied robot USD: {path}")
    calibration, limits = {}, {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        bodies = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if not all(len(body) == 1 for body in bodies):
            continue
        def matrix(position, quaternion):
            return pose_matrix(position.Get(), [*quaternion.Get().GetImaginary(), quaternion.Get().GetReal()])
        kind = ("revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else
                "prismatic" if prim.IsA(UsdPhysics.PrismaticJoint) else
                "fixed" if prim.IsA(UsdPhysics.FixedJoint) else None)
        if kind is None:
            continue
        axis = {"X": [1., 0., 0.], "Y": [0., 1., 0.], "Z": [0., 0., 1.]}.get(
            prim.GetAttribute("physics:axis").Get(), [1., 0., 0.])
        name = prim.GetName()
        calibration[name] = {"parent": bodies[0][0].name, "child": bodies[1][0].name,
            "parent_joint": matrix(joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()).tolist(),
            "child_joint": matrix(joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()).tolist(),
            "axis": axis, "kind": kind}
        if kind != "fixed":
            lo, hi = [prim.GetAttribute(f"physics:{side}Limit").Get() for side in ("lower", "upper")]
            if lo is not None and hi is not None and np.isfinite([lo, hi]).all():
                limits[name] = (np.deg2rad([lo, hi]) if kind == "revolute" else np.array([lo, hi])).tolist()
    return calibration, limits


class _Robot:
    def __init__(self, row, options):
        try:
            from worldharness.core.camera_geometry import RobotKinematics, axis_rotation, pose_matrix
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
            import yaml
        except ImportError as error:
            raise AuditUnavailable(f"CPU kinematics dependency unavailable: {error}") from error
        self.pose_matrix, self.axis_rotation = pose_matrix, axis_rotation
        self.least_squares, self.Rotation = least_squares, Rotation
        self.options = options
        self.urdf = Path(row["urdf_path"]).expanduser()
        self.kin = RobotKinematics(self.urdf, root="base_link")
        nominal = {joint[0]: (parent, joint) for parent, children in self.kin.children.items() for joint in children}
        self.seed = {name: float(value) for name, value in row["joints"].items()}
        self.limits = {name: list(value) for name, value in row.get("joint_limits", {}).items()}
        calibration = dict(row.get("joint_calibration", {}))
        # The URDF lacks verified fixed EEF mounts and differs from installed USD
        # for moving joints. Missing calibrated chains must never silently use it.
        required = {f"{arm}_arm_joint{i}" for arm in ("left", "right") for i in range(1, 7)}
        required |= {f"torso_joint{i}" for i in range(1, 5)}
        required |= {f"{arm}_gripper_axis{i}" for arm in ("left", "right") for i in (1, 2)}
        calibrated_children = {data["child"] for data in calibration.values()}
        if not required.issubset(calibration) or not {"left_eef_link", "right_eef_link"}.issubset(calibrated_children):
            usd = row.get("usd_path")
            if not usd or not Path(usd).is_file():
                raise AuditUnavailable("Complete moving-joint and fixed EEF calibration, or an existing robot USD, is required.")
            measured, usd_limits = _usd_calibration(usd, pose_matrix)
            measured.update(calibration)
            calibration = measured
            for name, limits in usd_limits.items():
                self.limits.setdefault(name, limits)
        self.calibration = calibration
        for name, data in calibration.items():
            old = nominal.get(name)
            kind = data.get("kind", old[1][1] if old else None)
            if kind is None:
                raise AuditUnavailable(f"Extra calibrated joint {name} needs an explicit verified joint kind.")
            parent, child = data["parent"], data["child"]
            if old and (old[0] != parent or old[1][2] != child):
                raise AuditUnavailable(f"Installed calibration changes the topology of {name}.")
            before, after = np.asarray(data["parent_joint"], float), np.linalg.inv(np.asarray(data["child_joint"], float))
            axis = np.asarray(data["axis"], float)
            if (before.shape != (4, 4) or after.shape != (4, 4) or axis.shape != (3,)
                    or not np.isfinite(np.r_[before.ravel(), after.ravel(), axis]).all()
                    or not np.isclose(np.linalg.norm(axis), 1)):
                raise AuditUnavailable(f"Invalid measured transforms for {name}.")
            self.kin.children[parent] = [item for item in self.kin.children.get(parent, []) if item[0] != name]
            self.kin.children[parent].append((name, kind, child, before, axis, after))
            self.kin.parents[child] = parent
        self.joint_kinds = {joint[0]: joint[1] for children in self.kin.children.values() for joint in children}
        for name in required:
            if name not in self.seed or name not in self.limits:
                raise AuditUnavailable(f"Measured encoder and native limits required for {name}.")
            lo, hi = self.limits[name]
            if not np.isfinite([lo, hi, self.seed[name]]).all() or not lo < hi:
                raise AuditUnavailable(f"Finite, nonempty measured limits required for {name}.")
        if joint_limit_margin(self.seed, self.limits) < -1e-5:
            raise AuditUnavailable("Measured initial robot posture violates supplied joint limits.")
        for arm in ("left", "right"):
            for i in (1, 2):
                name = f"{arm}_gripper_axis{i}"
                if not self.limits[name][0] <= options.aperture_m / 2 <= self.limits[name][1]:
                    raise AuditUnavailable("The explicit 0.1 m aperture does not fit measured gripper joint limits.")
        sphere_path = Path(row.get("collision_spheres_path", self.urdf.parent.parent /
                                   "curobo/r1_description_curobo_arm_no_torso.yaml"))
        if not sphere_path.is_file():
            raise AuditUnavailable(f"Installed R1 collision-sphere asset missing: {sphere_path}")
        config = yaml.safe_load(sphere_path.read_text())["robot_cfg"]["kinematics"]
        self.spheres = {name: rows for name, rows in config["collision_spheres"].items()
                        if not name.startswith("attached_object")}
        self.ignored = {frozenset((left, right)) for left, others in config.get("self_collision_ignore", {}).items()
                        for right in others}
        self.sphere_path = sphere_path
        self.chains = {arm: self.chain(f"{arm}_eef_link") for arm in ("left", "right")}
        self.kin.frames(self.seed, links=set(self.spheres) | {"left_eef_link", "right_eef_link"})
        self.position = np.asarray(row["position"], float)
        self.initial_pose = pose_matrix(self.position, row["orientation"])
        if abs(self.initial_pose[2, 2] - 1) > 1e-4:
            raise AuditUnavailable("Planar stance screening requires an upright measured robot root.")
        self.base_box = _box(row["base_aabb"])
        self.base_world_vertices, self.base_geometry_validation = _validated_base_vertices(row.get("base_geometry"), self.base_box)
        self.base_radius = float(np.max(np.linalg.norm(_corners(self.base_box)[:, :2] - self.position[:2], axis=1)))

    def chain(self, link):
        result, seen = [], set()
        while link != self.kin.root:
            if link in seen or link not in self.kin.parents:
                raise AuditUnavailable(f"Missing calibrated chain to {link}.")
            seen.add(link)
            parent = self.kin.parents[link]
            result.append(next(joint for joint in self.kin.children[parent] if joint[2] == link))
            link = parent
        return result[::-1]

    def fk(self, arm, joints):
        result = np.eye(4)
        for name, kind, _, before, axis, after in self.chains[arm]:
            motion = np.eye(4)
            if kind in ("revolute", "continuous"):
                motion[:3, :3] = self.axis_rotation(axis, joints[name])
            elif kind == "prismatic":
                motion[:3, 3] = axis * joints[name]
            result = result @ before @ motion @ after
        return result

    def torso_presets(self):
        names = [f"torso_joint{i}" for i in range(1, 5)]
        result = [{name: self.seed[name] for name in names}]
        # Two deterministic interior presets interpolate from the measured
        # posture toward each measured range's midpoint, without copied angles.
        for fraction in (.4, .8):
            result.append({name: (1 - fraction) * self.seed[name] + fraction * np.mean(self.limits[name])
                           for name in names})
        return result

    def solve(self, arm, joints, position, orientation, attempt_index):
        names = [f"{arm}_arm_joint{i}" for i in range(1, 7)]
        if self.options.torso_search == "continuous":
            names = [f"torso_joint{i}" for i in range(1, 5)] + names
        lower, upper = np.asarray([self.limits[name] for name in names]).T
        seed = np.asarray([joints[name] for name in names])
        if self.options.torso_search == "fixed_presets" and attempt_index % 2:
            seed = .5 * seed + .5 * (lower + upper) / 2
        def residual(values):
            pose = self.fk(arm, dict(joints, **dict(zip(names, values))))
            return np.r_[pose[:3, 3] - position,
                .25 * self.Rotation.from_matrix(orientation.T @ pose[:3, :3]).as_rotvec()]
        solution = self.least_squares(residual, np.clip(seed, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper), max_nfev=self.options.max_ik_evaluations,
            ftol=1e-7, xtol=1e-7, gtol=1e-7)
        values = dict(joints, **dict(zip(names, map(float, solution.x))))
        pose = self.fk(arm, values)
        return values, float(np.linalg.norm(pose[:3, 3] - position)), float(np.rad2deg(
            self.Rotation.from_matrix(orientation.T @ pose[:3, :3]).magnitude()))

    def cloud(self, joints, base):
        frames = self.kin.frames(joints, links=set(self.spheres))
        centers, radii, links = [], [], []
        for link, rows in self.spheres.items():
            for row in rows:
                centers.append(transform_points(np.asarray([row["center"]]), base @ frames[link])[0])
                radii.append(row["radius"])
                links.append(link)
        return np.asarray(centers), np.asarray(radii), links

    def gripper_dimensions(self, arm, joints):
        frames = self.kin.frames(joints, links={f"{arm}_eef_link", f"{arm}_arm_link6",
                                               f"{arm}_gripper_link1", f"{arm}_gripper_link2"})
        eef_inverse = np.linalg.inv(frames[f"{arm}_eef_link"])
        bounds = []
        for link, rows in self.spheres.items():
            if link in {f"{arm}_gripper_link1", f"{arm}_gripper_link2"}:
                for sphere in rows:
                    center = transform_points([sphere["center"]], eef_inverse @ frames[link])[0]
                    bounds.extend([center - sphere["radius"], center + sphere["radius"]])
        bounds = np.asarray(bounds)
        # Opening direction comes from the calibrated prismatic finger axis.
        joint = next(item for items in self.kin.children.values() for item in items
                     if item[0] == f"{arm}_gripper_axis1")
        axis = eef_inverse[:3, :3] @ frames[f"{arm}_gripper_link1"][:3, :3] @ joint[4]
        # The calibrated wrist -> fixed EEF offset measures the tool's forward
        # direction. For installed R1 it is approximately EEF +Z; no finger-X
        # convention is assumed. The sphere envelope is reported separately.
        approach = eef_inverse[:3, :3] @ (frames[f"{arm}_eef_link"][:3, 3] - frames[f"{arm}_arm_link6"][:3, 3])
        if np.linalg.norm(approach) < 1e-8:
            raise AuditUnavailable("A nonzero calibrated wrist-to-EEF offset is required for tool approach direction.")
        approach /= np.linalg.norm(approach)
        return {"aperture_m": self.options.aperture_m, "finger_joint_values_m": [self.options.aperture_m / 2] * 2,
                "opening_axis_eef": axis.tolist(), "approach_axis_eef": approach.tolist(),
                "approach_axis_source": "calibrated_wrist_to_fixed_eef_offset", "sphere_envelope_eef_m":
                [bounds.min(axis=0).tolist(), bounds.max(axis=0).tolist()],
                "sphere_envelope_dimensions_m": np.ptp(bounds, axis=0).tolist()}


def _convex_hull_2d(points):
    points = sorted(set(map(tuple, np.asarray(points, float))))
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def polygon_aabb_intersections(polygon, boxes, inflation=0.):
    """Conservative closed SAT intersections with inflated obstacle XY AABBs."""
    polygon, boxes = np.asarray(polygon, float), np.asarray(boxes, float).reshape(-1, 2, 2)
    if not len(boxes):
        return np.zeros(0, dtype=bool)
    center = boxes.mean(axis=1)
    extent = (boxes[:, 1] - boxes[:, 0]) / 2 + inflation
    edges = np.roll(polygon, -1, axis=0) - polygon
    axes = np.vstack((np.eye(2), np.column_stack((-edges[:, 1], edges[:, 0]))))
    hit = np.ones(len(boxes), dtype=bool)
    for axis in axes:
        projection = polygon @ axis
        middle, radius = center @ axis, extent @ np.abs(axis)
        hit &= (middle + radius >= projection.min()) & (middle - radius <= projection.max())
        if not np.any(hit):
            break
    return hit


def _angle_delta(after, before):
    return float((after - before + np.pi) % (2 * np.pi) - np.pi)


class _SE2Navigation:
    """Bounded 16-heading holonomic lattice with conservative swept-foot checks."""
    mode = "se2_rectangle"
    heading_count = 16

    def __init__(self, robot, objects, targets, options):
        self.options = options
        self.start = robot.position[:2].copy()
        self.initial_yaw = float(np.arctan2(robot.initial_pose[1, 0], robot.initial_pose[0, 0]))
        self.yaw_step = 2 * np.pi / self.heading_count
        measured = getattr(robot, "base_world_vertices", None)
        self.footprint_evidence = deepcopy(getattr(robot, "base_geometry_validation",
                                                  _validated_base_vertices(None, robot.base_box)[1]))
        if measured is not None and self.footprint_evidence.get("base_geometry_validated") is not True:
            measured = None
            self.footprint_evidence["base_geometry_fallback_reason"] = "Native vertices lack _Robot validation; measured AABB fallback retained."
        if measured is None:
            measured = _corners(robot.base_box)
            self.footprint_evidence.update(footprint_source="measured_base_aabb_corners", base_geometry_validated=False)
            if not self.footprint_evidence.get("base_geometry_fallback_reason"):
                self.footprint_evidence["base_geometry_fallback_reason"] = "Validated native base vertices are absent on the robot."
        self.footprint_evidence["footprint_vertex_count"] = len(measured)
        self.footprint_evidence["footprint_projection"] = "inverse_full_robot_pose_union_yaw_only_projection"
        # Use validated native vertices, or the unchanged eight-corner AABB
        # fallback. Neither supplied local bounds nor chassis guesses are used.
        local = transform_points(measured, np.linalg.inv(robot.initial_pose))[:, :2]
        self.local_bounds = np.stack((local.min(axis=0), local.max(axis=0)))
        # The lattice is planar. Union the same measured corners' yaw-only
        # projection so measured roll/pitch cannot shrink their world footprint.
        yaw_rotation = self._rotation(self.initial_yaw)
        projected = (measured[:, :2] - self.start) @ yaw_rotation
        self.local_bounds[0] = np.minimum(self.local_bounds[0], projected.min(axis=0))
        self.local_bounds[1] = np.maximum(self.local_bounds[1], projected.max(axis=0))
        low, high = self.local_bounds
        self.local_polygon = np.array([[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]])
        self.radius = float(np.linalg.norm(self.local_polygon, axis=1).max())
        points = np.asarray([self.start, *[_box(row["aabb"]).mean(axis=0)[:2] for row in targets]])
        minimum, maximum = points.min(axis=0) - 2., points.max(axis=0) + 2.
        self.state_cap = min(options.max_grid_cells, 250000)
        self.resolution = max(options.grid_resolution_m,
            np.sqrt(self.heading_count * np.prod(maximum - minimum) / self.state_cap))
        while True:
            self.lower = self.start - np.ceil((self.start - minimum) / self.resolution) * self.resolution
            shape_xy = np.ceil((maximum - self.lower) / self.resolution).astype(int) + 1
            self.shape = (self.heading_count, int(shape_xy[1]), int(shape_xy[0]))
            if np.prod(self.shape) <= self.state_cap:
                break
            self.resolution *= 1.01
        self.obstacles = []
        for row in objects:
            box = _box(row["aabb"])
            if box[0, 2] <= robot.base_box[1, 2] and box[1, 2] >= robot.base_box[0, 2]:
                self.obstacles.append((box[0, :2], box[1, :2], row["name"]))
        reach_low = self.lower - self.radius - options.collision_margin_m
        reach_high = self.lower + (shape_xy - 1) * self.resolution + self.radius + options.collision_margin_m
        self.boxes = np.asarray([[low, high] for low, high, _ in self.obstacles
            if np.all(high >= reach_low) and np.all(low <= reach_high)], float).reshape(-1, 2, 2)
        self.polygons = [self.local_polygon @ self._rotation(self.heading(k)).T for k in range(self.heading_count)]
        self.translation_shapes = {}
        for k, direction in product(range(self.heading_count), ((-1, 0), (1, 0), (0, -1), (0, 1))):
            delta = np.array(direction[::-1]) * self.resolution
            self.translation_shapes[k, direction] = _convex_hull_2d(np.vstack((self.polygons[k], self.polygons[k] + delta)))
        self.rotation_shapes = {}
        self.max_rotation_sample_step_rad = 0.
        for k, sign in product(range(self.heading_count), (-1, 1)):
            self.rotation_shapes[k, sign] = self._rotation_samples(self.heading(k), self.heading(k) + sign * self.yaw_step)
        self.stance_positions, self.stance_yaws = {}, {}
        self.previous, self.pose_cache = {}, {}
        self.failure = None
        self.initial = self.node(self.start, self.initial_yaw)
        if self.pose_blocked(self.start, self.initial_yaw):
            self.failure = "initial_base_rectangle_collision"
            return
        if self.pose_blocked(self.grid_position(self.initial), self.initial_yaw) or self.translation_blocked(
                self.start, self.grid_position(self.initial), self.initial_yaw):
            self.failure = "actual_start_to_lattice_sweep_blocked"
            return
        self.previous[self.initial] = None
        queue = deque([self.initial])
        while queue:
            node = queue.popleft()
            k, y, x = node
            origin = self.grid_position(node)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                other = (k, y + dy, x + dx)
                if (not 0 <= other[1] < self.shape[1] or not 0 <= other[2] < self.shape[2]
                        or other in self.previous or self._node_blocked(other)):
                    continue
                if not self._blocked(self.translation_shapes[k, (dy, dx)] + origin):
                    self.previous[other] = node
                    queue.append(other)
            for sign in (-1, 1):
                other = ((k + sign) % self.heading_count, y, x)
                if other in self.previous or self._node_blocked(other):
                    continue
                if not self._sampled_rotation_blocked(origin, self.rotation_shapes[k, sign]):
                    self.previous[other] = node
                    queue.append(other)

    @staticmethod
    def _rotation(yaw):
        return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

    def heading(self, k):
        return self.initial_yaw + k * self.yaw_step

    def node(self, point, yaw=None):
        x, y = np.rint((np.asarray(point) - self.lower) / self.resolution).astype(int)
        k = 0 if yaw is None else int(np.rint(_angle_delta(yaw, self.initial_yaw) / self.yaw_step)) % self.heading_count
        return k, int(y), int(x)

    def grid_position(self, node):
        return self.lower + np.array([node[2], node[1]]) * self.resolution

    def position(self, node):
        return self.stance_positions.get(node, self.grid_position(node))

    def _blocked(self, polygon, extra_inflation=0.):
        inflation = self.options.collision_margin_m + extra_inflation
        if not len(self.boxes):
            return False
        near = np.all(self.boxes[:, 1] + inflation >= polygon.min(axis=0), axis=1) & np.all(
            self.boxes[:, 0] - inflation <= polygon.max(axis=0), axis=1)
        return bool(np.any(polygon_aabb_intersections(polygon, self.boxes[near], inflation)))

    def pose_blocked(self, xy, yaw):
        return self._blocked(self.local_polygon @ self._rotation(yaw).T + xy)

    def _node_blocked(self, node):
        if node not in self.pose_cache:
            self.pose_cache[node] = self._blocked(self.polygons[node[0]] + self.grid_position(node))
        return self.pose_cache[node]

    def translation_blocked(self, begin, end, yaw):
        polygon = self.local_polygon @ self._rotation(yaw).T
        return self._blocked(_convex_hull_2d(np.vstack((polygon + begin, polygon + end))))

    def _rotation_samples(self, begin, end):
        delta = _angle_delta(end, begin)
        count = max(1, int(np.ceil(abs(delta) / np.deg2rad(5.))))
        step = abs(delta) / count
        self.max_rotation_sample_step_rad = max(self.max_rotation_sample_step_rad, step)
        inflation = 2 * self.radius * np.sin(step / 4)
        return ([self.local_polygon @ self._rotation(begin + fraction * delta).T
                 for fraction in np.linspace(0., 1., count + 1)], float(inflation))

    def _sampled_rotation_blocked(self, xy, samples):
        polygons, inflation = samples
        return any(self._blocked(polygon + xy, inflation) for polygon in polygons)

    def rotation_blocked(self, xy, begin, end):
        return self._sampled_rotation_blocked(xy, self._rotation_samples(begin, end))

    def route(self, node):
        current, result = node, []
        while current is not None:
            result.append([*self.grid_position(current).tolist(), self.heading(current[0])])
            current = self.previous[current]
        result = [[*self.start.tolist(), self.initial_yaw], *result[::-1]]
        if node in self.stance_positions:
            result.extend([[*self.stance_positions[node].tolist(), self.heading(node[0])],
                           [*self.stance_positions[node].tolist(), self.stance_yaws[node]]])
        for index in range(1, len(result)):
            result[index][2] = result[index - 1][2] + _angle_delta(result[index][2], result[index - 1][2])
        return result

    @staticmethod
    def _same_stance(point, yaw, other_point, other_yaw):
        return bool(np.allclose(point, other_point, rtol=0., atol=1e-9) and abs(_angle_delta(yaw, other_yaw)) <= 1e-9)

    def _stance_proposals(self, target, objects, options):
        center = _box(target["aabb"]).mean(axis=0)
        box, _ = _support_bounds(target, objects)
        proposals, raw_counts = [], Counter()
        def add(point, provenance):
            raw_counts[provenance["kind"]] += 1
            difference = center[:2] - point
            yaw = float(np.arctan2(difference[1], difference[0])) if np.linalg.norm(difference) > 1e-12 else self.initial_yaw
            for proposal in proposals:
                if self._same_stance(point, yaw, proposal["point"], proposal["yaw"]):
                    proposal["provenance"].append(provenance)
                    return
            proposals.append({"point": np.asarray(point, float).copy(), "yaw": yaw, "provenance": [provenance]})
        for axis, sign, fraction in product((0, 1), (-1, 1), (.2, .5, .8)):
            edge = box[0 if sign < 0 else 1, axis]
            point = center[:2].copy()
            point[axis] = edge + sign * (self.radius + .14)
            point[1 - axis] = (1 - fraction) * box[0, 1 - axis] + fraction * box[1, 1 - axis]
            for _ in range(4):
                yaw = float(np.arctan2(center[1] - point[1], center[0] - point[0]))
                projection = (self.local_polygon @ self._rotation(yaw).T)[:, axis]
                point[axis] = edge + sign * .14 - (projection.min() if sign > 0 else projection.max())
            add(point, {"kind": "edge", "axis": axis, "sign": sign, "fraction": fraction})
        if np.linalg.norm(self.start - center[:2]) <= options.max_stance_target_distance_m:
            add(self.start, {"kind": "initial_root"})
        offset = float(np.arctan2(self.start[1] - center[1], self.start[0] - center[0]))
        for radius, index in product((.55, .75, .95, 1.15), range(16)):
            angle = offset + index * 2 * np.pi / 16
            add(center[:2] + radius * np.array([np.cos(angle), np.sin(angle)]),
                {"kind": "radial", "radius_m": radius, "direction_index": index, "angle_rad": angle})
        proposals.sort(key=lambda proposal: np.linalg.norm(proposal["point"] - center[:2]) +
                       .05 * np.linalg.norm(proposal["point"] - self.start))
        return proposals, raw_counts, offset

    def _stance_connection(self, point, yaw, reserved):
        _, y, x = self.node(point)
        nodes = []
        for node in product(range(self.heading_count), range(y - 1, y + 2), range(x - 1, x + 2)):
            if node not in self.previous:
                continue
            if node in self.stance_positions and not self._same_stance(
                    point, yaw, self.stance_positions[node], self.stance_yaws[node]):
                continue
            if node in reserved and not self._same_stance(point, yaw, *reserved[node]):
                continue
            nodes.append(node)
        nodes.sort(key=lambda node: np.linalg.norm(self.grid_position(node) - point) +
                   .1 * abs(_angle_delta(self.heading(node[0]), yaw)))
        for node in nodes:
            heading = self.heading(node[0])
            if not self.translation_blocked(self.grid_position(node), point, heading) and not self.rotation_blocked(point, heading, yaw):
                if node in self.stance_positions:
                    return node, self.stance_positions[node], self.stance_yaws[node]
                return node, point, yaw
        return None

    def stances(self, target, objects, options):
        center = _box(target["aabb"]).mean(axis=0)
        proposals, raw_counts, offset = self._stance_proposals(target, objects, options)
        feasible, reserved, pose_counts, connected_counts = [], {}, Counter(), Counter()
        for proposal in proposals:
            point, yaw = proposal["point"], proposal["yaw"]
            kinds = {source["kind"] for source in proposal["provenance"]}
            if np.linalg.norm(point - center[:2]) > options.max_stance_target_distance_m or self.pose_blocked(point, yaw):
                continue
            pose_counts.update(kinds)
            connection = self._stance_connection(point, yaw, reserved)
            if connection is None:
                continue
            node, point, yaw = connection
            reserved[node] = (point.copy(), yaw)
            connected_counts.update(kinds)
            feasible.append((proposal, node, point, yaw))
        feasible.sort(key=lambda candidate: np.linalg.norm(candidate[2] - center[:2]) +
                      .05 * np.linalg.norm(candidate[2] - self.start))
        selected = feasible[:options.max_stances]
        selected_counts, records, result = Counter(), [], []
        for proposal, node, point, yaw in selected:
            if node not in self.stance_positions:
                self.stance_positions[node], self.stance_yaws[node] = point.copy(), yaw
            # Existing mappings are canonical and are never overwritten, even
            # when a later target proposes a numerically equivalent pose.
            point, yaw = self.stance_positions[node], self.stance_yaws[node]
            selected_counts.update({source["kind"] for source in proposal["provenance"]})
            records.append({"lattice_node": list(node), "position_world_xy_m": point.tolist(), "yaw_rad": yaw,
                            "provenance": deepcopy(proposal["provenance"])})
            result.append((node, yaw))
        def source_counts(counter):
            return {kind: counter[kind] for kind in ("edge", "initial_root", "radial")}
        self.last_stance_sampling = {"method": "support_edges_initial_root_and_target_radials",
            "radial_radii_m": [.55, .75, .95, 1.15], "radial_direction_count": 16, "radial_angle_offset_rad": offset,
            "raw_proposal_counts": source_counts(raw_counts), "raw_proposal_count": sum(raw_counts.values()),
            "deduplicated_proposal_count": len(proposals), "pose_clear_counts_by_source": source_counts(pose_counts),
            "connected_feasible_counts_by_source": source_counts(connected_counts), "connected_feasible_count": len(feasible),
            "selected_counts_by_source": source_counts(selected_counts), "selected_count": len(selected),
            "max_stances": options.max_stances, "ranking": "target_distance_m + 0.05 * start_distance_m",
            "selected_stances": records,
            "scope": "Only sampled collision-free poses connected by checked footprint sweeps; full-body travel and unsampled stance feasibility remain unverified."}
        return result


def _navigation_evidence(navigation, options):
    result = {"mode": options.navigation_mode, "grid_shape": list(navigation.shape), "resolution_m": navigation.resolution,
              "reachable_cell_count": len(navigation.previous), "failure": navigation.failure}
    if options.navigation_mode == "se2_rectangle":
        result.update(navigation.footprint_evidence)
        result.update(heading_count=navigation.heading_count, heading_offset_rad=navigation.initial_yaw,
            reachable_state_count=len(navigation.previous),
            total_lattice_states=int(np.prod(navigation.shape)), state_cap=navigation.state_cap,
            local_rectangle_bounds_m=navigation.local_bounds.tolist(),
            local_rectangle_center_offset_m=navigation.local_bounds.mean(axis=0).tolist(),
            maximum_corner_radius_m=navigation.radius,
            maximum_rotation_sample_spacing_deg=float(np.rad2deg(navigation.max_rotation_sample_step_rad)),
            rotation_missed_motion_inflation_formula="2 * maximum_corner_radius_m * sin(actual_sample_step_rad / 4)")
    return result
