"""Adapt a direct MJCF import to the installed OmniGibson link layout.

Run with python -m worldleague.prepare_robot native --help.
The raw USD remains unchanged; this module produces a separate derived asset.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def prepare_asset(raw_path, source_dir, output):
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    manifest = json.loads((source_dir / "manifest.json").read_text())
    physics = json.loads((source_dir / "physics.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    raw = Usd.Stage.Open(str(Path(raw_path).resolve()))
    while True:
        instances = [p for p in raw.Traverse() if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    flat = raw.Flatten()
    source = Usd.Stage.Open(flat)
    stage = Usd.Stage.CreateNew(str(output / "pandaomron.usda"))
    root = UsdGeom.Xform.Define(stage, "/PandaOmron").GetPrim()
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    old_bodies = {p.GetName(): p for p in source.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    old_paths = {str(p.GetPath()): f"/PandaOmron/{name}" for name, p in old_bodies.items()}
    raw_root = source.GetDefaultPrim()
    body_container = next(iter(old_bodies.values())).GetParent()
    for prim in raw_root.GetChildren():
        if prim.GetPath() != body_container.GetPath() and prim.GetName() not in ("joints", "worldBody"):
            Sdf.CopySpec(flat, prim.GetPath(), stage.GetRootLayer(), f"/PandaOmron/{prim.GetName()}")
    for name, prim in old_bodies.items():
        Sdf.CopySpec(flat, prim.GetPath(), stage.GetRootLayer(), old_paths[str(prim.GetPath())])
    for prim in source.Traverse():
        if not prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() == "PhysicsJoint":
            continue
        joint = UsdPhysics.Joint(prim)
        parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if not parents:
            continue  # The installed BaseObject creates its own world-to-dummy-root joint.
        parent = old_paths[str(parents[0])] if parents else "/PandaOmron/robot0_base"
        name = prim.GetName() if parents else "rootJoint"
        target = parent + "/" + name
        Sdf.CopySpec(flat, prim.GetPath(), stage.GetRootLayer(), target)
        copied = UsdPhysics.Joint(stage.GetPrimAtPath(target))
        copied.GetBody0Rel().SetTargets([old_paths[str(p)] for p in parents])
        copied.GetBody1Rel().SetTargets([old_paths[str(p)] for p in children])
    # Remap material, site and other internal targets after flattening link paths.
    for prim in stage.Traverse():
        for relationship in prim.GetRelationships():
            targets = []
            for target in relationship.GetTargets():
                for old, new in sorted(old_paths.items(), key=lambda x: -len(x[0])):
                    if target.HasPrefix(Sdf.Path(old)):
                        target = target.ReplacePrefix(Sdf.Path(old), Sdf.Path(new))
                        break
                targets.append(target)
            relationship.SetTargets(targets)
    # Replace the importer's D6 joint with the source's serial x/y/yaw sequence.
    # Anchor offsets preserve rotation about the source's (-0.21, 0, 0) pivot.
    pivot = next(j["position_body"] for j in physics["joints"]
                 if j["name"] == "mobilebase0_joint_mobile_yaw")
    for name in ("panda_base_x", "panda_base_y"):
        prim = UsdGeom.Xform.Define(stage, f"/PandaOmron/{name}").GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*pivot))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    sequence = [
        ("mobilebase0_joint_mobile_forward", "Prismatic", "X", "robot0_base", "panda_base_x", pivot, [0., 0., 0.]),
        ("mobilebase0_joint_mobile_side", "Prismatic", "Y", "panda_base_x", "panda_base_y", [0., 0., 0.], [0., 0., 0.]),
        ("mobilebase0_joint_mobile_yaw", "Revolute", "Z", "panda_base_y", "mobilebase0_base", [0., 0., 0.], pivot),
    ]
    for name, kind, axis, parent, child, pos0, pos1 in sequence:
        cls = UsdPhysics.PrismaticJoint if kind == "Prismatic" else UsdPhysics.RevoluteJoint
        joint = cls.Define(stage, f"/PandaOmron/{parent}/{name}")
        joint.CreateBody0Rel().SetTargets([f"/PandaOmron/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"/PandaOmron/{child}"])
        joint.CreateAxisAttr(axis)
        joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
        joint.CreateCollisionEnabledAttr(False)
        PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear" if kind == "Prismatic" else "angular")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear" if kind == "Prismatic" else "angular")
        actuator = next(a for a in physics["actuators"] if a["joint"] == name)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.)
        drive.CreateDampingAttr(float(actuator["gainprm"][0]))
        drive.CreateMaxForceAttr(float(max(abs(v) for v in actuator["forcerange"])))
    # The old-API JointPrim requires both state and drive schemas to be authored
    # before it loads any active joint.
    actuator_values = {a["joint"]: a for a in physics["actuators"]}
    joint_values = {value["name"]: value for value in physics["joints"]}
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.RevoluteJoint)):
            continue
        kind = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
        PhysxSchema.JointStateAPI.Apply(prim, kind)
        PhysxSchema.PhysxJointAPI.Apply(prim).CreateArmatureAttr(joint_values[prim.GetName()]["armature"])
        drive = UsdPhysics.DriveAPI.Apply(prim, kind)
        actuator = actuator_values[prim.GetName()]
        limits = actuator["forcerange"] if actuator["forcelimited"] else actuator["ctrlrange"]
        drive.CreateMaxForceAttr(float(max(abs(v) for v in limits)))
    for name in old_bodies:
        container = stage.GetPrimAtPath(f"/PandaOmron/{name}/collisions")
        if not container:
            continue
        source_friction = {tuple(g["friction"]) for g in physics["geoms"]
                           if g["body"] == name and (g["contype"] or g["conaffinity"])}
        geometries = [p for p in Usd.PrimRange(container) if p.IsA(UsdGeom.Gprim)]
        if not geometries:
            continue
        if len(source_friction) != 1:
            raise ValueError(f"Geometry-specific friction needs an explicit mapping on {name}.")
        friction = next(iter(source_friction))
        material = UsdShade.Material.Define(stage, f"/PandaOmron/Looks/physics_{name}")
        properties = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        properties.CreateStaticFrictionAttr(friction[0])
        properties.CreateDynamicFrictionAttr(friction[0])
        properties.CreateRestitutionAttr(0.)
        # The MJCF importer puts collision schemas on the grouping Xform. The
        # installed RigidPrim discovers collision schemas on each actual Gprim.
        for geom in geometries:
            UsdPhysics.CollisionAPI.Apply(geom).CreateCollisionEnabledAttr(True)
            if geom.IsA(UsdGeom.Mesh):
                UsdPhysics.MeshCollisionAPI.Apply(geom).CreateApproximationAttr("convexHull")
            UsdShade.MaterialBindingAPI.Apply(geom).Bind(material, materialPurpose="physics")
        container.RemoveAPI(UsdPhysics.CollisionAPI)
        container.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        container.RemoveAPI(PhysxSchema.PhysxMeshMergeCollisionAPI)
    # Restore source-compiled inertias, including masses inferred from geometry.
    body_values = {body["name"]: body for body in manifest["bodies"]}
    for name in [*old_bodies, "panda_base_x", "panda_base_y"]:
        prim = stage.GetPrimAtPath(f"/PandaOmron/{name}")
        value = body_values.get(name)
        mass = UsdPhysics.MassAPI.Apply(prim)
        if value and value["mass_kg"] > 0:
            mass.CreateMassAttr(value["mass_kg"])
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*value["inertia_kg_m2"]))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*value["inertial_position"]))
            q = value["inertial_quaternion_wxyz"]
            mass.CreatePrincipalAxesAttr(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        else:
            # Installed OmniGibson assigns 1e-6 kg to virtual links too.
            mass.CreateMassAttr(1e-6)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-8, 1e-8, 1e-8))
    # Remove importer camera guesses, then author the declared MuJoCo optical frames.
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdGeom.Camera):
            stage.RemovePrim(prim.GetPath())
    for name, values in manifest["cameras"].items():
        camera = UsdGeom.Camera.Define(stage, f"/PandaOmron/{values['parent_body']}/{name}")
        transform = UsdGeom.Xformable(camera)
        transform.AddTranslateOp().Set(Gf.Vec3d(*values["pos"]))
        q = values["quat"]
        transform.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
        fovy = float(values.get("camera_attribs", {}).get("fovy", 45.))
        aperture = 20.955
        focal_length = aperture / (2 * np.tan(np.deg2rad(fovy) / 2))
        camera.CreateHorizontalApertureAttr(aperture)
        camera.CreateVerticalApertureAttr(aperture)
        camera.CreateFocalLengthAttr(float(focal_length))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100.))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    articulation = stage.GetPrimAtPath("/PandaOmron/robot0_base")
    UsdPhysics.ArticulationRootAPI.Apply(articulation)
    PhysxSchema.PhysxArticulationAPI.Apply(articulation).CreateEnabledSelfCollisionsAttr(True)
    # USDObject marks imported transforms as preloaded in this SDK. Normalize
    # every geometry/link op without changing its local matrix.
    for prim in stage.Traverse():
        xform = UsdGeom.Xformable(prim)
        if not xform:
            continue
        before = xform.GetLocalTransformation()
        transform = Gf.Transform(before)
        reset = xform.GetResetXformStack()
        xform.ClearXformOpOrder()
        for name in list(prim.GetPropertyNames()):
            if name.startswith("xformOp:"):
                prim.RemoveProperty(name)
        translation = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orientation = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        scale = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        translation.Set(transform.GetTranslation())
        orientation.Set(transform.GetRotation().GetQuat())
        scale.Set(transform.GetScale())
        xform.SetResetXformStack(reset)
    stage.GetRootLayer().Save()
    return output / "pandaomron.usda"
