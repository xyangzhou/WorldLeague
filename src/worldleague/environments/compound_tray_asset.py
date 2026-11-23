"""Build a local compound robot asset from native loaded source geometry.

Run with python -m worldleague.prepare_robot compound --help.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from worldleague.environments.compound_tray import MODE, array, combine_mass_properties, native_mass_properties


def _matrix(position, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = position
    return matrix


def _author_matrix(prim, matrix):
    from pxr import Gf, UsdGeom
    xform = UsdGeom.Xformable(prim)
    value = Gf.Transform(Gf.Matrix4d(matrix.T.tolist()))
    xform.ClearXformOpOrder()
    for name in list(prim.GetPropertyNames()):
        if name.startswith("xformOp:"):
            prim.RemoveProperty(name)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(value.GetTranslation())
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(value.GetRotation().GetQuat())
    xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(value.GetScale())
    xform.SetResetXformStack(False)


def build_compound_asset(robot, environment, spec, source_asset, output):
    """Author only geometry descendants of the actual moving chassis body."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
    import omnigibson as og
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    support = robot.links[spec["support_link"]]
    stage = Usd.Stage.Open(str(source_asset))
    flattened = stage.Flatten()
    path = output / "pandaomron_compound.usda"
    flattened.Export(str(path))
    target = Usd.Stage.Open(str(path))
    root = target.GetDefaultPrim()
    parent_path = str(root.GetPath()) + "/" + spec["support_link"]
    parent = target.GetPrimAtPath(parent_path)
    live = og.sim.stage
    live_flat = live.Flatten()
    xforms = UsdGeom.XformCache()
    source_tray = environment.scene.object_registry("name", "collection_plate")
    tray_position, tray_quaternion = map(array, source_tray.get_position_orientation())
    boundaries = np.concatenate([array(link.collision_boundary_points_world) for link in source_tray.links.values()
                                 if link.collision_boundary_points_world is not None])
    local = Rotation.from_quat(tray_quaternion).inv().apply(boundaries - tray_position)
    low, high = local.min(0), local.max(0)
    tray_offset = spec["tray_center_support_xy_m"] + [spec["deck_top_support_z_m"] - float(low[2]) + .001]
    declarations = [dict(part, role="support") for part in spec["parts"]]
    declarations.append({"name": "collection_plate", "position_support_m": tray_offset,
                         "mass_kg": spec["tray_mass_kg"], "role": "tray"})
    masses = [native_mass_properties(support)]
    components, materials = [], {}

    def material(source):
        source_path = str(source.GetPath())
        if source_path in materials:
            return UsdShade.Material(target.GetPrimAtPath(materials[source_path]))
        destination = str(root.GetPath()) + "/CompoundLooks/m_" + hashlib.sha256(source_path.encode()).hexdigest()[:12]
        target.DefinePrim(str(root.GetPath()) + "/CompoundLooks", "Scope")
        Sdf.CopySpec(live_flat, source.GetPath(), target.GetRootLayer(), Sdf.Path(destination))
        materials[source_path] = destination
        for prim in Usd.PrimRange(target.GetPrimAtPath(destination)):
            for attribute in prim.GetAttributes():
                connections = attribute.GetConnections()
                if connections:
                    attribute.SetConnections([Sdf.Path(str(p).replace(source_path, destination, 1)) for p in connections])
        return UsdShade.Material(target.GetPrimAtPath(destination))

    for declaration in declarations:
        obj = environment.scene.object_registry("name", declaration["name"])
        if declaration["mass_kg"] is not None:
            obj.root_link.mass = float(declaration["mass_kg"])
        origin, quaternion = map(array, obj.get_position_orientation())
        inverse = np.linalg.inv(_matrix(origin, quaternion))
        part_transform = _matrix(declaration["position_support_m"], [0., 0., 0., 1.])
        group_name = "compound_" + declaration["name"]
        group_path = parent_path + "/" + group_name
        group = UsdGeom.Xform.Define(target, group_path).GetPrim()
        _author_matrix(group, part_transform)
        collision_paths, collision_points, index = [], [], 0
        mass_before = len(masses)
        for link in obj.links.values():
            link_position, link_quaternion = map(array, link.get_position_orientation())
            link_transform = inverse @ _matrix(link_position, link_quaternion)
            masses.append(native_mass_properties(link, part_transform @ link_transform))
            link_world = xforms.GetLocalToWorldTransform(link.prim)
            scale = np.asarray(Gf.Transform(link_world).GetScale())
            for kind, meshes in (("collision", link.collision_meshes), ("visual", link.visual_meshes)):
                for mesh in meshes.values():
                    source = mesh.prim
                    mesh_to_link = np.asarray(xforms.ComputeRelativeTransform(source, link.prim)[0]).T
                    scaled = np.eye(4)
                    scaled[:3, :3] = np.diag(scale)
                    # The SDK aggregates mesh.points_in_parent_frame directly
                    # into each rigid link's collision/visual hull. Keep shapes
                    # directly under that link so no group offset is lost.
                    matrix = part_transform @ link_transform @ scaled @ mesh_to_link
                    name = f"{group_name}_{kind}_{index:03d}"
                    destination = parent_path + "/" + name
                    Sdf.CopySpec(live_flat, source.GetPath(), target.GetRootLayer(), Sdf.Path(destination))
                    prim = target.GetPrimAtPath(destination)
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    prim.RemoveAPI(UsdPhysics.MassAPI)
                    _author_matrix(prim, matrix)
                    for relationship in list(prim.GetRelationships()):
                        if relationship.GetName().startswith("material:binding"):
                            prim.RemoveProperty(relationship.GetName())
                    for purpose in ("", "physics"):
                        bound, _ = UsdShade.MaterialBindingAPI(source).ComputeBoundMaterial(materialPurpose=purpose)
                        if bound:
                            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(bound.GetPrim()), materialPurpose=purpose)
                    if kind == "collision":
                        collision_paths.append(name)
                    index += 1
            if link.collision_boundary_points_world is not None:
                collision_points.extend(Rotation.from_quat(quaternion).inv().apply(
                    array(link.collision_boundary_points_world) - origin).tolist())
        components.append({"name": obj.name, "role": declaration["role"],
            "frame_relative_path": group_name, "position_support_m": declaration["position_support_m"],
            "orientation_support_xyzw": [0., 0., 0., 1.],
            "collider_relative_paths": collision_paths, "collision_points_local_m": collision_points,
            "mass_kg": sum(p["mass_kg"] for p in masses[mass_before:]),
            "independent_rigid_body": False})
    combined = combine_mass_properties(masses)
    mass_api = UsdPhysics.MassAPI.Apply(parent)
    mass_api.CreateMassAttr(combined["mass_kg"])
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*combined["com_m"]))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*combined["principal_inertia_kg_m2"]))
    q = combined["principal_quaternion_xyzw"]
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    PhysxSchema.PhysxContactReportAPI.Apply(parent).CreateThresholdAttr(0.)
    # Persist texture dependencies used by the local licensed source material.
    for prim in target.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if not value:
                continue
            dependency = Path(value.resolvedPath or value.path)
            if not dependency.is_file() or dependency.suffix.lower() == ".mdl":
                continue
            folder = output / "textures"
            folder.mkdir(exist_ok=True)
            copied = folder / (hashlib.sha256(dependency.read_bytes()).hexdigest()[:16] + dependency.suffix)
            shutil.copy2(dependency, copied)
            attribute.Set(Sdf.AssetPath(str(copied)))
    target.GetRootLayer().Save()
    receipt = {"schema": MODE,  "support_link": spec["support_link"],
        "tray_identity": {"category": source_tray.category, "model": source_tray.model},
        "tray_collision_bounds_local_m": [low.tolist(), high.tolist()], "components": components,
        "mass_parts": masses, "combined_mass_properties": combined, "cargo_attachment": False}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
