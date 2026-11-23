"""Cache native mesh calculations during reset. Used by worldleague scene."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import hashlib
import sys
import threading

import numpy as np


def release_variant_preprocess(task, module_name, question_path, state):
    """Load the explicitly selected variant; filename conventions are not task rules."""
    if (state.get('skip_reason') == 'storage_active_single_only_skips_variant_json'
            and module_name == 'tasks.physical_structure.rigid_containment'):
        return {**{key: value for key, value in state.items() if key != 'skip_reason'},
                'preprocess_policy': 'explicit_selected_variant'}
    return state


class MeshGeometryCache:
    """Reuse native triangulation and exact-frame volume/COM results.

    Geometry content, rather than object names, identifies identical meshes.
    World results additionally include the complete native world transform.
    Native transforms, mass properties and convex-hull fallbacks are retained.
    """

    def __init__(self, volume_function, mesh_function, world_pose):
        self.volume_function = volume_function
        self.mesh_function = mesh_function
        self.world_pose = world_pose
        self.meshes, self.volumes = {}, {}
        self.local = threading.local()
        self.stats = defaultdict(int)

    def geometry_key(self, prim):
        mesh_type = prim.GetTypeName()
        attributes = {
            "Mesh": ("points", "faceVertexCounts", "faceVertexIndices"),
            "Sphere": ("radius",), "Cube": ("size",),
            "Cone": ("radius", "height"), "Cylinder": ("radius", "height"),
        }.get(mesh_type)
        if attributes is None:
            return None
        digest = hashlib.blake2b(digest_size=20)
        digest.update(mesh_type.encode())
        for name in attributes:
            value = np.ascontiguousarray(prim.GetAttribute(name).Get())
            digest.update(name.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
            self.stats["geometry_bytes_hashed"] += value.nbytes
        return digest.digest()

    def volume(self, mesh_prim, world_frame=False):
        self.stats["volume_calls"] += 1
        key = self.geometry_key(mesh_prim)
        if key is None:
            return self.volume_function(mesh_prim, world_frame=world_frame)
        transform = None
        if world_frame:
            matrix = np.ascontiguousarray(self.world_pose(mesh_prim.GetPath().pathString), dtype=np.float64)
            transform = matrix.tobytes()
        result_key = (key, transform)
        result = self.volumes.get(result_key)
        if result is None:
            self.stats["volume_misses"] += 1
            previous = getattr(self.local, "geometry_key", None)
            self.local.geometry_key = key
            try:
                result = self.volume_function(mesh_prim, world_frame=world_frame)
            finally:
                self.local.geometry_key = previous
            self.volumes[result_key] = (result[0], result[1].clone())
        else:
            self.stats["world_volume_hits" if world_frame else "local_volume_hits"] += 1
        return result[0], result[1].clone()

    def mesh(self, mesh_prim, include_normals=True, include_texcoord=True, world_frame=False):
        key = getattr(self.local, "geometry_key", None)
        # Other mesh consumers, including deformable geometry, retain native behavior.
        if key is None or include_normals or include_texcoord:
            return self.mesh_function(mesh_prim, include_normals, include_texcoord, world_frame)
        self.stats["mesh_calls"] += 1
        canonical = self.meshes.get(key)
        if canonical is None:
            self.stats["mesh_misses"] += 1
            canonical = self.mesh_function(mesh_prim, False, False, False)
            self.meshes[key] = canonical
        else:
            self.stats["mesh_hits"] += 1
        # The original native calculation can mutate its returned mesh.
        result = canonical.copy()
        if world_frame:
            result.apply_transform(self.world_pose(mesh_prim.GetPath().pathString))
        return result


@contextmanager
def native_loading_context():
    """Reuse identical native mesh conversions within one scene load."""
    import omnigibson.utils.usd_utils as usd

    cache = MeshGeometryCache(usd.get_mesh_volume_and_com, usd.mesh_prim_to_trimesh_mesh,
                              usd.PoseAPI.get_world_pose_with_scale)
    changes = []
    for name, replacement in (("get_mesh_volume_and_com", cache.volume), ("mesh_prim_to_trimesh_mesh", cache.mesh)):
        original = getattr(usd, name)
        for module_name, module in tuple(sys.modules.items()):
            if module is not None and module_name.startswith("omnigibson.") and vars(module).get(name) is original:
                changes.append((module, name, original))
                setattr(module, name, replacement)
    try:
        yield
    finally:
        for module, name, original in reversed(changes):
            setattr(module, name, original)
