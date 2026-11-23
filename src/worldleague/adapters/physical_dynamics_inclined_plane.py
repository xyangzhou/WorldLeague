"""Convert original inclined planes into real grasp, placement and release trials.

Run CPU batching from the repository root with the behavior environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
Native hooks are read-only and run through the existing guarded robot runtime.
"""
from __future__ import annotations


import math

import numpy as np


from worldleague.tasks.inclined_apparatus import NATIVE_RECEIPT, materialize_apparatus


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite native or source measurement.')
    return value


def _pose(obj):
    position, quaternion = [_array(value) for value in obj.get_position_orientation()]
    low, high = [_array(value) for value in obj.aabb]
    kinematic = bool(getattr(obj, 'kinematic_only', False))
    try:
        linear = float(np.linalg.norm(_array(obj.get_linear_velocity())))
        angular = float(np.linalg.norm(_array(obj.get_angular_velocity())))
        velocity_source = 'native_dynamic_velocity'
    except AttributeError:
        if not (kinematic and obj.fixed_base):
            raise
        # Installed RigidKinematicPrim has no velocity state. Its continuously
        # checked pose establishes preservation; do not invent measured zeros.
        linear = angular = None
        velocity_source = 'fixed_native_kinematic_body_without_dynamic_velocity_state'
    return {'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
            'scale': _array(obj.scale).tolist(),
            'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
            'kinematic_only': kinematic, 'velocity_source': velocity_source,
            'linear_speed_m_s': linear, 'angular_speed_rad_s': angular}


def _friction(obj):
    rows = []
    for link in obj.links.values():
        for mesh in link.collision_meshes.values():
            material = mesh.get_applied_physics_material()
            if material is None:
                raise ValueError('Source collision mesh has no measurable applied friction material.')
            values = [material.get_static_friction(), material.get_dynamic_friction()]
            if not all(value is not None and math.isfinite(float(value)) for value in values):
                raise ValueError('Native friction material is not measurable.')
            rows.append({'mesh': mesh.prim_path, 'static_friction': float(values[0]),
                         'dynamic_friction': float(values[1])})
    if not rows:
        raise ValueError('Source object has no native collision meshes.')
    return rows


def _native_time():
    import omnigibson as og
    result = {'time_s': float(og.sim.current_time), 'physics_dt_s': float(og.sim.get_physics_dt()),
              'sim_step_dt_s': float(og.sim.get_sim_step_dt())}
    if not all(math.isfinite(value) for value in result.values()) or min(
            result['physics_dt_s'], result['sim_step_dt_s']) <= 0:
        raise ValueError('Native clock and time increments must be measurable and positive.')
    return result


def measure_source_apparatus(session, target, slope, parameters):
    """Read pre-translation source physics for root's native setup integration."""
    rows = {}
    for key, obj in (('target', target), ('slope', slope)):
        rows[key] = {**_pose(obj), 'name': obj.name, 'category': obj.category,
            'model': getattr(obj, 'model', None), 'mass_kg': float(obj.mass), 'materials': _friction(obj)}
    proposal = parameters['apparatus_proposal']
    envelope = materialize_apparatus(proposal, pickup_bounds=[rows['target']['aabb_min'], rows['target']['aabb_max']])
    return {'schema': NATIVE_RECEIPT, 'source_identity': parameters['source_identity'],
        'source_baseline': rows, 'envelope': envelope,
        'source_camera_position_m': _array(session.scene_camera_position).tolist(),
        'source_camera_quaternion_xyzw': _array(session.scene_camera_quaternion).tolist()}
