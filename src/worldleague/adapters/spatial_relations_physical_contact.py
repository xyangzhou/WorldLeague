"""Convert original contact pairs into pending, measured robot operations.

Used by worldleague scene --help.
"""
from __future__ import annotations


from worldleague.tasks.interaction_setup import array as _array


import numpy as np


OBJECTS = ('obj1', 'obj2')
# Provisional solver tolerance; Core contact traces must calibrate it natively.
PENETRATION_TOLERANCE_M = .001


def _snapshot(obj):
    """Read current native geometry, physical flags and velocity; never set state."""
    position, quaternion = (_array(value) for value in obj.get_position_orientation())
    low, high = (_array(value) for value in obj.aabb)
    linear, angular = (_array(value) for value in (obj.get_linear_velocity(), obj.get_angular_velocity()))
    if any(value.shape != (3,) for value in (position, low, high, linear, angular)) or (
            quaternion.shape != (4,) or np.linalg.norm(quaternion) == 0 or np.any(high <= low)):
        raise ValueError('Invalid measured native object state.')
    return {'name': obj.name, 'category': obj.category, 'model': obj.model,
        'position': position.tolist(), 'quaternion_xyzw': quaternion.tolist(),
        'aabb_min': low.tolist(), 'aabb_max': high.tolist(),
        'visual_only': bool(obj.visual_only), 'fixed_base': bool(obj.fixed_base),
        'linear_speed_m_s': float(np.linalg.norm(linear)), 'angular_speed_rad_s': float(np.linalg.norm(angular))}


def _current_pair_separations(api, scene_idx, row_paths, column_paths):
    """Read current native pair buffers when the impulse-gated wrapper is empty.

    No cached witness, wake, physics step, or geometry approximation is used.
    Unsupported native APIs remain unavailable rather than fabricating samples.
    """
    views = getattr(api, '_CONTACT_VIEW', {})
    if scene_idx not in views:
        return []
    rows = [api.get_body_row_idx(path) for path in row_paths]
    columns = [api.get_body_col_idx(path) for path in column_paths]
    if any(scene != scene_idx for scene, _ in rows + columns):
        raise ValueError('Contact buffer scene identity mismatch.')
    forces, points, normals, distances, counts, starts = views[scene_idx].get_contact_data(dt=1.0)
    values = []
    capacity = min(len(forces), len(points), len(normals), len(distances))
    for _, row in rows:
        for _, column in columns:
            count = int(_array(counts[row, column]).item())
            start = int(_array(starts[row, column]).item())
            if count < 0 or (count and (start < 0 or start + count > capacity)):
                raise ValueError('Incomplete native pair contact buffer.')
            if count:
                pair = _array(distances[start:start + count]).reshape(-1)
                if len(pair) != count or not np.isfinite(pair).all():
                    raise ValueError('Invalid native pair separation samples.')
                values.extend(float(value) for value in pair)
    return values


def _contact_geometry(objects, native_contact, gap, tolerance=PENETRATION_TOLERANCE_M):
    """Read PhysX separation; AABB overlap alone never establishes penetration."""
    if not native_contact and gap > 0.:
        return {'available': True, 'relation': 'separated', 'penetrating': False,
            'minimum_separation_m': gap, 'separation_source': 'positive_aabb_clearance',
            'penetration_tolerance_m': tolerance}
    if not native_contact:
        return {'available': False, 'relation': 'unresolved', 'penetrating': None,
            'reason': 'No native contact and no strictly positive conservative clearance.'}
    try:
        from omnigibson.utils.usd_utils import RigidContactAPI
        first, second = (objects[key] for key in OBJECTS)
        first_paths = [str(link.prim_path) for link in first.links.values()]
        second_paths = [str(link.prim_path) for link in second.links.values()]
        separations = []
        sample_access = 'impulse_gated_pair_query'
        query_errors = []
        # Static bodies may have contact-sensor rows but no dynamic filter column.
        # Query the reverse orientation before declaring the pair unavailable.
        for row_paths, column_paths in ((first_paths, second_paths), (second_paths, first_paths)):
            try:
                contacts = RigidContactAPI.get_contact_data(first.scene.idx,
                    row_prim_paths=row_paths, column_prim_paths=column_paths)
                separations = [float(_array(row[5]).reshape(-1)[0]) for row in contacts]
                if not separations:
                    separations = _current_pair_separations(
                        RigidContactAPI, first.scene.idx, row_paths, column_paths)
                    if separations:
                        sample_access = 'current_pair_buffer_without_impulse_gate'
            except KeyError as error:
                query_errors.append(f'KeyError: {error}')
                continue
            if separations:
                break
        if not separations:
            if len(query_errors) == 2:
                raise ValueError('Neither native pair orientation is registered: ' + '; '.join(query_errors))
            raise ValueError('Native contact has no pair-specific separation samples.')
        minimum = min(separations)
        penetrating = minimum < -tolerance
        return {'available': True, 'relation': 'penetrating' if penetrating else 'touching',
            'penetrating': penetrating, 'minimum_separation_m': minimum,
            'separations_m': separations, 'separation_source': 'native_physx_contact_data',
            'native_sample_access': sample_access,
            'penetration_tolerance_m': tolerance}
    except Exception as error:
        return {'available': False, 'relation': 'unresolved', 'penetrating': None,
            'reason': f'{type(error).__name__}: {error}'}
