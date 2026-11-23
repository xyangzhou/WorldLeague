"""Convert source triangles into supported robot perimeter and vertex inspections.

Run CPU generation from the repository root with the behavior Python environment:
CategoryConverter().convert_all(sources, ConversionContext(batch_root)).
Set OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2.
The campaign handles screenshots; this module never launches simulation or actions.
"""
from __future__ import annotations


import hashlib
import math

import numpy as np


LAYOUT_VERSION = 'seeded_aabb_centers_v1'
INITIAL_SHAPE_MARGIN = .025  # Half the existing 5% side-equality scoring tolerance.


def shape_geometry(centres, shape):
    """Evaluate source-derived shapes in world XY, normalized by longest span."""
    points = np.asarray(centres, dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError('Three finite horizontal native centers are required.')
    pairs = ((0, 1), (0, 2), (1, 2))
    lengths = np.array([np.linalg.norm(points[i] - points[j]) for i, j in pairs])
    longest = float(lengths.max())
    if longest <= 0:
        return {'satisfied': False, 'longest_span_m': longest, 'reason': 'collapsed_centers'}
    i, j = pairs[int(lengths.argmax())]
    direction = (points[j] - points[i]) / longest
    residual = float(np.abs((points - points[i]) @ [-direction[1], direction[0]]).max())
    differences = np.array([abs(lengths[i] - lengths[j]) / longest for i, j in pairs])
    area = abs(float(np.cross(points[1] - points[0], points[2] - points[0]))) / 2
    distinct = bool(lengths.min() > max(.03, .05 * longest))
    if shape == 'collinear':
        satisfied = distinct and residual <= min(.02, .02 * longest)
    elif shape == 'equilateral':
        satisfied = distinct and 2 * area / longest ** 2 >= .05 and differences.max() <= .05
    elif shape == 'isosceles':
        satisfied = distinct and 2 * area / longest ** 2 >= .05 and differences.min() <= .05 < differences.max()
    elif shape == 'random triangle':
        satisfied = distinct and 2 * area / longest ** 2 >= .05 and differences.min() > .05
    else:
        raise ValueError('Unsupported declared source shape.')
    return {'satisfied': bool(satisfied), 'shape': shape, 'plane': 'world_xy',
        'pairwise_distances_m': lengths.tolist(), 'longest_span_m': longest,
        'minimum_center_separation_m': max(.03, .05 * longest), 'normalized_area': 2 * area / longest ** 2,
        'normalized_side_differences': differences.tolist(), 'maximum_perpendicular_residual_m': residual,
        'line_angular_residual_rad': math.atan2(residual, longest),
        'line_maximum_angular_residual_rad': math.atan(.02)}


def initial_shape_is_wrong(centres, target_shape, margin=INITIAL_SHAPE_MARGIN):
    """Reject the target by a side-ratio margin, retaining scoring's area bound.

    The margin is half the existing 0.05 equality band. For a scalene target
    (source name ``random triangle``), require a pair inside the equality band
    by that margin. The other two targets start from distinctly scalene sides.
    This changes initialization only; historical scoring tolerances are retained.
    """
    geometry = shape_geometry(centres, target_shape)
    if geometry['satisfied'] or geometry.get('normalized_area', 0.) < .05:
        return False
    differences = geometry.get('normalized_side_differences', [])
    if not differences:
        return False
    return bool(min(differences) <= .05 - margin if target_shape == 'random triangle'
                else min(differences) >= .05 + margin)


def seeded_initial_layout(*, object_keys, aabb_half_extents_xy, root_to_aabb_center_xy,
                          root_z_m, support_aabb_xy, target_shape, seed,
                          source_identity, minimum_clearance_m=.005, max_attempts=4096):
    """Pure supported-layout recipe, called after native asset measurements.

    Shared setup must preserve orientations/scales, supply the horizontal support
    rectangle and measured root-to-AABB offsets, place only during initialization,
    then store this returned receipt under
    ``category_interaction_setup['geometric_initial_layout']``. Add
    ``accepted=True`` only after native support/grasp-access/settling checks.
    ``minimum_clearance_m`` must include the setup's grasp-space requirement;
    5 mm is only the historical object-clearance floor, not a graspability proof.
    The qualifier recomputes the recipe and checks the settled native geometry.
    """
    keys = list(object_keys)
    extents = np.asarray(aabb_half_extents_xy, dtype=float)
    offsets = np.asarray(root_to_aabb_center_xy, dtype=float)
    bounds = np.asarray(support_aabb_xy, dtype=float)
    z = np.asarray(root_z_m, dtype=float)
    if (len(keys) != 3 or len(set(keys)) != 3 or extents.shape != (3, 2) or
            offsets.shape != (3, 2) or bounds.shape != (2, 2) or z.shape != (3,) or
            not all(np.isfinite(a).all() for a in (extents, offsets, bounds, z)) or
            np.any(extents <= 0) or minimum_clearance_m < .005 or
            target_shape not in ('equilateral', 'isosceles', 'random triangle')):
        raise ValueError('Invalid measured seeded-layout inputs.')
    inputs = {'object_keys': keys, 'aabb_half_extents_xy': extents.tolist(),
        'root_to_aabb_center_xy': offsets.tolist(), 'root_z_m': z.tolist(),
        'support_aabb_xy': bounds.tolist(), 'target_shape': target_shape,
        'seed': int(seed), 'source_identity': str(source_identity),
        'minimum_clearance_m': float(minimum_clearance_m), 'max_attempts': int(max_attempts)}
    token = hashlib.sha256(f'{seed}\0{source_identity}\0{LAYOUT_VERSION}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], 'big'))
    low, high = bounds[0] + extents + minimum_clearance_m, bounds[1] - extents - minimum_clearance_m
    if np.any(high <= low):
        raise ValueError('Insufficient support footprint for seeded geometry.')
    for attempt in range(max_attempts):
        if target_shape == 'random triangle':
            # Equilateral starts provide a separated, robustly non-scalene class.
            angle = rng.uniform(0., 2 * math.pi)
            radius = rng.uniform(.2, .48) * float(np.min(bounds[1] - bounds[0]))
            center = rng.uniform(np.max(low, axis=0), np.min(high, axis=0))
            angles = angle + np.arange(3) * 2 * math.pi / 3
            centers = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
        else:
            centers = rng.uniform(low, high)
        if np.any(centers < low) or np.any(centers > high):
            continue
        gaps = [float(np.linalg.norm(np.maximum(np.abs(centers[i] - centers[j]) -
            extents[i] - extents[j], 0.))) for i, j in ((0, 1), (0, 2), (1, 2))]
        geometry = shape_geometry(centers, target_shape)
        if (min(gaps) < minimum_clearance_m or
                min(geometry['pairwise_distances_m']) <= geometry['minimum_center_separation_m'] or
                not initial_shape_is_wrong(centers, target_shape)):
            continue
        roots = np.column_stack((centers - offsets, z))
        return {'version': LAYOUT_VERSION, 'inputs': inputs, 'seed': int(seed),
            'source_identity': str(source_identity), 'attempt': attempt,
            'object_keys': keys, 'initial_positions': roots.tolist(),
            'initial_aabb_centers_xy': centers.tolist(), 'initial_geometry': geometry,
            'initial_false_margin': INITIAL_SHAPE_MARGIN, 'horizontal_clearances_m': gaps}
    raise ValueError('No supported nonoverlapping initially-wrong seeded layout found.')
