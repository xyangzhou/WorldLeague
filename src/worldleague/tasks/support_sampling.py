"""Screen private pickup placements on native same-room support surfaces.

Run through src/worldleague/runtime/commands/sample_esi_supports.py inside the retained GPU allocation.
This module never supplies a navigation or manipulation policy to the Agent.
"""
from __future__ import annotations


import numpy as np
from scipy.ndimage import distance_transform_edt

from worldleague.native.utils import _array


def footprint_clearance(free, resolution, radius):
    """Erode by a full circular robot footprint, including map-edge clearance."""
    padded = np.pad(free, 1, constant_values=False)
    distances = distance_transform_edt(padded)[1:-1, 1:-1] * resolution
    return free & (distances > radius + resolution / 2)


def chassis_footprint(robot):
    """Measure R1's physical chassis rather than its empty articulation root."""
    name = robot.base_footprint_link_name
    low, high = (_array(value) for value in robot.links[name].aabb)
    extent = high - low
    if not np.isfinite(extent).all() or np.any(extent[:2] <= 0):
        raise ValueError("Physical chassis bounds are unavailable.")
    return name, extent, float(np.linalg.norm(extent[:2]) / 2 + .03)


def same_body(path, body):
    return body == path or body.startswith(path + "/")


def upward_layers(hits, support_path, minimum, maximum):
    """Keep actual upward-facing layers rather than the furniture AABB maximum."""
    heights = sorted(float(_array(hit["position"])[2]) for hit in hits
        if hit.get("hit") and same_body(support_path, hit["rigidBody"])
        and float(_array(hit["normal"])[2]) > .95
        and minimum <= float(_array(hit["position"])[2]) <= maximum)
    return [z for i, z in enumerate(heights) if i == 0 or z - heights[i - 1] > .005]


def support_edge_points(low, high, margin, stance_xy, robot_xy, maximum_distance=.85):
    """Project free stance cells onto the usable footprint of a support surface."""
    lower, upper = np.asarray(low)[:2] + margin, np.asarray(high)[:2] - margin
    if np.any(lower >= upper) or not len(stance_xy):
        return np.empty((0, 2)), float('inf')
    points = np.clip(stance_xy, lower, upper)
    distances = np.linalg.norm(points - stance_xy, axis=1)
    valid = distances <= maximum_distance
    order = np.argsort(np.linalg.norm(stance_xy - robot_xy, axis=1), kind='stable')
    points = points[order[valid[order]]]
    if len(points):
        _, indices = np.unique(np.round(points, 6), axis=0, return_index=True)
        points = points[np.sort(indices)][:12]
    return points, float(distances.min())
