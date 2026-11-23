"""ESI-Bench scene helpers used by worldleague eval and continuous."""
from __future__ import annotations
import math
import numpy as np

def _normalize_bbox_xyxy(bbox_xyxy, name="bbox", min_extent: float = 1e-3):
    arr = np.array(bbox_xyxy, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"{name} must contain 4 numbers: [xmin, ymin, xmax, ymax], got {bbox_xyxy}")
    xmin, ymin, xmax, ymax = [float(v) for v in arr.tolist()]
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if abs(xmax - xmin) < float(min_extent):
        pad = max(float(min_extent) - abs(xmax - xmin), float(min_extent)) * 0.5
        xmin -= pad
        xmax += pad
    if abs(ymax - ymin) < float(min_extent):
        pad = max(float(min_extent) - abs(ymax - ymin), float(min_extent)) * 0.5
        ymin -= pad
        ymax += pad
    return xmin, ymin, xmax, ymax

def _expand_bbox_xyxy(bbox_xyxy, expansion_ratio: float, expansion_min: float):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    pad_x = max(float(expansion_min), width * float(expansion_ratio))
    pad_y = max(float(expansion_min), height * float(expansion_ratio))
    return (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)

def _bboxes_intersect_xy(bbox_a_xyxy, bbox_b_xyxy, eps: float = 1e-6) -> bool:
    axmin, aymin, axmax, aymax = _normalize_bbox_xyxy(bbox_a_xyxy)
    bxmin, bymin, bxmax, bymax = _normalize_bbox_xyxy(bbox_b_xyxy)
    return not (
        axmax < bxmin - float(eps)
        or bxmax < axmin - float(eps)
        or aymax < bymin - float(eps)
        or bymax < aymin - float(eps)
    )

def _expand_bbox_until_wall_touch(
    bbox_xyxy,
    wall_bboxes_xyxy: list[tuple[float, float, float, float]],
    expansion_ratio: float,
    expansion_min: float,
    step_ratio: float = 0.05,
    max_steps_per_direction: int = 256,
):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    target_pad_x = max(float(expansion_min), width * float(expansion_ratio))
    target_pad_y = max(float(expansion_min), height * float(expansion_ratio))
    step_x = max(1e-3, target_pad_x * float(step_ratio))
    step_y = max(1e-3, target_pad_y * float(step_ratio))

    remaining = {
        "left": target_pad_x,
        "right": target_pad_x,
        "down": target_pad_y,
        "up": target_pad_y,
    }
    finished = {direction: remaining[direction] <= 1e-8 for direction in remaining}

    for _ in range(int(max_steps_per_direction)):
        if all(finished.values()):
            break

        progressed = False
        for direction in ("left", "right", "down", "up"):
            if finished[direction]:
                continue

            delta = min(
                remaining[direction],
                step_x if direction in {"left", "right"} else step_y,
            )
            candidate = [xmin, ymin, xmax, ymax]
            if direction == "left":
                candidate[0] -= delta
            elif direction == "right":
                candidate[2] += delta
            elif direction == "down":
                candidate[1] -= delta
            else:
                candidate[3] += delta

            hit_wall = any(_bboxes_intersect_xy(candidate, wall_bbox) for wall_bbox in wall_bboxes_xyxy)
            if hit_wall:
                finished[direction] = True
                remaining[direction] = 0.0
                continue

            xmin, ymin, xmax, ymax = candidate
            remaining[direction] -= delta
            progressed = True
            if remaining[direction] <= 1e-8:
                finished[direction] = True

        if not progressed and not all(finished.values()):
            break

    return xmin, ymin, xmax, ymax

def _room_bbox_center_xy(bbox_xyxy):
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    return np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5], dtype=float)

def _distance_xy(a, b) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def _distance_point_to_bbox_xy(point_xy, bbox_xyxy) -> float:
    xmin, ymin, xmax, ymax = _normalize_bbox_xyxy(bbox_xyxy)
    px = float(point_xy[0])
    py = float(point_xy[1])
    dx = 0.0 if xmin <= px <= xmax else min(abs(px - xmin), abs(px - xmax))
    dy = 0.0 if ymin <= py <= ymax else min(abs(py - ymin), abs(py - ymax))
    return math.hypot(dx, dy)
