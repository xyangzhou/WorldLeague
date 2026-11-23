"""CPU helpers for human-only whole-floor review.

Used by worldleague scene --help.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


MAPPING_TASKS = {"connectivity", "long_term_navigation", "regional_boundary", "traversable_passage"}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _review_raster(segmentation):
    """Prefer the original native layout PNG without changing its runtime map."""
    native = _array(getattr(segmentation, "room_ins_map", None))
    scene_dir = getattr(segmentation, "scene_dir", None)
    if scene_dir is not None:
        path = Path(scene_dir) / "layout" / "floor_insseg_0.png"
        if path.is_file():
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.ndim == 2:
                # Official SegmentationMap._load_map uses 0.01 m as its
                # original PNG scale before nearest-neighbour downsampling.
                return raw, .01, str(path), native
    return native, float(segmentation.map_resolution), None, native


def is_cognitive_mapping_session(session):
    """Recognize native task modules, canonical payload labels, or source paths."""
    module = getattr(session, "task_module", None)
    module_name = getattr(module, "__name__", "").lower()
    if ".cognitive_mapping." in module_name:
        return True
    payload = getattr(session, "payload", {}) or {}
    for key in ("big_task", "_hf_big_task", "category"):
        if str(payload.get(key, "")).lower().replace("_", " ") == "cognitive mapping":
            return True
    for key in ("question_path", "metadata_path"):
        path = str(getattr(session, key, "")).replace("\\", "/").lower()
        if "/cognitive mapping/" in path or "/cognitive_mapping/" in path:
            return True
    return (getattr(module, "TASK_NAME", None) in MAPPING_TASKS
            and bool(getattr(module, "FULL_SCENE", False)))


def room_map_geometry(segmentation):
    """Read every nonzero room ID using the installed map-to-world convention.

    Bounds include cell edges rather than just cell centres. The installed
    native SegmentationMap contains floor_insseg_0 only, even in scene models
    whose names mention another floor. Floor heights are declared map metadata,
    not measured contact/support evidence.
    """
    if segmentation is None:
        raise ValueError("native_room_map_unavailable")
    raster, resolution, original_path, native = _review_raster(segmentation)
    if (raster.ndim != 2 or not raster.size or (original_path and raster.shape[0] != raster.shape[1])
            or not np.issubdtype(raster.dtype, np.number)
            or not np.isfinite(raster).all() or np.any(raster < 0)
            or np.any(raster != np.floor(raster))):
        raise ValueError("invalid_native_room_map")
    occupied = np.argwhere(raster != 0)
    if not len(occupied):
        raise ValueError("native_room_map_has_no_rooms")
    native_resolution = float(segmentation.map_resolution)
    heights = _array(segmentation.floor_heights).astype(float).reshape(-1)
    if not math.isfinite(resolution) or resolution <= 0 or not len(heights) or not np.isfinite(heights).all():
        raise ValueError("native_room_map_frame_unavailable")

    def world(points):
        # BaseMap.map_to_world takes tensors and flips (row, col) into (x, y).
        import torch
        points = torch.as_tensor(points, dtype=torch.float64)
        if original_path is not None:
            # Preserve original PNG cell-to-world alignment, including any
            # outer cells lost by the runtime map's integer resize dimensions.
            result = _array(torch.flip((points - raster.shape[0] / 2.) * resolution, dims=(-1,))).astype(float)
        else:
            result = _array(segmentation.map_to_world(points)).astype(float)
        if result.shape != np.asarray(points).shape or not np.isfinite(result).all():
            raise ValueError("invalid_native_map_to_world")
        return result

    axes = world([[0., 0.], [1., 0.], [0., 1.]])
    if (not np.allclose(axes[1] - axes[0], [0., resolution], atol=1e-5)
            or not np.allclose(axes[2] - axes[0], [resolution, 0.], atol=1e-5)):
        raise ValueError("unsupported_native_map_orientation")
    lower, upper = occupied.min(axis=0), occupied.max(axis=0)
    corners = world([[lower[0] - .5, lower[1] - .5], [upper[0] + .5, upper[1] + .5]])
    names = {int(key): str(value) for key, value in
             (getattr(segmentation, "room_ins_id_to_ins_name", {}) or {}).items()}
    names.update({int(value): str(key) for key, value in
                  (getattr(segmentation, "room_ins_name_to_ins_id", {}) or {}).items()})
    rooms = []
    for identifier in sorted(int(value) for value in np.unique(raster) if value != 0):
        cells = np.argwhere(raster == identifier)
        # Choose an actual cell nearest the centroid; labels never mark an
        # empty hole or another room in a concave or disconnected region.
        anchor = cells[np.argmin(np.sum((cells - cells.mean(axis=0)) ** 2, axis=1))]
        rooms.append({"id": identifier, "name": names.get(identifier, f"room_id_{identifier}"),
                      "name_available": identifier in names, "cell_count": len(cells),
                      "label_map_row_col": anchor.tolist(), "label_world_xy_m": world(anchor.tolist()).tolist()})
    registered = sorted(set(names) - {0})
    native_ids = sorted(int(value) for value in np.unique(native) if value != 0) if native.ndim == 2 else []
    original_ids = [room["id"] for room in rooms]
    return {"bounds_world_xy_m": [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
            "occupied_map_bounds_row_col": [lower.tolist(), upper.tolist()],
            "map_shape_row_col": list(raster.shape), "map_resolution_m": resolution,
            "native_map_resolution_m": native_resolution, "original_map_resolution_m": .01 if original_path else None,
            "original_segmentation_path": original_path,
            "raster_source": "original_native_layout_png" if original_path else "native_runtime_room_map",
            "native_raster_room_ids": native_ids,
            "original_room_ids_lost_by_native_resize": sorted(set(original_ids) - set(native_ids)) if original_path else None,
            "unregistered_room_ids": sorted(set(original_ids) - set(names)),
            "map_origin_world_xy_m": axes[0].tolist(), "floor_idx": 0,
            "floor_z_m": float(heights[0]), "declared_floor_heights_m": heights.tolist(),
            "floor_reference": "native seg_map.floor_heights[0]; declared, not measured support",
            "rooms": rooms, "all_raster_room_ids": [room["id"] for room in rooms],
            "registered_room_ids_missing_from_raster": sorted(set(registered) - {room["id"] for room in rooms}),
            "orientation": "world +X right, world +Y up; native row increases +Y, column increases +X",
            "coverage": "All nonzero original native floor-0 PNG room IDs." if original_path else
                "All nonzero runtime floor-0 raster room IDs; original PNG unavailable.",
            "limitations": ["Segmentation is not a native render or a traversability/doorway graph.",
                "Room adjacency does not prove a real R1 passage or collision clearance.",
                "The installed native map loads floor_insseg_0 only; other floors are not reconstructed.",
                "Original room IDs absent from the native dictionary retain explicit unregistered ID labels.",
                "Unlabelled cells, walls, doors and furniture are not classified by this room raster."]}


def mapping_camera_geometry(session):
    """Return whole native map bounds, with explicit structural fallback limits."""
    scene = session.env.scene
    try:
        geometry = room_map_geometry(getattr(scene, "seg_map", None))
        geometry["bounds_source"] = "original_native_room_png_all_nonzero_ids" if geometry["original_segmentation_path"] else "native_room_instance_map_all_nonzero_ids"
        if geometry["original_segmentation_path"]:
            # Include both original and runtime map cell envelopes; integer
            # resizing may shift edge bounds by a fraction of a coarse cell.
            native_only = SimpleNamespace()
            for key in ("room_ins_map", "map_resolution", "floor_heights", "room_ins_id_to_ins_name",
                        "room_ins_name_to_ins_id", "map_to_world"):
                setattr(native_only, key, getattr(scene.seg_map, key, {} if "name" in key else None))
            try:
                native_geometry = room_map_geometry(native_only)
            except (ValueError, AttributeError, TypeError) as error:
                geometry["native_bounds_error"] = f"{type(error).__name__}:{error}"
            else:
                original_bounds, native_bounds = (np.asarray(value["bounds_world_xy_m"]) for value in (geometry, native_geometry))
                geometry["original_bounds_world_xy_m"] = original_bounds.tolist()
                geometry["native_bounds_world_xy_m"] = native_bounds.tolist()
                geometry["bounds_world_xy_m"] = [np.minimum(original_bounds[0], native_bounds[0]).tolist(),
                                                  np.maximum(original_bounds[1], native_bounds[1]).tolist()]
        return geometry
    except (ValueError, AttributeError, TypeError, IndexError) as error:
        failure = f"{type(error).__name__}:{error}"
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    bounds = []
    floors = []
    for obj in objects:
        if getattr(obj, "category", "") not in ("floor", "floors", "wall", "walls"):
            continue
        low, high = (_array(value).astype(float) for value in obj.aabb)
        if low.shape == high.shape == (3,) and np.isfinite([low, high]).all() and np.all(high >= low):
            bounds.extend((low, high))
            if obj.category in ("floor", "floors"):
                floors.append(float(high[2]))
    if not bounds or not floors:
        raise ValueError("whole_floor_camera_bounds_unavailable:" + failure)
    points = np.asarray(bounds)
    return {"bounds_world_xy_m": [points[:, :2].min(axis=0).tolist(), points[:, :2].max(axis=0).tolist()],
            "floor_z_m": min(floors), "floor_idx": None,
            "floor_reference": "minimum native floor AABB upper surface; support unmeasured",
            "bounds_source": "native_structure_aabb_fallback", "room_map_error": failure,
            "limitations": ["Room map unavailable; structural bounds cannot certify all room IDs or floor layers."]}


def fit_mapping_camera(geometry, direction, *, horizontal_aperture, focal_length,
                       image_width, image_height, vertical_aperture=None, overhead=False, margin=1.10):
    """Fit every floor-box corner to a perspective sensor's actual optics."""
    low, high = np.asarray(geometry["bounds_world_xy_m"], float)
    values = [horizontal_aperture, focal_length, image_width, image_height, margin, geometry["floor_z_m"]]
    if (low.shape != (2,) or high.shape != (2,) or not np.isfinite([low, high]).all()
            or np.any(high <= low) or not np.isfinite(values).all()
            or min(values[:4]) <= 0 or margin < 1):
        raise ValueError("invalid_whole_floor_camera_geometry")
    backward = np.asarray([0., 0., 1.] if overhead else direction, float)
    if backward.shape != (3,) or not np.isfinite(backward).all() or np.linalg.norm(backward) < 1e-6:
        raise ValueError("invalid_whole_floor_camera_direction")
    backward /= np.linalg.norm(backward)
    up = np.array([0., 1., 0.]) if abs(backward[2]) > .99 else np.array([0., 0., 1.])
    right = np.cross(up, backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    floor_z = float(geometry["floor_z_m"])
    top_z = floor_z + 1.6
    target = np.array([*((low + high) / 2), floor_z + .8])
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                        for z in (floor_z, top_z)])
    relative = corners - target
    tangent_x = float(horizontal_aperture) / (2 * float(focal_length))
    tangent_y = tangent_x * float(image_height) / float(image_width)
    if vertical_aperture is not None:
        if not math.isfinite(float(vertical_aperture)) or vertical_aperture <= 0:
            raise ValueError("invalid_vertical_camera_aperture")
        # Fit conservatively for both the physical aperture and render-product
        # aspect; differing USD aperture/resolution aspects must not crop rooms.
        physical_y = float(vertical_aperture) / (2 * float(focal_length))
        tangent_x = min(tangent_x, physical_y * float(image_width) / float(image_height))
        tangent_y = min(tangent_y, physical_y)
    axial = relative @ backward
    distance = max(float(np.max(axial + margin * np.abs(relative @ right) / tangent_x)),
                   float(np.max(axial + margin * np.abs(relative @ up) / tangent_y)),
                   float(np.max(axial)) + .5, 2.)
    position = target + distance * backward
    depth = distance - axial
    near = max(.001, float(position[2] - top_z)) if overhead else .001
    far = max(float(np.max(depth)) + 2., near + 1.)
    return {"position_world_m": position.tolist(), "target_world_m": target.tolist(),
            "distance_m": distance, "horizontal_fov_deg": math.degrees(2 * math.atan(tangent_x)),
            "vertical_fov_deg": math.degrees(2 * math.atan(tangent_y)),
            "horizontal_aperture_mm": float(horizontal_aperture), "focal_length_mm": float(focal_length),
            "vertical_aperture_mm": float(vertical_aperture) if vertical_aperture is not None else None,
            "image_width": int(image_width), "image_height": int(image_height), "framing_margin": margin,
            "optics_fit_policy": "Conservative intersection of native apertures and render-product aspect.",
            "fitted_z_interval_m": [floor_z, top_z], "clipping_range_m": [near, far],
            "cutaway_plane_world_z_m": top_z if overhead else None,
            "coverage": "All floor-map bounds fit the frustum; occlusion is not measured.",
            "limitations": ["Walls and tall furniture may occlude diagonal views.",
                "Overhead near clipping removes geometry above floor + 1.6 m for this human sensor only."
                if overhead else "No cutaway clipping is applied to this view."]}
