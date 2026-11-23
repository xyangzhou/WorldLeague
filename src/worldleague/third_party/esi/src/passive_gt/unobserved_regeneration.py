"""ESI-Bench scene helpers used by worldleague eval and continuous."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from .views import extract_view_specs

PROVENANCE = "saved_gt_camera_deterministic_generator_replay"

KNOWN_IDENTITY_CONTAINER_ASSETS = {
    ("carton", "cdmmwy"),
    ("cedar_chest", "fwstpx"),
    ("cedar_chest", "gbdzls"),
}

PHASE_KEYS = ("image1", "image2")

CONTENT_NAME_PREFIX = "render_unobserved_content_"

class InsufficientUnobservedMetadata(ValueError):
    """Raised when a strict saved-view replay would require guessing state."""

@dataclass(frozen=True)
class ContentReplaySpec:
    category: str
    model: str
    bbox_size_m: tuple[float, float, float]
    display_name: str | None = None
    sampling_source: str | None = None

    def runtime_payload(self) -> dict[str, Any]:
        """Return the field names consumed by the original phase spawner."""
        return {
            "category": self.category,
            "representative_model": self.model,
            "bbox_size_m": list(self.bbox_size_m),
            "display_name": self.display_name,
            "sampling_source": self.sampling_source,
        }

@dataclass(frozen=True)
class BoxReplaySpec:
    box_index: int
    position_label: str
    change_type: str
    container_name: str
    container_category: str
    container_model: str
    container_position: tuple[float, float, float]
    container_orientation_xyzw: tuple[float, float, float, float]
    container_bbox_min: tuple[float, float, float]
    container_bbox_max: tuple[float, float, float]
    placement_type: str
    phase1_content: ContentReplaySpec | None
    phase2_content: ContentReplaySpec | None

    def content_for_phase(self, phase: str) -> ContentReplaySpec | None:
        if phase == "image1":
            return self.phase1_content
        if phase == "image2":
            return self.phase2_content
        raise KeyError(f"Unknown Unobserved Change phase: {phase}")

@dataclass(frozen=True)
class SavedGTView:
    view_id: str
    phase: str
    box_index: int
    image_path: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    look_target: tuple[float, float, float]
    fov_deg: float
    view_direction: str
    container_name: str
    metadata_pointer: str

@dataclass(frozen=True)
class MissingGTCapture:
    phase: str
    box_index: int
    container_name: str
    reason: str = "original_gt_capture_missing_camera_and_image"

@dataclass(frozen=True)
class UnobservedReplayPlan:
    scene: str
    room: str
    task_type: str
    question_index: int
    seed: int
    boxes: tuple[BoxReplaySpec, ...]
    views: tuple[SavedGTView, ...]
    missing_captures: tuple[MissingGTCapture, ...]

    @property
    def theoretical_capture_count(self) -> int:
        return len(PHASE_KEYS) * len(self.boxes)

    @property
    def saved_capture_count(self) -> int:
        return len(self.views)

    @property
    def saved_view_metadata_replayable(self) -> bool:
        # Construction already validated every saved camera and every state
        # field needed by the original placement helper.  This does not claim
        # pixel identity or that an unsaved whole-scene runtime snapshot exists.
        return bool(self.views)

    @property
    def original_capture_complete(self) -> bool:
        return not self.missing_captures

    def audit_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "room": self.room,
            "task_type": self.task_type,
            "question_index": self.question_index,
            "seed": self.seed,
            "box_count": len(self.boxes),
            "theoretical_capture_count": self.theoretical_capture_count,
            "saved_capture_count": self.saved_capture_count,
            "missing_capture_count": len(self.missing_captures),
            "saved_view_metadata_replayable": self.saved_view_metadata_replayable,
            "original_capture_complete": self.original_capture_complete,
            "missing_captures": [asdict(item) for item in self.missing_captures],
            "provenance": PROVENANCE,
        }

def _float_tuple(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise InsufficientUnobservedMetadata(f"{field} must contain {length} numeric values")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InsufficientUnobservedMetadata(f"{field} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise InsufficientUnobservedMetadata(f"{field} contains a non-finite value")
    return result

def _keyed_items(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        return [(str(key), item) for key, item in value.items() if isinstance(item, dict)]
    if not isinstance(value, list):
        return []
    output = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        output.append((str(item.get("_key", index)), item))
    return output

def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise InsufficientUnobservedMetadata(f"{field} is missing")
    return text

def _content_spec(value: Any, field: str) -> ContentReplaySpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InsufficientUnobservedMetadata(f"{field} must be an object or null")
    return ContentReplaySpec(
        category=_required_text(value.get("category"), f"{field}.category"),
        model=_required_text(value.get("model"), f"{field}.model"),
        bbox_size_m=_float_tuple(value.get("bbox_size_m"), 3, f"{field}.bbox_size_m"),
        display_name=str(value.get("display_name") or "").strip() or None,
        sampling_source=str(value.get("sampling_source") or "").strip() or None,
    )

def _container_orientation(category: str, model: str) -> tuple[float, float, float, float]:
    asset = (category, model)
    if asset not in KNOWN_IDENTITY_CONTAINER_ASSETS:
        raise InsufficientUnobservedMetadata(
            "Container orientation was not serialized and cannot be inferred safely for "
            f"{category}/{model}. The audited generator uses identity only for {sorted(KNOWN_IDENTITY_CONTAINER_ASSETS)}."
        )
    return (0.0, 0.0, 0.0, 1.0)

def _box_spec(value: Any, fallback_index: int) -> BoxReplaySpec:
    if not isinstance(value, dict):
        raise InsufficientUnobservedMetadata(f"question_data.boxes[{fallback_index}] is not an object")
    box_index = int(value.get("box_index", fallback_index))
    container = value.get("container") or {}
    if not isinstance(container, dict):
        raise InsufficientUnobservedMetadata(f"box {box_index} container is not an object")
    category = _required_text(container.get("category"), f"box {box_index} container.category")
    model = _required_text(container.get("model"), f"box {box_index} container.model")
    placement = container.get("placement") or {}
    if not isinstance(placement, dict):
        raise InsufficientUnobservedMetadata(f"box {box_index} container.placement is not an object")
    bbox = container.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 2:
        raise InsufficientUnobservedMetadata(f"box {box_index} container.bbox must contain min and max")
    bbox_min = _float_tuple(bbox[0], 3, f"box {box_index} container.bbox[0]")
    bbox_max = _float_tuple(bbox[1], 3, f"box {box_index} container.bbox[1]")
    if any(high <= low for low, high in zip(bbox_min, bbox_max)):
        raise InsufficientUnobservedMetadata(f"box {box_index} container.bbox has a non-positive extent")
    return BoxReplaySpec(
        box_index=box_index,
        position_label=_required_text(value.get("position_label"), f"box {box_index} position_label"),
        change_type=_required_text(value.get("change_type"), f"box {box_index} change_type"),
        container_name=_required_text(container.get("name"), f"box {box_index} container.name"),
        container_category=category,
        container_model=model,
        container_position=_float_tuple(placement.get("position"), 3, f"box {box_index} container.placement.position"),
        container_orientation_xyzw=_container_orientation(category, model),
        container_bbox_min=bbox_min,
        container_bbox_max=bbox_max,
        placement_type=_required_text(placement.get("placement_type"), f"box {box_index} placement_type"),
        phase1_content=_content_spec(value.get("phase1_content"), f"box {box_index} phase1_content"),
        phase2_content=_content_spec(value.get("phase2_content"), f"box {box_index} phase2_content"),
    )

def _saved_view(
    item: dict[str, Any],
    *,
    phase: str,
    item_index: int,
    phase_fov: float,
    boxes_by_index: dict[int, BoxReplaySpec],
) -> SavedGTView:
    box_index = int(item.get("box_index", -1))
    box = boxes_by_index.get(box_index)
    if box is None:
        raise InsufficientUnobservedMetadata(f"{phase} GT view references unknown box_index={box_index}")
    container_name = _required_text(item.get("container_name"), f"{phase} view {item_index} container_name")
    if container_name != box.container_name:
        raise InsufficientUnobservedMetadata(
            f"{phase} box {box_index} container mismatch: view={container_name}, box={box.container_name}"
        )
    image_path = _required_text(item.get("image_path"), f"{phase} box {box_index} image_path")
    image_name = Path(image_path.replace("\\", "/")).name
    if not image_name:
        raise InsufficientUnobservedMetadata(f"{phase} box {box_index} image_path has no filename")
    pose = item.get("camera_pose") or {}
    if not isinstance(pose, dict):
        raise InsufficientUnobservedMetadata(f"{phase} box {box_index} camera_pose is not an object")
    return SavedGTView(
        view_id=f"gt_view/{phase}/{image_name}",
        phase=phase,
        box_index=box_index,
        image_path=image_path,
        position=_float_tuple(pose.get("position"), 3, f"{phase} box {box_index} camera position"),
        quaternion_xyzw=_float_tuple(
            pose.get("quaternion_xyzw"), 4, f"{phase} box {box_index} camera quaternion_xyzw"
        ),
        look_target=_float_tuple(item.get("look_target"), 3, f"{phase} box {box_index} look_target"),
        fov_deg=float(item.get("fov_deg") or phase_fov),
        view_direction=_required_text(item.get("view_direction"), f"{phase} box {box_index} view_direction"),
        container_name=container_name,
        metadata_pointer=f"/question_data/render/gt_view/{phase}/images/{item_index}",
    )

def build_unobserved_replay_plan(
    payload: dict[str, Any],
    *,
    validate_selector: bool = True,
) -> UnobservedReplayPlan:
    """Validate and extract the exact state/camera inputs for one sample.

    Only views that the original generator successfully saved are replayable as
    strict GT.  Empty phase/box slots are reported in ``missing_captures`` and
    are never assigned a newly invented camera.
    """
    question_data = payload.get("question_data") or {}
    if not isinstance(question_data, dict):
        raise InsufficientUnobservedMetadata("question_data is missing")
    task_type = _required_text(question_data.get("task_type") or payload.get("task_type"), "task_type")
    boxes = tuple(
        sorted(
            (_box_spec(value, index) for index, value in enumerate(question_data.get("boxes") or [])),
            key=lambda item: item.box_index,
        )
    )
    if not boxes:
        raise InsufficientUnobservedMetadata("question_data.boxes is empty")
    if [box.box_index for box in boxes] != list(range(len(boxes))):
        raise InsufficientUnobservedMetadata("box_index values must be contiguous from zero")
    boxes_by_index = {box.box_index: box for box in boxes}

    gt_view = ((question_data.get("render") or {}).get("gt_view") or {})
    if not isinstance(gt_view, dict):
        raise InsufficientUnobservedMetadata("question_data.render.gt_view is missing")
    views: list[SavedGTView] = []
    missing: list[MissingGTCapture] = []
    for phase in PHASE_KEYS:
        phase_record = gt_view.get(phase) or {}
        if not isinstance(phase_record, dict):
            raise InsufficientUnobservedMetadata(f"gt_view.{phase} is not an object")
        phase_fov = float(phase_record.get("fov_deg") or 55.0)
        phase_views = [
            _saved_view(
                item,
                phase=phase,
                item_index=item_index,
                phase_fov=phase_fov,
                boxes_by_index=boxes_by_index,
            )
            for item_index, (_key, item) in enumerate(_keyed_items(phase_record.get("images")))
        ]
        seen_box_indices = {view.box_index for view in phase_views}
        if len(seen_box_indices) != len(phase_views):
            raise InsufficientUnobservedMetadata(f"gt_view.{phase} has duplicate box captures")
        views.extend(phase_views)
        for box in boxes:
            if box.box_index not in seen_box_indices:
                missing.append(
                    MissingGTCapture(
                        phase=phase,
                        box_index=box.box_index,
                        container_name=box.container_name,
                    )
                )

    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise InsufficientUnobservedMetadata("GT view filenames are not unique within their phase")
    if validate_selector:
        # Rule-backed fallback cameras are valid inputs for a fresh GT state
        # replay, but they are not historical captures and therefore do not
        # belong in this saved-capture reconstruction plan.
        selector_ids = {
            spec.view_id
            for spec in extract_view_specs(payload)
            if spec.role != "gt_rule_camera"
        }
        if selector_ids != set(view_ids):
            raise InsufficientUnobservedMetadata(
                "Replay plan does not match the Passive-GT selector: "
                f"metadata_only={sorted(set(view_ids) - selector_ids)}, selector_only={sorted(selector_ids - set(view_ids))}"
            )

    question_index_value = question_data.get("candidate_index", payload.get("question_index", 0))
    return UnobservedReplayPlan(
        scene=_required_text(payload.get("scene"), "scene"),
        room=_required_text(payload.get("room"), "room"),
        task_type=task_type,
        question_index=int(question_index_value or 0),
        seed=int(payload.get("seed", 7) or 7),
        boxes=boxes,
        views=tuple(views),
        missing_captures=tuple(missing),
    )

def _preloaded_object_configs(plan: UnobservedReplayPlan) -> list[dict[str, Any]]:
    """Load replay assets with the environment instead of adding live prims.

    The installed OmniGibson build can segfault in ``scene.add_object`` when a
    visual-only DatasetObject is added after a full scene starts playing.  The
    same assets are stable when supplied through Environment config.  Names
    intentionally match the producer helpers so they reuse the initialized
    objects from the scene registry.
    """
    objects: list[dict[str, Any]] = []
    for box in plan.boxes:
        objects.append(
            {
                "type": "DatasetObject",
                "name": box.container_name,
                "category": box.container_category,
                "model": box.container_model,
                "position": list(box.container_position),
                "orientation": list(box.container_orientation_xyzw),
                "fixed_base": True,
                "visual_only": True,
            }
        )
        for phase_key, content in (
            ("phase1_content", box.phase1_content),
            ("phase2_content", box.phase2_content),
        ):
            if content is None:
                continue
            objects.append(
                {
                    "type": "DatasetObject",
                    "name": f"{CONTENT_NAME_PREFIX}{phase_key}_{content.category}_{box.box_index}",
                    "category": content.category,
                    "model": content.model,
                    "position": [1000.0 + 2.5 * box.box_index, 1000.0, 120.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "fixed_base": True,
                    "visual_only": True,
                }
            )
    return objects
