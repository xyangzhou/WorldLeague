"""ESI-Bench scene helpers used by worldleague eval and continuous."""
from __future__ import annotations
import inspect
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OG_DISABLE_EMITTER_APIS", "1")

import numpy as np
import omnigibson as og
import yaml
from omnigibson.macros import gm
from tasks._registry import module_for_name, module_for_payload

gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False

@dataclass
class ActiveExploreConfig:
    task: str
    metadata: Path
    question_index: int
    json_root: Path | None
    results_root: Path
    step_image_root: Path
    provider: str
    model: str | None
    api_key: str | None
    base_url: str | None
    history_image_window: int
    max_steps: int
    min_steps: int
    threshold: float
    max_new_tokens: int
    temperature: float
    top_p: float
    robot: str
    overwrite: bool

@dataclass
class SelectedQuestion:
    payload: dict[str, Any]
    source_path: Path
    source_label: str

def load_task_module_for_payload(task_name: str, payload: dict[str, Any]):
    module_name = module_for_payload(payload) or module_for_name(task_name)
    return importlib.import_module(module_name)

def resolve_json_path(raw_path: str, metadata_path: Path, json_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    roots = [metadata_path.parent]
    if json_root is not None:
        roots.append(json_root)
    roots.append(Path.cwd())
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve question JSON path: {raw_path}")

def decode_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value

def is_hf_question_row(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "metadata_json" in value
        and "big_task" in value
        and "small_task" in value
        and ("question" in value or "runner_task" in value)
    )

def payload_from_hf_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = decode_json_field(row.get("metadata_json"))
    payload = dict(metadata) if isinstance(metadata, dict) else {}
    row_answer = decode_json_field(row.get("answer"))
    row_options = decode_json_field(row.get("options_json"))
    row_image_paths = decode_json_field(row.get("image_paths_json"))

    payload["scene"] = row.get("scene") or payload.get("scene") or "unknown_scene"
    payload["room"] = row.get("room") or payload.get("room") or "unknown_room"
    if row.get("question") and not payload.get("_question"):
        payload["_question"] = row["question"]
    if row.get("answer") not in (None, "") and "_ground_truth" not in payload:
        payload["_ground_truth"] = row_answer
    if row.get("answer") not in (None, "") and "answer" not in payload:
        payload["answer"] = row_answer

    payload["_hf_id"] = row.get("id")
    payload["_hf_big_task"] = row.get("big_task")
    payload["_hf_small_task"] = row.get("small_task")
    payload["_hf_runner_task"] = row.get("runner_task")
    payload["_hf_options"] = row_options if row_options is not None else []
    payload["_hf_image_paths"] = row_image_paths if row_image_paths is not None else []
    return payload

def load_question_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if is_hf_question_row(payload):
        return payload_from_hf_row(payload)
    return payload

def select_question(metadata_path: Path, question_index: int, json_root: Path | None) -> SelectedQuestion:
    if metadata_path.suffix.lower() == ".jsonl":
        raise ValueError("questions.jsonl is an HF export artifact, not a runner input. Use dataset/json_clean/*.json or dataset/json_clean/<Big Task>.json.")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if is_hf_question_row(metadata):
        return SelectedQuestion(payload=payload_from_hf_row(metadata), source_path=metadata_path, source_label=str(metadata_path))
    if isinstance(metadata, dict) and isinstance(metadata.get("json_paths"), list):
        json_paths = metadata["json_paths"]
    elif isinstance(metadata, list):
        json_paths = metadata
    else:
        return SelectedQuestion(payload=metadata, source_path=metadata_path, source_label=str(metadata_path))
    if not json_paths:
        raise ValueError(f"No json_paths found in metadata: {metadata_path}")
    if question_index < 0 or question_index >= len(json_paths):
        raise IndexError(f"question_index={question_index} out of range for {len(json_paths)} paths")
    selected = json_paths[question_index]
    question_json = resolve_json_path(str(selected), metadata_path, json_root)
    return SelectedQuestion(payload=load_question_json(question_json), source_path=question_json, source_label=str(question_json))

def set_camera(pos: np.ndarray, quat: np.ndarray) -> None:
    og.sim._viewer_camera.set_position_orientation(position=pos, orientation=quat)
    for _ in range(10):
        og.sim.render()

def build_env_config(
    scene_name: str,
    room_name: str | None,
    robot: str,
    objects: list[dict[str, Any]],
    full_scene: bool = False,
) -> dict[str, Any]:
    cfg_file = Path(og.example_config_path) / f"{robot.lower()}_primitives.yaml"
    if cfg_file.exists():
        with cfg_file.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = {"scene": {"type": "InteractiveTraversableScene"}, "robots": [], "objects": []}
    config.setdefault("scene", {})
    config["scene"]["scene_model"] = scene_name
    config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]
    if room_name and not full_scene:
        config["scene"]["load_room_instances"] = [room_name]
    config["objects"] = objects
    return config

def task_build_env_config(
    task_module,
    scene_name: str,
    room_name: str | None,
    robot: str,
    objects: list[dict[str, Any]],
    full_scene: bool = False,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hook = getattr(task_module, "build_env_config", None)
    if hook is not None:
        return call_with_supported_args(
            hook,
            scene_name,
            room_name,
            robot,
            objects,
            full_scene=full_scene,
            payload=payload,
        )
    return build_env_config(scene_name, room_name, robot, objects, full_scene=full_scene)

def task_initial_settle_steps(task_module) -> int:
    if getattr(task_module, "SKIP_INITIAL_SETTLE", False):
        return 0
    return int(getattr(task_module, "INITIAL_SETTLE_STEPS", 30))

def call_with_supported_args(func, *args, **kwargs):
    params = inspect.signature(func).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in params}
    return func(*args, **filtered_kwargs)

def task_preprocess(task_module, payload: dict[str, Any], question_json: Path, config: ActiveExploreConfig) -> dict[str, Any]:
    hook = getattr(task_module, "preprocess", None)
    if hook is None:
        return {}
    state = call_with_supported_args(hook, payload, source_json=question_json, config=config)
    return state if isinstance(state, dict) else {}

def task_postprocess_env(task_module, env, payload: dict[str, Any], camera_info: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any]:
    hook = getattr(task_module, "postprocess_env", None)
    if hook is None:
        return {}
    result = call_with_supported_args(hook, env, payload, camera_info, task_state=task_state)
    return result if isinstance(result, dict) else {}

def task_cleanup_runtime(task_module, env, payload: dict[str, Any], task_state: dict[str, Any]) -> bool:
    hook = getattr(task_module, "cleanup_runtime", None)
    if hook is None:
        return False
    result = call_with_supported_args(hook, env, payload, task_state=task_state)
    return bool(result)
