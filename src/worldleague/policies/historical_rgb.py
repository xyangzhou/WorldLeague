"""Recall previously delivered decision images in the chat client. Run worldleague eval TASK --config FILE."""

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory

TOOL = {"type": "function", "function": {
    "name": "read_historical_rgb",
    "description": "Recall a previously delivered RGB view by its decision number and camera name.",
    "parameters": {"type": "object", "properties": {
        "decision_step": {"type": "integer", "minimum": 1},
        "camera": {"type": "string"}},
        "required": ["decision_step", "camera"], "additionalProperties": False}}}
MAX_ROUNDS = 4
MAX_IMAGES = 4


class HistoricalRGB:
    """Keep the actual transmitted PNGs for this episode in temporary storage."""

    def __init__(self):
        self.storage = TemporaryDirectory(prefix="worldleague-rgb-")
        self.steps = []

    def remember(self, views, observation):
        step = len(self.steps) + 1
        paths = {}
        for index, (name, part) in enumerate(views.items()):
            path = Path(self.storage.name) / f"{step}-{index}.png"
            path.write_bytes(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
            source = (getattr(observation, "initial_overview", None) if name == "initial_overview_rgb"
                      else observation)
            paths[name] = (path, {key: getattr(source, key) for key in ("snapshot_id", "captured_at")
                                  if hasattr(source, key)})
        self.steps.append(paths)
        return step

    def recall(self, function, remaining):
        if function.get("name") != "read_historical_rgb":
            return {"error": "Unknown tool."}, None
        try:
            args = json.loads(function["arguments"])
        except (KeyError, TypeError, ValueError):
            args = None
        if not isinstance(args, dict):
            return {"error": "Supply decision_step and camera."}, None
        step, camera = args.get("decision_step"), args.get("camera")
        if type(step) is not int or not 1 <= step < len(self.steps) or not isinstance(camera, str):
            return {"error": "Choose an earlier decision in this episode and a delivered camera."}, None
        view = self.steps[step - 1].get(camera)
        if view is None or remaining <= 0:
            return {"error": "Camera unavailable or four recalls already used this decision."}, None
        path, metadata = view
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"decision_step": step, "camera": camera, **metadata}, {
            "type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}

    def close(self):
        self.storage.cleanup()
