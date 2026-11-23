"""Record native observation boundaries with worldleague eval or continuous --record-trajectory."""

import json
from pathlib import Path

from PIL import Image


class TrajectoryRecorder:
    def __init__(self, output):
        self.output = Path(output)
        self.directory = self.output / "trajectory_frames"
        self.directory.mkdir()
        self.last_key = self.last_images = self.overview = None
        self.sequence = 0

    def boundary(self, side, turn, observation, *, snapshot_id=None):
        if isinstance(observation, dict):
            images = observation["images"]
            metadata = {k: observation[k] for k in ("control_step", "simulation_time_s")}
            metadata.update(state=observation["state"].tolist(), snapshot_id=snapshot_id)
            key = id(images)
        else:
            images = observation.rgb_images()
            metadata = {k: getattr(observation, k) for k in ("action_step", "snapshot_id", "captured_at")}
            feedback = observation.last_action_feedback
            metadata["feedback"] = feedback.to_dict() if feedback is not None else None
            metadata["task_events"] = [event.to_dict() for event in observation.public_task_events]
            key = observation.snapshot_id
        if isinstance(observation, dict):
            # Retain the object so its id cannot be recycled across adjacent boundaries.
            self.latest_arrays = images
        if key != self.last_key:
            self.sequence += 1
            paths = {}
            for index, (camera, frame) in enumerate(images.items()):
                if frame is None:
                    continue
                if camera == "initial_overview_rgb" and self.overview is not None:
                    paths[camera] = self.overview
                    continue
                path = self.directory / f"{self.sequence:06d}-{index}.png"
                Image.fromarray(frame).save(path, compress_level=1)
                paths[camera] = str(path.relative_to(self.output))
                if camera == "initial_overview_rgb":
                    self.overview = paths[camera]
            self.last_key, self.last_images = key, paths
        offsets = {p.name: p.stat().st_size for p in (self.output / "private_evaluation").glob("*.jsonl")}
        row = {"turn": turn, "side": side, **metadata, "images": self.last_images, "physical_bytes": offsets}
        with (self.output / "trajectory_frames.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
