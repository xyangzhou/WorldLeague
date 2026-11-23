"""Load a benchmark task. Run: worldleague tasks."""

from dataclasses import dataclass
import json
from pathlib import Path

import yaml

from worldleague.resources import DATASET_ROOT
from .schema import CategoryTaskDefinition


@dataclass(frozen=True)
class LoadedTask:
    definition: CategoryTaskDefinition
    path: Path
    source_path: Path
    variant: str
    seed: int
    scene_name: str
    room_name: str
    run_mode: str = "benchmark"

    release_v1: bool = True
    alias: str = ""

    @property
    def task_text(self):
        return self.definition.public.task_text()

    @property
    def public_payload(self):
        return {
            "task_id": self.definition.id,
            "task_version": self.definition.version,
            **self.definition.public.model_dump(mode="json"),
        }

    def manifest(self, episode_id):
        return {
            "episode_id": episode_id,
            "alias": self.alias,
            "task_id": self.definition.id,
            "task_version": self.definition.version,
            "scene_seed": self.seed,
            "scene_variant": self.variant,
            "scene": self.scene_name,
            "room": self.room_name,
            "public_task": self.public_payload,
        }


def catalogue(root=DATASET_ROOT):
    return json.loads((Path(root) / "catalogue.json").read_text())["tasks"]


def instruction_protocol():
    """Report the policy-facing instruction protocol after loading the selected dataset."""
    label = json.loads((DATASET_ROOT / "catalogue.json").read_text()).get("instruction_protocol")
    return "task_instructions_v3" if label in (None, "separate_physical_and_qa_v2") else label


def load_task(
    selection,
    *,
    variant=None,
    seed=None,
    run_mode="benchmark",
    data_root=None,
    task_override=None,
    release_support_edges=True,
):
    rows = catalogue()
    path = Path(selection)
    if path.is_file():
        row = next(
            (
                row
                for row in rows
                if (DATASET_ROOT / row["task_file"]).resolve() == path.resolve()
            ),
            None,
        )
    else:
        matches = [row for row in rows if str(selection) in (row["alias"], row["task_id"])]
        if len(matches) > 1:
            aliases = ", ".join(row["alias"] for row in matches)
            raise ValueError(f"Task ID {selection!s} is shared. Select a benchmark alias from {aliases}.")
        row = next(iter(matches), None)
    if row is None:
        raise ValueError(f"Unknown benchmark task {selection!s}")
    path = DATASET_ROOT / row["task_file"]
    value = yaml.safe_load(path.read_text())
    value.update(yaml.safe_load((DATASET_ROOT / value.pop("scene_file")).read_text()))
    if task_override:
        value.update(yaml.safe_load((DATASET_ROOT / task_override).read_text()))
    definition = CategoryTaskDefinition.model_validate(value)
    identity = definition.source.json_path
    source_root = Path(data_root) if data_root else DATASET_ROOT / "sources/json_clean"
    source = source_root / identity
    raw = json.loads(source.read_text())
    return LoadedTask(
        definition,
        path,
        source,
        variant or row["variant"],
        definition.defaults.seed if seed is None else seed,
        raw["scene"],
        raw["room"],
        run_mode,
        release_support_edges,
        row["alias"],
    )
