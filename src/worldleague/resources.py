"""Locate installed configs and scene builders. Set WORLDLEAGUE_BENCHMARK for an external dataset."""

import os
from pathlib import Path
import sysconfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs"
if not (CONFIG_ROOT / "esi_base.json").is_file():
    CONFIG_ROOT = Path(sysconfig.get_path("data")) / "share/worldleague/configs"
DATASET_ROOT = Path(
    os.environ.get("WORLDLEAGUE_BENCHMARK", ROOT / "benchmark")
).resolve()
ESI_ROOT = Path(os.environ.get("WORLDLEAGUE_ESI_ROOT", Path(__file__).parent / "third_party/esi")).resolve()


def resolve_resource_path(path, root=ROOT):
    value = str(path)
    if value.startswith("behavior://"):
        return Path(os.environ["OMNIGIBSON_DATA_PATH"]) / value.removeprefix(
            "behavior://"
        )
    if value.startswith("dataset://"):
        return DATASET_ROOT / value.removeprefix("dataset://")
    return Path(root) / path


def evaluation_config(value):
    """Accept a local YAML path or a bundled VLM preset name."""
    path = Path(value)
    if path.is_file():
        return path
    if path.parent == Path(".") and not path.suffix:
        preset = CONFIG_ROOT / "vlm" / (path.name + ".yaml")
        if preset.is_file():
            return preset
    raise FileNotFoundError(f"Unknown evaluation configuration: {value}")
