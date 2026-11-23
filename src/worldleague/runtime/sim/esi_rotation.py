"""R1 local-axis wrist rotation configuration; no simulator imports or public parameters."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


from worldleague.runtime.sim.esi_omnigibson import WORLDHARNESS_ROOT


@dataclass(frozen=True)
class RotationControlConfig:
    rotation_deg: float = 15.0
    rotation_tolerance_deg: float = 1.0
    max_rotation_speed_deg_s: float = 15.0
    max_step_rotation_deg: float = 2.0


    @classmethod
    def load(cls, path: str | Path = WORLDHARNESS_ROOT / "configs/esi_rotation.json"):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values.pop("path_tolerance_deg", None)  # Inert historical configuration.
        return cls(**values)


def rotation_axis_sign(token):
    """Return a local axis index and signed angle multiplier for an R1 token."""
    return {"ROTATE_CW": (2, 1), "ROTATE_CCW": (2, -1),
            "ROTATE_PITCH_POS": (1, 1), "ROTATE_PITCH_NEG": (1, -1),
            "ROTATE_ROLL_POS": (0, 1), "ROTATE_ROLL_NEG": (0, -1)}[token]
