"""YAML-controlled action chunk processing.

Used by worldleague continuous --help.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class ConditionalResampler:
    """Resample only when the configured gripper window permits it."""

    def __init__(self, config, *, robot_config=None, gripper_table=None):
        self.config = config
        self.robot_config = robot_config
        self.first_closure = {}
        if config["gripper_correction"]["enabled"]:
            import json
            table = json.loads(Path(gripper_table).read_text())
            fraction = config["gripper_correction"]["min_closed_fraction"]
            for idx, name in table["task_names"].items():
                stages = table["per_task"].get(str(idx), {}).get("stages", {})
                self.first_closure[name] = {
                    side: min((int(s) for s, v in stages.items()
                               if v[f"{side}_closed"] >= fraction * max(v["frames"], 1)), default=None)
                    for side in ("left", "right")}

    def correct_grippers(self, actions, observation):
        """Match the author's task/stage rule before resampling; unknown tasks pass through."""
        cfg = self.config["gripper_correction"]
        info = {"enabled": cfg["enabled"], "fired": False}
        if not cfg["enabled"]:
            return actions, info
        task = observation["prompt"]
        rule = self.first_closure.get(task)
        info.update(task=task, matched_task=rule is not None, stage=cfg["stage"])
        if rule is None:
            return actions, info
        from worldleague.policies.continuous_websocket import r1pro_state23
        state = np.asarray(r1pro_state23(observation["state"], self.robot_config), dtype=np.float64)
        indices = self.config["gripper_indices"]
        state[indices] = state[indices] * cfg["width_scale"] + cfg["width_offset"]
        fixes = []
        for side, index in zip(("left", "right"), indices):
            first = rule[side]
            expected = first is not None and (cfg["stage"] is None or cfg["stage"] >= first)
            fixes.append(bool(state[index] < cfg["closed_threshold"] and not expected))
        if any(fixes):
            actions = np.tile(state.astype(actions.dtype), (len(actions), 1))
            for index, fix in zip(indices, fixes):
                if fix:
                    actions[:, index] = cfg["open_command"]
            info.update(fired=True, left=fixes[0], right=fixes[1])
        return actions, info

    def process(self, actions, observation):
        from scipy.interpolate import interp1d

        actions, correction_info = self.correct_grippers(np.asarray(actions), observation)
        cfg = self.config
        n_from, n_to = cfg["source_steps"], cfg["execution_steps"]
        base, grippers = cfg["base_velocity_indices"], cfg["gripper_indices"]
        if max(base + grippers) >= actions.shape[1]:
            raise ValueError("Configured processing indices exceed the received action dimension.")
        variation = np.ptp(actions[:n_from, grippers], axis=0)
        minimum = {"linear": 2, "quadratic": 3, "cubic": 4}[cfg["interpolation"]]
        compressed = (n_to < n_from and len(actions) >= n_from and n_from >= minimum
                      and not np.any(variation > cfg["gripper_variation_threshold"]))
        factor = cfg["base_velocity_scale"]["numerator"] / cfg["base_velocity_scale"]["denominator"]
        if compressed:
            source = np.linspace(0, n_from - 1, n_from)
            target = np.linspace(0, n_from - 1, n_to)
            out = np.empty((n_to, actions.shape[1]), dtype=np.float64)
            for dim in range(actions.shape[1]):
                out[:, dim] = interp1d(source, actions[:n_from, dim], kind=cfg["interpolation"])(target)
            out[:, base] *= factor
            # The reference helper preserves prediction precision before its buffer cast.
            out = out.astype(actions.dtype)
        else:
            out = actions[:n_to].copy()
            if len(out) < n_to:
                pad = np.repeat(out[-1:], n_to-len(out), axis=0)
                pad[:, base] = 0
                out = np.concatenate((out, pad), axis=0)
        out = out.astype(cfg["output_dtype"])
        if not np.isfinite(out).all():
            raise ValueError("Action processing produced nonfinite commands.")
        return out, {"type": cfg["type"], "compressed": bool(compressed),
                     "source_steps": n_from, "execution_steps": n_to,
                     "gripper_variation": variation.tolist(),
                     "base_velocity_scale_applied": factor if compressed else 1.0,
                     "output_dtype": cfg["output_dtype"], "gripper_correction": correction_info}
