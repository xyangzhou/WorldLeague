"""Load an explicit continuous policy profile and public task instructions.

Used by worldleague continuous --help.
"""
from __future__ import annotations


def prompt_observation(observation, prompts):
    key = "initialization" if observation["control_step"] == 0 else "step"
    return {**observation, "prompt": prompts[key]}
