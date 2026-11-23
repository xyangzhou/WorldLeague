"""Prepare one native-policy episode. Run: worldleague continuous MODEL TASK --url ws://HOST:PORT."""

from pathlib import Path
import os
import yaml
from .resources import CONFIG_ROOT, DATASET_ROOT


def resolve_robot_assets(value):
    if isinstance(value, dict):
        return {key: resolve_robot_assets(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_robot_assets(item) for item in value]
    if isinstance(value, str) and value.startswith("robot://"):
        return str(
            Path(os.environ.get("WORLDLEAGUE_ROBOT_ASSETS", Path.home() / ".cache/worldleague/robots"))
            / value.removeprefix("robot://")
        )
    return value


def plan(model, alias, url, output, seed, *, plugin=None):
    cfg = CONFIG_ROOT / "continuous"
    # These embodiment interfaces reuse the published task setups and accept
    # executable actions directly from a new policy.
    setup_presets = {"R1Pro": "pi0.5", "PandaOmron": "GR00T-N1.5-RoboCasa365"}
    setup_preset = setup_presets.get(model, model)
    selections = yaml.safe_load((cfg / "models" / f"{setup_preset}.yaml").read_text())
    if alias not in selections:
        raise ValueError(f"{model} has no continuous-control setup for {alias}.")
    selection = selections[alias]
    setup = yaml.safe_load((DATASET_ROOT / "continuous" / f"{alias}.yaml").read_text())[
        selection["setup"] - 1
    ]
    policy = yaml.safe_load(
        (cfg / "policies" / f"{selection['policy']}.yaml").read_text()
    )
    if model in setup_presets:
        priming = policy["profile"]["config"]["alignment"]
        direct = {"type": "direct"}
        if model == "PandaOmron":
            direct.update(history_length=1, history_interval_steps=1,
                          reference_control_hz=setup["environment"]["control_hz"])
        policy.update(reset_mode="none", max_chunk_steps=32, execution_horizon=32,
                      action_processing={"config": direct},
                      profile={"config": {"alignment": priming, "action_processing": direct}})
    policy["interface"] = model
    if plugin is None:
        policy["url"] = url
    else:
        policy.update(backend="python", plugin=plugin)
    table = policy["action_processing"].get("gripper_table")
    if table:
        table["path"] = str(cfg / table["path"])
    robot = resolve_robot_assets(
        yaml.safe_load((cfg / "robots" / f"{setup['robot']}.yaml").read_text())
    )
    env = {**setup["environment"], "robot": {"config": robot}}
    task = {
        "path": alias,
        "variant": setup["variant"],
        "seed": seed,
        "task_override": setup.get("task_override"),
        "release_support_edges": setup["release_support_edges"],
        "policy_prompts": setup["policy_prompts"],
    }
    return {
        "policy": policy,
        "task": task,
        "environment": env,
        "evaluation": {**setup["evaluation"], "world": "run"},
        "output_dir": str(output),
    }
