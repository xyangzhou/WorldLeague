"""Own one continuous native scene. Run: python -m worldleague.runtime.commands.run_continuous_episode --plan FILE --gpu-info FILE."""

import argparse
import json
import os
from pathlib import Path


def main(argv=None, *, launch_time=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gpu-info", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    gpu = json.loads(args.gpu_info.read_text())
    from worldleague.runtime.device import install_gpu_selection, isolate_texture_cache

    install_gpu_selection(gpu)
    from omnigibson import simulator

    isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    from worldleague.policies.continuous import run_episode

    summary = run_episode(plan, launch_time=launch_time)
    if summary["execution_validity"] != "complete" or summary["termination_reason"] == "evaluation_error":
        return 1
    return 0
