"""Open the WorldHarness R1 environment. Run: worldleague scene T04 --gpu 0."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
import random
import time

import numpy as np
import torch

from worldleague.resources import CONFIG_ROOT, ESI_ROOT
from worldleague.tasks.loader import load_task
from worldleague.tasks.scene import DeclarativeScene


def main(argv=None, *, initial_observation_capture=None, episode=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--variant")
    parser.add_argument("--gpu-info", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8615)
    args = parser.parse_args(argv)
    from worldleague.runtime.device import install_gpu_selection, isolate_texture_cache

    install_gpu_selection(json.loads(args.gpu_info.read_text()))
    from omnigibson import simulator

    isolate_texture_cache(simulator, os.environ["OMNIGIBSON_APPDATA_PATH"])
    from worldharness.r1 import (ESISession, GripperControlConfig, ObservationConfig,
                                 PerformanceCounters, instrument_native_session, create_controller)

    task = load_task(args.task_file, seed=args.scene_seed, variant=args.variant)
    cfg = CONFIG_ROOT
    gripper = GripperControlConfig.load(cfg / "esi_gripper.json")
    random.seed(task.seed)
    np.random.seed(task.seed)
    torch.manual_seed(task.seed)
    adapter = DeclarativeScene(task)
    session = ESISession(
        task.source_path, esi_root=ESI_ROOT,
        grasp_mode=gripper.grasp_mode,
        **adapter.observation_settings(),
    )
    session.scene_adapter = adapter
    session.loaded_task = task
    session.public_task_text = task.task_text
    session.standby_until_agent = episode is None
    session.agent_observation_config = ObservationConfig.load(
        cfg / "esi_agent_observation.json"
    )
    session.performance = PerformanceCounters()
    session.reuse_step_observations = session.serialization_cache = True
    out = args.output_dir
    out.mkdir(exist_ok=True, parents=True)
    (out / "sim_pid.txt").write_text(str(os.getpid()))
    try:
        reset_started = time.monotonic()
        session.reset()
        session.scene_load_s = time.monotonic() - reset_started
        controller, executor = create_controller(session)
        source = session.observation_source
        source.human_depth_visualizations = False
        instrument_native_session(session)
        session.episode_id = (
            json.loads(args.episode_manifest.read_text())["episode_id"]
            if args.episode_manifest
            else out.name
        )
        if initial_observation_capture:
            initial_observation_capture(session)
        else:
            source.capture(action_boundary=True)
        session.task_evaluator = adapter.create_evaluator(
            session, controller, out / "private_evaluation"
        )
        if task.definition.private.evaluation.operation in (
            "establish_final_relation",
            "insert_or_verify_inside",
        ):

            def stability_window():
                for _ in range(15):
                    session.step_control(
                        {session.robot.name: controller.hold_action.copy()}
                    )
                    if controller.on_step:
                        controller.on_step()
                session.initial_stability_evidence = {
                    "native_control_steps": 15,
                    "simulated_duration_s": 15 * controller.dt,
                    "agent_actions": 0,
                    "policy": "unchanged_controller_hold_after_begin_episode",
                }

            session.initial_stability_window = stability_window
        if episode:
            return episode(session, controller, executor)
        from worldharness.sim.esi_web import run_esi_control

        from worldharness.r1 import control_metadata
        metadata = control_metadata(controller, executor, task.task_text)
        def record_port(port):
            pending = out / "port.tmp"
            pending.write_text(str(port))
            pending.replace(out / "port")

        run_esi_control(
            session,
            metadata,
            controller,
            executor,
            host="127.0.0.1",
            port=args.port,
            preview_fps=0, on_listening=record_port,
        )
    finally:
        session.close()
    return 0
