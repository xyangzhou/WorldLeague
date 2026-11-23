"""Run the WorldHarness episode on WorldLeague. Run worldleague eval T04 --config FILE --gpu N --output DIR."""
import asyncio
import json
import subprocess
import sys
import time

from ..tasks.loader import instruction_protocol, load_task
from ..tasks.temporal_policy import initial_survey_limit, survey_policy
from ..tasks.terminal_qa import grade, operation_progress, public_questions


def public_episode(task, action_mode, episode_id):
    parameters = task.definition.private.evaluation.parameters
    spec = parameters.get("terminal_qa")
    return dict(episode_id=episode_id, public_task=task.public_payload, instruction=task.task_text,
                max_steps=task.definition.defaults.max_steps,
                action_mode="fixed" if action_mode == "fixed-repeat" else action_mode,
                questions=public_questions(spec) if spec else [],
                survey_limit=initial_survey_limit(task, task.definition.defaults.max_steps),
                temporal_policy=survey_policy(parameters) if "phase_1_max_decisions" in parameters else None)


def evaluate(args, config):
    from worldharness.runner import run

    scene = args.output / "scene"
    port = config.get("environment", {}).get("port", 0)
    command = [sys.executable, "-u", "-m", "worldleague", "scene", args.task, "--seed", str(args.seed),
               "--port", str(port), "--gpu", str(args.gpu), "--output", str(scene)]
    with (args.output / "simulator.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ready = False
    try:
        port_file = scene / "port"
        while process.poll() is None and not port_file.exists():
            time.sleep(0.25)
        if process.poll() is not None:
            raise RuntimeError(f"Scene exited with status {process.returncode}. See {log.name}")
        url = f"http://127.0.0.1:{int(port_file.read_text())}"
        ready = True
        task = load_task(args.task, seed=args.seed)
        public = public_episode(task, args.action_mode, args.output.name)
        recorder = None
        if args.record_trajectory:
            from .trajectory import TrajectoryRecorder
            trajectory = TrajectoryRecorder(args.output)
            def recorder(observation):
                trajectory.boundary("observation", observation.action_step, observation)
        summary, answer = asyncio.run(run(url, public, config, args.output, on_observation=recorder))
        result = {"task": task.alias, "seed": task.seed, "instruction_protocol": instruction_protocol(), "decisions": summary["agent_decisions"],
                  "termination": summary["termination_reason"], "simulator_pid": process.pid, "url": url,
                  "interactive_score": None, "physical_success": None, "terminal_qa": None,
                  "decision_budget": public["max_steps"], "world_advances_during_inference": True,
                  "action_mode": args.action_mode, "timings": summary["timings"]}
        state = scene / "private_evaluation/state.json"
        if state.is_file() and config.get("evaluation", {}).get("terminal_qa", True):
            terminal = json.loads(state.read_text())
            physical = operation_progress(terminal)
            (args.output / "physical_score.json").write_text(json.dumps(physical, indent=2))
            result["interactive_score"] = physical["score_percent"] if physical.get("evaluation_validity") == "complete" else None
            result["physical_success"] = physical["task_success"]
            result["scoring_version"] = physical["scoring_version"]
            if answer is not None:
                spec = task.definition.private.evaluation.parameters["terminal_qa"]
                qa = grade(spec, answer, terminal["native_result"])
                result["terminal_qa"] = 100 * qa["fraction"] if qa["fraction"] is not None else None
                (args.output / "terminal_qa.json").write_text(json.dumps(qa, indent=2))
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return int(summary["termination_reason"] in ("runtime_error", "model_error"))
    finally:
        if not ready or not config.get("environment", {}).get("keep_simulator", False):
            if process.poll() is None:
                process.terminate()
                process.wait()
