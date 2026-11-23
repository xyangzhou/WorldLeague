"""Continuous action loop. Run: worldleague continuous --help."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from worldleague.evaluation.clock import InferenceClock
from worldleague.evaluation.trajectory import TrajectoryRecorder
from worldleague.tasks.temporal_policy import initial_survey_limit
from pathlib import Path
import time
from worldleague.tasks.loader import instruction_protocol


def run_episode(plan, *, environment_factory=None, policy_factory=None, launch_time=None):
    """Execute explicitly budgeted native actions; diagnostic completion is not task success."""
    from worldleague.environments.continuous import ContinuousEnvironment, action_chunk
    from worldleague.policies.continuous_websocket import WebsocketPolicy
    processing = plan["policy"].get("action_processing")
    processor = None
    if processing is not None and processing["config"]["type"] == "conditional_resample":
        from worldleague.policies.action_processing import ConditionalResampler
        processor = ConditionalResampler(processing["config"], robot_config=plan["environment"]["robot"]["config"],
                                        gripper_table=processing.get("gripper_table", {}).get("path"))
    output = Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    trace = TrajectoryRecorder(output) if plan.get("record_trajectory") else None
    started = time.monotonic()
    setup_start = started if launch_time is None else launch_time
    environment = (environment_factory or ContinuousEnvironment)(plan)
    if policy_factory is None:
        from worldleague.plugins import LocalActionPolicy

        policy_factory = LocalActionPolicy if plan["policy"].get("backend") == "python" else WebsocketPolicy
    policy = policy_factory(plan["policy"], plan["environment"]["robot"]["config"])
    calls = steps = 0
    scene_load_s = episode_start = None
    inference_s = execution_s = 0.0
    stage = "model_error"
    summary = {"run_mode": "benchmark", "evaluation_validity": "not_evaluated",
               "task_success": None, "score_percent": None, "completion_score": None,
               "agent_failure_countable": False, "task_progress_is_final": False,
               "terminal_qa_enabled": False,
               "budget_unit": "native_control_steps", "cost": None,
               "max_control_steps": plan["evaluation"]["max_control_steps"],
               "control_budget": plan["evaluation"]["control_budget"]}
    worker = ThreadPoolExecutor(max_workers=1)
    limit = None
    survey_steps = idle_steps = 0
    surveying = False
    try:
        worker.submit(policy.start).result()
        (output / "policy_metadata.json").write_text(json.dumps(getattr(policy, "metadata", {}), indent=2) + "\n")
        stage = "environment_error"
        reset_started = time.monotonic()
        observation = environment.reset()
        scene_load_s = time.monotonic() - reset_started
        if "policy_prompts" in plan["task"]:
            from worldleague.policies.continuous_profile import prompt_observation
            observation = prompt_observation(observation, plan["task"]["policy_prompts"])
            (output / "policy_prompts.json").write_text(json.dumps(plan["task"]["policy_prompts"], indent=2) + "\n")
        environment.save_metadata(output)
        episode_start = time.monotonic()
        if hasattr(environment, "save_observation"):
            environment.save_observation(output, observation)
        if hasattr(environment, 'prepare_policy_observation'):
            observation = environment.prepare_policy_observation(observation)
        clock = InferenceClock(environment.dt)
        budget = plan["evaluation"]["max_control_steps"]
        limit = initial_survey_limit(environment.task, budget)
        surveying = limit is not None and not environment.delivery_opened
        reason = "control_budget_exhausted"
        with (output / "actions.jsonl").open("a") as log:
            while steps < budget:
                if surveying and steps >= limit:
                    reason = "survey_incomplete"
                    break
                stage = "model_error"
                calls += 1
                if trace:
                    trace.boundary("before", calls, observation, snapshot_id=getattr(environment, "snapshot_id", None))
                response, ticks, ended, elapsed = clock.request(
                    worker, lambda: policy.infer(observation), environment.step_during_inference)
                idle_steps += ticks
                inference_s += elapsed
                if ended:
                    reason = "native_environment_terminated"
                    with (output / "requests.jsonl").open("a") as requests:
                        requests.write(json.dumps({"model_call": calls, "inference_s": elapsed, "inference_steps": ticks,
                                                   "executed_actions": 0}) + "\n")
                    break
                raw_chunk = action_chunk(response, key=plan["policy"]["action_key"],
                                     action_dim=environment.action_dim,
                                     max_chunk_steps=plan["policy"]["max_chunk_steps"])
                chunk, processing_info = (processor.process(raw_chunk, observation) if processor is not None
                                          else (raw_chunk.astype("float32"), None))
                stage = "execution_error"
                executed = 0
                terminated = False
                begin = time.monotonic()
                horizon = plan["policy"].get("execution_horizon", len(chunk))
                if surveying:
                    horizon = min(horizon, limit - steps)
                if hasattr(environment, "begin_action_chunk"):
                    environment.begin_action_chunk(min(budget - steps, horizon, len(chunk)))
                for action in chunk[:min(budget - steps, horizon)]:
                    terminated = environment.step(action)
                    steps += 1
                    if surveying:
                        survey_steps += 1
                    executed += 1
                    # Flush each executed action, including a partially completed chunk.
                    log.write(json.dumps({"control_step": steps, "model_call": calls,
                                          "action": action.tolist(),
                                          **getattr(environment, "last_action_metadata", {})}) + "\n")
                    log.flush()
                    if plan["policy"]["observation_format"] == "xr1_robocasa365":
                        if hasattr(environment, "observe_control_boundary"):
                            environment.observe_control_boundary(terminated=terminated)
                        else:
                            environment.observe()
                    if terminated:
                        break
                execution_s += time.monotonic() - begin
                with (output / "requests.jsonl").open("a") as requests:
                    requests.write(json.dumps({"model_call": calls, "inference_s": elapsed, "inference_steps": ticks,
                                               "received_actions": len(raw_chunk), "executed_actions": executed,
                                               "predicted_actions": raw_chunk.tolist(),
                                               "processed_actions": chunk.tolist(),
                                               "action_processing": processing_info,
                                               "server_timing": response.get("server_timing"),
                                               "policy_timing": response.get("policy_timing")}) + "\n")
                stage = "observation_error"
                observation = environment.observe()
                if "policy_prompts" in plan["task"]:
                    observation = prompt_observation(observation, plan["task"]["policy_prompts"])
                if hasattr(environment, "save_observation"):
                    environment.save_observation(output, observation)
                if hasattr(environment, 'after_action_chunk'):
                    observation = environment.after_action_chunk(observation, terminated=terminated)
                    if 'policy_prompts' in plan['task']:
                        observation = prompt_observation(observation, plan['task']['policy_prompts'])
                if hasattr(environment, 'prepare_policy_observation'):
                    observation = environment.prepare_policy_observation(observation)
                if trace:
                    trace.boundary("after", calls, observation, snapshot_id=getattr(environment, "snapshot_id", None))
                if surveying and environment.delivery_opened:
                    surveying = False
                if terminated:
                    reason = "native_environment_terminated"
                    break
        if surveying and steps >= limit and reason == "control_budget_exhausted":
            reason = "survey_incomplete"
        summary.update(termination_reason=reason, execution_validity="complete")
        if plan["evaluation"].get("operation_scoring"):
            stage = "evaluation_error"
            operation = environment.finalize_evaluation(reason)
            if operation is None:
                raise RuntimeError("Requested native operation scoring produced no result.")
            (output / "physical_score.json").write_text(json.dumps(operation, indent=2, allow_nan=False))
            summary.update(operation)
            summary["operation"] = operation
            summary["task_progress_is_final"] = True
            summary["scoring_note"] = "Physical goal predicates scored at the native control frequency."
    except Exception as error:
        import traceback
        (output / "error.txt").write_text(traceback.format_exc())
        summary.update(termination_reason=stage,
                       execution_validity="complete" if stage == "evaluation_error" else "interrupted",
                       error=f"{type(error).__name__}: {error}")
        if stage == "evaluation_error":
            summary.update(evaluation_validity="unavailable", terminal_qa_enabled=False,
                           score_percent=None, completion_score=None,
                           scoring_error=f"{type(error).__name__}: {error}")
    finally:
        # Native Kit shutdown can exit Python without returning. Persist episode
        # evidence before releasing the simulator, then let the parent record exit.
        episode_end = time.monotonic()
        summary.update(model_calls=calls, decision_calls=calls, executed_actions=steps,
                       control_steps=steps, simulation_time_s=(steps + idle_steps) * getattr(environment, "dt", 0.0),
                       inference_steps=idle_steps, inference_simulation_s=idle_steps * getattr(environment, "dt", 0.0),
                       world_advances_during_inference=True,
                       cleanup_state="pending",
                       timings={"wall_s": episode_end - started, "inference_s": inference_s,
                                "scene_load_s": scene_load_s,
                                "evaluation_total_s": episode_end - setup_start if episode_start is not None else None,
                                "setup_s": episode_start - setup_start if episode_start is not None else None,
                                "episode_s": episode_end - episode_start if episode_start is not None else None,
                                "action_execution_s": execution_s})
        if trace and getattr(environment, "terminal_observation", None) is not None:
            trace.boundary("terminal", calls, environment.terminal_observation, snapshot_id=getattr(environment, "snapshot_id", None))
        if "timing" in getattr(environment, "metadata", {}):
            summary["control_timing"] = environment.metadata["timing"]
        if hasattr(environment, "observation_performance"):
            summary["observation_performance"] = dict(environment.observation_performance)
        if plan['evaluation'].get('container_motion_policy'):
            summary['container_motion_policy'] = plan['evaluation']['container_motion_policy']
        temporal = getattr(environment, 'temporal', None)
        if temporal is not None:
            summary['temporal_survey'] = {'protocol': plan['evaluation']['temporal_survey'],
                'policy': temporal.policy, 'phase_1_decisions': temporal.decisions,
                'final_phase': temporal.controller.phase, 'transition': temporal.transition}
        operation = summary.get("operation", {})
        complete = summary.get("execution_validity") == "complete" and operation.get("evaluation_validity") == "complete"
        result = {"task": plan["task"]["path"], "seed": plan["task"]["seed"],
                  "interactive_score": operation.get("score_percent") if complete else None,
                  "physical_success": operation.get("task_success") if complete else None, "terminal_qa": None,
                  "scoring_version": operation.get("scoring_version"),
                  "instruction_protocol": instruction_protocol(),
                  "termination": summary["termination_reason"],
                  "control_steps": steps, "control_budget": plan["evaluation"]["max_control_steps"]}
        result.update(world_advances_during_inference=True, inference_steps=idle_steps,
                      inference_simulation_s=summary["inference_simulation_s"])
        if limit is not None:
            result["survey"] = {"limit":limit, "used":survey_steps, "unit":"control_steps", "delivery_opened":not surveying}
            summary["survey"] = result["survey"]
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        (output/"result.json").write_text(json.dumps(result, indent=2))
        for owned in (policy, environment):
            try:
                if owned is policy:
                    worker.submit(policy.close).result()
                else:
                    environment.close()
            except Exception as error:
                summary.setdefault("cleanup_errors", []).append(str(error))
        worker.shutdown()
        summary["cleanup_state"] = "returned"
        summary["timings"]["wall_s"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
