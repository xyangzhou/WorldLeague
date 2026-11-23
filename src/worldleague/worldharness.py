"""Run a VLM in the R1 action loop. Run: worldleague eval TASK --policy module:factory."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from worldleague.tasks.terminal_qa import grade, operation_progress, public_questions

from .decisions import InvalidDecision, dispatch
from .evaluation.clock import InferenceClock
from .evaluation.trajectory import TrajectoryRecorder
from .tasks.temporal_policy import initial_survey_limit, survey_policy
from .tasks.wia_history_collection import IncompleteSurvey
from .evaluation.requests import request_recorder
from .tasks.loader import instruction_protocol


def evaluate(session, controller, executor, policy, output, *, world="run", action_mode, record_trajectory=False):
    """Policies receive public observations and commands. Native measurements stay with the scorer."""
    trace = TrajectoryRecorder(output) if record_trajectory else None
    source = session.observation_source
    task = session.loaded_task
    obs = source.begin_episode(task.task_text)
    phase_budget = survey_policy(task.definition.private.evaluation.parameters)[
        "phase_1_max_decisions"
    ]
    if hasattr(policy, "reset_episode_memory"):
        policy.reset_episode_memory()
    history = getattr(session, "candidate_public_history", None)
    if history:
        policy.set_history(history["frames"])
        session.candidate_history_delivery_receipt = (
            session.candidate_public_history_delivery()
        )
    notes = []
    actions = tuple(executor.available_actions)
    started = time.monotonic()
    if world != "run":
        raise ValueError("WorldLeague evaluations use --world run.")
    limit = initial_survey_limit(task, task.definition.defaults.max_steps)
    survey_turns = 0
    survey_open = limit is not None
    idle_steps = 0
    clock = InferenceClock(controller.dt)
    reason = "max_steps"
    decision_s = execution_s = 0.0
    feedback = None
    with (
        ThreadPoolExecutor(max_workers=1) as worker,
        (Path(output) / "decisions.jsonl").open("w") as stream,
        (Path(output) / "decision_events.jsonl").open("w") as events,
    ):
        decisions = 0
        for step in range(task.definition.defaults.max_steps):
            if survey_open and limit == 0:
                reason = "survey_incomplete"
                break
            decisions = step + 1
            if trace:
                trace.boundary("before", decisions, obs)
            if survey_open:
                survey_turns += 1
            if hasattr(policy, "set_feedback"):
                policy.set_feedback(feedback)
            if hasattr(policy, "set_request_recorder"):
                policy.set_request_recorder(request_recorder(output, "interaction", obs.action_step))
            feedback = None
            decision = None
            def request():
                try:
                    return policy.act(obs, actions, tuple(notes)), None
                except InvalidDecision as error:
                    return None, str(error)
            (decision, feedback), ticks, _, inference_s = clock.request(worker, request, controller.step_base_debug)
            idle_steps += ticks
            decision_s += inference_s
            stream.write(json.dumps(decision) + "\n")
            stream.flush()
            phase = getattr(session, "temporal_phase_controller", None)
            in_survey = phase is not None and phase.phase == "phase_1"
            fields = decision if isinstance(decision, dict) else {}
            before = time.monotonic()
            try:
                if feedback is not None:
                    raise InvalidDecision(feedback)
                event = dispatch(
                    decision,
                    obs,
                    session,
                    executor,
                    actions,
                    action_mode,
                    survey_deadline=survey_open and survey_turns == limit,
                    end_survey=in_survey and step + 1 >= phase_budget,
                )
                feedback = event.get("message")
                if fields.get("note"):
                    notes.append(str(fields["note"])[:128])
            except InvalidDecision as error:
                feedback = str(error)
                event = {
                    "outcome": "invalid",
                    "status": "invalid_decision",
                    "message": feedback,
                }
                if in_survey and step + 1 >= phase_budget:
                    event["phase_transition"] = "phase_1_budget_exhausted"
                    source.report_temporal(
                        "phase_1_budget_exhausted",
                        None,
                        snapshot_id=obs.snapshot_id,
                        action_step=obs.action_step,
                    )
            elapsed = time.monotonic() - before
            if event["outcome"] in ("completed", "stopped"):
                execution_s += elapsed
            event.update(
                decision=step + 1,
                snapshot_id=obs.snapshot_id,
                action_step=obs.action_step,
                command={
                    k: fields[k]
                    for k in ("action", "distance_m", "angle_deg", "repeat")
                    if k in fields
                },
                decision_s=inference_s,
                dispatch_s=elapsed,
            )
            if (ticks and event["status"] != "model_done" and
                    source.agent_observation().snapshot_id == obs.snapshot_id):
                source.capture(action_boundary=True)
            if survey_open:
                if event["status"] == "survey_complete" or any(
                    entry.get("kind") == "history_followup"
                    for entry in getattr(session, "public_task_events", ())
                ):
                    survey_open = False
                elif survey_turns == limit:
                    boundary_obs = source.agent_observation()
                    if event["status"] == "survey_rejected" and boundary_obs.snapshot_id == obs.snapshot_id:
                        reason = "survey_incomplete"
                    else:
                        try:
                            source.report_answer("SURVEY_COMPLETE", snapshot_id=boundary_obs.snapshot_id,
                                                 action_step=boundary_obs.action_step)
                        except IncompleteSurvey:
                            reason = "survey_incomplete"
                        else:
                            survey_open = False
                            feedback = None
                if survey_turns == limit:
                    event["survey_deadline"] = "incomplete" if survey_open else "delivery"
            event["inference_steps"] = ticks
            events.write(json.dumps(event) + "\n")
            events.flush()
            if trace:
                trace.boundary("after", decisions, source.agent_observation())
            if reason == "survey_incomplete":
                break
            if event["status"] == "model_done":
                reason = "model_done"
                break
            obs = source.agent_observation()
    obs = source.agent_observation()
    source.finalize_episode(
        snapshot_id=obs.snapshot_id, action_step=obs.action_step, reason=reason
    )
    terminal = session.task_evaluator.done_result
    physical = operation_progress(terminal)
    (Path(output) / "physical_score.json").write_text(
        json.dumps(physical, indent=2, allow_nan=False)
    )
    obs = source.agent_observation()
    if trace:
        trace.boundary("terminal", decisions, obs)
    spec = task.definition.private.evaluation.parameters["terminal_qa"]
    complete = physical.get("evaluation_validity") == "complete"
    result = {
        "interactive_score": physical["score_percent"] if complete else None,
        "physical_success": physical["task_success"],
        "terminal_qa": None,
        "scoring_version": physical["scoring_version"],
        "world_advances_during_inference": True,
        "inference_steps": idle_steps,
        "inference_simulation_s": idle_steps * controller.dt,
        "action_mode": action_mode,
        "decisions": decisions,
        "termination": reason,
        "decision_budget": task.definition.defaults.max_steps,
        "instruction_protocol": instruction_protocol(),
    }
    if limit is not None:
        result["survey"] = {"limit": limit, "used": survey_turns, "unit": "decisions", "delivery_opened": not survey_open}
    if hasattr(policy, "set_request_recorder"):
        policy.set_request_recorder(request_recorder(output, "terminal_qa", obs.action_step))
    qa_start = time.monotonic()
    try:
        answer = policy.answer(obs, public_questions(spec), tuple(notes))
    except Exception as error:
        result.update(
            qa_error=f"{type(error).__name__}: {error}",
            runtime_s=time.monotonic() - started,
        )
        result["timings"] = {
            "decision_s": decision_s,
            "qa_s": time.monotonic() - qa_start,
            "action_execution_s": execution_s,
        }
        return result
    qa_elapsed = time.monotonic() - qa_start
    qa = grade(
        spec,
        json.dumps(answer) if isinstance(answer, dict) else answer,
        terminal["native_result"],
    )
    (Path(output) / "terminal_qa.json").write_text(json.dumps(qa, indent=2))
    result.update(
        terminal_qa=100 * qa["fraction"] if qa["fraction"] is not None else None,
        runtime_s=time.monotonic() - started,
    )
    result["timings"] = {
        "decision_s": decision_s,
        "qa_s": qa_elapsed,
        "action_execution_s": execution_s,
    }
    return result
