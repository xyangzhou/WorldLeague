"""Dispatch one public decision. Run with worldleague eval TASK --config FILE."""

from worldleague.tasks.wia_history_collection import IncompleteSurvey

from worldharness.core.action_compiler import ParameterizedAction, compile_action


from worldharness.protocol import InvalidDecision


def dispatch(
    decision, observation, session, executor, actions, mode, *, end_survey=False, survey_deadline=False
):
    if not isinstance(decision, dict):
        raise InvalidDecision("Return a JSON object containing an action.")
    action = decision.get("action")
    survey = decision.get("survey_complete") or decision.get("phase_1_complete")
    for key in ("survey_complete", "phase_1_complete"):
        if key in decision and type(decision[key]) is not bool:
            raise InvalidDecision(f"{key} must be true or false.")
    if action != "DONE" and action not in actions:
        raise InvalidDecision("Choose an available action or DONE.")
    phase = getattr(session, "temporal_phase_controller", None)
    operation = getattr(
        session.loaded_task.definition.private.evaluation, "operation", None
    )
    legacy_phase = (
        decision.get("phase_1_complete") is True and phase and phase.phase == "phase_1"
    )
    if survey and (
        (action != "DONE" and not legacy_phase) or decision.get("answer") is not None
    ):
        raise InvalidDecision("To finish a survey use DONE with no answer.")
    command = None
    repeat = decision.get("repeat", 1)
    if action != "DONE" and not legacy_phase:
        if mode == "fixed-repeat":
            if type(repeat) is not int or repeat not in (1, 2, 4, 8):
                raise InvalidDecision("Use a repeat count of 1, 2, 4 or 8.")
        else:
            payload = {
                k: decision[k]
                for k in ("action", "angle_deg", "distance_m")
                if k in decision
            }
            try:
                command = ParameterizedAction.from_payload(payload, actions)
                compile_action(
                    command,
                    executor.capabilities,
                    executor.primitive_resolution,
                    executor.macro_limits,
                )
            except (ValueError, TypeError) as error:
                raise InvalidDecision(str(error)) from error

    source = session.observation_source
    boundary = {
        "snapshot_id": observation.snapshot_id,
        "action_step": observation.action_step,
    }
    try:
        if survey and operation == "history_multi_retrieval_v1":
            try:
                source.report_answer("SURVEY_COMPLETE", **boundary)
            except IncompleteSurvey:
                return {
                    "outcome": "declaration", "status": "survey_rejected",
                    "message": "Survey completion was not accepted. Continue inspecting and preserve the targets.",
                }
            return {"outcome": "declaration", "status": "survey_complete"}
        if survey and not (phase and phase.phase == "phase_1"):
            raise InvalidDecision("This turn has no open survey.")
        if decision.get("arrived_rooms") and not (
            isinstance(decision["arrived_rooms"], list)
            and all(isinstance(r, str) for r in decision["arrived_rooms"])
        ):
            raise InvalidDecision("arrived_rooms must be a list of room names.")
        if decision.get("arrived_rooms"):
            source.report_arrivals(decision["arrived_rooms"], **boundary)
        if decision.get("observed_objects"):
            source.report_observed_objects(
                decision["observed_objects"], list(observation.rgb_images()), **boundary
            )
        if decision.get("answer") is not None:
            if phase and phase.phase == "phase_2":
                source.report_temporal("answer", decision["answer"], **boundary)
            else:
                source.report_answer(decision["answer"], **boundary)
    except ValueError as error:
        raise InvalidDecision(str(error)) from error

    if action == "DONE" or legacy_phase:
        if phase and phase.phase == "phase_1":
            source.report_temporal("phase_1_complete", None, **boundary)
            return {"outcome": "declaration", "status": "phase_1_complete"}
        if survey_deadline:
            return {"outcome": "declaration", "status": "survey_deadline"}
        source.report_done(**boundary)
        return {"outcome": "declaration", "status": "model_done"}
    if end_survey:
        source.report_temporal("phase_1_budget_exhausted", None, **boundary)
        return {"outcome": "declaration", "status": "phase_1_budget_exhausted"}
    if command is not None:
        result = executor.execute_macro(
            command,
            resolution=executor.primitive_resolution,
            limits=executor.macro_limits,
        )
    else:
        for _ in range(repeat):
            result = executor.execute(action)
            if not result["completed"]:
                break
    return {
        "outcome": "completed" if result["completed"] else "stopped",
        "status": result["reason"],
    }
