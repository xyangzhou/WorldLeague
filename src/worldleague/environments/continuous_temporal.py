"""Bounded continuous surveys using the existing native two-phase event.

Used by worldleague continuous --help.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from worldleague.tasks.temporal_policy import survey_policy


PROTOCOL = "task_bounded_auto_v1"


class ContinuousTemporal:
    def __init__(self, environment):
        self.environment = environment
        self.controller = environment.session.temporal_phase_controller
        self.policy = survey_policy(environment.task.definition.private.evaluation.parameters)
        if not self.policy["phase_1_auto_transition"]:
            raise ValueError("Task survey does not authorize automatic transitions.")
        horizon = environment.plan["policy"]["execution_horizon"]
        if self.policy["phase_1_max_decisions"] * horizon >= environment.plan["evaluation"]["max_control_steps"]:
            raise ValueError("Survey limit must leave controls for Phase 2.")
        self.r1pro = not environment.is_panda
        self.action_gate_receipt = None
        if self.r1pro:
            if (environment.plan["policy"].get("observation_format") != "r1pro_joint23"
                    or environment.plan["environment"]["robot"]["config"]["model"] != "r1pro"):
                raise ValueError("Continuous temporal survey requires the explicit XR-1 or R1Pro joint23 action contract.")
            self._capture_r1pro_hold()
        self.decisions = 0
        self.transition = None
        self.output = Path(environment.plan["output_dir"]) / "temporal"
        self.output.mkdir(parents=True, exist_ok=True)
        protocol = {"protocol": PROTOCOL, "policy": self.policy,
            "decision_unit": "one_policy_action_chunk", "execution_horizon": horizon,
            "total_control_budget_unchanged": True, "terminal_qa_enabled": False,
            "phase_1_action_gate": "Hold arm target and finger targets; permit base and torso controls."}
        if self.action_gate_receipt is not None:
            protocol["action_gate_capture"] = self.action_gate_receipt
        self._write("protocol.json", protocol)

    def _capture_r1pro_hold(self):
        from worldleague.environments.continuous import array

        robot = self.environment.robot
        layout = {"base": range(3), "trunk": range(3, 7), "arm_left": range(7, 14),
                  "gripper_left": range(14, 15), "arm_right": range(15, 22),
                  "gripper_right": range(22, 23)}
        if robot.action_dim != 23 or any(
                not np.array_equal(array(robot.controller_action_idx.get(name, [])), list(indices))
                for name, indices in layout.items()):
            raise ValueError("R1Pro temporal survey requires the validated native 23-action layout.")
        control = robot.get_control_dict()
        self.held_commands = {}
        captured = {}
        for name in ("arm_left", "gripper_left", "arm_right", "gripper_right"):
            indices = np.asarray(list(layout[name]), dtype=int)
            command = array(robot.controllers[name].compute_no_op_action(control)).copy()
            if command.shape != indices.shape or not np.isfinite(command).all():
                raise ValueError(f"Invalid R1Pro measured no-op command for {name}.")
            self.held_commands[name] = (indices, command)
            captured[name] = {"action_indices": indices.tolist(), "command": command.tolist()}
        self.action_gate_receipt = {
            "source": "robot.get_control_dict / controller.compute_no_op_action",
            "control_step": self.environment.control_steps,
            "fixed_for_phase_1": True, "controllers": captured}

    def _write(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    @property
    def surveying(self):
        return self.controller.phase == "phase_1"

    def gate_action(self, action):
        requested = np.asarray(action)
        if self.r1pro:
            if not self.surveying:
                return requested.copy()
            dtype = np.result_type(requested.dtype, *(command.dtype for _, command in self.held_commands.values()))
            applied = requested.astype(dtype, copy=True)
            for indices, command in self.held_commands.values():
                applied[indices] = command
            return applied
        applied = requested.copy()
        if self.surveying:
            applied[:6] = 0
            # convert() ramps fingers for either binary command. Environment.step
            # separately restores its held progress and measured native targets.
            applied[11] = 1
        return applied

    def prompt(self, observation):
        phase = self.controller.phase
        if phase == "phase_1":
            remaining = max(0, self.policy["phase_1_max_decisions"] - self.decisions)
            instruction = (f"Current phase: Phase 1 read-only survey. {remaining} action chunks remain before "
                           "the automatic unobserved event. Only base and torso controls execute; "
                           "arm and gripper targets remain held.")
        else:
            instruction = ("Current phase: Phase 2. The unobserved event has occurred. Compare the current "
                           "scene with your earlier observations and perform the task's applicable physical operation. "
                           "Arm and gripper controls are now enabled. No answer submission is required.")
        return {**observation, "prompt": observation["prompt"] + "\n" + instruction}

    def after_chunk(self, observation, *, terminated=False):
        if not self.surveying:
            return observation
        self.decisions += 1
        if terminated or self.decisions < self.policy["phase_1_max_decisions"]:
            return observation
        env = self.environment
        if env.scoring is None:
            raise RuntimeError("Native temporal transitions require the active native evaluator.")
        boundary = env.scoring.boundary(observation)
        before_time = float(env.session.pipeline.og.sim.current_time)
        receipt = {"protocol": PROTOCOL, "reason": "phase_1_budget_exhausted",
            "phase_1_decisions": self.decisions, "control_step": env.control_steps,
            "before_snapshot_id": boundary.snapshot_id, "native_time_s": before_time,
            "model_declared_done": False, "terminal_qa_enabled": False}
        self._write("phase_1_boundary.json", receipt)
        from PIL import Image
        evidence = self.output / 'phase_1_final'
        evidence.mkdir(exist_ok=False)
        for role, pixels in observation['images'].items():
            Image.fromarray(pixels).save(evidence / (role + '.png'))
        self._write('phase_1_final/state.json', {
            'control_step': env.control_steps, 'state': observation['state'].tolist(),
            'snapshot_id': boundary.snapshot_id, 'prompt': observation['prompt']})
        try:
            env.scoring.evaluator.prepare_temporal_transition(boundary)
            event = self.controller.advance()
            if self.controller.phase != "phase_2":
                raise RuntimeError("Native source event did not enter Phase 2.")
            if float(env.session.pipeline.og.sim.current_time) != before_time:
                raise RuntimeError("Native external event unexpectedly advanced simulation time.")
            event["reason"] = "budget_exhausted"
            # Replace only the current, pre-event image. Earlier timed history
            # remains exact; the next request observes the real event result.
            if env.history is not None and env.history.samples[-1]["control_step"] == env.control_steps:
                env.history.samples.pop()
            env._latest_observation = None
            env._observation_epoch += 1
            refreshed = env.observe()
            env.save_observation(env.plan["output_dir"], refreshed)
            receipt.update(native_event=deepcopy(event), after_snapshot_id=env.scoring.boundary(refreshed).snapshot_id,
                           completed=True, physics_steps=0, robot_actions=0)
            self.transition = receipt
            self._write("transition.json", receipt)
            return refreshed
        except Exception as error:
            receipt.update(completed=False, error=f"{type(error).__name__}: {error}")
            self._write("transition.json", receipt)
            raise
