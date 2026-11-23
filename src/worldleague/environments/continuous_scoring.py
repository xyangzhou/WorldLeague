"""Read-only continuous-control bridge to existing native WIA scorers.

Used by worldleague continuous --help.
"""
from types import SimpleNamespace
from datetime import datetime, timezone
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from worldleague.runtime.sim.robot_measurements import BaseControlConfig, GripperControlConfig, base_pose, ee_pose
from worldleague.environments.continuous import array, boundary_snapshot_id


class ScoringController:
    """Expose measured state without installing or invoking macro controllers."""

    def __init__(self, environment):
        self.environment = environment
        self.session, self.robot, self.dt = environment.session, environment.robot, environment.dt
        self.working_arm = "right" if "right" in self.robot.arm_names else self.robot.arm_names[0]
        self.on_step = None
        self.config = BaseControlConfig.load()
        if hasattr(environment, "action_adapter"):
            velocity = np.max(np.abs(environment.action_adapter.base_limits), axis=1)
            self.config = replace(self.config,
                max_translation_speed_m_s=float(np.linalg.norm(velocity[:2])),
                max_rotation_speed_deg_s=float(np.rad2deg(velocity[2])))
        self.gripper_config = GripperControlConfig.load()

    def pose(self):
        return base_pose(self.robot)

    def ee_pose(self):
        return ee_pose(self.robot, self.working_arm)

    def physical_grasp_measurement(self):
        from worldleague.environments.mobile_tray import evaluator_grasp
        now = float(self.session.pipeline.og.sim.current_time)
        cached = getattr(self, "_physical_grasp_cache", None)
        if cached is not None and cached[0] == now:
            return cached[1]
        rows = {}
        for arm in self.robot.arm_names:
            measured = evaluator_grasp(self.robot, self.session.env.scene, arm, native_time_s=now)
            indices = array(self.robot.gripper_control_idx[arm]).astype(int)
            positions = array(self.robot.get_joint_positions())[indices]
            if hasattr(self.environment, "action_adapter"):
                limits = self.environment.action_adapter.gripper_limits
                opened = np.array([limits[0, 1], limits[1, 0]])
            else:
                raise ValueError("Physical finger opening targets are not declared.")
            rows[arm] = {"available": True, "object_name": measured["holding"],
                         "holding": not measured["empty"], "constraint_active": False,
                         "release_pending": False,
                         "gripper_open": bool(np.max(np.abs(positions - opened)) <= self.gripper_config.joint_tolerance_m),
                         "physical_grasp_established": measured["holding"] is not None,
                         "joint_positions_m": positions.tolist(), "physical_contact_evidence": measured}
        working = rows[self.working_arm]
        held = sorted({row["object_name"] for row in rows.values() if row["object_name"] is not None})
        detached = all(not row["holding"] for row in rows.values())
        raw = {"available": True, "working_arm": self.working_arm, "arms": rows,
               "object_name": working["object_name"], "holding": not detached,
               "held_names": held, "constraint_active": False, "release_pending": False,
               "gripper_open": working["gripper_open"],
               "all_grippers_open": all(row["gripper_open"] for row in rows.values()),
               "pickup_eligible": working["object_name"] is not None,
               "physical_grasp_established": working["physical_grasp_established"],
               "measurement": "native_physical_contacts_no_assisted_constraint", "measurement_errors": []}
        result = ({"available": True, "held_names": held, "detached": detached,
                   "released": detached and working["gripper_open"]}, raw)
        self._physical_grasp_cache = (now, result)
        return result

    def debug_gripper_state(self):
        if self.robot.grasping_mode == "physical":
            _, raw = self.physical_grasp_measurement()
            return {**raw, "grasp_mode": "physical", "gripper_closed": False}
        from worldleague.adapters.rigid_partial_insertion import _aggregate_grasp
        return _aggregate_grasp(self.robot, self)

    def geometric_rearrangement_evidence(self):
        from worldleague.runtime.sim.esi_interaction_history import rearrangement_evidence
        # The continuous step is the complete public action; no macro is executed.
        self._public_action_context = {"sequence": self.environment.control_steps, "active": True}
        return rearrangement_evidence(self)


class ContinuousScoring:
    def __init__(self, environment, output):
        self.environment, self.output = environment, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        environment.session._wia_action_only_scoring = True
        self.controller = ScoringController(environment)
        self.evaluator = environment.session.scene_adapter.create_evaluator(
            environment.session, self.controller, self.output)
        environment.session.task_evaluator = self.evaluator
        from worldleague.runtime.sim.esi_visual_evidence import install_visual_evidence
        install_visual_evidence(environment.session, self.controller)
        self.latest = None

    def boundary(self, observation=None):
        env = self.environment
        images = {} if observation is None else observation["images"]
        boundary = SimpleNamespace(snapshot_id=boundary_snapshot_id(env),
            action_step=env.control_steps, task_text=env.task.task_text, action_history=(),
            captured_at=datetime.now(timezone.utc).isoformat(), rgb_images=lambda: images)
        env.session._observation_source = SimpleNamespace(
            latest_snapshot=SimpleNamespace(observation=boundary))
        return boundary

    def begin(self, observation):
        boundary = self.boundary(observation)
        self.evaluator.capture_observation(boundary)
        self.latest = self.evaluator.begin_episode(boundary)

    def sample_step(self):
        self.evaluator.sample_step()

    def record(self, observation):
        boundary = self.boundary(observation)
        self.evaluator.capture_observation(boundary)
        self.latest = self.evaluator.record(boundary, action=None)
        from worldleague.tasks.terminal_qa import operation_progress
        live = operation_progress(self.latest, require_text_confirmation=False)
        live.update(action_step=self.environment.control_steps, evaluation_finalized=False,
                    terminal_qa_enabled=False,
                    declaration_policy="native_evidence_without_text_v1")
        (self.output / "live_operation_score.json").write_text(json.dumps(live, indent=2, allow_nan=False) + "\n")

    def finalize(self, reason, observation):
        boundary = self.boundary(observation)
        from worldleague.environments.continuous_declarations import freeze_native_boundary
        if getattr(self.evaluator, "failure", None) is None:
            try:
                freeze_native_boundary(getattr(self.evaluator, "native", None), boundary)
            except Exception as error:
                self.evaluator._fail("action_only_terminal_boundary", error)
        self.latest = self.evaluator.finalize_episode(boundary, reason=reason)
        from worldleague.tasks.terminal_qa import operation_progress
        operation = operation_progress(self.latest, require_text_confirmation=False)
        operation.update(declaration_policy="native_evidence_without_text_v1", primary_metric="operation_only_100",
                         terminal_qa_enabled=False, evaluation_finalized=True,
                         model_declared_done=False, termination_reason=reason)
        (self.output / "operation_score.json").write_text(json.dumps(operation, indent=2, allow_nan=False) + "\n")
        return operation
