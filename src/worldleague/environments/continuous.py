"""WIA scenes with native continuous control.

Used by worldleague continuous --help.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def action_chunk(response, *, key, action_dim, max_chunk_steps):
    """Validate the whole chunk before executing any part of it."""
    if not isinstance(response, dict) or key not in response:
        raise ValueError(f"Policy response requires {key!r}.")
    # Preserve model-output precision until policy processing and its execution cast.
    chunk = np.asarray(response[key])
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if (chunk.ndim != 2 or chunk.shape[1] != action_dim
            or not 1 <= len(chunk) <= max_chunk_steps or chunk.dtype.kind not in "iuf"
            or not np.isfinite(chunk).all() or np.any(np.abs(chunk.astype(np.float64)) > np.finfo(np.float32).max)):
        raise ValueError("Policy action shape, chunk length, or finite-value contract failed.")
    return chunk


def configure_initialization(session, evaluation, *, is_panda):
    """Install explicitly selected reset adapters before native reset begins."""
    for name in ('initial_posture_adaptation', 'initial_camera_adaptation',
                 'initial_base_adaptation', 'initial_viewpoint_adaptation', 'container_motion_policy'):
        setattr(session, name, evaluation.get(name))
    viewpoint = session.initial_viewpoint_adaptation
    if viewpoint is None:
        return
    previous = getattr(session, '_initialization_stage_observer', None)

    def observe_stage(active_session, stage, env_config):
        if previous is not None:
            previous(active_session, stage, env_config)
        if stage == 'adapted':
            from worldleague.environments.pandaomron_viewpoint import initialize_optical_viewpoint
            initialize_optical_viewpoint(active_session, viewpoint['reference'])

    session._initialization_stage_observer = observe_stage


def boundary_snapshot_id(env):
    return f"continuous_{env.session.native_reset_id}_{env.control_steps:012d}_{getattr(env, '_observation_epoch', 0):04d}"


class ContinuousEnvironment:
    """Own one native WIA scene; no macro controller or model-specific preprocessing."""

    def __init__(self, plan):
        self.plan = plan
        self.session = None
        self.control_steps = 0
        self.inference_steps = 0
        self._inference_hold = None
        self.metadata = {}
        self.history = None
        self.last_action_metadata = {}
        self._latest_observation = None
        self.collection_evaluator = None
        self.scoring = None
        self.temporal = None
        self._observation_epoch = 0
        self.observation_capture_mode = plan["environment"].get("observation_capture_mode", "all_controls")
        self._capture_controls = set()
        self._terminal_observation = False
        self.observation_performance = {"captures": 0, "render_updates": 0,
                                        "render_s": 0., "sensor_read_s": 0., "control_step_s": 0.}

    def reset(self):
        if self.session is not None:
            self.close()
        self.control_steps = 0
        self.inference_steps = 0
        self._inference_hold = None
        self.metadata = {}
        self.history = None
        self.last_action_metadata = {}
        self._latest_observation = None
        self.collection_evaluator = None
        self.scoring = None
        self.temporal = None
        self._observation_epoch = 0
        self._capture_controls = set()
        self._terminal_observation = False
        self.observation_performance.update({key: 0 for key in self.observation_performance})
        import random
        import torch
        from worldleague.tasks.loader import load_task
        from worldleague.tasks.scene import DeclarativeScene
        from worldleague.runtime.sim.esi_omnigibson import ESISession

        task_config = self.plan["task"]
        self.task = load_task(task_config["path"], variant=task_config["variant"], seed=task_config["seed"],
                              task_override=task_config.get("task_override"), release_support_edges=task_config["release_support_edges"], run_mode="benchmark")
        random.seed(self.task.seed)
        np.random.seed(self.task.seed)
        torch.manual_seed(self.task.seed)
        adapter = DeclarativeScene(self.task)
        settings = adapter.observation_settings()
        # The robot file owns optics; do not inherit the R1 eight-camera preset.
        settings.pop("camera_size", None)
        config = self.plan["environment"]["robot"]["config"]
        self.is_panda = config["model"] == "pandaomron"
        timing = None
        run = self.plan["environment"]
        if "control_hz" in run:
            timing = {"action_frequency": run["control_hz"],
                      "rendering_frequency": run.get("rendering_hz", run["control_hz"])}
            if "physics_hz" in run:
                timing["physics_frequency"] = run["physics_hz"]
        self.session = ESISession(self.task.source_path, robot_config=config, native_timing=timing, **settings)
        self.session.loaded_task = self.task
        self.session.scene_adapter = adapter
        configure_initialization(self.session, self.plan['evaluation'], is_panda=self.is_panda)
        if self.session.initial_viewpoint_adaptation is not None:
            self.session.initial_camera_calibration_settings = deepcopy(
                self.plan['policy']['profile']['config']['alignment']['camera_calibration'])
        self.session.reset()
        self.robot = self.session.robot
        groups = {name: {"indices": array(indices).astype(int).tolist(),
                         "controller": type(self.robot.controllers[name]).__name__,
                         "motor_type": getattr(self.robot.controllers[name], "motor_type",
                                               getattr(self.robot.controllers[name], "_motor_type", None)),
                         "use_delta_commands": getattr(self.robot.controllers[name], "use_delta_commands", None)}
                  for name, indices in self.robot.controller_action_idx.items()}
        self.action_dim = self.robot.action_dim
        self.native_action_dim = self.robot.action_dim
        if self.is_panda:
            from worldleague.policies.xr1_robocasa365 import ObservationHistory, PandaOmronActionAdapter
            physics = json.loads(Path(config["source_manifest"]).with_name("physics.json").read_text())
            self.action_adapter = PandaOmronActionAdapter(self.robot, physics)
            self.action_dim = self.action_adapter.policy_action_dim
            processing = self.plan["policy"]["profile"]["config"]["action_processing"]
            self.history = ObservationHistory(processing["history_length"], processing["history_interval_steps"],
                                              allow_sparse=self.observation_capture_mode == "policy_history")
        self.dt = 1.0 / float(self.session.env.env_config["action_frequency"])
        reference_hz = self.plan["evaluation"]["control_budget"]["control_hz"]
        self.roles = config["eval"]["camera_sensor_names"]
        if self.plan["evaluation"].get("operation_scoring"):
            self.session.robot_camera_registry = dict(self.roles)
            # Private instance masks are read by existing visibility scorers only.
            for name in self.roles.values():
                self.robot.sensors[name].add_modality("seg_instance")
            self.session.env.load_observation_space()
        self.metadata = {"robot_type": type(self.robot).__name__, "action_dim": self.action_dim,
                         "controllers": groups, "control_dt_s": self.dt,
                         "joint_names": list(self.robot.joints),
                         "joint_positions_at_reset": array(self.robot.get_joint_positions()).tolist(),
                         "proprio_obs": config["proprio_obs"],
                         "embodiment_settings": self.session.embodiment_settings,
                         "cameras": {role: {"name": name,
                            "image_width": int(self.robot.sensors[name].image_width),
                            "image_height": int(self.robot.sensors[name].image_height),
                            "horizontal_aperture": float(self.robot.sensors[name].horizontal_aperture),
                            "focal_length": float(self.robot.sensors[name].focal_length)}
                            for role, name in self.roles.items()},
                         "constructor_adapter": f"model:{config['model']} -> type:{type(self.robot).__name__}",
                         "native_action_dim": self.native_action_dim,
                         "timing": {"effective_control_hz": 1.0 / self.dt, "budget_reference_hz": reference_hz,
                                    "matches_shared_budget_frequency": bool(np.isclose(1.0 / self.dt, reference_hz)),
                                    "diagnostic_timing_reason": self.plan["evaluation"].get("diagnostic_timing_reason")},
                         "task": self.task.manifest("continuous_" + self.session.native_reset_id),
                         "scene_patches": adapter.applied}
        self.metadata["timing"].update(
            physics_hz=float(self.session.env.env_config["physics_frequency"]),
            rendering_hz=float(self.session.env.env_config["rendering_frequency"]),
            observation_capture_mode=self.observation_capture_mode, render_updates_per_capture=4)
        if self.is_panda:
            self.metadata["timing"].update(
                model_reference_control_hz=processing["reference_control_hz"],
                matches_model_reference_frequency=bool(np.isclose(1.0 / self.dt, processing["reference_control_hz"])),
                history_interval_controls=processing["history_interval_steps"],
                effective_history_interval_s=processing["history_interval_steps"] * self.dt)
        if hasattr(self.robot, "mobile_tray_mount"):
            self.metadata.update(protocol="single_arm_mobile_tray_v1",
                                 mobile_tray_mount=self.robot.mobile_tray_mount,
                                 parent_task=self.task.definition.repair.get("control_budget_parent", self.task.definition.id))
        if "profile" in self.plan["policy"]:
            from worldleague.environments.robot_config import prime_cameras
            calibration = self.plan["policy"]["profile"]["config"]["alignment"]["camera_calibration"]
            self.metadata["camera_render_updates"] = prime_cameras(
                self.robot, self.session.pipeline.og.sim, self.roles, calibration)
        return self.observe()

    def _capture_observation(self):
        if self.history is not None and self._latest_observation is not None and self._latest_observation["control_step"] == self.control_steps:
            return self._latest_observation
        # Flush sensor transforms at the current observation boundary.
        started = time.perf_counter()
        for _ in range(4):
            self.session.pipeline.og.sim.render()
        self.observation_performance["render_s"] += time.perf_counter() - started
        self.observation_performance["render_updates"] += 4
        started = time.perf_counter()
        images = {}
        for role, name in self.roles.items():
            raw, _ = self.robot.sensors[name].get_obs()
            image = array(raw["rgb"])[..., :3].copy()
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise RuntimeError("Native RGB must be a nonempty uint8 HWC image.")
            if not image.size:
                raise RuntimeError("Native camera buffer is empty.")
            images[role] = image
        if self.is_panda:
            from worldleague.policies.xr1_robocasa365 import native_measurements, state14
            state = state14(native_measurements(self.robot))
        else:
            state, _ = self.robot.get_proprioception()
            state = array(state).astype(np.float32, copy=True)
        if state.ndim != 1 or not np.isfinite(state).all():
            raise RuntimeError("Native proprioception must be a finite vector.")
        # No private task bindings, object poses, segmentation, or scorer data.
        observation = {"images": images, "state": state, "prompt": self.task.task_text,
                       "control_step": self.control_steps, "simulation_time_s": (self.control_steps + self.inference_steps) * self.dt}
        if self.history is not None:
            self.history.append(observation)
            self._latest_observation = observation
        self.observation_performance["sensor_read_s"] += time.perf_counter() - started
        self.observation_performance["captures"] += 1
        return observation

    def observe(self):
        observation = self._capture_observation()
        if self.history is not None and "history" not in observation:
            try:
                observation["history"] = self.history.sample(observation["prompt"])
            except ValueError as error:
                if not self._terminal_observation:
                    raise
                # An unplanned early native termination has no next inference.
                # Keep its fresh terminal image without fabricating past frames.
                observation["terminal_history_unavailable"] = str(error)
        return observation

    def step_during_inference(self):
        """Advance native physics without spending a policy control or repeating its last velocity."""
        import torch
        if self._inference_hold is None:
            state = self.robot.get_control_dict()
            hold = torch.zeros(self.robot.action_dim, dtype=torch.float32)
            last = self.robot.dump_action()
            for name, controller in self.robot.controllers.items():
                indices = self.robot.controller_action_idx[name]
                if name.startswith("gripper") and last is not None:
                    command = last[indices]
                else:
                    command = controller.compute_no_op_action(state)
                hold[array(indices).astype(int)] = torch.as_tensor(array(command), dtype=hold.dtype)
            self._inference_hold = hold
        result = self.session.step_control(self._inference_hold)
        self.inference_steps += 1
        if self.scoring is not None:
            self.scoring.sample_step()
        terminated = bool(result[2] or result[3])
        if terminated:
            self._terminal_observation = True
            self._latest_observation = None
            self._observation_epoch += 1
            if self.history is not None and self.history.samples and self.history.samples[-1]["control_step"] == self.control_steps:
                self.history.samples.pop()
        return terminated

    def begin_action_chunk(self, controls):
        self._inference_hold = None
        if self.observation_capture_mode != "policy_history":
            return
        from worldleague.policies.xr1_robocasa365 import required_history_captures
        self._capture_controls = required_history_captures(
            self.control_steps, controls, self.plan["policy"]["execution_horizon"],
            self.plan["evaluation"]["max_control_steps"], self.history.length, self.history.interval_steps)

    def observe_control_boundary(self, *, terminated=False):
        self._terminal_observation = terminated
        if self.observation_capture_mode == "all_controls":
            return self.observe()
        if terminated or self.control_steps in self._capture_controls:
            return self._capture_observation()

    def step(self, action):
        import torch
        if np.asarray(action).shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Invalid continuous action.")
        if self.is_panda:
            requested_action = np.asarray(action).copy()
            survey = self.temporal is not None and self.temporal.surveying
            if survey:
                action = self.temporal.gate_action(action)
                progress = self.action_adapter.gripper_progress.copy()
                finger_controller = self.robot.controllers['gripper_right']
                goal = finger_controller._goal
                held_fingers = (array(goal['target']).copy() if goal is not None else
                    array(self.robot.get_joint_positions())[array(self.robot.gripper_control_idx['right']).astype(int)].copy())
            action, self.last_action_metadata = self.action_adapter.convert(action)
            if survey:
                self.action_adapter.gripper_progress = progress
                action[self.action_adapter.indices['gripper_right']] = held_fingers
                self.last_action_metadata.update(temporal_phase='phase_1',
                    phase_1_action_gate=True, requested_policy_action=requested_action.tolist(),
                    gripper_progress=progress.tolist(), native_command=action.tolist(), gripper_close=None)
        elif self.temporal is not None:
            requested_action = np.asarray(action).copy()
            action = self.temporal.gate_action(action)
            self.last_action_metadata = {
                'temporal_phase': self.temporal.controller.phase,
                'phase_1_action_gate': self.temporal.surveying,
                'requested_policy_action': requested_action.tolist(),
                'native_command': action.tolist()}
        started = time.perf_counter()
        result = self.session.step_control(torch.as_tensor(action, dtype=torch.float32))
        self.observation_performance["control_step_s"] += time.perf_counter() - started
        if getattr(self.robot, "native_controller_errors", []):
            raise RuntimeError(f"Native controller callback failed: {self.robot.native_controller_errors[-1]}")
        self.control_steps += 1
        if self.scoring is not None:
            self.scoring.sample_step()
        elif self.collection_evaluator is not None:
            self.collection_evaluator.sample_step()
        if self.is_panda:
            self.last_action_metadata.update(
                measured_base_pose=[array(value).tolist() for value in self.robot.get_position_orientation()],
                measured_eef_pose=[array(value).tolist() for value in self.robot.get_eef_pose("right")],
                measured_joint_positions=array(self.robot.get_joint_positions()).tolist(),
                simulation_time_s=(self.control_steps + self.inference_steps) * self.dt)
        # Native DummyTask termination is distinct from a WIA physical-goal score.
        return bool(result[2] or result[3])

    @property
    def snapshot_id(self):
        return boundary_snapshot_id(self)

    def save_metadata(self, output):
        if self.plan["evaluation"].get("operation_scoring") and self.scoring is None:
            from worldleague.environments.continuous_scoring import ContinuousScoring
            self.scoring = ContinuousScoring(self, Path(output) / "private_evaluation")
            self.scoring.begin(self.observe())
            if self.plan['evaluation'].get('temporal_survey'):
                from worldleague.environments.continuous_temporal import ContinuousTemporal
                self.temporal = ContinuousTemporal(self)
                self.metadata['temporal_survey'] = {
                    'protocol': self.plan['evaluation']['temporal_survey'],
                    'policy': self.temporal.policy, 'decision_unit': 'one_policy_action_chunk',
                    'terminal_qa_enabled': False}
                if self.temporal.action_gate_receipt is not None:
                    self.metadata['temporal_survey']['action_gate_capture'] = self.temporal.action_gate_receipt
            if hasattr(self.robot, "mobile_tray_mount"):
                self.collection_evaluator = self.scoring.evaluator.native
            self.metadata["scoring"] = {"protocol": "continuous_operation_only_v1", "terminal_qa_enabled": False}
        Path(output, "environment.json").write_text(json.dumps(self.metadata, indent=2) + "\n")

    def save_observation(self, output, observation):
        if self.scoring is not None:
            self.scoring.record(observation)

    def finalize_evaluation(self, reason):
        if self.scoring is not None:
            observation = self.observe()
            if self.plan.get("record_trajectory"):
                self.terminal_observation = observation
            return self.scoring.finalize(reason, observation)

    @property
    def delivery_opened(self):
        return any(event.get("kind") == "history_followup" for event in getattr(self.session, "public_task_events", ()))

    def prepare_policy_observation(self, observation):
        if self.temporal is not None:
            return self.temporal.prompt(observation)
        if self.scoring is not None:
            advance = getattr(self.scoring.evaluator.native, "advance_action_only_survey", None)
            if advance is not None:
                request = advance(self.scoring.boundary(observation))
                if request:
                    return {**observation, "prompt": observation["prompt"] + "\n" + request}
        return observation

    def after_action_chunk(self, observation, *, terminated=False):
        if self.temporal is not None:
            return self.temporal.after_chunk(observation, terminated=terminated)
        return observation

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
