"""Model-independent BEHAVIOR WebSocket transport.

Used by worldleague continuous --help.
"""
from __future__ import annotations

import numpy as np


def encode_array(value):
    """Numeric MessagePack wire schema used by the BEHAVIOR policy server."""
    if isinstance(value, np.ndarray) and value.dtype.kind not in "OVc":
        return {b"__ndarray__": True, b"dtype": value.dtype.str,
                b"shape": value.shape, b"data": value.tobytes()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported wire value: {type(value).__name__}")


def decode_array(value):
    if b"__ndarray__" in value:
        dtype = np.dtype(value[b"dtype"])
        if dtype.kind in "OVc":
            raise ValueError("Only real numeric arrays are accepted from the policy.")
        return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"]).copy()
    if b"__npgeneric__" in value:
        return value[b"data"]
    return value


def r1pro_state23(state, robot_config):
    """Extract measurements shared by the wire adapter and policy processing."""
    expected = ["base_qvel", "arm_left_qpos", "arm_left_qvel", "eef_left_pos", "eef_left_quat",
                    "gripper_left_qpos", "gripper_left_qvel", "arm_right_qpos", "arm_right_qvel",
                    "eef_right_pos", "eef_right_quat", "gripper_right_qpos", "gripper_right_qvel",
                    "trunk_qpos", "trunk_qvel"]
    state = np.asarray(state, dtype=np.float32)
    if robot_config["proprio_obs"] != expected or state.shape != (61,):
        raise ValueError("The R1Pro joint23 adapter requires the declared 61-dimensional proprio layout.")
    return np.concatenate((state[:3], state[53:57], state[3:10],
                           state[24:26].sum(keepdims=True), state[28:35],
                           state[49:51].sum(keepdims=True)))


def policy_observation(observation, robot_config, policy):
    if policy["observation_format"] == "xr1_robocasa365":
        return {**observation["history"], "prompt": observation["prompt"]}
    if policy["observation_format"] == "r1pro_joint23":
        state23 = r1pro_state23(observation["state"], robot_config)
        return {"images": {role: resize_with_pad(observation["images"][role], policy["image_size"])
                           for role in ("head", "left_wrist", "right_wrist")},
                "state": state23, "prompt": observation["prompt"]}
    if policy["observation_format"] == "wia":
        return {key: observation[key] for key in
                ("images", "state", "prompt", "control_step", "simulation_time_s")}
    namespace = policy["observation_namespace"]
    name = robot_config["name"]
    roles = robot_config["eval"]["camera_sensor_names"]
    result = {f"{namespace}::proprio": observation["state"], "prompt": observation["prompt"]}
    for role, sensor in roles.items():
        suffix = sensor.removeprefix(name + ":")
        result[f"{namespace}::{namespace}:{suffix}::rgb"] = observation["images"][role]
    return result


def resize_with_pad(image, size):
    """Match OpenPI client's PIL bilinear resize and centered zero padding."""
    from PIL import Image
    image = Image.fromarray(image)
    width, height = image.size
    ratio = max(width / size, height / size)
    resized = image.resize((int(width / ratio), int(height / ratio)), resample=Image.Resampling.BILINEAR)
    result = Image.new(image.mode, (size, size), 0)
    result.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return np.asarray(result)


class WebsocketPolicy:
    def __init__(self, config, robot_config):
        self.config, self.robot_config = config, robot_config
        self.connection = None

    def start(self):
        import msgpack
        from websockets.sync.client import connect

        self.connection = connect(self.config["url"], compression=None, max_size=None,
                                  open_timeout=self.config["timeout_s"])
        self.metadata = self._receive()
        # BEHAVIOR reset is one-way; waiting for a reset reply would deadlock.
        if self.config.get("reset_mode", "behavior") == "behavior":
            self.connection.send(msgpack.packb({"reset": True}))

    def _receive(self):
        import msgpack
        raw = self.connection.recv(timeout=self.config["timeout_s"])
        if isinstance(raw, str):
            raise RuntimeError("Policy server returned a text error: " + raw[:2000])
        value = msgpack.unpackb(raw, object_hook=decode_array, strict_map_key=False)
        if not isinstance(value, dict):
            raise ValueError("Policy server must return a mapping.")
        return value

    def infer(self, observation):
        import msgpack
        value = policy_observation(observation, self.robot_config, self.config)
        self.last_request = value
        self.connection.send(msgpack.packb(value, default=encode_array))
        # Do not replay an ambiguous in-flight request after a connection failure.
        return self._receive()

    def close(self):
        if self.connection is not None:
            self.connection.close()
