"""Use a multimodal Chat Completions endpoint via worldleague eval --config FILE."""

import base64
import io
import json
import os
from pathlib import Path
import time

import requests
from PIL import Image

from worldleague.decisions import InvalidDecision
from worldleague.evaluation.requests import request_recorder
from .historical_rgb import HistoricalRGB, MAX_IMAGES, MAX_ROUNDS, TOOL


def image_part(frame):
    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": url}}


class ChatPolicy:
    def __init__(self, config, *, action_mode, output=None):
        self.action_mode = action_mode
        self.history = []
        self.rgb = HistoricalRGB()
        self.feedback = None
        self.output = output
        self.record_request = None
        self.config = config
        self.session = requests.Session()
        if config.get("api_key_env"):
            self.session.headers["Authorization"] = (
                "Bearer " + os.environ[config["api_key_env"]]
            )

    def reset_episode_memory(self):
        self.rgb.close()
        self.rgb = HistoricalRGB()
        self.history = []
        self.feedback = None
        self.record_request = None

    def close(self):
        self.rgb.close()
        self.session.close()

    def set_request_recorder(self, recorder):
        self.record_request = recorder

    def set_feedback(self, message):
        self.feedback = message

    def set_history(self, frames):
        self.history = list(frames)

    def request(self, observation, instruction, notes, phase):
        parts = [
            {"type": "text", "text": observation.task_text + "\n" + instruction},
            {
                "type": "text",
                "text": json.dumps(
                    {"notes": notes, "observation": observation.metadata()}
                ),
            },
        ]
        views = observation.rgb_images()
        for historical in self.history:
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": f"Historical head view {historical['index']} at {historical['simulation_time_s']} s",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64," + historical["png_base64"]
                        },
                    },
                ]
            )
        delivered = {name: image_part(frame) for name, frame in views.items() if frame is not None}
        for name, part in delivered.items():
            parts.extend([{"type": "text", "text": name}, part])
        messages = [{"role": "user", "content": parts}]
        if phase == "interaction":
            decision_step = self.rgb.remember(delivered, observation)
            parts.append({"type": "text", "text": f"Decision {decision_step}. Earlier delivered views may be recalled by decision number and camera name. Up to four images in four tool rounds."})
        remaining = MAX_IMAGES
        for round_index in range(MAX_ROUNDS + 1):
            enabled = phase == "interaction" and len(self.rgb.steps) > 1 and round_index < MAX_ROUNDS and remaining > 0
            message = self._complete(messages, phase, observation.action_step, enabled)
            calls = message.get("tool_calls") or []
            if not calls:
                content = message.get("content")
                try:
                    return json.loads(content)
                except (TypeError, json.JSONDecodeError) as error:
                    if phase == "terminal_qa":
                        return content
                    raise InvalidDecision("Return one JSON object containing an action.") from error
            if not enabled:
                raise InvalidDecision("Historical lookup is unavailable on this request. Choose an action.")
            messages.append({"role": "assistant", **{key: message[key] for key in (
                "content", "tool_calls", "reasoning_content", "reasoning_details") if key in message}})
            images = []
            for call in calls:
                receipt, image = self.rgb.recall(call["function"], remaining)
                if self.output is not None:
                    row = {"decision": decision_step, "name": call["function"].get("name"),
                           "arguments": call["function"].get("arguments"), "round": round_index + 1,
                           "status": "executed" if image is not None else "rejected"}
                    with (Path(self.output) / "tool_calls.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(receipt)})
                if image is not None:
                    remaining -= 1
                    images.extend([{"type":"text", "text":f"Historical decision {receipt['decision_step']}, camera {receipt['camera']}."}, image])
            if images:
                messages.append({"role": "user", "content": images})
        raise InvalidDecision("Return a final action after the last lookup.")

    def _complete(self, messages, phase, action_step, enabled):
        params = {
            key: self.config[key]
            for key in ("model", "temperature", "reasoning_effort", "max_tokens")
            if key in self.config
        }
        record = self.record_request
        if record is None and self.output is not None:
            record = request_recorder(self.output, phase, action_step)
        status, usage = None, None
        started = time.monotonic()
        try:
            response = self.session.post(
                self.config["base_url"].rstrip("/") + "/chat/completions",
                json={**params, "messages": messages, **({"tools": [TOOL]} if enabled else {})},
                timeout=self.config.get("timeout_s", 1800),
            )
            status = response.status_code
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage")
            gateway = (data.get("provider_metadata") or {}).get("gateway") or {}
            charge = gateway.get("cost", gateway.get("gatewayCost"))
            if charge is not None and (usage or {}).get("cost") is None:
                usage = {**(usage or {}), "cost": charge}
            return data["choices"][0]["message"]
        finally:
            if record is not None:
                record(time.monotonic() - started, usage, http_status=status)

    def act(self, observation, actions, notes):
        prompt = (
            "Choose one of these commands: "
            + ", ".join(actions)
            + ", DONE. Return one JSON object "
            "with action, distance_m for translation or angle_deg for rotation when required. "
            "Optional fields are note, arrived_rooms, observed_objects, answer and phase_1_complete. "
            "When instructed to finish a survey, pair survey_complete=true with DONE and answer=null."
        )
        if self.action_mode == "fixed-repeat":
            prompt = (
                "Choose an action from "
                + ", ".join(actions)
                + ", DONE. Return JSON with action and repeat (1, 2, 4 or 8). "
                "Optional fields: note, arrived_rooms, observed_objects, survey_complete and phase_1_complete."
            )
        if self.feedback:
            prompt += "\nPrevious decision was rejected: " + self.feedback
        return self.request(observation, prompt, notes, "interaction")

    def answer(self, observation, questions, notes):
        prompt = (
            'Answer the terminal questions with {"answers":{question_id:value}}.\n'
            + json.dumps(questions)
        )
        return self.request(observation, prompt, notes, "terminal_qa")
