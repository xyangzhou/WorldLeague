"""Record policy request measurements for worldleague analyze --system NAME=DIRECTORY."""

import json
from pathlib import Path


def request_recorder(output, phase, action_step):
    path = Path(output) / "requests.jsonl"

    def record(inference_s, usage=None, *, http_status=None):
        row = {
            "phase": phase,
            "action_step": action_step,
            "inference_s": inference_s,
            "usage": usage,
            "http_status": http_status,
        }
        with path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    return record
