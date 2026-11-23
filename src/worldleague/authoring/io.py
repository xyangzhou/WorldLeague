"""Benchmark JSON and identity helpers.

Used by worldleague scene --help.
"""
import json
import os
from pathlib import Path
import uuid

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default

def write_json(path, value):
    """Publish a complete document; writers never share temporary filenames."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
