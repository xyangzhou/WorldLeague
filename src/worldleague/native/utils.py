"""Small simulator-independent native helpers.

Used by worldleague scene --help.
"""
import threading
from typing import Any
import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _main_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('R1 control and sensor updates must run on the main thread.')
