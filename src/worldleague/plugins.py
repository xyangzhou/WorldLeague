"""Public plugin adapters. Run worldleague eval --help or worldleague continuous --help."""

from dataclasses import dataclass
from importlib import import_module
from typing import Callable


def load_plugin(path):
    """Resolve an installed Python module:callable."""
    module, name = path.split(":", 1)
    return getattr(import_module(module), name)


@dataclass(frozen=True)
class HarnessTurn:
    """Public data at a decision or terminal QA boundary."""

    observation: object
    actions: tuple
    notes: tuple
    history: tuple
    action_mode: str
    max_decisions: int
    questions: tuple | None = None
    feedback: str | None = None
    record_request: Callable | None = None


class HarnessDriver:
    def __init__(self, play, *, action_mode, max_decisions):
        self.play = play
        self.action_mode = action_mode
        self.max_decisions = max_decisions
        self.episode = None
        self.history = ()
        self.feedback = None
        self.record_request = None

    def set_request_recorder(self, recorder):
        self.record_request = recorder

    def set_feedback(self, message):
        self.feedback = message

    def set_history(self, frames):
        self.history = tuple(frames)

    def act(self, observation, actions, notes):
        turn = HarnessTurn(observation, actions, notes, self.history, self.action_mode, self.max_decisions, feedback=self.feedback, record_request=self.record_request)
        try:
            if self.episode is None:
                self.episode = self.play(turn)
                return next(self.episode)
            return self.episode.send(turn)
        except StopIteration as error:
            raise ValueError("Yield a decision (use DONE to finish interaction).") from error

    def answer(self, observation, questions, notes):
        turn = HarnessTurn(observation, (), notes, self.history, self.action_mode, self.max_decisions, tuple(questions), record_request=self.record_request)
        try:
            self.episode.send(turn)
        except StopIteration as end:
            return end.value
        raise ValueError("Return the answers on the terminal QA turn.")

    def close(self):
        if self.episode is not None:
            self.episode.close()


class LocalActionPolicy:
    """Apply the preset's wire-format conversion before invoking a local predictor."""

    def __init__(self, config, robot_config):
        self.config = config
        self.robot_config = robot_config
        self.predict = None

    def start(self):
        self.predict = load_plugin(self.config["plugin"])()

    def infer(self, observation):
        from .policies.continuous_websocket import policy_observation

        request = policy_observation(observation, self.robot_config, self.config)
        return {self.config["action_key"]: self.predict(request)}

    def close(self):
        close = getattr(self.predict, "close", None)
        if close is not None:
            close()
