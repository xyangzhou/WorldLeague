"""Pace native physics during inference. Used by worldleague eval and continuous."""

import time
from concurrent.futures import wait


class InferenceClock:
    """Carry the fractional control interval across requests within an episode."""

    def __init__(self, dt):
        self.dt = dt
        self.remaining = dt

    def request(self, worker, request, tick):
        def timed_request():
            reply = request()
            return reply, time.monotonic()

        started = time.monotonic()
        deadline = started + self.remaining
        future = worker.submit(timed_request)
        steps = 0
        ended = False
        while not wait((future,), timeout=max(0.0, deadline - time.monotonic())).done:
            deadline = time.monotonic() + self.dt
            ended = bool(tick())
            steps += 1
            if ended:
                break
        reply, finished = future.result()
        self.remaining = max(0.0, min(self.dt, deadline - finished))
        return reply, steps, ended, finished - started
