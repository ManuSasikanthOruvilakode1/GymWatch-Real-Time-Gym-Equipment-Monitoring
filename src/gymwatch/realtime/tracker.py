"""Temporal smoothing of per-frame occupancy probabilities.

Single-frame classifiers flicker (a person leaning forward, motion blur, someone walking
past). The original project printed the raw argmax of every frame. Here each camera has a
small state machine:

* exponential moving average of ``p(in_use)``,
* hysteresis (separate enter / exit thresholds) so the state does not toggle around 0.5,
* a minimum dwell time: a new state must hold for ``min_dwell_s`` before it is committed,
  and the session boundary is back-dated to when the change actually started.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gymwatch.config import SmoothingConfig

FREE, IN_USE, UNKNOWN = "free", "in_use", "unknown"


@dataclass(frozen=True)
class Transition:
    state: str  # the new committed state
    at: float  # back-dated time the change began
    previous: str


@dataclass
class OccupancyTracker:
    cfg: SmoothingConfig
    state: str = UNKNOWN
    smoothed: float | None = None
    since: float | None = None  # when the committed state began
    _candidate: str | None = field(default=None, repr=False)
    _candidate_since: float | None = field(default=None, repr=False)

    def update(self, p_in_use: float, t: float) -> Transition | None:
        a = self.cfg.ema_alpha
        self.smoothed = (
            p_in_use if self.smoothed is None else a * p_in_use + (1 - a) * self.smoothed
        )

        if self.smoothed >= self.cfg.enter_threshold:
            proposal = IN_USE
        elif self.smoothed <= self.cfg.exit_threshold:
            proposal = FREE
        else:
            proposal = self._candidate or self.state  # inside the hysteresis band: no new evidence

        if proposal == self.state:
            self._candidate = self._candidate_since = None
            return None
        if proposal != self._candidate:
            self._candidate, self._candidate_since = proposal, t
        if t - self._candidate_since >= self.cfg.min_dwell_s:
            return self._commit()
        return None

    def _commit(self) -> Transition:
        assert self._candidate is not None and self._candidate_since is not None
        tr = Transition(state=self._candidate, at=self._candidate_since, previous=self.state)
        self.state, self.since = self._candidate, self._candidate_since
        self._candidate = self._candidate_since = None
        return tr

    @property
    def pending(self) -> str | None:
        return self._candidate
