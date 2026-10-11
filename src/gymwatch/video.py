"""Offline video analysis and evaluation against annotated segments.

Uses the same predictor and tracker as the live monitor, but drives them with the video's
own timestamps so results are deterministic and independent of machine speed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from gymwatch.config import SmoothingConfig
from gymwatch.predictor import OnnxPredictor
from gymwatch.realtime.tracker import IN_USE, OccupancyTracker


@dataclass
class Timeline:
    t: np.ndarray
    p_raw: np.ndarray
    p_smooth: np.ndarray
    state: list[str]  # live state shown at each moment (lags by the dwell time)
    logged: list[str]  # state as recorded in the database (transitions are back-dated)

    def raw_state(self, threshold: float = 0.5) -> list[str]:
        return ["in_use" if p >= threshold else "free" for p in self.p_raw]


def analyze_video(
    path: str | Path,
    predictor: OnnxPredictor,
    smoothing: SmoothingConfig,
    sample_fps: float = 6.0,
    roi: tuple[float, float, float, float] | None = None,
    batch_size: int = 16,
) -> Timeline:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / sample_fps))
    times, frames, probs = [], [], []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                times.append(idx / fps)
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                if len(frames) == batch_size:
                    probs += [p.p_in_use for p in predictor.predict(frames, [roi] * len(frames))]
                    frames = []
        idx += 1
    cap.release()
    if frames:
        probs += [p.p_in_use for p in predictor.predict(frames, [roi] * len(frames))]

    tracker = OccupancyTracker(smoothing)
    smooth, states, transitions = [], [], []
    for t, p in zip(times, probs, strict=True):
        tr = tracker.update(p, t)
        if tr is not None:
            transitions.append(tr)
        smooth.append(tracker.smoothed)
        states.append(tracker.state)
    logged = []
    for t in times:
        state = "unknown"
        for tr in transitions:
            if tr.at <= t:
                state = tr.state
        logged.append(state)
    return Timeline(np.array(times), np.array(probs), np.array(smooth), states, logged)


def load_video_labels(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["videos"]


def label_at(segments: list[dict], t: float) -> str | None:
    for s in segments:
        if s["start"] <= t < s["end"]:
            return s["label"]
    return None


def _flips(states: list[str]) -> int:
    committed = [s for s in states if s in {"in_use", "free"}]
    return sum(a != b for a, b in itertools.pairwise(committed))


def score_timeline(tl: Timeline, segments: list[dict]) -> dict:
    """Frame accuracy of raw argmax vs. smoothed state on annotated (non-ignored) frames."""
    raw = tl.raw_state()
    keep = [i for i, t in enumerate(tl.t) if label_at(segments, t) in {"in_use", "free"}]
    truth = [label_at(segments, tl.t[i]) for i in keep]
    # Frames before the tracker's first commit are "unknown"; count them as errors to be strict.
    raw_acc = (
        float(np.mean([raw[i] == y for i, y in zip(keep, truth, strict=True)]))
        if keep
        else float("nan")
    )
    smooth_acc = (
        float(np.mean([tl.state[i] == y for i, y in zip(keep, truth, strict=True)]))
        if keep
        else float("nan")
    )
    first_commit = next((float(tl.t[i]) for i, s in enumerate(tl.state) if s != "unknown"), None)
    warm = [(i, y) for i, y in zip(keep, truth, strict=True) if tl.state[i] != "unknown"]
    after_warmup = float(np.mean([tl.state[i] == y for i, y in warm])) if warm else float("nan")
    logged_acc = (
        float(np.mean([tl.logged[i] == y for i, y in zip(keep, truth, strict=True)]))
        if keep
        else float("nan")
    )
    return {
        "logged_accuracy": logged_acc,
        "smoothed_accuracy_after_warmup": after_warmup,
        "frames_scored": len(keep),
        "frames_total": len(tl.t),
        "raw_accuracy": raw_acc,
        "smoothed_accuracy": smooth_acc,
        "raw_state_flips": _flips(raw),
        "smoothed_state_flips": _flips(tl.state),
        "time_to_first_state_s": first_commit,
        "fraction_in_use_smoothed": float(np.mean([s == IN_USE for s in tl.state])),
    }
