from __future__ import annotations

import numpy as np
import pytest

from gymwatch.config import SmoothingConfig
from gymwatch.preprocess import center_crop_to_aspect, crop_roi, prepare
from gymwatch.realtime.store import EventStore
from gymwatch.realtime.tracker import FREE, IN_USE, UNKNOWN, OccupancyTracker

CFG = SmoothingConfig(ema_alpha=0.5, enter_threshold=0.65, exit_threshold=0.35, min_dwell_s=1.0)


def run(tracker, probs, fps=5.0):
    transitions = []
    for i, p in enumerate(probs):
        tr = tracker.update(p, i / fps)
        if tr:
            transitions.append(tr)
    return transitions


def test_tracker_commits_after_dwell_and_backdates():
    tr = OccupancyTracker(CFG)
    out = run(tr, [0.95] * 10)
    assert [t.state for t in out] == [IN_USE]
    assert out[0].previous == UNKNOWN
    assert out[0].at == 0.0  # back-dated to when the evidence started
    assert tr.state == IN_USE


def test_tracker_ignores_short_flicker():
    # 0.6 s of "free" evidence is shorter than the 1 s dwell: no transition.
    out = run(OccupancyTracker(CFG), [0.95] * 10 + [0.0] * 3 + [0.95] * 10)
    assert [t.state for t in out] == [IN_USE]


def test_tracker_hysteresis_band_holds_state():
    tr = OccupancyTracker(CFG)
    run(tr, [0.95] * 10)
    assert run(tr, [0.5] * 50) == []
    assert tr.state == IN_USE


def test_tracker_full_session():
    out = run(OccupancyTracker(CFG), [0.0] * 10 + [1.0] * 20 + [0.0] * 20)
    assert [t.state for t in out] == [FREE, IN_USE, FREE]
    assert out[2].at - out[1].at == pytest.approx(4.0, abs=0.5)


def test_store_sessions_and_stats():
    store = EventStore(":memory:")
    store.record_state("cam", "pec_fly", "in_use", 100.0)
    store.record_state("cam", "pec_fly", "free", 160.0)
    store.record_state("cam", "pec_fly", "in_use", 200.0)  # still open
    sessions = store.sessions("cam")
    assert len(sessions) == 2
    assert sessions[1]["duration_s"] == pytest.approx(60.0)
    assert sessions[0]["ended_at"] is None
    [row] = store.stats(window_s=200.0, now=300.0)  # window [100, 300]: busy 60 + 100
    assert row["sessions"] == 2
    assert row["utilisation"] == pytest.approx(160 / 200)


def test_store_window_clips_partial_sessions():
    store = EventStore(":memory:")
    store.record_state("cam", "leg_extension", "in_use", 0.0)
    store.record_state("cam", "leg_extension", "free", 100.0)
    [row] = store.stats(window_s=50.0, now=120.0)  # window [70, 120]: busy 70..100
    assert row["utilisation"] == pytest.approx(30 / 50)


def test_preprocess_shapes_and_crops():
    frame = np.zeros((1920, 1080, 3), np.uint8)
    assert center_crop_to_aspect(frame, 0.75).shape[:2] == (1440, 1080)
    assert center_crop_to_aspect(np.zeros((100, 400, 3), np.uint8), 0.75).shape[:2] == (100, 75)
    assert crop_roi(frame, (0.25, 0.5, 0.75, 1.0)).shape[:2] == (960, 540)
    x = prepare(frame, (320, 240))
    assert x.shape == (3, 320, 240)
    assert x.dtype == np.float32
