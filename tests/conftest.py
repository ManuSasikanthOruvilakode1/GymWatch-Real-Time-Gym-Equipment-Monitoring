from __future__ import annotations

from datetime import datetime, timedelta
from typing import ClassVar

import numpy as np
import pytest

from gymwatch.data.manifest import Record, assign_sessions
from gymwatch.predictor import FramePrediction


def make_records(seed: int = 0) -> list[Record]:
    """Synthetic burst-captured dataset: 2 machines x 3 sessions, ~10 frames/s, alternating use."""
    rng = np.random.default_rng(seed)
    records = []
    base = datetime(2024, 7, 6, 21, 0, 0)
    for e_i, equipment in enumerate(("leg_extension", "pec_fly")):
        for s in range(3):
            start = base + timedelta(hours=e_i, minutes=10 * s)
            for sec in range(60):
                occupancy = "in_use" if (sec // 15) % 2 == 0 else "free"
                for k in range(int(rng.integers(1, 4))):
                    t = start + timedelta(seconds=sec)
                    records.append(
                        Record(
                            path=f"{equipment}/{s}_{sec}_{k}.jpg",
                            cache_path="",
                            equipment=equipment,
                            occupancy=occupancy,
                            timestamp=t.isoformat(),
                            dhash=f"{int(rng.integers(0, 2**62)):016x}",
                        )
                    )
    assign_sessions(records, session_gap_s=30, block_s=2)
    return records


@pytest.fixture
def records() -> list[Record]:
    return make_records()


class StubPredictor:
    """Stands in for OnnxPredictor: p(in use) = mean brightness of the frame."""

    meta: ClassVar[dict] = {"backbone": "stub", "image_size": [32, 24]}

    def predict(self, frames, rois=None):
        return [FramePrediction(float(f.mean() / 255.0), "pec_fly", 0.99) for f in frames]
