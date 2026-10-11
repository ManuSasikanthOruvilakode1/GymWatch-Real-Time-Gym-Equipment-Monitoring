"""Multi-camera monitor: frame sources -> batched inference -> trackers -> store / overlays."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2

from gymwatch.config import CameraConfig, MonitorConfig
from gymwatch.predictor import OnnxPredictor
from gymwatch.realtime.render import annotate
from gymwatch.realtime.sources import FrameSource
from gymwatch.realtime.store import EventStore
from gymwatch.realtime.tracker import OccupancyTracker

log = logging.getLogger(__name__)


@dataclass
class CameraRuntime:
    cfg: CameraConfig
    source: FrameSource
    tracker: OccupancyTracker
    last_frame_time: float = 0.0
    p_raw: float = 0.0
    equipment_seen: str = ""
    equipment_confidence: float = 0.0
    jpeg: bytes | None = None
    history: deque = field(default_factory=lambda: deque(maxlen=600))

    def status(self, now: float) -> dict:
        t = self.tracker
        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "equipment": self.cfg.equipment,
            "state": t.state,
            "pending": t.pending,
            "state_since": t.since,
            "state_age_s": round(now - t.since, 1) if t.since else 0.0,
            "p_in_use": round(self.p_raw, 4),
            "p_in_use_smoothed": round(t.smoothed or 0.0, 4),
            "equipment_seen": self.equipment_seen,
            "equipment_confidence": round(self.equipment_confidence, 4),
            # The classifier also predicts which machine it sees; a mismatch flags a moved camera.
            "equipment_mismatch": bool(self.equipment_seen)
            and self.equipment_seen != self.cfg.equipment
            and self.equipment_confidence > 0.9,
            "source_error": self.source.error,
            "frame_age_s": round(now - self.last_frame_time, 2) if self.last_frame_time else None,
        }


class MonitorHub:
    def __init__(
        self,
        cfg: MonitorConfig,
        predictor: OnnxPredictor | None = None,
        store: EventStore | None = None,
    ):
        self.cfg = cfg
        self.predictor = predictor or OnnxPredictor(cfg.model_path)
        self.store = store or EventStore(cfg.database)
        self.cameras: dict[str, CameraRuntime] = {
            c.id: CameraRuntime(
                c,
                FrameSource(c.source, loop=c.loop),
                OccupancyTracker(c.smoothing or cfg.smoothing),
            )
            for c in cfg.cameras
        }
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="monitor-inference", daemon=True)
        self._lock = threading.Lock()
        self.inference_ms: deque[float] = deque(maxlen=100)

    def start(self) -> MonitorHub:
        for cam in self.cameras.values():
            cam.source.start()
        self._thread.start()
        log.info(
            "monitoring %d camera(s) at %.1f inferences/s",
            len(self.cameras),
            self.cfg.inference_fps,
        )
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        for cam in self.cameras.values():
            cam.source.stop()
        now = time.time()
        for cam in self.cameras.values():  # close open sessions cleanly
            if cam.tracker.state == "in_use":
                self.store.record_state(cam.cfg.id, cam.cfg.equipment, "free", now)
        self.store.close()

    # -- read side -------------------------------------------------------------------------
    def snapshot(self) -> list[dict]:
        now = time.time()
        with self._lock:
            return [cam.status(now) for cam in self.cameras.values()]

    def jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            cam = self.cameras.get(camera_id)
            return cam.jpeg if cam else None

    def history(self, camera_id: str) -> list[tuple[float, float, float]]:
        with self._lock:
            cam = self.cameras.get(camera_id)
            return list(cam.history) if cam else []

    def mean_inference_ms(self) -> float:
        return sum(self.inference_ms) / len(self.inference_ms) if self.inference_ms else 0.0

    # -- inference loop --------------------------------------------------------------------
    def step(self) -> None:
        """Run one batched inference over every camera that produced a new frame."""
        batch = []
        for cam in self.cameras.values():
            frame, ts = cam.source.latest()
            if frame is not None and ts > cam.last_frame_time:
                batch.append((cam, frame, ts))
        if not batch:
            return
        t0 = time.perf_counter()
        preds = self.predictor.predict([f for _, f, _ in batch], [c.cfg.roi for c, _, _ in batch])
        self.inference_ms.append((time.perf_counter() - t0) * 1000)
        now = time.time()
        for (cam, frame, ts), pred in zip(batch, preds, strict=True):
            transition = cam.tracker.update(pred.p_in_use, ts)
            if transition is not None and transition.state in {"in_use", "free"}:
                if not (transition.previous == "unknown" and transition.state == "free"):
                    self.store.record_state(
                        cam.cfg.id, cam.cfg.equipment, transition.state, transition.at
                    )
                log.info("%s -> %s", cam.cfg.id, transition.state)
            bgr = annotate(
                frame,
                name=cam.cfg.name,
                state=cam.tracker.state,
                p_raw=pred.p_in_use,
                p_smooth=cam.tracker.smoothed or 0.0,
                state_age_s=now - cam.tracker.since if cam.tracker.since else 0.0,
                roi=cam.cfg.roi,
                width=self.cfg.stream_width,
            )
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with self._lock:
                cam.last_frame_time = ts
                cam.p_raw = pred.p_in_use
                cam.equipment_seen = pred.equipment
                cam.equipment_confidence = pred.equipment_confidence
                cam.history.append((ts, pred.p_in_use, cam.tracker.smoothed or 0.0))
                if ok:
                    cam.jpeg = buf.tobytes()

    def _loop(self) -> None:
        period = 1.0 / self.cfg.inference_fps
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.step()
            except Exception:
                log.exception("inference step failed")
            self._stop.wait(max(0.0, period - (time.perf_counter() - t0)))
