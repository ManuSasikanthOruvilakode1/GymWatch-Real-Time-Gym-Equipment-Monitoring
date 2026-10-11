"""Frame sources: video files (played back in real time), webcams and network streams."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def open_capture(source: str) -> cv2.VideoCapture:
    target: int | str = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source {source!r}")
    return cap


class FrameSource:
    """Continuously reads a source on a background thread and exposes the latest RGB frame.

    Files are paced to their native frame rate (dropping frames rather than lagging), so a
    recorded clip behaves like a live camera. Network/webcam sources reconnect on failure.
    """

    def __init__(self, source: str, loop: bool = True):
        self.source = source
        self.loop = loop
        self.is_file = Path(source).exists()
        self._frame: np.ndarray | None = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"source:{source}", daemon=True)
        self.error: str | None = None

    def start(self) -> FrameSource:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._frame_time

    def _publish(self, bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._frame, self._frame_time = rgb, time.time()

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                cap = open_capture(self.source)
                self.error = None
                backoff = 1.0
                if self.is_file:
                    finished = self._play_file(cap)
                    if finished and not self.loop:
                        return
                else:
                    self._read_live(cap)
            except Exception as exc:
                self.error = str(exc)
                log.warning("source %s: %s (retrying in %.0fs)", self.source, exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def _play_file(self, cap: cv2.VideoCapture) -> bool:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start, index = time.perf_counter(), 0  # index = number of frames consumed so far
        try:
            while not self._stop.is_set():
                due = int((time.perf_counter() - start) * fps)
                while index < due:  # skip frames we are late for without decoding them
                    if not cap.grab():
                        return True
                    index += 1
                ok, frame = cap.read()
                if not ok:
                    return True
                index += 1
                self._publish(frame)
                self._stop.wait(max(0.0, (index / fps) - (time.perf_counter() - start)))
            return False
        finally:
            cap.release()

    def _read_live(self, cap: cv2.VideoCapture) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("stream read failed")
                self._publish(frame)
        finally:
            cap.release()
