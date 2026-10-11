"""HTTP API and live dashboard."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from gymwatch import __version__
from gymwatch.config import Config
from gymwatch.realtime.hub import MonitorHub

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def create_app(cfg: Config, hub: MonitorHub | None = None) -> FastAPI:
    """Build the app. ``hub`` can be injected (tests); otherwise one is started on lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        owned = hub is None
        app.state.hub = hub or MonitorHub(cfg.monitor).start()
        try:
            yield
        finally:
            if owned:
                app.state.hub.stop()

    app = FastAPI(title="GymWatch", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def _hub() -> MonitorHub:
        return app.state.hub

    def _camera(camera_id: str):
        cam = _hub().cameras.get(camera_id)
        if cam is None:
            raise HTTPException(404, f"unknown camera {camera_id!r}")
        return cam

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        h = _hub()
        return {
            "status": "ok",
            "version": __version__,
            "model": h.predictor.meta,
            "cameras": len(h.cameras),
            "mean_inference_ms": round(h.mean_inference_ms(), 2),
        }

    @app.get("/api/cameras")
    def cameras() -> list[dict]:
        return _hub().snapshot()

    @app.get("/api/cameras/{camera_id}/frame.jpg")
    def frame(camera_id: str) -> Response:
        _camera(camera_id)
        jpeg = _hub().jpeg(camera_id)
        if jpeg is None:
            raise HTTPException(503, "no frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera_id}/stream.mjpg")
    async def stream(camera_id: str) -> StreamingResponse:
        _camera(camera_id)
        period = 1.0 / max(cfg.monitor.inference_fps, 1.0)

        async def frames():
            last = None
            while True:
                jpeg = _hub().jpeg(camera_id)
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(period / 2)

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/cameras/{camera_id}/history")
    def history(camera_id: str, seconds: float = Query(60.0, gt=0, le=600)) -> dict:
        _camera(camera_id)
        cutoff = time.time() - seconds
        pts = [p for p in _hub().history(camera_id) if p[0] >= cutoff]
        return {
            "t": [round(p[0], 2) for p in pts],
            "p_raw": [round(p[1], 4) for p in pts],
            "p_smooth": [round(p[2], 4) for p in pts],
        }

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        """Server-sent events: a camera snapshot every 500 ms."""

        async def gen():
            while True:
                yield f"data: {json.dumps(_hub().snapshot())}\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/sessions")
    def sessions(camera_id: str | None = None, limit: int = Query(50, ge=1, le=1000)) -> list[dict]:
        return _hub().store.sessions(camera_id, limit)

    @app.get("/api/stats")
    def stats(window_s: float = Query(3600.0, gt=0, le=30 * 86400)) -> list[dict]:
        rows = {r["camera_id"]: r for r in _hub().store.stats(window_s)}
        out = []
        for cam in _hub().cameras.values():
            out.append(
                rows.get(cam.cfg.id)
                or {
                    "camera_id": cam.cfg.id,
                    "equipment": cam.cfg.equipment,
                    "sessions": 0,
                    "utilisation": 0.0,
                    "mean_session_s": 0.0,
                    "window_s": window_s,
                }
            )
        return out

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image too large")
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "could not decode image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = _hub().predictor.predict([rgb])[0]
        return {**pred.to_dict(), "state": "in_use" if pred.p_in_use >= 0.5 else "free"}

    @app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
    def metrics() -> str:
        """Prometheus exposition format."""
        h = _hub()
        lines = [
            "# TYPE gymwatch_inference_ms gauge",
            f"gymwatch_inference_ms {h.mean_inference_ms():.3f}",
            "# TYPE gymwatch_in_use gauge",
            "# TYPE gymwatch_p_in_use gauge",
        ]
        for s in h.snapshot():
            lbl = f'{{camera="{s["id"]}",equipment="{s["equipment"]}"}}'
            lines.append(f"gymwatch_in_use{lbl} {1 if s['state'] == 'in_use' else 0}")
            lines.append(f"gymwatch_p_in_use{lbl} {s['p_in_use_smoothed']}")
        return "\n".join(lines) + "\n"

    return app
