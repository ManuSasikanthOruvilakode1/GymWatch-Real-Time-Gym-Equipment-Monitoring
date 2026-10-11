from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from gymwatch.api import create_app
from gymwatch.config import CameraConfig, Config, ModelConfig, MonitorConfig, SmoothingConfig
from gymwatch.export import export_onnx, parity_check
from gymwatch.model import GymNet, load_checkpoint, save_checkpoint
from gymwatch.predictor import OnnxPredictor
from gymwatch.realtime.hub import MonitorHub
from gymwatch.realtime.store import EventStore
from tests.conftest import StubPredictor

TINY = ModelConfig(backbone="mobilenetv4_conv_small", pretrained=False, image_size=(64, 48))


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = GymNet(TINY).eval()
    model.temperature.fill_(1.5)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    save_checkpoint(model, path, epoch=1)
    return path


def test_forward_shapes(checkpoint):
    model, ckpt = load_checkpoint(checkpoint)
    occ, eq = model(torch.randn(3, 3, 64, 48))
    assert occ.shape == (3, 2) and eq.shape == (3, 2)
    assert float(model.temperature) == pytest.approx(1.5)
    assert ckpt["epoch"] == 1


def test_onnx_export_parity_and_predictor(checkpoint, tmp_path):
    out = tmp_path / "m.onnx"
    meta = export_onnx(checkpoint, out)
    assert meta["image_size"] == [64, 48]
    batch = np.random.default_rng(0).normal(size=(4, 3, 64, 48)).astype(np.float32)
    assert parity_check(checkpoint, out, batch) < 1e-4

    pred = OnnxPredictor(out, providers=["CPUExecutionProvider"])
    assert pred.temperature == pytest.approx(1.5)
    frames = [np.full((1920, 1080, 3), v, np.uint8) for v in (0, 128, 255)]
    res = pred.predict(frames, [None, (0.1, 0.1, 0.9, 0.9), None])
    assert len(res) == 3
    assert all(
        0.0 <= r.p_in_use <= 1.0 and r.equipment in {"leg_extension", "pec_fly"} for r in res
    )


@pytest.fixture
def client():
    cams = [
        CameraConfig(id="a", name="Cam A", source="missing-a.mp4", equipment="pec_fly"),
        CameraConfig(id="b", name="Cam B", source="missing-b.mp4", equipment="leg_extension"),
    ]
    mcfg = MonitorConfig(cameras=cams, smoothing=SmoothingConfig(min_dwell_s=0.0, ema_alpha=1.0))
    hub = MonitorHub(mcfg, predictor=StubPredictor(), store=EventStore(":memory:"))
    with TestClient(create_app(Config(monitor=mcfg), hub=hub)) as c:
        c.hub = hub
        yield c


def _feed(hub, camera_id, value):
    hub.cameras[camera_id].source._publish(np.full((40, 30, 3), value, np.uint8))
    hub.step()


def test_api_state_flow(client):
    assert client.get("/api/health").json()["cameras"] == 2
    assert {c["state"] for c in client.get("/api/cameras").json()} == {"unknown"}

    _feed(client.hub, "a", 255)  # bright frame -> stub says in use
    _feed(client.hub, "b", 0)
    states = {c["id"]: c["state"] for c in client.get("/api/cameras").json()}
    assert states == {"a": "in_use", "b": "free"}

    _feed(client.hub, "a", 0)
    sessions = client.get("/api/sessions").json()
    assert (
        len(sessions) == 1
        and sessions[0]["camera_id"] == "a"
        and sessions[0]["ended_at"] is not None
    )
    stats = {s["camera_id"]: s for s in client.get("/api/stats").json()}
    assert stats["a"]["sessions"] == 1 and stats["b"]["sessions"] == 0

    r = client.get("/api/cameras/a/frame.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert len(client.get("/api/cameras/a/history").json()["t"]) == 2
    assert 'gymwatch_in_use{camera="a"' in client.get("/metrics").text
    assert client.get("/api/cameras/nope/frame.jpg").status_code == 404


def test_api_predict_upload(client):
    _ok, buf = cv2.imencode(".jpg", np.full((80, 60, 3), 255, np.uint8))
    r = client.post("/api/predict", files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["state"] == "in_use"
    bad = client.post("/api/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 400


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "GymWatch" in r.text
    assert client.get("/static/app.js").status_code == 200
