"""Runtime predictor backed by ONNX Runtime (no PyTorch needed at inference time)."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gymwatch.preprocess import prepare, softmax

Roi = tuple[float, float, float, float] | None


@dataclass(frozen=True)
class FramePrediction:
    p_in_use: float
    equipment: str
    equipment_confidence: float

    def to_dict(self) -> dict:
        return {
            "p_in_use": round(self.p_in_use, 4),
            "equipment": self.equipment,
            "equipment_confidence": round(self.equipment_confidence, 4),
        }


class OnnxPredictor:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path} (run `gymwatch export` first)")
        available = ort.get_available_providers()
        if providers is None:
            providers = [
                p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
            ]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        meta = {
            k: json.loads(v) for k, v in self.session.get_modelmeta().custom_metadata_map.items()
        }
        self.meta = meta
        self.image_size: tuple[int, int] = tuple(meta["image_size"])
        self.temperature: float = float(meta.get("temperature", 1.0))
        self.occupancy_labels: list[str] = meta["occupancy_labels"]
        self.equipment_labels: list[str] = meta["equipment_labels"]
        self._in_use_idx = self.occupancy_labels.index("in_use")
        self._lock = threading.Lock()

    def predict(
        self, frames_rgb: list[np.ndarray], rois: list[Roi] | None = None
    ) -> list[FramePrediction]:
        if not frames_rgb:
            return []
        rois = rois or [None] * len(frames_rgb)
        batch = np.stack(
            [prepare(f, self.image_size, r) for f, r in zip(frames_rgb, rois, strict=True)]
        )
        with self._lock:  # ORT sessions are thread-safe, but serialising keeps GPU memory flat
            occ_logits, eq_logits = self.session.run(None, {"image": batch})
        occ = softmax(occ_logits, self.temperature)
        eq = softmax(eq_logits)
        return [
            FramePrediction(
                p_in_use=float(occ[i, self._in_use_idx]),
                equipment=self.equipment_labels[int(eq[i].argmax())],
                equipment_confidence=float(eq[i].max()),
            )
            for i in range(len(frames_rgb))
        ]
