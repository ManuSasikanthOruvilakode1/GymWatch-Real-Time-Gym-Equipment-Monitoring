"""Typed configuration loaded from YAML.

Every pipeline stage reads one ``Config`` object so that a run is fully described by
a single file that is also copied next to its artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    root: Path = Path("data")
    cache_dir: Path = Path(".cache/images")
    manifest: Path = Path("artifacts/manifest.csv")
    cache_short_side: int = 384
    # Burst photos are ~10 fps, so neighbours are near-duplicates. Images are grouped into
    # contiguous time blocks and whole blocks are assigned to a split.
    session_gap_s: float = 30.0
    block_s: float = 2.0
    # Drop training images this close (in time) to any val/test image of the same session.
    purge_s: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


class ModelConfig(BaseModel):
    backbone: str = "mobilenetv4_conv_medium.e500_r224_in1k"
    pretrained: bool = True
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1
    # (height, width): the photos are 3:4 portrait.
    image_size: tuple[int, int] = (320, 240)


class TrainConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    label_smoothing: float = 0.1
    occupancy_loss_weight: float = 1.0
    equipment_loss_weight: float = 0.5
    ema_decay: float = 0.998
    patience: int = 8
    amp: bool = True
    num_workers: int = 4
    seed: int = 42
    output_dir: Path = Path("runs")


class SmoothingConfig(BaseModel):
    """Temporal filter turning noisy per-frame probabilities into a stable state."""

    ema_alpha: float = 0.35
    enter_threshold: float = 0.65
    exit_threshold: float = 0.35
    min_dwell_s: float = 1.5


class CameraConfig(BaseModel):
    id: str
    name: str
    source: str  # file path, RTSP/HTTP URL, or webcam index as a string ("0")
    equipment: Literal["leg_extension", "pec_fly"]
    # Optional region of interest as fractions of the frame: [x0, y0, x1, y1].
    roi: tuple[float, float, float, float] | None = None
    loop: bool = True
    # Per-camera thresholds/dwell. Useful for a machine with little training data, where the
    # model ranks frames well (high AUC) but its default 0.5 operating point is off.
    smoothing: SmoothingConfig | None = None

    @model_validator(mode="after")
    def _check_roi(self) -> CameraConfig:
        if self.roi is not None:
            x0, y0, x1, y1 = self.roi
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"camera {self.id}: roi must satisfy 0<=x0<x1<=1, 0<=y0<y1<=1")
        return self


class MonitorConfig(BaseModel):
    model_path: Path = Path("models/gymwatch.onnx")
    inference_fps: float = 6.0
    database: Path = Path("artifacts/events.sqlite")
    stream_width: int = 360
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> MonitorConfig:
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("camera ids must be unique")
        return self


class Config(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
        raw: dict = {}
        if path is not None:
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        for item in overrides or []:
            _apply_override(raw, item)
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)


def _apply_override(raw: dict, item: str) -> None:
    """Apply a ``section.key=value`` override, parsing the value as YAML."""
    if "=" not in item:
        raise ValueError(f"override must look like section.key=value, got {item!r}")
    dotted, value = item.split("=", 1)
    node = raw
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = yaml.safe_load(value)
