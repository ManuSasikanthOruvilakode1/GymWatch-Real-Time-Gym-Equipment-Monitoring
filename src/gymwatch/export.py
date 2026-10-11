"""Export a trained checkpoint to a self-describing ONNX file."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import torch

from gymwatch import __version__
from gymwatch.labels import EQUIPMENT, OCCUPANCY
from gymwatch.model import load_checkpoint


def export_onnx(checkpoint: Path, output: Path, opset: int = 18) -> dict:
    model, ckpt = load_checkpoint(checkpoint)
    h, w = model.cfg.image_size
    dummy = torch.randn(1, 3, h, w)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["image"],
        output_names=["occupancy_logits", "equipment_logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=64)},),
        opset_version=opset,
        dynamo=True,
        external_data=False,
    )
    meta = {
        "gymwatch_version": __version__,
        "backbone": model.cfg.backbone,
        "pretrained_init": model.cfg.pretrained,
        "image_size": [h, w],
        "occupancy_labels": list(OCCUPANCY),
        "equipment_labels": list(EQUIPMENT),
        "temperature": float(model.temperature),
        "epoch": ckpt.get("epoch"),
    }
    proto = onnx.load(str(output))
    onnx.checker.check_model(proto)
    del proto.metadata_props[:]
    for k, v in meta.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = k, json.dumps(v)
    onnx.save(proto, str(output))
    return meta


def parity_check(checkpoint: Path, onnx_path: Path, batch: np.ndarray) -> float:
    """Max absolute logit difference between PyTorch and ONNX Runtime on ``batch``."""
    import onnxruntime as ort

    model, _ = load_checkpoint(checkpoint)
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(batch))]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": batch})
    return float(max(np.abs(a - b).max() for a, b in zip(ref, got, strict=True)))
