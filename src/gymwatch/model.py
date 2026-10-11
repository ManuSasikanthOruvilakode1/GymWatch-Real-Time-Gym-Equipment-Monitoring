"""Multi-task classifier: shared backbone with an occupancy head and an equipment head."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from gymwatch.config import ModelConfig
from gymwatch.labels import EQUIPMENT, OCCUPANCY


class GymNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            drop_path_rate=cfg.drop_path_rate,
        )
        # Width of the pooled pre-logits vector (differs from num_features for e.g. MobileNetV4).
        dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.dropout = nn.Dropout(cfg.drop_rate)
        self.occupancy_head = nn.Linear(dim, len(OCCUPANCY))
        self.equipment_head = nn.Linear(dim, len(EQUIPMENT))
        # Post-hoc calibration (temperature scaling) for the occupancy head; fitted on val.
        self.register_buffer("temperature", torch.ones(()))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Last spatial feature map (B, C, H, W); the Grad-CAM target."""
        return self.backbone.forward_features(x)

    def heads(self, fmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.dropout(self.backbone.forward_head(fmap, pre_logits=True))
        return self.occupancy_head(z), self.equipment_head(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(self.features(x))

    def param_groups(self, lr: float, backbone_lr_mult: float, weight_decay: float) -> list[dict]:
        """Lower LR for the (pretrained) backbone; no weight decay on norms and biases."""
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = name.startswith("backbone.")
            no_decay = p.ndim <= 1
            groups.setdefault((is_backbone, no_decay), []).append(p)
        mult = backbone_lr_mult if self.cfg.pretrained else 1.0
        return [
            {
                "params": params,
                "lr": lr * (mult if is_backbone else 1.0),
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
            for (is_backbone, no_decay), params in groups.items()
        ]


def save_checkpoint(model: GymNet, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_cfg": model.cfg.model_dump(mode="json"), "state_dict": model.state_dict(), **extra},
        path,
    )


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> tuple[GymNet, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig.model_validate({**ckpt["model_cfg"], "pretrained": False})
    model = GymNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    # Keep the original flag for reporting (weights are already loaded).
    model.cfg = ModelConfig.model_validate(ckpt["model_cfg"])
    model.eval()
    return model, ckpt
