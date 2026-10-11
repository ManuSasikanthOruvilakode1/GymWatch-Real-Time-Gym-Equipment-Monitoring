"""Classification and calibration metrics."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from gymwatch.labels import EQUIPMENT, JOINT_CLASSES, OCCUPANCY
from gymwatch.preprocess import softmax


@dataclass
class Predictions:
    occ_logits: np.ndarray
    eq_logits: np.ndarray
    occ_true: np.ndarray
    eq_true: np.ndarray


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = True
) -> Predictions:
    model.eval()
    occ, eq, yo, ye = [], [], [], []
    for x, o, e in loader:
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
        ):
            lo, le = model(x.to(device, non_blocking=True))
        occ.append(lo.float().cpu())
        eq.append(le.float().cpu())
        yo.append(o)
        ye.append(e)
    return Predictions(
        torch.cat(occ).numpy(), torch.cat(eq).numpy(), torch.cat(yo).numpy(), torch.cat(ye).numpy()
    )


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in itertools.pairwise(edges):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.5, 5.0)
) -> float:
    """Single-parameter temperature scaling minimising validation NLL.

    When the validation set is perfectly separated, NLL keeps falling as T -> 0, which would
    make the deployed model wildly overconfident. ``bounds`` keeps T in a sane range.
    """
    lg = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), *bounds))


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean())


def summarize(p: Predictions, temperature: float = 1.0) -> dict:
    occ_prob = softmax(p.occ_logits, temperature)
    eq_prob = softmax(p.eq_logits)
    occ_pred = occ_prob.argmax(1)
    eq_pred = eq_prob.argmax(1)
    prec, rec, f1, support = precision_recall_fscore_support(
        p.occ_true, occ_pred, labels=range(len(OCCUPANCY)), zero_division=0
    )
    joint_true = p.eq_true * len(OCCUPANCY) + p.occ_true
    joint_pred = eq_pred * len(OCCUPANCY) + occ_pred
    out = {
        "n": len(p.occ_true),
        "occupancy": {
            "accuracy": float((occ_pred == p.occ_true).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(p.occ_true, occ_pred)),
            "macro_f1": float(f1_score(p.occ_true, occ_pred, average="macro", zero_division=0)),
            "per_class": {
                c: {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, c in enumerate(OCCUPANCY)
            },
            "confusion_matrix": confusion_matrix(
                p.occ_true, occ_pred, labels=range(len(OCCUPANCY))
            ).tolist(),
            "ece": expected_calibration_error(occ_prob, p.occ_true),
            "nll": _nll(occ_prob, p.occ_true),
            "brier": float(((occ_prob[:, 1] - p.occ_true) ** 2).mean()),
        },
        "equipment": {
            "accuracy": float((eq_pred == p.eq_true).mean()),
            "confusion_matrix": confusion_matrix(
                p.eq_true, eq_pred, labels=range(len(EQUIPMENT))
            ).tolist(),
        },
        "joint": {
            "accuracy": float((joint_pred == joint_true).mean()),
            "macro_f1": float(f1_score(joint_true, joint_pred, average="macro", zero_division=0)),
            "labels": list(JOINT_CLASSES),
            "confusion_matrix": confusion_matrix(
                joint_true, joint_pred, labels=range(len(JOINT_CLASSES))
            ).tolist(),
        },
        "temperature": temperature,
    }
    if len(np.unique(p.occ_true)) == 2:
        out["occupancy"]["roc_auc"] = float(roc_auc_score(p.occ_true, occ_prob[:, 1]))
    return out
