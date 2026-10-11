"""Figures for the evaluation report: curves, confusion matrices, calibration, Grad-CAM, timelines."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch

from gymwatch.preprocess import MEAN, STD, softmax

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_history(history: list[dict], path: Path) -> Path:
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    a.plot(
        ep,
        [h["train_loss"] for h in history],
        color=BLUE,
        label="train loss (both heads, label-smoothed)",
    )
    a.plot(ep, [h["val_occ_nll"] for h in history], color=ORANGE, label="val occupancy NLL")
    a.set_xlabel("epoch")
    a.set_title("Loss", loc="left")
    a.legend()
    b.plot(ep, [h["val_occ_f1"] for h in history], color=BLUE, label="val occupancy macro-F1")
    b.plot(ep, [h["val_eq_acc"] for h in history], color=ORANGE, label="val equipment accuracy")
    b.set_xlabel("epoch")
    b.set_ylim(min(0.5, min(h["val_occ_f1"] for h in history)), 1.01)
    b.set_title("Validation", loc="left")
    b.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion(
    cm: Sequence[Sequence[int]], labels: Sequence[str], title: str, path: Path
) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    m = np.asarray(cm, dtype=float)
    norm = m / np.clip(m.sum(1, keepdims=True), 1, None)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.3 * n + 1.6, 1.1 * n + 1.0))
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{int(m[i, j])}\n{norm[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.6 else INK,
                fontsize=8,
            )
    ax.set_xticks(range(n), labels, rotation=30, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, loc="left")
    return _save(fig, path)


def plot_reliability(
    logits: np.ndarray, labels: np.ndarray, temperature: float, path: Path, bins: int = 10
) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    for t, colour, name in [
        (1.0, ORANGE, "uncalibrated"),
        (temperature, BLUE, f"temperature {temperature:.2f}"),
    ]:
        probs = softmax(logits, t)
        conf, correct = probs.max(1), probs.argmax(1) == labels
        edges = np.linspace(0.5, 1, bins // 2 + 1)
        xs, ys = [], []
        for lo, hi in itertools.pairwise(edges):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() >= 3:
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())
        ax.plot(xs, ys, marker="o", markersize=5, color=colour, label=name)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Occupancy calibration (test)", loc="left")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_split(records, path: Path) -> Path:
    """Capture timeline per session, coloured by split: shows the blocked, purged split."""
    colours = {"train": BLUE, "val": ORANGE, "test": AQUA, "purged": "#b9b8b2"}
    sessions = sorted({r.session for r in records})
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(sessions) + 1.2))
    for row, s in enumerate(sessions):
        recs = [r for r in records if r.session == s]
        t0 = min(r.time for r in recs)
        for split, c in colours.items():
            xs = [(r.time - t0).total_seconds() for r in recs if r.split == split]
            ys = [
                row + (0.15 if r.occupancy == "in_use" else -0.15) for r in recs if r.split == split
            ]
            ax.scatter(xs, ys, s=10, color=c, label=split if row == 0 else None, linewidths=0)
    ax.set_yticks(range(len(sessions)), sessions)
    ax.set_xlabel("seconds since session start (upper dots: in use, lower: free)")
    ax.set_title(
        "Leakage-free split: whole time blocks per split, boundary frames purged", loc="left"
    )
    ax.legend(ncols=4, loc="upper right", bbox_to_anchor=(1, 1.12 + 0.6 / len(sessions)))
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_timeline(tl, segments: list[dict], smoothing, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    end = float(tl.t.max()) if len(tl.t) else 1.0
    for s in segments:
        colour = {"in_use": BLUE, "free": "#ffffff", "ignore": "#d9d8d3"}[s["label"]]
        ax.axvspan(
            s["start"],
            min(s["end"], end),
            ymin=0.94,
            ymax=1.0,
            color=colour,
            alpha=0.9,
            linewidth=0,
        )
    ax.plot(tl.t, tl.p_raw, color=GRID, linewidth=1.2, label="per-frame p(in use)", zorder=1)
    ax.plot(tl.t, tl.p_smooth, color=BLUE, label="smoothed", zorder=2)
    state = np.array([1.0 if s == "in_use" else 0.0 if s == "free" else np.nan for s in tl.state])
    ax.step(tl.t, state * 0.9 + 0.02, where="post", color=ORANGE, label="committed state", zorder=3)
    for y in (smoothing.enter_threshold, smoothing.exit_threshold):
        ax.axhline(y, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(0, end)
    ax.set_xlabel("seconds  (top band: ground truth, blue = in use, white = free, grey = ignored)")
    ax.set_title(title, loc="left")
    ax.legend(ncols=3, loc="lower left", bbox_to_anchor=(0, 1.08))
    return _save(fig, path)


# -- Grad-CAM -------------------------------------------------------------------------------
def gradcam(model, x: torch.Tensor, target: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Grad-CAM over the last feature map for the occupancy head. Returns (cams, p_in_use)."""
    model.eval()
    fmap = model.features(x)
    fmap.retain_grad()
    occ, _ = model.heads(fmap)
    idx = occ.argmax(1) if target is None else torch.full((len(x),), target)
    occ.gather(1, idx.view(-1, 1)).sum().backward()
    weights = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * fmap).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )
    cam = cam.squeeze(1)
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1)
    p = torch.softmax(occ.detach(), 1)[:, 1]
    return cam.detach().numpy(), p.numpy()


def plot_gradcam_grid(
    model, images: torch.Tensor, titles: Sequence[str], path: Path, cols: int = 6
) -> Path:
    cams, probs = gradcam(model, images.clone().requires_grad_(False))
    rgb = (images.permute(0, 2, 3, 1).numpy() * STD + MEAN).clip(0, 1)
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.45))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for i, ax in enumerate(np.atleast_1d(axes).flat[: len(images)]):
        ax.imshow(rgb[i])
        ax.imshow(cams[i], cmap="inferno", alpha=0.45)
        ax.set_title(f"{titles[i]}\np(in use)={probs[i]:.2f}", fontsize=7)
    fig.suptitle(
        "Grad-CAM: where the occupancy head looks (held-out test images)", x=0.01, ha="left"
    )
    return _save(fig, path)
