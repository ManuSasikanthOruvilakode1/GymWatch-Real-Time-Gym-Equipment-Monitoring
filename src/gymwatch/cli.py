"""Command-line interface: ``gymwatch <command> --help``."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from gymwatch.config import Config

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="Real-time gym equipment monitoring."
)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config file.")]
OverrideOpt = Annotated[
    list[str] | None, typer.Option("--set", "-s", help="Override, e.g. train.epochs=5")
]
DEFAULT_CONFIG = Path("configs/default.yaml")


@app.callback()
def _setup(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    # Windows consoles/redirects default to cp1252; libraries (e.g. torch.onnx) print emoji.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def _load(config: Path, overrides: list[str] | None) -> Config:
    return Config.load(config if config.exists() else None, overrides)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


@app.command()
def prepare(
    config: ConfigOpt = DEFAULT_CONFIG, overrides: OverrideOpt = None, workers: int = 8
) -> None:
    """Scan photos, build the image cache, and write a leakage-free train/val/test manifest."""
    from gymwatch.data.manifest import assign_sessions, build_records, write_manifest
    from gymwatch.data.split import assign_splits, leakage_report
    from gymwatch.report import plot_split

    cfg = _load(config, overrides)
    d = cfg.data
    t0 = time.perf_counter()
    records = build_records(d.root, d.cache_dir, d.cache_short_side, workers)
    assign_sessions(records, d.session_gap_s, d.block_s)
    assign_splits(records, d.val_fraction, d.test_fraction, d.purge_s, d.seed)
    write_manifest(records, d.manifest)
    report = leakage_report(records)
    _write_json(d.manifest.with_name("split_report.json"), report)
    plot_split(records, d.manifest.with_name("split.png"))

    table = Table(title=f"{len(records)} images in {time.perf_counter() - t0:.0f}s")
    classes = sorted({c for s in report["counts"].values() for c in s})
    table.add_column("split")
    table.add_column("blocks", justify="right")
    for c in classes:
        table.add_column(c, justify="right")
    for split, counts in report["counts"].items():
        table.add_row(
            split, str(report["blocks"][split]), *[str(counts.get(c, 0)) for c in classes]
        )
    console.print(table)
    for name, g in report["heldout_to_nearest_train_frame"].items():
        console.print(
            f"{name:>15}: held-out frames with a training frame <=1 s away: {g['share_within_1s']:.0%}"
            f"  (same second: {g['share_within_0s']:.0%}, median gap {g['median_gap_s']} s)"
        )


def _records(cfg: Config):
    from gymwatch.data.manifest import read_manifest

    if not cfg.data.manifest.exists():
        raise typer.BadParameter(f"{cfg.data.manifest} not found - run `gymwatch prepare` first")
    return read_manifest(cfg.data.manifest)


@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG,
    overrides: OverrideOpt = None,
    name: Annotated[str, typer.Option(help="Run name under train.output_dir.")] = "main",
) -> None:
    """Train on the train split, select and calibrate on val."""
    from gymwatch.report import plot_history
    from gymwatch.train import train_model

    cfg = _load(config, overrides)
    records = _records(cfg)
    out = cfg.train.output_dir / name
    result = train_model(
        cfg,
        [r for r in records if r.split == "train"],
        [r for r in records if r.split == "val"],
        out,
    )
    plot_history(result["history"], out / "report" / "history.png")
    v = result["val"]["occupancy"]
    console.print(
        f"[bold]best epoch {result['best_epoch']}[/]  val occupancy F1 {v['macro_f1']:.4f}  "
        f"ECE {v['ece']:.4f}  T={result['temperature']:.2f}  -> {out / 'best.pt'}"
    )


@app.command()
def evaluate(
    run: Annotated[Path, typer.Argument(help="Run directory containing best.pt")] = Path(
        "runs/main"
    ),
    config: ConfigOpt = DEFAULT_CONFIG,
    overrides: OverrideOpt = None,
) -> None:
    """Score the held-out test split and write figures (confusion, calibration, Grad-CAM)."""
    import torch

    from gymwatch.data.dataset import GymDataset
    from gymwatch.labels import EQUIPMENT, JOINT_CLASSES, OCCUPANCY
    from gymwatch.metrics import predict, summarize
    from gymwatch.model import load_checkpoint
    from gymwatch.report import plot_confusion, plot_gradcam_grid, plot_reliability
    from gymwatch.train import get_device, make_loader

    cfg = _load(config, overrides)
    test = [r for r in _records(cfg) if r.split == "test"]
    model, _ = load_checkpoint(run / "best.pt")
    device = get_device()
    model.to(device)
    preds = predict(model, make_loader(test, cfg, train=False), device)
    t = float(model.temperature)
    metrics = summarize(preds, t)
    metrics["uncalibrated_ece"] = summarize(preds, 1.0)["occupancy"]["ece"]
    _write_json(run / "test_metrics.json", metrics)

    rep = run / "report"
    plot_confusion(
        metrics["occupancy"]["confusion_matrix"],
        OCCUPANCY,
        "Occupancy (test)",
        rep / "cm_occupancy.png",
    )
    plot_confusion(
        metrics["joint"]["confusion_matrix"],
        JOINT_CLASSES,
        "Equipment x occupancy (test)",
        rep / "cm_joint.png",
    )
    plot_reliability(preds.occ_logits, preds.occ_true, t, rep / "calibration.png")

    # Grad-CAM on a stratified sample: misclassified first, then correct ones per class.
    model.cpu().float()
    occ_pred = preds.occ_logits.argmax(1)
    wrong = [i for i in range(len(test)) if occ_pred[i] != preds.occ_true[i]]
    picks = wrong[:4]
    for e in EQUIPMENT:
        for o in OCCUPANCY:
            picks += [
                i
                for i in range(len(test))
                if test[i].equipment == e and test[i].occupancy == o and i not in picks
            ][::7][:2]
    ds = GymDataset([test[i] for i in picks], cfg.model.image_size, train=False)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    titles = [
        ("WRONG " if i in wrong else "") + f"{test[i].equipment}/{test[i].occupancy}" for i in picks
    ]
    plot_gradcam_grid(model, imgs, titles, rep / "gradcam.png")

    o = metrics["occupancy"]
    console.print(
        f"[bold]test n={metrics['n']}[/]  occupancy acc {o['accuracy']:.4f}  macro-F1 {o['macro_f1']:.4f}  "
        f"AUC {o.get('roc_auc', float('nan')):.4f}  ECE {o['ece']:.4f} (uncal. {metrics['uncalibrated_ece']:.4f})  "
        f"equipment acc {metrics['equipment']['accuracy']:.4f}  joint acc {metrics['joint']['accuracy']:.4f}"
    )


@app.command()
def cv(
    config: ConfigOpt = DEFAULT_CONFIG,
    overrides: OverrideOpt = None,
    k: int = 5,
    by: Annotated[
        str, typer.Option(help="'block': K folds of time blocks; 'session': leave-one-session-out")
    ] = "block",
    name: str = "cv",
) -> None:
    """Grouped, purged K-fold cross-validation (mean +/- std)."""
    import numpy as np

    from gymwatch.data.split import cv_folds, session_folds
    from gymwatch.train import train_model

    cfg = _load(config, overrides)
    records = _records(cfg)
    if by not in {"block", "session"}:
        raise typer.BadParameter("--by must be 'block' or 'session'")
    folds = (
        session_folds(records)
        if by == "session"
        else cv_folds(records, k, cfg.data.purge_s, cfg.data.seed)
    )
    k = len(folds)
    rows = []
    for i, (tr, va) in enumerate(folds):
        held_sessions = sorted({records[j].session for j in va}) if by == "session" else []
        console.rule(f"fold {i + 1}/{k}  train={len(tr)} val={len(va)} {' '.join(held_sessions)}")
        res = train_model(
            cfg,
            [records[j] for j in tr],
            [records[j] for j in va],
            cfg.train.output_dir / name / f"fold{i}",
        )
        v = res["val"]
        rows.append(
            {
                "fold": i,
                "held_out_sessions": held_sessions,
                "n_val": len(va),
                "occ_accuracy": v["occupancy"]["accuracy"],
                "occ_macro_f1": v["occupancy"]["macro_f1"],
                "occ_auc": v["occupancy"].get("roc_auc", float("nan")),
                "equipment_accuracy": v["equipment"]["accuracy"],
                "joint_accuracy": v["joint"]["accuracy"],
            }
        )
    keys = [k_ for k_ in rows[0] if isinstance(rows[0][k_], float)]
    # nan-aware: a held-out session can contain a single class, where AUC is undefined.
    summary = {
        k_: {
            "mean": float(np.nanmean([r[k_] for r in rows])),
            "std": float(np.nanstd([r[k_] for r in rows])),
        }
        for k_ in keys
    }
    _write_json(
        cfg.train.output_dir / name / "cv_summary.json",
        {"by": by, "folds": rows, "summary": summary, "config": cfg.model.model_dump(mode="json")},
    )
    table = Table(
        title=f"{k}-fold CV by {by} ({cfg.model.backbone}, pretrained={cfg.model.pretrained})"
    )
    table.add_column("metric")
    table.add_column("mean +/- std", justify="right")
    for k_, s in summary.items():
        table.add_row(k_, f"{s['mean']:.4f} +/- {s['std']:.4f}")
    console.print(table)


@app.command()
def export(
    run: Annotated[Path, typer.Argument()] = Path("runs/main"),
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("models/gymwatch.onnx"),
    config: ConfigOpt = DEFAULT_CONFIG,
) -> None:
    """Export best.pt to ONNX (with metadata) and verify numerical parity."""
    import numpy as np
    import torch

    from gymwatch.data.dataset import GymDataset
    from gymwatch.export import export_onnx, parity_check

    cfg = _load(config, None)
    meta = export_onnx(run / "best.pt", output)
    val = [r for r in _records(cfg) if r.split in {"val", "test"}][:16]
    batch = torch.stack(
        [GymDataset(val, tuple(meta["image_size"]), train=False)[i][0] for i in range(len(val))]
    ).numpy()
    diff = parity_check(run / "best.pt", output, batch.astype(np.float32))
    size_mb = output.stat().st_size / 1e6
    console.print(f"exported {output} ({size_mb:.1f} MB)  max |torch - onnx| = {diff:.2e}")
    if diff > 1e-3:
        raise typer.Exit(code=1)


@app.command()
def predict(
    images: Annotated[list[Path], typer.Argument(help="Image files")],
    model: Annotated[Path, typer.Option()] = Path("models/gymwatch.onnx"),
) -> None:
    """Classify individual images with the exported model."""
    import numpy as np

    from gymwatch.data.manifest import load_oriented
    from gymwatch.predictor import OnnxPredictor

    pred = OnnxPredictor(model)
    table = Table()
    for col in ("image", "p(in use)", "state", "equipment"):
        table.add_column(col)
    for path in images:
        p = pred.predict([np.asarray(load_oriented(path, 640))])[0]
        state = "IN USE" if p.p_in_use >= 0.5 else "free"
        table.add_row(
            path.name, f"{p.p_in_use:.3f}", state, f"{p.equipment} ({p.equipment_confidence:.2f})"
        )
    console.print(table)


@app.command("video-eval")
def video_eval(
    labels: Annotated[Path, typer.Option()] = Path("configs/video_labels.yaml"),
    model: Annotated[Path, typer.Option()] = Path("models/gymwatch.onnx"),
    config: ConfigOpt = DEFAULT_CONFIG,
    out: Annotated[Path, typer.Option()] = Path("artifacts/video_eval"),
    sample_fps: float = 6.0,
) -> None:
    """Run the deployed predictor + tracker over annotated videos; compare raw vs. smoothed."""
    from gymwatch.predictor import OnnxPredictor
    from gymwatch.report import plot_timeline
    from gymwatch.video import analyze_video, load_video_labels, score_timeline

    cfg = _load(config, None)
    pred = OnnxPredictor(model)
    results = {}
    table = Table(title="Video evaluation (annotated, non-ignored frames)")
    for col in ("video", "frames", "raw acc", "live acc", "logged acc", "raw flips", "live flips"):
        table.add_column(col)
    for v in load_video_labels(labels):
        tl = analyze_video(v["path"], pred, cfg.monitor.smoothing, sample_fps)
        s = score_timeline(tl, v["segments"])
        stem = Path(v["path"]).stem
        results[stem] = {"equipment": v["equipment"], **s}
        plot_timeline(
            tl,
            v["segments"],
            cfg.monitor.smoothing,
            f"{v['equipment']} - {stem}",
            out / f"{stem}.png",
        )
        table.add_row(
            stem,
            str(s["frames_scored"]),
            f"{s['raw_accuracy']:.3f}",
            f"{s['smoothed_accuracy']:.3f}",
            f"{s['logged_accuracy']:.3f}",
            str(s["raw_state_flips"]),
            str(s["smoothed_state_flips"]),
        )
    _write_json(out / "video_metrics.json", results)
    console.print(table)


@app.command()
def bench(
    model: Annotated[Path, typer.Option()] = Path("models/gymwatch.onnx"),
    batch: int = 1,
    iters: int = 200,
) -> None:
    """Measure end-to-end latency (preprocess + inference) on a 1080x1920 frame."""
    import numpy as np

    from gymwatch.predictor import OnnxPredictor

    pred = OnnxPredictor(model)
    frames = [np.random.default_rng(0).integers(0, 255, (1920, 1080, 3), dtype=np.uint8)] * batch
    for _ in range(10):
        pred.predict(frames)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pred.predict(frames)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    console.print(
        f"providers={pred.session.get_providers()}  batch={batch}  "
        f"p50={times[len(times) // 2]:.2f} ms  p95={times[int(len(times) * 0.95)]:.2f} ms  "
        f"throughput={1000 * batch / (sum(times) / len(times)):.0f} frames/s"
    )


@app.command()
def monitor(
    config: ConfigOpt = DEFAULT_CONFIG, overrides: OverrideOpt = None, seconds: float = 0
) -> None:
    """Run the multi-camera monitor headless, printing live state (Ctrl+C to stop)."""
    from rich.live import Live

    from gymwatch.realtime.hub import MonitorHub

    cfg = _load(config, overrides)
    hub = MonitorHub(cfg.monitor).start()
    t_end = time.time() + seconds if seconds else float("inf")

    def render() -> Table:
        t = Table(title=f"GymWatch  ({hub.mean_inference_ms():.1f} ms / batch)")
        for col in ("camera", "state", "for", "p raw", "p smooth"):
            t.add_column(col)
        for s in hub.snapshot():
            colour = {"in_use": "red", "free": "green"}.get(s["state"], "grey50")
            t.add_row(
                s["name"],
                f"[{colour}]{s['state']}[/]",
                f"{s['state_age_s']:.0f}s",
                f"{s['p_in_use']:.2f}",
                f"{s['p_in_use_smoothed']:.2f}",
            )
        return t

    try:
        with Live(render(), console=console, refresh_per_second=4) as live:
            while time.time() < t_end:
                time.sleep(0.25)
                live.update(render())
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop()


@app.command()
def serve(
    config: ConfigOpt = DEFAULT_CONFIG,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the API + live dashboard (http://HOST:PORT)."""
    import uvicorn

    from gymwatch.api import create_app

    uvicorn.run(create_app(_load(config, None)), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
