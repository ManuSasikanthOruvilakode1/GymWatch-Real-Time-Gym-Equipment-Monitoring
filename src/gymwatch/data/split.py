"""Leakage-free splitting for burst-captured photos.

A random image-level split (what the original notebook did via ``validation_split``) puts
frames taken 0.1 s apart into both train and validation, which inflates accuracy. Here:

1. whole time *blocks* are assigned to one split (``StratifiedGroupKFold``), and
2. training images within ``purge_s`` of any held-out image of the same session are dropped
   ("purged"), so no held-out frame has a near-twin in training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from gymwatch.data.manifest import Record


def _joint(r: Record) -> str:
    return f"{r.equipment}/{r.occupancy}"


def _first_fold(records: Sequence[Record], n_splits: int, seed: int) -> set[int]:
    y = [_joint(r) for r in records]
    groups = [r.block for r in records]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(np.zeros(len(records)), y, groups))
    return set(held.tolist())


def purge(
    records: Sequence[Record], train_idx: Sequence[int], held_idx: Sequence[int], purge_s: float
) -> list[int]:
    """Return ``train_idx`` minus images within ``purge_s`` of a held-out image in the same session."""
    held_times: dict[str, list[float]] = defaultdict(list)
    for i in held_idx:
        held_times[records[i].session].append(records[i].time.timestamp())
    sorted_times = {k: np.sort(np.array(v)) for k, v in held_times.items()}
    keep = []
    for i in train_idx:
        times = sorted_times.get(records[i].session)
        if times is not None:
            t = records[i].time.timestamp()
            pos = np.searchsorted(times, t)
            near = [times[j] for j in (pos - 1, pos) if 0 <= j < len(times)]
            if any(abs(t - u) <= purge_s for u in near):
                continue
        keep.append(i)
    return keep


def assign_splits(
    records: list[Record], val_fraction: float, test_fraction: float, purge_s: float, seed: int
) -> None:
    """Set ``record.split`` to train / val / test / purged in place."""
    n = len(records)
    test = _first_fold(records, max(2, round(1 / test_fraction)), seed)
    rest = [i for i in range(n) if i not in test]
    rest_records = [records[i] for i in rest]
    val_local = _first_fold(
        rest_records, max(2, round((1 - test_fraction) / val_fraction)), seed + 1
    )
    val = {rest[i] for i in val_local}
    train = [i for i in rest if i not in val]
    kept = set(purge(records, train, sorted(val | test), purge_s))
    for i, r in enumerate(records):
        if i in test:
            r.split = "test"
        elif i in val:
            r.split = "val"
        elif i in kept:
            r.split = "train"
        else:
            r.split = "purged"


def cv_folds(
    records: Sequence[Record], k: int, purge_s: float, seed: int
) -> list[tuple[list[int], list[int]]]:
    """Grouped (by time block), stratified, purged K-fold over the non-test records."""
    pool = [i for i, r in enumerate(records) if r.split != "test"]
    y = [_joint(records[i]) for i in pool]
    groups = [records[i].block for i in pool]
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for tr, va in sgkf.split(np.zeros(len(pool)), y, groups):
        train_idx = [pool[i] for i in tr]
        val_idx = [pool[i] for i in va]
        folds.append((purge(records, train_idx, val_idx, purge_s), val_idx))
    return folds


def session_folds(
    records: Sequence[Record], min_size: int = 50
) -> list[tuple[list[int], list[int]]]:
    """Leave-one-session-out over every session with at least ``min_size`` images.

    Each held-out session is a separate capture (different day or camera placement), so this
    measures generalisation to a new deployment rather than to nearby moments of the same one.
    """
    by_session: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_session[r.session].append(i)
    folds = []
    for _session, held in sorted(by_session.items()):
        if len(held) >= min_size:
            held_set = set(held)
            folds.append(([i for i in range(len(records)) if i not in held_set], held))
    return folds


def nearest_train_gap(
    records: Sequence[Record], train_idx: Sequence[int], held_idx: Sequence[int]
) -> np.ndarray:
    """For each held-out image, seconds to the closest training image of the same session."""
    times: dict[str, list[float]] = defaultdict(list)
    for i in train_idx:
        times[records[i].session].append(records[i].time.timestamp())
    sorted_times = {k: np.sort(np.array(v)) for k, v in times.items()}
    gaps = []
    for i in held_idx:
        ts = sorted_times.get(records[i].session)
        if ts is None or len(ts) == 0:
            gaps.append(np.inf)
            continue
        t = records[i].time.timestamp()
        pos = np.searchsorted(ts, t)
        gaps.append(min(abs(t - ts[j]) for j in (pos - 1, pos) if 0 <= j < len(ts)))
    return np.array(gaps)


def leakage_report(records: Sequence[Record], seed: int = 0) -> dict:
    """Split sizes plus the temporal gap between held-out images and their nearest training frame.

    The same statistic is computed for a naive random image-level split (what the original
    notebook did) to quantify how much leakage the grouped + purged split removes.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        counts[r.split][_joint(r)] += 1
    train = [i for i, r in enumerate(records) if r.split == "train"]
    held = [i for i, r in enumerate(records) if r.split in {"val", "test"}]
    ours = nearest_train_gap(records, train, held)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(records))
    n_held = len(held)
    naive = nearest_train_gap(records, perm[n_held:].tolist(), perm[:n_held].tolist())

    def describe(g: np.ndarray) -> dict:
        finite = g[np.isfinite(g)]
        return {
            "median_gap_s": float(np.median(finite)) if len(finite) else None,
            "share_within_0s": float(np.mean(g <= 0.0)),
            "share_within_1s": float(np.mean(g <= 1.0)),
            "share_session_unseen": float(np.mean(~np.isfinite(g))),
        }

    return {
        "counts": {s: dict(c) for s, c in sorted(counts.items())},
        "blocks": {s: len({r.block for r in records if r.split == s}) for s in counts},
        "heldout_to_nearest_train_frame": {
            "grouped_purged": describe(ours),
            "random_split": describe(naive),
        },
    }
