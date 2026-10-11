"""Scan the raw photo folders into a manifest and build a downscaled image cache.

Raw photos are 12-24 MP phone JPEGs; decoding them every epoch is the main cost of the
old pipeline. ``prepare`` decodes each file once (using JPEG draft mode), applies the EXIF
orientation, and writes a small cached copy plus metadata used for leakage-free splitting.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from gymwatch.labels import parse_folder

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_FILENAME_TS = re.compile(r"(\d{4}-\d{2}-\d{2}) (\d{2})\.(\d{2})\.(\d{2})")


@dataclass
class Record:
    path: str
    cache_path: str
    equipment: str
    occupancy: str
    timestamp: str
    session: str = ""
    block: str = ""
    width: int = 0
    height: int = 0
    dhash: str = ""
    split: str = ""

    @property
    def time(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)


def capture_time(path: Path) -> datetime:
    """Capture time from the phone's filename, then EXIF, then file mtime."""
    m = _FILENAME_TS.search(path.name)
    if m:
        date, hh, mm, ss = m.groups()
        return datetime.fromisoformat(f"{date}T{hh}:{mm}:{ss}")
    try:
        exif = Image.open(path).getexif()
        raw = exif.get_ifd(0x8769).get(0x9003) or exif.get(0x0132)
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def load_oriented(path: Path, short_side: int | None = None) -> Image.Image:
    """Open an image as upright RGB, optionally downscaled so its short side is ``short_side``."""
    im = Image.open(path)
    if short_side is not None and im.format == "JPEG":
        # Draft mode lets libjpeg decode at 1/2, 1/4 or 1/8 scale: ~10x faster for 12 MP files.
        w, h = im.size
        scale = short_side / min(w, h)
        im.draft("RGB", (max(1, int(w * scale)), max(1, int(h * scale))))
    im = ImageOps.exif_transpose(im).convert("RGB")
    if short_side is not None:
        w, h = im.size
        scale = short_side / min(w, h)
        if scale < 1:
            im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    return im


def dhash(im: Image.Image, size: int = 8) -> str:
    """64-bit difference hash: a cheap content fingerprint stored in the manifest."""
    g = np.asarray(
        im.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.int16
    )
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def _process(args: tuple[Path, Path, Path, int]) -> Record:
    src, root, cache_root, short_side = args
    label = parse_folder(src.parent.name)
    rel = src.relative_to(root)
    cache_path = (cache_root / rel).with_suffix(".jpg")
    with Image.open(src) as probe:
        oriented = ImageOps.exif_transpose(probe)
        width, height = oriented.size
    if cache_path.exists():
        im = Image.open(cache_path).convert("RGB")
    else:
        im = load_oriented(src, short_side)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(cache_path, quality=94)
    return Record(
        path=rel.as_posix(),
        cache_path=cache_path.as_posix(),
        equipment=label.equipment,
        occupancy=label.occupancy,
        timestamp=capture_time(src).isoformat(),
        width=width,
        height=height,
        dhash=dhash(im),
    )


def scan(root: Path) -> list[Path]:
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no images found under {root}")
    return files


def build_records(root: Path, cache_root: Path, short_side: int, workers: int = 8) -> list[Record]:
    jobs = [(p, root, cache_root, short_side) for p in scan(root)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_process, jobs, chunksize=8))


def assign_sessions(records: list[Record], session_gap_s: float, block_s: float) -> None:
    """Group records into capture sessions and fixed-length time blocks, per machine.

    Sessions are formed per *equipment* (not per class) because in-use and free shots of the
    same machine were taken back-to-back, so they share near-identical backgrounds.
    """
    by_equipment: dict[str, list[Record]] = {}
    for r in records:
        by_equipment.setdefault(r.equipment, []).append(r)
    for equipment, recs in by_equipment.items():
        recs.sort(key=lambda r: (r.time, r.path))
        session_idx, start, prev = -1, None, None
        for r in recs:
            t = r.time
            if prev is None or (t - prev).total_seconds() > session_gap_s:
                session_idx += 1
                start = t
            prev = t
            r.session = f"{equipment}#s{session_idx}"
            r.block = f"{r.session}#b{int((t - start).total_seconds() // block_s)}"


def write_manifest(records: Iterable[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(Record)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=names)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def read_manifest(path: Path) -> list[Record]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        Record(**{**row, "width": int(row["width"]), "height": int(row["height"])}) for row in rows
    ]
