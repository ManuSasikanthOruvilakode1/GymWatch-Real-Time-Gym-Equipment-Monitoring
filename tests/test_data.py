from __future__ import annotations

import pytest

from gymwatch.config import CameraConfig, Config, MonitorConfig
from gymwatch.data.split import (
    assign_splits,
    cv_folds,
    leakage_report,
    nearest_train_gap,
    session_folds,
)
from gymwatch.labels import Label, parse_folder


@pytest.mark.parametrize(
    ("folder", "expected"),
    [
        ("IN-USE-LEG-EXTENSION", Label("leg_extension", "in_use")),
        ("NOT-IN-USE-LEG-EXTENSION", Label("leg_extension", "free")),
        ("IN-USE-PEC-FLY", Label("pec_fly", "in_use")),
        ("not-in-use-pec-fly", Label("pec_fly", "free")),
    ],
)
def test_parse_folder(folder, expected):
    assert parse_folder(folder) == expected


@pytest.mark.parametrize("folder", ["IN-USE-SQUAT-RACK", "LEG-EXTENSION", ""])
def test_parse_folder_rejects_unknown(folder):
    with pytest.raises(ValueError):
        parse_folder(folder)


def test_sessions_are_per_equipment(records):
    for r in records:
        assert r.session.startswith(r.equipment)
    assert len({r.session for r in records}) == 6


def test_split_keeps_blocks_whole_and_purges_neighbours(records):
    assign_splits(records, val_fraction=0.15, test_fraction=0.15, purge_s=1.0, seed=0)
    splits_per_block: dict[str, set[str]] = {}
    for r in records:
        if r.split != "purged":
            splits_per_block.setdefault(r.block, set()).add(r.split)
    assert all(len(s) == 1 for s in splits_per_block.values())
    assert {r.split for r in records} >= {"train", "val", "test"}

    train = [i for i, r in enumerate(records) if r.split == "train"]
    held = [i for i, r in enumerate(records) if r.split in {"val", "test"}]
    assert nearest_train_gap(records, train, held).min() > 1.0


def test_split_is_deterministic(records):
    from tests.conftest import make_records

    other = make_records()
    assign_splits(records, 0.15, 0.15, 1.0, seed=3)
    assign_splits(other, 0.15, 0.15, 1.0, seed=3)
    assert [r.split for r in records] == [r.split for r in other]


def test_leakage_report_shows_random_split_leaks(records):
    assign_splits(records, 0.15, 0.15, 1.0, seed=0)
    rep = leakage_report(records)["heldout_to_nearest_train_frame"]
    assert rep["grouped_purged"]["share_within_1s"] == 0.0
    assert rep["random_split"]["share_within_1s"] > 0.9


def test_cv_folds_disjoint_and_exclude_test(records):
    assign_splits(records, 0.15, 0.15, 1.0, seed=0)
    folds = cv_folds(records, k=4, purge_s=1.0, seed=0)
    assert len(folds) == 4
    all_val = [i for _, va in folds for i in va]
    assert len(all_val) == len(set(all_val))
    for tr, va in folds:
        assert not set(tr) & set(va)
        assert all(records[i].split != "test" for i in tr + va)
        assert nearest_train_gap(records, tr, va).min() > 1.0


def test_config_overrides_and_validation(tmp_path):
    cfg = Config.load(
        None, ["train.epochs=3", "model.image_size=[64, 48]", "model.pretrained=false"]
    )
    assert cfg.train.epochs == 3
    assert cfg.model.image_size == (64, 48)
    assert cfg.model.pretrained is False
    cfg.dump(tmp_path / "c.yaml")
    assert Config.load(tmp_path / "c.yaml") == cfg

    with pytest.raises(ValueError):
        CameraConfig(id="a", name="a", source="0", equipment="pec_fly", roi=(0.5, 0, 0.4, 1))
    cam = CameraConfig(id="a", name="a", source="0", equipment="pec_fly")
    with pytest.raises(ValueError):
        MonitorConfig(cameras=[cam, cam])


def test_session_folds_hold_out_whole_sessions(records):
    folds = session_folds(records, min_size=10)
    assert len(folds) == 6
    for tr, va in folds:
        held = {records[i].session for i in va}
        assert len(held) == 1
        assert not held & {records[i].session for i in tr}
        assert len(tr) + len(va) == len(records)
