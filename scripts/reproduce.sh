#!/usr/bin/env bash
# Reproduce every number in the README from the raw photos and videos.
# Usage: bash scripts/reproduce.sh        (≈15 min on an RTX 4060 laptop GPU)
set -euo pipefail
cd "$(dirname "$0")/.."

SCRATCH=(-s model.pretrained=false -s train.epochs=80 -s train.patience=20 -s train.lr=0.002)

uv run gymwatch prepare

# Main models (train/val/test split)
uv run gymwatch train --name pretrained
uv run gymwatch evaluate runs/pretrained
uv run gymwatch train --name scratch "${SCRATCH[@]}"
uv run gymwatch evaluate runs/scratch "${SCRATCH[@]}"

# Cross-validation: 5 folds of time blocks, and leave-one-capture-session-out
uv run gymwatch cv --by block --name cv_block_pretrained
uv run gymwatch cv --by block --name cv_block_scratch "${SCRATCH[@]}"
uv run gymwatch cv --by session --name cv_session_pretrained
uv run gymwatch cv --by session --name cv_session_scratch "${SCRATCH[@]}"

# Deployable model + video evaluation + latency
uv run gymwatch export runs/pretrained -o models/gymwatch.onnx
uv run gymwatch export runs/scratch -o models/gymwatch_scratch.onnx
uv run gymwatch video-eval --model models/gymwatch.onnx --out artifacts/video_eval
uv run gymwatch video-eval --model models/gymwatch_scratch.onnx --out artifacts/video_eval_scratch
uv run gymwatch bench --model models/gymwatch.onnx
