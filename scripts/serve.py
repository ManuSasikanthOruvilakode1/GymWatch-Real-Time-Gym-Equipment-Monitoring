"""Minimal server entry point for the inference-only Docker image (no typer/torch needed)."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from gymwatch.api import create_app
from gymwatch.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(create_app(Config.load(args.config)), host=args.host, port=args.port)
