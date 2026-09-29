#!/usr/bin/env python3
"""Run SSL method/backbone jobs serially; each job uses all visible GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([entry for entry in visible.split(",") if entry.strip()])
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return len([entry for entry in output.splitlines() if entry.strip()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["simsiam", "mocov2", "byol", "fcmae"],
        choices=("simsiam", "mocov2", "byol", "fcmae"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8s", "yolov10s", "yolo12s"],
        choices=("yolov8s", "yolov10s", "yolo12s"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = gpu_count()
    if count < 1:
        raise RuntimeError("No NVIDIA GPU is visible")
    output_dir = args.output_dir.expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train_ssl.py")

    for method in args.methods:
        for model in args.models:
            command = (
                ["torchrun", "--standalone", f"--nproc_per_node={count}"]
                if count > 1
                else [sys.executable]
            )
            command += [
                str(script),
                "--method", method,
                "--model", model,
                "--data-root", str(args.data_root.expanduser().resolve()),
                "--image-list", str(args.image_list.expanduser().resolve()),
                "--output-dir", str(output_dir),
                "--epochs", str(args.epochs),
                "--seed", str(args.seed),
                "--workers", str(args.workers),
                "--micro-batch", str(args.micro_batch),
            ]
            if args.force:
                command.append("--force")
            log_path = log_dir / f"{method}_{model}.log"
            print(f"RUN {' '.join(command)} -> {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise SystemExit(f"Training failed; inspect {log_path}")


if __name__ == "__main__":
    main()

