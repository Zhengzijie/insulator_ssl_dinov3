#!/usr/bin/env python3
"""Run linear-probe seeds in parallel, one per visible GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from models.feature_backbones import BACKBONES


DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / "configs" / "classes.yaml"


def gpu_indices() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [entry.strip() for entry in visible.split(",") if entry.strip()]
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return [entry.strip() for entry in output.splitlines() if entry.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument("--dinov3-weight", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--types", nargs="+", choices=("GI", "PI", "CI"), default=["GI", "PI", "CI"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = gpu_indices()
    if not gpus:
        raise RuntimeError("No NVIDIA GPU is visible")
    output_dir = args.output_dir.expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("train_linear_probe.py")
    failures: list[str] = []

    for backbone in args.backbones:
        for type_name in args.types:
            pending = deque(args.seeds)
            active: dict[str, tuple[subprocess.Popen, object, str]] = {}
            while pending or active:
                for gpu in gpus:
                    if gpu in active or not pending:
                        continue
                    seed = pending.popleft()
                    task = f"{backbone}_{type_name}_s{seed}"
                    command = [
                        sys.executable,
                        str(script),
                        "--backbone", backbone,
                        "--type", type_name,
                        "--seed", str(seed),
                        "--data-root", str(args.data_root.expanduser().resolve()),
                        "--index-csv", str(args.index_csv.expanduser().resolve()),
                        "--classes", str(args.classes.expanduser().resolve()),
                        "--cache-dir", str(args.cache_dir.expanduser().resolve()),
                        "--output-dir", str(output_dir),
                        "--epochs", str(args.epochs),
                        "--batch", str(args.batch),
                        "--workers", str(args.workers),
                        "--device", "0",
                    ]
                    if backbone.startswith("dinov3"):
                        if args.dinov3_repo is None or args.dinov3_weight is None:
                            raise ValueError(
                                "DINOv3 tasks require --dinov3-repo and --dinov3-weight"
                            )
                        command += [
                            "--dinov3-repo", str(args.dinov3_repo.expanduser().resolve()),
                            "--dinov3-weight", str(args.dinov3_weight.expanduser().resolve()),
                        ]
                    if args.force:
                        command.append("--force")
                    log_path = log_dir / f"{task}.log"
                    handle = log_path.open("a", encoding="utf-8")
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = gpu
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                    active[gpu] = (process, handle, task)
                    print(f"GPU {gpu}: START {task} pid={process.pid}", flush=True)
                if active:
                    time.sleep(1)
                for gpu, (process, handle, task) in list(active.items()):
                    return_code = process.poll()
                    if return_code is None:
                        continue
                    handle.close()
                    del active[gpu]
                    if return_code:
                        failures.append(task)
                        print(f"GPU {gpu}: FAILED {task}", flush=True)
                    else:
                        print(f"GPU {gpu}: DONE {task}", flush=True)
            if failures:
                raise SystemExit(f"Linear-probe failures: {failures}")


if __name__ == "__main__":
    main()
