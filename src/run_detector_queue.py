#!/usr/bin/env python3
"""Run one resumable detector fine-tuning task per visible GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


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
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--ssl-dir", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--inits",
        nargs="+",
        default=["baseline", "simsiam", "mocov2", "byol", "fcmae"],
        choices=("baseline", "simsiam", "mocov2", "byol", "fcmae"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8s", "yolov10s", "yolo12s"],
        choices=("yolov8s", "yolov10s", "yolo12s"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpus = gpu_indices()
    if not gpus:
        raise RuntimeError("No NVIDIA GPU is visible")
    project = args.project.expanduser().resolve()
    ssl_dir = args.ssl_dir.expanduser().resolve()
    log_dir = project / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "queue_state.json"
    script = Path(__file__).with_name("train_detector.py")
    tasks = deque(
        (model, initialization, seed)
        for initialization in args.inits
        for model in args.models
        for seed in args.seeds
    )
    running: dict[str, tuple[subprocess.Popen, tuple[str, str, int], object, Path]] = {}
    failures: list[dict[str, object]] = []

    def save_state() -> None:
        payload = {
            "pending": [
                {"model": model, "init": initialization, "seed": seed}
                for model, initialization, seed in tasks
            ],
            "running": [
                {
                    "gpu": gpu,
                    "model": task[0],
                    "init": task[1],
                    "seed": task[2],
                    "pid": process.pid,
                }
                for gpu, (process, task, _, _) in running.items()
            ],
            "failures": failures,
        }
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    while tasks or running:
        for gpu in gpus:
            if gpu in running or not tasks:
                continue
            model, initialization, seed = tasks.popleft()
            name = f"{model}_{initialization}_s{seed}"
            complete = project / name / "weights" / "best.pt"
            if complete.exists() and not args.force:
                print(f"SKIP {complete}", flush=True)
                continue
            command = [
                sys.executable,
                str(script),
                "--model", model,
                "--init", initialization,
                "--seed", str(seed),
                "--data", str(args.data.expanduser().resolve()),
                "--project", str(project),
                "--name", name,
                "--epochs", str(args.epochs),
                "--batch", str(args.batch),
                "--workers", str(args.workers),
                "--device", "0",
            ]
            if initialization != "baseline":
                command += [
                    "--ssl-checkpoint",
                    str(ssl_dir / f"{initialization}_{model}.pt"),
                ]
            if args.force:
                command.append("--force")
            log_path = log_dir / f"{name}.log"
            handle = log_path.open("a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (process, (model, initialization, seed), handle, log_path)
            print(f"GPU {gpu}: START {name} pid={process.pid}", flush=True)
        save_state()
        if not running:
            continue
        time.sleep(2)
        for gpu, (process, task, handle, log_path) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del running[gpu]
            if return_code:
                failure = {
                    "model": task[0],
                    "init": task[1],
                    "seed": task[2],
                    "returncode": return_code,
                    "log": str(log_path),
                }
                failures.append(failure)
                print(f"GPU {gpu}: FAILED {failure}", flush=True)
            else:
                print(f"GPU {gpu}: DONE {task}", flush=True)
    save_state()
    if failures:
        raise SystemExit(f"Fine-tuning failures recorded in {state_path}")


if __name__ == "__main__":
    main()

