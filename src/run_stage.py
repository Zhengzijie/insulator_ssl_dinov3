#!/usr/bin/env python3
"""Translate the paper configuration into ordered executable stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from utils.config import configured_path, load_config, workspace_path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"


def run(command: list[str]) -> None:
    print("COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def strings(values) -> list[str]:
    return [str(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "ssl", "detectors", "probes", "components", "pipeline", "weather"),
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    python = sys.executable

    if args.stage == "prepare":
        run([python, str(SOURCE / "data" / "prepare_detection_data.py"), "--config", str(config_path)])
        run([python, str(SOURCE / "data" / "prepare_probe_index.py"), "--config", str(config_path)])
    elif args.stage == "ssl":
        section = config["ssl"]
        run(
            [
                python,
                str(SOURCE / "run_ssl_queue.py"),
                "--data-root", str(configured_path(config, "ssl_source", must_exist=True)),
                "--image-list", str(workspace_path(config, "detection", "splits", "ssl_list.txt")),
                "--output-dir", str(workspace_path(config, "checkpoints", "ssl")),
                "--methods", *strings(section["methods"]),
                "--models", *strings(section["models"]),
                "--epochs", str(section["epochs"]),
                "--seed", str(section["seed"]),
                "--workers", str(section["workers"]),
                "--micro-batch", str(section["micro_batch"]),
            ]
        )
    elif args.stage == "detectors":
        section = config["detector"]
        run(
            [
                python,
                str(SOURCE / "run_detector_queue.py"),
                "--data", str(workspace_path(config, "detection", "data.yaml")),
                "--ssl-dir", str(workspace_path(config, "checkpoints", "ssl")),
                "--project", str(workspace_path(config, "runs", "detector")),
                "--inits", *strings(section["initializations"]),
                "--models", *strings(section["models"]),
                "--seeds", *strings(section["seeds"]),
                "--epochs", str(section["epochs"]),
                "--batch", str(section["batch"]),
                "--workers", str(section["workers"]),
            ]
        )
    elif args.stage == "probes":
        section = config["probe"]
        run(
            [
                python,
                str(SOURCE / "run_probe_queue.py"),
                "--data-root", str(configured_path(config, "crop_source", must_exist=True)),
                "--index-csv", str(workspace_path(config, "probe", "splits", "train_balanced.csv")),
                "--classes", str(ROOT / "configs" / "classes.yaml"),
                "--dinov3-repo", str(configured_path(config, "dinov3_repo", must_exist=True)),
                "--dinov3-weight", str(configured_path(config, "dinov3_weight", must_exist=True)),
                "--cache-dir", str(workspace_path(config, "cache")),
                "--output-dir", str(workspace_path(config, "runs", "probes")),
                "--backbones", *strings(section["backbones"]),
                "--types", *strings(section["types"]),
                "--seeds", *strings(section["seeds"]),
                "--epochs", str(section["epochs"]),
                "--batch", str(section["batch"]),
                "--workers", str(section["workers"]),
            ]
        )
    elif args.stage == "components":
        run([python, str(SOURCE / "evaluation" / "evaluate_detector.py"), "--config", str(config_path)])
        run([python, str(SOURCE / "evaluation" / "evaluate_probe.py"), "--config", str(config_path)])
    elif args.stage == "pipeline":
        for seed in config["pipeline"]["seeds"]:
            run(
                [
                    python,
                    str(SOURCE / "pipeline" / "run_pipeline.py"),
                    "--config", str(config_path),
                    "--seed", str(seed),
                ]
            )
        run([python, str(SOURCE / "evaluation" / "evaluate_pipeline.py"), "--config", str(config_path)])
    else:
        run([python, str(SOURCE / "robustness" / "make_weather.py"), "--config", str(config_path)])
        run([python, str(SOURCE / "robustness" / "evaluate_weather.py"), "--config", str(config_path)])


if __name__ == "__main__":
    main()

