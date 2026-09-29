#!/usr/bin/env python3
"""Evaluate trained detectors without aggregating across experimental runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.grn_yolo import register_grn
from utils.config import load_config, workspace_path
from utils.io import write_json


def metric_payload(metrics, class_names: list[str]) -> dict:
    all_ap = np.asarray(metrics.box.all_ap, dtype=float)
    class_indices = np.asarray(metrics.box.ap_class_index, dtype=int)
    by_id = {int(class_id): all_ap[index] for index, class_id in enumerate(class_indices)}
    per_class = {}
    for class_id, name in enumerate(class_names):
        values = by_id.get(class_id)
        per_class[name] = {
            "ap50": float(values[0]) if values is not None else None,
            "ap50_95": float(values.mean()) if values is not None else None,
        }
    return {
        "per_class": per_class,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="0")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--initializations", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    detector_config = config["detector"]
    models = args.models or detector_config["models"]
    initializations = args.initializations or detector_config["initializations"]
    seeds = args.seeds or detector_config["seeds"]
    class_mapping = config["classes"]["types"]
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: int(item[1]))]
    data_yaml = workspace_path(config, "detection", "data.yaml")
    run_root = workspace_path(config, "runs", "detector")
    output_root = workspace_path(config, "evaluation", "components", "detector")
    output_root.mkdir(parents=True, exist_ok=True)

    register_grn()
    from ultralytics import YOLO

    for model_name in models:
        for initialization in initializations:
            for seed in seeds:
                run_name = f"{model_name}_{initialization}_s{seed}"
                checkpoint = run_root / run_name / "weights" / "best.pt"
                output = output_root / f"{run_name}_{args.split}.json"
                if output.exists() and not args.force:
                    print(f"SKIP {output}")
                    continue
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                model = YOLO(str(checkpoint))
                metrics = model.val(
                    data=str(data_yaml),
                    split=args.split,
                    imgsz=int(detector_config["image_size"]),
                    batch=int(detector_config["batch"]),
                    device=args.device,
                    workers=int(detector_config["workers"]),
                    plots=False,
                    save_json=False,
                    verbose=True,
                )
                write_json(
                    output,
                    {
                        "model": model_name,
                        "initialization": initialization,
                        "seed": int(seed),
                        "split": args.split,
                        "checkpoint": str(checkpoint),
                        "metrics": metric_payload(metrics, class_names),
                    },
                )
                print(f"SAVED {output}")


if __name__ == "__main__":
    main()
