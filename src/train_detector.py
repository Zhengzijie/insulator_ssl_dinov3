#!/usr/bin/env python3
"""Fine-tune one GRN-augmented detector from random or SSL initialization."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from models.grn_yolo import (
    build_yolo,
    extract_backbone,
    load_backbone,
    register_grn,
)


os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def best_existing_map50(csv_path: Path) -> float:
    if not csv_path.exists():
        return -float("inf")
    values = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if key and "mAP50(B)" in key and "50-95" not in key:
                    try:
                        values.append(float(value))
                    except (TypeError, ValueError):
                        pass
    return max(values, default=-float("inf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("yolov8s", "yolov10s", "yolo12s"))
    parser.add_argument(
        "--init",
        required=True,
        choices=("baseline", "simsiam", "mocov2", "byol", "fcmae"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset YAML")
    parser.add_argument("--ssl-checkpoint", type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    if args.init == "baseline" and args.ssl_checkpoint is not None:
        raise ValueError("--ssl-checkpoint must not be supplied for baseline training")
    if args.init != "baseline" and args.ssl_checkpoint is None:
        raise ValueError("--ssl-checkpoint is required for SSL-initialized training")

    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    project = args.project.expanduser().resolve()
    name = args.name or f"{args.model}_{args.init}_s{args.seed}"
    run_dir = project / name
    weights_dir = run_dir / "weights"
    last_checkpoint = weights_dir / "last.pt"
    map50_checkpoint = weights_dir / "best_map50.pt"
    best_checkpoint = weights_dir / "best.pt"
    initialization_report = run_dir / "ssl_init_report.json"

    seed_everything(args.seed)
    register_grn()
    from ultralytics import YOLO
    import ultralytics.engine.trainer as ultralytics_trainer

    # Ultralytics 8.3.87 otherwise downloads a pretrained reference model only
    # to test AMP compatibility. The training model itself remains AMP-enabled.
    ultralytics_trainer.check_amp = lambda _model: True

    if last_checkpoint.exists() and not args.force:
        model = YOLO(str(last_checkpoint))
        resume: str | bool = str(last_checkpoint)
        if args.init != "baseline":
            if not initialization_report.is_file():
                raise RuntimeError(
                    "Cannot resume an SSL run without its initialization verification file"
                )
            report = json.loads(initialization_report.read_text(encoding="utf-8"))
            if not report.get("trainer_handoff_verified"):
                raise RuntimeError("The saved run has no verified SSL backbone handoff")
        print(f"RESUME {last_checkpoint}")
    else:
        model = build_yolo(args.model, verbose=True)
        resume = False
        if args.init != "baseline":
            checkpoint = args.ssl_checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            report = load_backbone(model, checkpoint)
            expected = {
                key: value.detach().cpu().clone()
                for key, value in extract_backbone(model).state_dict().items()
            }

            # A YAML-created Ultralytics wrapper otherwise rebuilds the model
            # inside the trainer. This marker preserves the in-memory weights.
            model.ckpt = {"model": model.model, "source": str(checkpoint)}

            def verify_trainer_handoff(trainer) -> None:
                trained = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                actual = extract_backbone(trained).state_dict()
                missing = sorted(set(expected) - set(actual))
                unexpected = sorted(set(actual) - set(expected))
                unequal = sorted(
                    key
                    for key in set(expected) & set(actual)
                    if not torch.equal(expected[key], actual[key].detach().cpu())
                )
                verified = not (missing or unexpected or unequal)
                report.update(
                    {
                        "trainer_handoff_verified": verified,
                        "trainer_handoff_missing": missing,
                        "trainer_handoff_unexpected": unexpected,
                        "trainer_handoff_unequal": unequal,
                    }
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                initialization_report.write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    "TRAINER_BACKBONE_VERIFY "
                    + json.dumps(
                        {
                            "verified": verified,
                            "keys": len(expected),
                            "missing": missing,
                            "unexpected": unexpected,
                            "unequal": unequal,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if not verified:
                    raise RuntimeError("SSL backbone changed before the first training epoch")

            model.add_callback("on_pretrain_routine_end", verify_trainer_handoff)

    tracker = {
        "best": -float("inf")
        if args.force
        else best_existing_map50(run_dir / "results.csv")
    }

    def save_map50_checkpoint(trainer) -> None:
        current = float(trainer.metrics.get("metrics/mAP50(B)", -float("inf")))
        if current > tracker["best"]:
            tracker["best"] = current
            weights_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trainer.last, map50_checkpoint)
            print(
                f"MAP50_BEST epoch={trainer.epoch + 1} "
                f"map50={current:.8f} path={map50_checkpoint}",
                flush=True,
            )

    model.add_callback("on_model_save", save_map50_checkpoint)
    training = dict(
        data=str(data_yaml),
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch,
        lr0=0.01,
        optimizer="auto",
        amp=True,
        seed=args.seed,
        deterministic=True,
        pretrained=False,
        project=str(project),
        name=name,
        exist_ok=True,
        device=args.device,
        workers=args.workers,
        plots=False,
        val=True,
        save=True,
        verbose=True,
    )
    if resume:
        training["resume"] = resume
    model.train(**training)

    if not map50_checkpoint.exists():
        shutil.copy2(last_checkpoint, map50_checkpoint)
    shutil.copy2(map50_checkpoint, best_checkpoint)
    print(f"SAVED validation-mAP50 checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
