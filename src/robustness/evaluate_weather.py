#!/usr/bin/env python3
"""Evaluate detector and DINOv3 probes under clean and synthetic weather."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.feature_backbones import load_backbone
from models.grn_yolo import register_grn
from utils.config import configured_path, load_config, workspace_path
from utils.io import write_json
from utils.metrics import classification_metrics


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FolderCrops(Dataset):
    def __init__(self, root: Path, classes: list[str]):
        self.samples = []
        for class_id, class_name in enumerate(classes):
            paths = sorted(
                path
                for path in (root / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            self.samples.extend((path, class_id) for path in paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, class_id = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), class_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    conditions = ["clean", *config["weather"]["conditions"]]
    seeds = config["pipeline"]["seeds"]
    model_name = config["pipeline"]["detector_model"]
    initialization = config["pipeline"]["detector_initialization"]
    output_root = workspace_path(config, "evaluation", "weather")
    output_root.mkdir(parents=True, exist_ok=True)

    register_grn()
    from ultralytics import YOLO

    for condition in conditions:
        data_yaml = (
            workspace_path(config, "detection", "data.yaml")
            if condition == "clean"
            else workspace_path(config, "weather", condition, "detection", "data.yaml")
        )
        for seed in seeds:
            output = output_root / f"det_{condition}_s{seed}.json"
            if output.exists() and not args.force:
                continue
            checkpoint = workspace_path(
                config,
                "runs",
                "detector",
                f"{model_name}_{initialization}_s{seed}",
                "weights",
                "best.pt",
            )
            metrics = YOLO(str(checkpoint)).val(
                data=str(data_yaml),
                split="val",
                imgsz=int(config["detector"]["image_size"]),
                conf=0.001,
                iou=float(config["pipeline"]["nms_iou_threshold"]),
                device=args.device,
                plots=False,
                save_json=False,
                verbose=True,
            )
            write_json(
                output,
                {
                    "condition": condition,
                    "seed": int(seed),
                    "model": model_name,
                    "initialization": initialization,
                    "metrics": {
                        "map50": float(metrics.box.map50),
                        "map50_95": float(metrics.box.map),
                        "per_class_ap50": [float(value) for value in metrics.box.ap50],
                    },
                },
            )

    device = torch.device(f"cuda:{args.device}")
    loaded = load_backbone(
        "dinov3",
        cache_dir=workspace_path(config, "cache"),
        dinov3_repo=configured_path(config, "dinov3_repo", must_exist=True),
        dinov3_weight=configured_path(config, "dinov3_weight", must_exist=True),
    )
    extractor = loaded.extractor.to(device).eval()
    clean_crop_root = configured_path(config, "crop_source", must_exist=True) / "val"
    for condition in conditions:
        crop_root = (
            clean_crop_root
            if condition == "clean"
            else workspace_path(config, "weather", condition, "classification")
        )
        for type_name in config["probe"]["types"]:
            classes = list(config["classes"]["fine"][type_name])
            loader = DataLoader(
                FolderCrops(crop_root, classes),
                batch_size=int(config["probe"]["batch"]),
                shuffle=False,
                num_workers=int(config["probe"]["workers"]),
                pin_memory=True,
            )
            for seed in seeds:
                output = output_root / f"cls_{condition}_{type_name}_s{seed}.json"
                if output.exists() and not args.force:
                    continue
                checkpoint = workspace_path(
                    config,
                    "runs",
                    "probes",
                    f"dinov3_{type_name}_s{seed}",
                    "last.pt",
                )
                state = torch.load(checkpoint, map_location="cpu")
                head = nn.Linear(loaded.feature_dim, len(classes)).to(device)
                head.load_state_dict(state["head"], strict=True)
                head.eval()
                targets, predictions = [], []
                with torch.no_grad():
                    for images, labels in loader:
                        images = images.to(device, non_blocking=True)
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            features = extractor(images)
                        predicted = head(features.float()).argmax(dim=1).cpu().tolist()
                        targets.extend(labels.tolist())
                        predictions.extend(predicted)
                write_json(
                    output,
                    {
                        "condition": condition,
                        "type": type_name,
                        "seed": int(seed),
                        "metrics": classification_metrics(targets, predictions, classes),
                    },
                )
    print(f"SAVED weather evaluation: {output_root}")


if __name__ == "__main__":
    main()
