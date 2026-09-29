#!/usr/bin/env python3
"""Evaluate final-epoch frozen-backbone linear probes on validation crops."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.feature_backbones import load_backbone
from utils.config import configured_path, load_config, workspace_path
from utils.io import write_json
from utils.metrics import classification_metrics


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ValidationCrops(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        root: Path,
        classes: list[str],
    ):
        self.rows = rows
        self.root = root
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[row["subclass"]]


def validation_rows(index_csv: Path, type_name: str) -> list[dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["type"] == type_name and row["split"] == "val"
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--backbones", nargs="+")
    parser.add_argument("--types", nargs="+", choices=("GI", "PI", "CI"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    probe_config = config["probe"]
    backbones = args.backbones or probe_config["backbones"]
    type_names = args.types or probe_config["types"]
    seeds = args.seeds or probe_config["seeds"]
    crop_root = configured_path(config, "crop_source", must_exist=True)
    index_csv = workspace_path(config, "probe", "splits", "index.csv")
    run_root = workspace_path(config, "runs", "probes")
    output_root = workspace_path(config, "evaluation", "components", "probes")
    cache_dir = workspace_path(config, "cache")
    device = torch.device(f"cuda:{args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP16 feature extraction")

    for backbone_name in backbones:
        loaded = load_backbone(
            backbone_name,
            cache_dir=cache_dir,
            dinov3_repo=configured_path(config, "dinov3_repo") if backbone_name.startswith("dinov3") else None,
            dinov3_weight=configured_path(config, "dinov3_weight") if backbone_name.startswith("dinov3") else None,
        )
        extractor = loaded.extractor.to(device).eval()
        for type_name in type_names:
            classes = list(config["classes"]["fine"][type_name])
            rows = validation_rows(index_csv, type_name)
            loader = DataLoader(
                ValidationCrops(rows, crop_root, classes),
                batch_size=int(probe_config["batch"]),
                shuffle=False,
                num_workers=int(probe_config["workers"]),
                pin_memory=True,
            )
            for seed in seeds:
                output = output_root / f"{backbone_name}_{type_name}_s{seed}.json"
                if output.exists() and not args.force:
                    print(f"SKIP {output}")
                    continue
                checkpoint = run_root / f"{backbone_name}_{type_name}_s{seed}" / "last.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
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
                        "backbone": backbone_name,
                        "type": type_name,
                        "seed": int(seed),
                        "classes": classes,
                        "checkpoint": str(checkpoint),
                        "metrics": classification_metrics(targets, predictions, classes),
                    },
                )
                print(f"SAVED {output}")


if __name__ == "__main__":
    main()
