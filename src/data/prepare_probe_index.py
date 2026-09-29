#!/usr/bin/env python3
"""Validate crop folders and create deterministic linear-probe indexes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import configured_path, load_config, workspace_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest(path: Path) -> str:
    value = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def image_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )


def valid_size(path: Path, minimum_side: int) -> bool:
    with Image.open(path) as image:
        return min(image.size) >= minimum_side


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "type", "subclass", "split"))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-directory-name")
    parser.add_argument("--minimum-side", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    crop_root = configured_path(config, "crop_source", must_exist=True)
    train_directory_name = args.train_directory_name or config["probe"].get(
        "train_directory_name", "train"
    )
    train_root = crop_root / train_directory_name
    validation_root = crop_root / "val"
    fine_classes = config["classes"]["fine"]
    expected = {name for values in fine_classes.values() for name in values}
    if not train_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError("Both crop train and val directories are required")
    if {path.name for path in train_root.iterdir() if path.is_dir()} != expected:
        raise ValueError("Training crop directories do not match configured fine classes")
    if {path.name for path in validation_root.iterdir() if path.is_dir()} != expected:
        raise ValueError("Validation crop directories do not match configured fine classes")

    validation_digests = set()
    validation_rows = []
    for type_name, class_names in fine_classes.items():
        for class_name in class_names:
            for path in image_files(validation_root / class_name):
                if not valid_size(path, args.minimum_side):
                    continue
                validation_digests.add(digest(path))
                validation_rows.append(
                    {
                        "path": path.relative_to(crop_root).as_posix(),
                        "type": type_name,
                        "subclass": class_name,
                        "split": "val",
                    }
                )

    training_rows = []
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for type_name, class_names in fine_classes.items():
        for class_name in class_names:
            for path in image_files(train_root / class_name):
                if not valid_size(path, args.minimum_side) or digest(path) in validation_digests:
                    continue
                row = {
                    "path": path.relative_to(crop_root).as_posix(),
                    "type": type_name,
                    "subclass": class_name,
                    "split": "train",
                }
                training_rows.append(row)
                grouped[(type_name, class_name)].append(row)

    random_state = int(config["split"]["random_state"])
    generator = random.Random(random_state)
    balanced_rows = []
    for type_name, class_names in fine_classes.items():
        target_size = min(len(grouped[(type_name, class_name)]) for class_name in class_names)
        if target_size == 0:
            raise RuntimeError(f"Cannot balance type {type_name}: one class is empty")
        for class_name in class_names:
            values = list(grouped[(type_name, class_name)])
            generator.shuffle(values)
            balanced_rows.extend(values[:target_size])

    output_root = workspace_path(config, "probe", "splits")

    def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
        return row["split"], row["type"], row["subclass"], row["path"]

    write_csv(output_root / "index.csv", sorted(training_rows + validation_rows, key=row_key))
    write_csv(output_root / "train_balanced.csv", sorted(balanced_rows, key=row_key))
    print(f"Prepared probe indexes: {output_root}")


if __name__ == "__main__":
    main()
