#!/usr/bin/env python3
"""Create group-aware detection splits and a leakage-filtered SSL image list."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from sklearn.model_selection import GroupShuffleSplit
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import configured_path, load_config, workspace_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TYPE_NAMES = ("GI", "PI", "CI")
DJI_PATTERN = re.compile(r"(?P<date>20\d{6})[^0-9]?(?P<hour>[0-2]\d)")


def image_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix(),
    )


def file_digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timestamp_group(path: Path, _: int) -> str | None:
    match = DJI_PATTERN.search(path.stem)
    return f"{match.group('date')}_{match.group('hour')}" if match else None


def prefix_group(path: Path, _: int) -> str | None:
    prefix, separator, _suffix = path.stem.rpartition("_")
    return prefix if separator and prefix else None


def block_group(_path: Path, index: int, block_size: int) -> str:
    return f"block_{index // block_size:08d}"


def choose_group_rule(
    paths: list[Path],
    fallback_size: int,
) -> tuple[str, dict[Path, str]]:
    candidates: list[tuple[str, Callable[[Path, int], str | None]]] = [
        ("timestamp_date_hour", timestamp_group),
        ("filename_prefix", prefix_group),
        ("contiguous_block", lambda path, index: block_group(path, index, fallback_size)),
    ]
    for name, function in candidates:
        values = [function(path, index) for index, path in enumerate(paths)]
        if any(value is None for value in values):
            continue
        counts = Counter(str(value) for value in values)
        maximum_fraction = max(counts.values()) / len(paths)
        if name == "contiguous_block" or (len(counts) >= 20 and maximum_fraction <= 0.30):
            return name, {path: str(value) for path, value in zip(paths, values)}
    raise RuntimeError("No valid grouping rule could be constructed")


def grouped_split(
    paths: list[Path],
    groups: dict[Path, str],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    random_state: int,
) -> dict[str, list[Path]]:
    group_values = [groups[path] for path in paths]
    first = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=random_state)
    train_indices, remainder_indices = next(first.split(paths, groups=group_values))
    remainder_paths = [paths[index] for index in remainder_indices]
    remainder_groups = [groups[path] for path in remainder_paths]
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_fraction,
        random_state=random_state,
    )
    validation_indices, test_indices = next(
        second.split(remainder_paths, groups=remainder_groups)
    )
    return {
        "train": sorted((paths[index] for index in train_indices), key=lambda path: path.as_posix()),
        "val": sorted(
            (remainder_paths[index] for index in validation_indices),
            key=lambda path: path.as_posix(),
        ),
        "test": sorted(
            (remainder_paths[index] for index in test_indices),
            key=lambda path: path.as_posix(),
        ),
    }


def remap_label(source: Path, destination: Path, type_id: int) -> None:
    rows = []
    if source.is_file():
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO row in {source}:{line_number}")
            rows.append(" ".join([str(type_id), *fields[1:]]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting symbolic link: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Workspace destination already exists: {destination}")
    destination.symlink_to(source.resolve())


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    source_root = configured_path(config, "detection_source", must_exist=True)
    ssl_root = configured_path(config, "ssl_source", must_exist=True)
    output_root = workspace_path(config, "detection")
    split_root = output_root / "splits"
    prepared_root = output_root / "dataset"
    split_config = config["split"]
    type_ids = {name: int(value) for name, value in config["classes"]["types"].items()}

    source_by_type: dict[str, list[Path]] = {}
    all_paths = []
    for type_name in TYPE_NAMES:
        images_root = source_root / type_name / "images"
        labels_root = source_root / type_name / "labels"
        paths = image_files(images_root)
        if not paths:
            raise RuntimeError(f"No images found in {images_root}")
        source_by_type[type_name] = paths
        all_paths.extend(paths)
        for source_image in paths:
            relative = source_image.relative_to(images_root)
            prepared_image = prepared_root / type_name / "images" / relative
            prepared_label = prepared_root / type_name / "labels" / relative.with_suffix(".txt")
            safe_symlink(source_image, prepared_image)
            remap_label(labels_root / relative.with_suffix(".txt"), prepared_label, type_ids[type_name])

    all_paths = sorted(all_paths, key=lambda path: path.as_posix())
    group_rule, groups = choose_group_rule(
        all_paths,
        int(split_config["fallback_group_size"]),
    )
    combined = {"train": [], "val": [], "test": []}
    source_to_prepared: dict[Path, Path] = {}
    for type_name, paths in source_by_type.items():
        images_root = source_root / type_name / "images"
        for path in paths:
            source_to_prepared[path] = prepared_root / type_name / "images" / path.relative_to(images_root)
        subsets = grouped_split(
            paths,
            groups,
            float(split_config["train_fraction"]),
            float(split_config["validation_fraction"]),
            float(split_config["test_fraction"]),
            int(split_config["random_state"]),
        )
        for subset, subset_paths in subsets.items():
            combined[subset].extend(subset_paths)

    output_root.mkdir(parents=True, exist_ok=True)
    safe_symlink(prepared_root, split_root / "dataset")
    for subset, paths in combined.items():
        relative = sorted(
            "./dataset/" + source_to_prepared[path].relative_to(prepared_root).as_posix()
            for path in paths
        )
        write_lines(split_root / f"{subset}.txt", relative)

    test_paths = combined["test"]
    test_digests = {file_digest(path) for path in test_paths}
    test_groups = {groups[path] for path in test_paths}
    ssl_values = []
    for path in image_files(ssl_root):
        relative = path.relative_to(ssl_root).as_posix()
        group_value = None
        if group_rule == "timestamp_date_hour":
            group_value = timestamp_group(path, 0)
        elif group_rule == "filename_prefix":
            group_value = prefix_group(path, 0)
        if file_digest(path) in test_digests or (group_value is not None and group_value in test_groups):
            continue
        ssl_values.append(relative)
    write_lines(split_root / "ssl_list.txt", sorted(ssl_values))

    dataset_yaml = {
        "path": str(output_root),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "test": "splits/test.txt",
        "names": {type_ids[name]: name for name in TYPE_NAMES},
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )
    manifest = {
        "group_rule": group_rule,
        "type_mapping": type_ids,
        "split_sha256": {
            path.name: file_digest(path, "sha256")
            for path in sorted(split_root.glob("*.txt"))
        },
    }
    (split_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared detection workspace: {output_root}")


if __name__ == "__main__":
    main()
