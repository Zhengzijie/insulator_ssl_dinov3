#!/usr/bin/env python3
"""Generate deterministic fog, rain, and glare images without geometry changes."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import configured_path, load_config, workspace_path
from utils.seed import path_seed


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def weather_transforms(config: dict) -> dict[str, A.BasicTransform]:
    weather = config["weather"]
    fog_values = weather["fog"]
    fog_signature = inspect.signature(A.RandomFog)
    if "fog_coef_range" in fog_signature.parameters:
        fog = A.RandomFog(
            fog_coef_range=tuple(fog_values["fog_coef_range"]),
            alpha_coef=float(fog_values["alpha_coef"]),
            p=1.0,
        )
    else:
        fog = A.RandomFog(
            fog_coef_lower=float(fog_values["fog_coef_range"][0]),
            fog_coef_upper=float(fog_values["fog_coef_range"][1]),
            alpha_coef=float(fog_values["alpha_coef"]),
            p=1.0,
        )
    rain_values = weather["rain"]
    rain_signature = inspect.signature(A.RandomRain)
    rain_kwargs = {
        "drop_length": int(rain_values["drop_length"]),
        "drop_width": int(rain_values["drop_width"]),
        "blur_value": int(rain_values["blur_value"]),
        "brightness_coefficient": float(rain_values["brightness_coefficient"]),
        "rain_type": None,
        "p": 1.0,
    }
    if "slant_range" in rain_signature.parameters:
        rain_kwargs["slant_range"] = tuple(rain_values["slant_range"])
    else:
        rain_kwargs["slant_lower"] = int(rain_values["slant_range"][0])
        rain_kwargs["slant_upper"] = int(rain_values["slant_range"][1])
    rain = A.RandomRain(**rain_kwargs)
    glare_values = weather["glare"]
    glare = A.RandomSunFlare(
        flare_roi=tuple(glare_values["flare_roi"]),
        src_radius=int(glare_values["source_radius"]),
        p=1.0,
    )
    return {"fog": fog, "rain": rain, "glare": glare}


def apply_transform(transform: A.BasicTransform, image: np.ndarray, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    composed = A.Compose([transform])
    if hasattr(composed, "set_random_seed"):
        composed.set_random_seed(seed)
    return composed(image=image)["image"]


def save_transformed(
    source: Path,
    destination: Path,
    transform: A.BasicTransform,
    seed_key: str,
    force: bool,
) -> None:
    if destination.exists() and not force:
        return
    with Image.open(source) as opened:
        image = np.asarray(opened.convert("RGB"))
    transformed = apply_transform(transform, image, path_seed(seed_key))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    Image.fromarray(transformed).save(temporary)
    temporary.replace(destination)


def safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting symbolic link: {destination}")
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.symlink_to(source.resolve())


def paired_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_positions = [index for index, value in enumerate(parts) if value == "images"]
    if not image_positions:
        raise ValueError(f"Detection image path has no images directory: {image_path}")
    parts[image_positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.as_posix(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    transforms_by_condition = weather_transforms(config)
    weather_root = workspace_path(config, "weather")
    clean_detection_root = workspace_path(config, "detection")
    validation_list = clean_detection_root / "splits" / "val.txt"
    listed_values = [
        line.strip()
        for line in validation_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    detection_sources = [
        Path(value) if Path(value).is_absolute() else (validation_list.parent / value).resolve()
        for value in listed_values
    ]
    crop_root = configured_path(config, "crop_source", must_exist=True) / "val"

    for condition, transform in transforms_by_condition.items():
        condition_root = weather_root / condition
        degraded_detection = condition_root / "detection"
        condition_values = []
        for source in detection_sources:
            relative = source.relative_to(clean_detection_root)
            destination = degraded_detection / relative
            save_transformed(source, destination, transform, relative.as_posix(), args.force)
            clean_label = paired_label(source)
            degraded_label = paired_label(destination)
            safe_symlink(clean_label, degraded_label)
            condition_values.append("./" + relative.as_posix())
        split_path = degraded_detection / "val.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text("".join(f"{value}\n" for value in condition_values), encoding="utf-8")
        data_yaml = {
            "path": str(degraded_detection),
            "val": "val.txt",
            "names": {
                int(value): name for name, value in config["classes"]["types"].items()
            },
        }
        (degraded_detection / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False),
            encoding="utf-8",
        )

        degraded_crops = condition_root / "classification"
        for class_name in [
            name for values in config["classes"]["fine"].values() for name in values
        ]:
            for source in image_files(crop_root / class_name):
                relative = source.relative_to(crop_root)
                save_transformed(
                    source,
                    degraded_crops / relative,
                    transform,
                    f"classification/{relative.as_posix()}",
                    args.force,
                )

    metadata = {
        "albumentations_version": A.__version__,
        "conditions": config["weather"],
        "seed_rule": "crc32(relative_path)",
        "geometry_changed": False,
    }
    weather_root.mkdir(parents=True, exist_ok=True)
    (weather_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared weather workspace: {weather_root}")


if __name__ == "__main__":
    main()
