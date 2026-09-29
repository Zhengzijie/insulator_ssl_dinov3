#!/usr/bin/env python3
"""Run detector, crop extraction, type routing, and DINOv3 classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.feature_backbones import load_backbone
from models.grn_yolo import register_grn
from utils.config import configured_path, load_config, workspace_path
from utils.geometry import expanded_bounds, greedy_match, yolo_to_xyxy
from utils.io import write_jsonl


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def flattened_classes(config: dict[str, Any]) -> list[str]:
    values = []
    for type_name in ("GI", "PI", "CI"):
        values.extend(config["classes"]["fine"][type_name])
    return values


def label_path(label_root: Path, image_path: Path) -> Path:
    for part in image_path.parts:
        if part in {"GI", "PI", "CI"}:
            candidate = label_root / part / f"{image_path.stem}.txt"
            if candidate.is_file():
                return candidate
    candidates = [label_root / type_name / f"{image_path.stem}.txt" for type_name in ("GI", "PI", "CI")]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected one fine-label file for {image_path}; found {existing}")
    return existing[0]


def read_targets(
    path: Path,
    image_width: int,
    image_height: int,
    class_names: list[str],
    type_ids: dict[str, int],
) -> list[dict[str, Any]]:
    targets = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid label row at {path}:{line_number}")
        class_id = int(fields[0])
        if class_id not in range(len(class_names)):
            raise ValueError(f"Fine class identifier is out of range at {path}:{line_number}")
        fine_name = class_names[class_id]
        type_name = fine_name.split("_", 1)[0]
        targets.append(
            {
                "box": yolo_to_xyxy(
                    *map(float, fields[1:]),
                    image_width=image_width,
                    image_height=image_height,
                ),
                "class_id": class_id,
                "fine": fine_name,
                "type": type_name,
                "type_id": type_ids[type_name],
            }
        )
    return targets


class CropClassifier:
    def __init__(self, config: dict[str, Any], seed: int, device: torch.device):
        self.device = device
        self.fine_classes = flattened_classes(config)
        self.extractor = load_backbone(
            "dinov3",
            cache_dir=workspace_path(config, "cache"),
            dinov3_repo=configured_path(config, "dinov3_repo", must_exist=True),
            dinov3_weight=configured_path(config, "dinov3_weight", must_exist=True),
        ).extractor.to(device).eval()
        self.heads: dict[str, nn.Linear] = {}
        for type_name in ("GI", "PI", "CI"):
            classes = list(config["classes"]["fine"][type_name])
            checkpoint = workspace_path(
                config,
                "runs",
                "probes",
                f"dinov3_{type_name}_s{seed}",
                "last.pt",
            )
            state = torch.load(checkpoint, map_location="cpu")
            head = nn.Linear(768, len(classes)).to(device)
            head.load_state_dict(state["head"], strict=True)
            self.heads[type_name] = head.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
        self.type_classes = config["classes"]["fine"]

    def __call__(self, crop: Image.Image, type_name: str) -> tuple[str, int, float]:
        tensor = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            features = self.extractor(tensor)
        probabilities = self.heads[type_name](features.float()).softmax(dim=1)[0]
        local_index = int(probabilities.argmax().item())
        fine_name = self.type_classes[type_name][local_index]
        return fine_name, self.fine_classes.index(fine_name), float(probabilities[local_index])


def classify_box(
    classifier: CropClassifier,
    image: Image.Image,
    box: list[float],
    type_name: str,
    type_id: int,
    type_confidence: float,
    crop_expansion: float,
    minimum_side: int,
) -> dict[str, Any] | None:
    bounds = expanded_bounds(box, image.width, image.height, crop_expansion)
    if bounds[2] - bounds[0] < minimum_side or bounds[3] - bounds[1] < minimum_side:
        return None
    fine_name, fine_id, fine_probability = classifier(image.crop(bounds), type_name)
    return {
        "box": [float(value) for value in box],
        "type_conf": float(type_confidence),
        "pred_type": type_name,
        "pred_type_id": int(type_id),
        "pred_fine": fine_name,
        "pred_fine_id": int(fine_id),
        "fine_prob": fine_probability,
        "confidence": float(type_confidence * fine_probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    pipeline_config = config["pipeline"]
    device = torch.device(f"cuda:{args.device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pipeline inference")
    model_name = pipeline_config["detector_model"]
    initialization = pipeline_config["detector_initialization"]
    checkpoint = workspace_path(
        config,
        "runs",
        "detector",
        f"{model_name}_{initialization}_s{args.seed}",
        "weights",
        "best.pt",
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = workspace_path(config, "evaluation", "pipeline", f"pred_s{args.seed}.jsonl")
    if output.exists() and not args.force:
        print(f"SKIP {output}")
        return
    detection_root = workspace_path(config, "detection")
    image_list = detection_root / "splits" / "val.txt"
    image_paths = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        path = Path(value)
        image_paths.append(path if path.is_absolute() else (image_list.parent / path).resolve())
    fine_label_root = configured_path(config, "fine_label_root", must_exist=True)
    fine_classes = flattened_classes(config)
    type_ids = {name: int(value) for name, value in config["classes"]["types"].items()}
    id_to_type = {value: name for name, value in type_ids.items()}

    register_grn()
    from ultralytics import YOLO

    detector = YOLO(str(checkpoint))
    classifier = CropClassifier(config, args.seed, device)
    records = []
    for image_path in image_paths:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        targets = read_targets(
            label_path(fine_label_root, image_path),
            image.width,
            image.height,
            fine_classes,
            type_ids,
        )
        result = detector.predict(
            source=str(image_path),
            imgsz=int(pipeline_config["image_size"]),
            conf=float(pipeline_config["confidence_threshold"]),
            iou=float(pipeline_config["nms_iou_threshold"]),
            max_det=int(pipeline_config["maximum_detections"]),
            device=args.device,
            verbose=False,
        )[0]
        detected = [
            {
                "box": box.tolist(),
                "confidence": float(confidence),
                "type_id": int(type_id),
            }
            for box, confidence, type_id in zip(
                result.boxes.xyxy.detach().cpu(),
                result.boxes.conf.detach().cpu(),
                result.boxes.cls.detach().cpu(),
            )
        ]
        matches = greedy_match(
            detected,
            targets,
            iou_threshold=float(pipeline_config["matching_iou_threshold"]),
            class_aware=False,
        )
        deployed, type_control, oracle = [], [], []
        for target in targets:
            prediction = classify_box(
                classifier,
                image,
                target["box"],
                target["type"],
                target["type_id"],
                1.0,
                float(pipeline_config["crop_expansion"]),
                int(pipeline_config["minimum_crop_side"]),
            )
            if prediction is not None:
                oracle.append(prediction)
        for detection_index, detection in enumerate(detected):
            predicted_type = id_to_type[int(detection["type_id"])]
            prediction = classify_box(
                classifier,
                image,
                detection["box"],
                predicted_type,
                int(detection["type_id"]),
                float(detection["confidence"]),
                float(pipeline_config["crop_expansion"]),
                int(pipeline_config["minimum_crop_side"]),
            )
            if prediction is not None:
                deployed.append(prediction)
            route_type = predicted_type
            route_id = int(detection["type_id"])
            if detection_index in matches:
                target = targets[matches[detection_index]]
                route_type, route_id = target["type"], target["type_id"]
            controlled = classify_box(
                classifier,
                image,
                detection["box"],
                route_type,
                route_id,
                float(detection["confidence"]),
                float(pipeline_config["crop_expansion"]),
                int(pipeline_config["minimum_crop_side"]),
            )
            if controlled is not None:
                controlled["detector_pred_type"] = predicted_type
                type_control.append(controlled)
        records.append(
            {
                "image": image_path.relative_to(detection_root).as_posix(),
                "targets": targets,
                "predictions": {
                    "oracle": oracle,
                    "predicted_boxes_gt_type": type_control,
                    "deployed": deployed,
                },
            }
        )
        print(f"DONE {image_path}", flush=True)
    write_jsonl(output, records)
    print(f"SAVED {output}")


if __name__ == "__main__":
    main()
