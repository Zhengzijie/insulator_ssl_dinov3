"""Bounding-box conversion, crop expansion, and deterministic matching."""

from __future__ import annotations

from typing import Any


def yolo_to_xyxy(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    box_width = width * image_width
    box_height = height * image_height
    x_center = center_x * image_width
    y_center = center_y * image_height
    return [
        x_center - box_width / 2,
        y_center - box_height / 2,
        x_center + box_width / 2,
        y_center + box_height / 2,
    ]


def box_iou(left: list[float], right: list[float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def expanded_bounds(
    box: list[float],
    image_width: int,
    image_height: int,
    fraction: float = 0.10,
) -> tuple[int, int, int, int]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (
        max(0, int(box[0] - width * fraction)),
        max(0, int(box[1] - height * fraction)),
        min(image_width, int(box[2] + width * fraction + 0.9999)),
        min(image_height, int(box[3] + height * fraction + 0.9999)),
    )


def greedy_match(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    iou_threshold: float = 0.50,
    class_aware: bool = False,
) -> dict[int, int]:
    """Map prediction indexes to target indexes in descending confidence order."""
    order = sorted(
        range(len(predictions)),
        key=lambda index: float(predictions[index].get("confidence", 0.0)),
        reverse=True,
    )
    used_targets: set[int] = set()
    matches: dict[int, int] = {}
    for prediction_index in order:
        prediction = predictions[prediction_index]
        candidates = []
        for target_index, target in enumerate(targets):
            if target_index in used_targets:
                continue
            if class_aware and prediction.get("class_id") != target.get("class_id"):
                continue
            overlap = box_iou(prediction["box"], target["box"])
            if overlap >= iou_threshold:
                candidates.append((overlap, target_index))
        if candidates:
            _, target_index = max(candidates)
            matches[prediction_index] = target_index
            used_targets.add(target_index)
    return matches

