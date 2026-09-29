"""Metric implementations shared by component and pipeline evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from utils.geometry import box_iou


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    class_names: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    per_class = {}
    total = int(matrix.sum())
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        denominator = true_negative + false_positive
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "specificity": float(true_negative / denominator) if denominator else 0.0,
            "support": int(support[index]),
        }
    return {
        "per_class": per_class,
        "macro_f1": float(f1.mean()),
        "accuracy": float(accuracy_score(targets, predictions)),
        "confusion_matrix": matrix.tolist(),
    }


def _interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    points = np.linspace(0.0, 1.0, 101)
    values = [precision[recall >= point].max() if np.any(recall >= point) else 0.0 for point in points]
    return float(np.mean(values))


def detection_ap50(
    targets: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    class_names: list[str] | tuple[str, ...],
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute class-aware AP using 101-point interpolation."""
    targets_by_image_class: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_image_class[(str(target["image_id"]), int(target["class_id"]))].append(target)
    per_class: dict[str, float | None] = {}
    valid_values = []
    for class_id, class_name in enumerate(class_names):
        total_targets = sum(
            len(items)
            for (image_id, candidate_class), items in targets_by_image_class.items()
            if candidate_class == class_id
        )
        if total_targets == 0:
            per_class[class_name] = None
            continue
        candidates = sorted(
            [item for item in predictions if int(item["class_id"]) == class_id],
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        claimed: dict[tuple[str, int], set[int]] = defaultdict(set)
        true_positives = []
        false_positives = []
        for prediction in candidates:
            key = (str(prediction["image_id"]), class_id)
            overlaps = [
                (box_iou(prediction["box"], target["box"]), target_index)
                for target_index, target in enumerate(targets_by_image_class.get(key, []))
                if target_index not in claimed[key]
            ]
            if overlaps and max(overlaps)[0] >= iou_threshold:
                _, target_index = max(overlaps)
                claimed[key].add(target_index)
                true_positives.append(1.0)
                false_positives.append(0.0)
            else:
                true_positives.append(0.0)
                false_positives.append(1.0)
        if not candidates:
            value = 0.0
        else:
            true_positive_sum = np.cumsum(true_positives)
            false_positive_sum = np.cumsum(false_positives)
            recall = true_positive_sum / total_targets
            precision = true_positive_sum / np.maximum(true_positive_sum + false_positive_sum, 1e-12)
            value = _interpolated_ap(recall, precision)
        per_class[class_name] = value
        valid_values.append(value)
    return {
        "per_class_ap50": per_class,
        "map50": float(np.mean(valid_values)) if valid_values else 0.0,
    }

