from __future__ import annotations

import pytest

from utils.metrics import classification_metrics, detection_ap50


def test_classification_metrics_perfect() -> None:
    metrics = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], ["a", "b"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_detection_ap_perfect() -> None:
    targets = [{"image_id": "image", "class_id": 0, "box": [0, 0, 10, 10]}]
    predictions = [
        {
            "image_id": "image",
            "class_id": 0,
            "box": [0, 0, 10, 10],
            "confidence": 0.9,
        }
    ]
    metrics = detection_ap50(targets, predictions, ["object"])
    assert metrics["map50"] == pytest.approx(1.0)


def test_detection_ap_penalizes_high_confidence_false_positive() -> None:
    targets = [{"image_id": "image", "class_id": 0, "box": [0, 0, 10, 10]}]
    predictions = [
        {"image_id": "image", "class_id": 0, "box": [20, 20, 30, 30], "confidence": 0.9},
        {"image_id": "image", "class_id": 0, "box": [0, 0, 10, 10], "confidence": 0.8},
    ]
    metrics = detection_ap50(targets, predictions, ["object"])
    assert metrics["map50"] == pytest.approx(0.5)

