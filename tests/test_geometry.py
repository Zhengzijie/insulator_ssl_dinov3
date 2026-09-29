from __future__ import annotations

import pytest

from utils.geometry import box_iou, expanded_bounds, greedy_match, yolo_to_xyxy


def test_yolo_conversion() -> None:
    assert yolo_to_xyxy(0.5, 0.5, 0.5, 0.5, 200, 100) == [50.0, 25.0, 150.0, 75.0]


def test_iou() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_expansion_clips_to_image() -> None:
    assert expanded_bounds([0, 0, 20, 20], 100, 100, 0.1) == (0, 0, 22, 22)


def test_greedy_order_uses_confidence() -> None:
    predictions = [
        {"box": [0, 0, 10, 10], "confidence": 0.2, "class_id": 0},
        {"box": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0},
    ]
    targets = [{"box": [0, 0, 10, 10], "class_id": 0}]
    assert greedy_match(predictions, targets) == {1: 0}

