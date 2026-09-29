#!/usr/bin/env python3
"""Evaluate two-stage records and attribute deployable-pipeline errors."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config, workspace_path
from utils.geometry import greedy_match
from utils.io import read_jsonl, write_json
from utils.metrics import detection_ap50


SETTINGS = ("oracle", "predicted_boxes_gt_type", "deployed")


def flatten_records(records: list[dict[str, Any]], setting: str) -> tuple[list[dict], list[dict]]:
    targets, predictions = [], []
    for record in records:
        image_id = record["image"]
        for target in record["targets"]:
            targets.append({**target, "image_id": image_id})
        for prediction in record["predictions"][setting]:
            predictions.append(
                {
                    **prediction,
                    "image_id": image_id,
                    "class_id": int(prediction["pred_fine_id"]),
                }
            )
    return targets, predictions


def instance_metrics(
    records: list[dict[str, Any]],
    setting: str,
    class_names: list[str],
    threshold: float,
) -> dict[str, Any]:
    true_positive = Counter()
    false_positive = Counter()
    false_negative = Counter()
    for record in records:
        targets = record["targets"]
        predictions = [
            {
                **prediction,
                "class_id": int(prediction["pred_fine_id"]),
            }
            for prediction in record["predictions"][setting]
        ]
        matches = greedy_match(predictions, targets, threshold, class_aware=True)
        matched_targets = set(matches.values())
        for prediction_index, prediction in enumerate(predictions):
            class_id = int(prediction["class_id"])
            if prediction_index in matches:
                true_positive[class_id] += 1
            else:
                false_positive[class_id] += 1
        for target_index, target in enumerate(targets):
            if target_index not in matched_targets:
                false_negative[int(target["class_id"])] += 1
    per_class = {}
    f1_values = []
    for class_id, class_name in enumerate(class_names):
        tp = true_positive[class_id]
        fp = false_positive[class_id]
        fn = false_negative[class_id]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_values.append(f1)
    total_tp = sum(true_positive.values())
    total_fp = sum(false_positive.values())
    total_fn = sum(false_negative.values())
    return {
        "precision": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "recall": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def error_attribution(records: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    errors = Counter()
    routing_by_type = defaultdict(Counter)
    type_confusion = defaultdict(Counter)
    pi_ci = 0
    routing_total = 0
    for record in records:
        targets = record["targets"]
        predictions = record["predictions"]["deployed"]
        matches = greedy_match(predictions, targets, threshold, class_aware=False)
        matched_targets = set(matches.values())
        errors["false_positive"] += len(predictions) - len(matches)
        errors["miss"] += len(targets) - len(matched_targets)
        for prediction_index, target_index in matches.items():
            prediction = predictions[prediction_index]
            target = targets[target_index]
            ground_truth_type = target["type"]
            predicted_type = prediction["pred_type"]
            type_confusion[ground_truth_type][predicted_type] += 1
            routing_by_type[ground_truth_type]["localized"] += 1
            if predicted_type != ground_truth_type:
                errors["routing"] += 1
                routing_by_type[ground_truth_type]["routing_error"] += 1
                routing_total += 1
                if {predicted_type, ground_truth_type} == {"PI", "CI"}:
                    pi_ci += 1
            elif prediction["pred_fine"] != target["fine"]:
                errors["classification"] += 1
            else:
                errors["correct"] += 1
    routing_rates = {
        type_name: (
            values["routing_error"] / values["localized"]
            if values["localized"]
            else 0.0
        )
        for type_name, values in routing_by_type.items()
    }
    return {
        "counts": dict(errors),
        "routing_error_rate_by_type": routing_rates,
        "pi_ci_fraction_of_routing_errors": pi_ci / routing_total if routing_total else 0.0,
        "type_confusion": {
            target: dict(predicted) for target, predicted in type_confusion.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = args.seeds or config["pipeline"]["seeds"]
    threshold = float(config["pipeline"]["matching_iou_threshold"])
    class_names = []
    for type_name in ("GI", "PI", "CI"):
        class_names.extend(config["classes"]["fine"][type_name])
    output_root = workspace_path(config, "evaluation", "pipeline")
    for seed in seeds:
        source = output_root / f"pred_s{seed}.jsonl"
        destination = output_root / f"metrics_s{seed}.json"
        if destination.exists() and not args.force:
            print(f"SKIP {destination}")
            continue
        records = read_jsonl(source)
        settings = {}
        for setting in SETTINGS:
            targets, predictions = flatten_records(records, setting)
            settings[setting] = {
                "detection": detection_ap50(targets, predictions, class_names, threshold),
                "instances": instance_metrics(records, setting, class_names, threshold),
            }
        write_json(
            destination,
            {
                "seed": int(seed),
                "iou_threshold": threshold,
                "settings": settings,
                "error_attribution": error_attribution(records, threshold),
            },
        )
        print(f"SAVED {destination}")


if __name__ == "__main__":
    main()
