# Repository map

## Configuration

- `configs/experiment.yaml`: complete protocol and all external paths.
- `configs/classes.yaml`: fine-grained class order used by linear heads.
- `configs/detection.example.yaml`: minimal Ultralytics dataset example.

## Data preparation

- `src/data/prepare_detection_data.py`: group-aware detector splits, type-label
  remapping, leakage-safe SSL list, and manifest digests.
- `src/data/prepare_probe_index.py`: crop validation, duplicate filtering, and
  deterministic within-type training balance.

## Models and training

- `src/models/grn_yolo.py`: GRN implementation, YAML rewriting, backbone
  extraction, and strict encoder transfer.
- `src/models/feature_backbones.py`: frozen DINOv3, DINOv2, and supervised
  feature extractors.
- `src/train_ssl.py`: SimSiam, MoCo v2, BYOL, and FCMAE optimization.
- `src/train_detector.py`: baseline or SSL-initialized full-network detector
  fine-tuning.
- `src/train_linear_probe.py`: type-specific frozen-feature linear probes.
- `src/run_*_queue.py`: resumable GPU scheduling.

## Inference and evaluation

- `src/evaluation/evaluate_detector.py`: component-level detector evaluation.
- `src/evaluation/evaluate_probe.py`: final-epoch crop-classifier evaluation.
- `src/pipeline/run_pipeline.py`: deployable and controlled two-stage inference.
- `src/evaluation/evaluate_pipeline.py`: matching, metrics, and ordered error
  attribution.
- `src/robustness/make_weather.py`: deterministic weather generation.
- `src/robustness/evaluate_weather.py`: component evaluation under degradation.

## Verification

- `tests/test_grn_yolo.py`: stage insertion, forward shape, and checkpoint
  transfer tests.
- `tests/test_geometry.py`: box conversion, crop expansion, IoU, and matching.
- `tests/test_metrics.py`: classification metric behavior.
- `scripts/`: commands in experimental order.

