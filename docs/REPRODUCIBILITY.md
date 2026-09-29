# Reproducibility checklist

## Environment

```bash
conda env create -f environment.yml
conda activate insulator-repro
python -m pytest -q
nvidia-smi
```

Record the Python, PyTorch, CUDA, cuDNN, Ultralytics, timm, and Albumentations
versions alongside each run. The scripts store training settings in their
checkpoints and write command logs to the configured workspace.

## Data integrity

```bash
python src/data/prepare_detection_data.py --config configs/experiment.yaml
python src/data/prepare_probe_index.py --config configs/experiment.yaml
```

Keep the generated manifest with the experiment. Split files use relative paths
and manifest entries include file digests. Re-running preparation with unchanged
inputs and configuration must produce the same split-file digests.

## Architecture checks

```bash
python src/models/grn_yolo.py
python -m pytest tests/test_grn_yolo.py -q
```

The test constructs every detector from YAML, verifies four GRN insertions,
checks stride-32 output, and performs a strict encoder round trip.

## Training checks

Run a short structural check before full optimization:

```bash
python src/train_ssl.py --help
python src/train_detector.py --help
python src/train_linear_probe.py --help
```

For SSL-initialized detectors, retain `ssl_init_report.json`. A valid report has
no missing, unexpected, shape-mismatched, or trainer-handoff-unequal keys.

## Evaluation checks

Component checkpoints must not be selected using the test set. Detector
checkpoint selection uses validation AP at IoU 0.5. Linear probes use the final
epoch. End-to-end records should be immutable JSON Lines files so matching and
metric calculations can be repeated without new model inference.

