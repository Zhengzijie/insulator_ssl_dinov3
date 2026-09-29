# Fine-Grained Detection Framework for Multi-Type Insulator Defects

Reproducibility package for the study:

> **Fine-Grained Detection Framework for Multi-Type Insulator Defects Based on
> Self-Supervised Learning Enhancement and DINOv3 Feature Extraction**

This repository documents the complete experimental workflow without
redistributing images, annotations, checkpoints, logs, or reported numerical
results. All paths are configurable and all generated artifacts are written
outside the source tree.

## Method overview

The framework separates localization/type recognition from fine-grained defect
recognition:

```text
unlabelled full images
        │
        ├── SimSiam / MoCo v2 / BYOL / FCMAE pretraining
        │                     │
        │             YOLO backbone + GRN
        │                     │
labelled full images ── detector fine-tuning ── boxes + GI/PI/CI type
                                                  │
                                                  ├── crop and expand 10%
                                                  │
                                                  └── type-routed DINOv3
                                                      frozen encoder
                                                           │
                                               type-specific linear head
                                                           │
                                                11 fine-grained states
```

The detector predicts three insulator types. A predicted box is expanded,
clipped to the image boundary, resized, and routed by its predicted type to one
of three linear classifiers. The classifiers share a frozen DINOv3 ViT-B/16
feature extractor and predict 3, 4, and 4 subclasses respectively.

## Repository contents

```text
configs/                 experiment settings and data-format examples
docs/                    protocol, data layout, and reproducibility notes
scripts/                 ordered shell entry points
src/data/                deterministic indexes and group-aware splits
src/models/              GRN-YOLO and frozen feature extractors
src/                     SSL, detector, and linear-probe training
src/pipeline/            two-stage end-to-end inference
src/evaluation/          detector, probe, and pipeline evaluation
src/robustness/          deterministic fog/rain/glare stress testing
src/utils/               common geometry, metrics, and seeding utilities
tests/                   architecture and metric checks
```

Detailed file descriptions are provided in
[`docs/REPOSITORY_MAP.md`](docs/REPOSITORY_MAP.md).

## Installation

```bash
conda env create -f environment.yml
conda activate insulator-repro
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
```

Place the official DINOv3 ViT-B/16 checkpoint outside the repository and set
its path in `configs/experiment.yaml`. DINOv3 weights are not redistributed.

## Data preparation

The repository accepts source images and YOLO labels without modifying them.
Generated indexes, remapped labels, and symbolic links are written under a
separate workspace.

1. Copy `configs/experiment.yaml` and edit only its path section.
2. Build the deterministic detection split and SSL image list:

   ```bash
   python src/data/prepare_detection_data.py --config configs/experiment.yaml
   ```

3. Build the crop-level index used by the linear probes:

   ```bash
   python src/data/prepare_probe_index.py --config configs/experiment.yaml
   ```

Expected source layouts and label conventions are described in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

## Training sequence

### Stage 1: self-supervised backbone pretraining

```bash
bash scripts/01_pretrain_ssl.sh configs/experiment.yaml
```

The queue trains each requested method/backbone combination. SSL checkpoints
contain only the encoder state. Effective global batch sizes remain fixed when
the number of visible GPUs changes; gradient accumulation supplies the
difference.

### Stage 2: detector fine-tuning

```bash
bash scripts/02_finetune_detectors.sh configs/experiment.yaml
```

Every detector is constructed from YAML with random initialization. SSL runs
receive a strict key-for-key backbone transfer; the neck and head are random.
The code verifies that the transferred backbone survives the Ultralytics
trainer handoff before the first epoch. All detector parameters are then
optimized.

### Stage 3: frozen DINOv3 linear probes

```bash
bash scripts/03_train_probes.sh configs/experiment.yaml
```

The feature extractor remains frozen in evaluation mode. Only a linear head is
trained for each type and random seed. The same workflow also supports DINOv2,
supervised ViT-B/16, ConvNeXt-B, ResNet-50, DINOv3 intermediate blocks, and the
concatenation of four DINOv3 class tokens.

## Evaluation sequence

```bash
bash scripts/04_evaluate_components.sh configs/experiment.yaml
bash scripts/05_run_pipeline.sh configs/experiment.yaml
bash scripts/06_weather_stress_test.sh configs/experiment.yaml
```

Component evaluation records detector AP and fine-grained classification
metrics. Pipeline evaluation compares three settings:

1. ground-truth boxes with ground-truth type routing;
2. predicted boxes with ground-truth type routing for matched detections;
3. deployable predicted boxes with predicted type routing.

The comparison separates localization, routing, and fine-grained
classification errors. The weather workflow applies deterministic synthetic
fog, rain, and sun glare without changing box geometry.

## Reproducibility safeguards

- Relative paths are stored in generated manifests.
- Source datasets are opened read-only and never renamed or overwritten.
- Seeds control model initialization, sampling order, and augmentation.
- Group-aware detector splits keep a collection group in one subset.
- Train/validation duplicate checks use file digests.
- SSL-to-detector transfer requires exact keys and tensor shapes.
- Existing completed checkpoints are skipped; interrupted runs resume.
- Each inference record retains boxes, type confidence, routed subclass, and
  subclass probability.
- Weather random state is derived from the relative image path.

See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the full protocol and
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for verification commands.

## Scope of the public package

The package contains methods and evaluation code only. It intentionally omits
the study images, annotations, model weights, machine-specific paths, numerical
result tables, statistical conclusions, and manuscript figures.
