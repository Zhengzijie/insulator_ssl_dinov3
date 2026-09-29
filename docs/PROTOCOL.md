# Experimental protocol

## 1. Detection data and self-supervised pool

The detection task uses three object types: GI, PI, and CI. Original fine-level
labels are remapped to the type identifier while preserving normalized YOLO box
coordinates. Remapped labels are written to the workspace; source labels remain
unchanged.

Images are sorted by relative path before splitting. Collection groups are
derived in the following order:

1. an acquisition date and hour embedded in a DJI-style filename;
2. the filename prefix before the final underscore;
3. a deterministic contiguous-block fallback.

The first grouping rule that provides adequate group diversity without one
dominant group is selected. GI, PI, and CI images are independently split by
group into 70% training, 15% validation, and 15% testing with random state 0.
No group may appear in more than one subset.

The SSL pool is assembled from unlabelled images. An SSL image is excluded when
its file digest matches a test image or its collection group occurs in the test
set. Split files contain paths relative to the configured data root.

## 2. GRN-augmented YOLO backbone

YOLOv8s, YOLOv10s, and YOLO12s are constructed from their Ultralytics YAML
definitions without pretrained detector weights. The YAML backbone section
defines the encoder boundary. Global response normalization is inserted after
the output of every backbone stage:

```text
Gx = ||x||2 over spatial dimensions
Nx = Gx / mean_channel(Gx)
y  = gamma * (x * Nx) + beta + x
```

`gamma` and `beta` are initialized to zero, so a newly inserted GRN layer is an
identity mapping. Absolute references in the detector head are rewritten after
insertion. The backbone output is the stride-32 feature map; representation
learning methods apply global average pooling when they require a vector.

## 3. Self-supervised pretraining

All methods use 224-pixel inputs, 100 epochs, automatic mixed precision, and a
cosine learning-rate schedule. Two independently augmented views are created
with random resized crop, horizontal flip, color jitter, and Gaussian blur.

### SimSiam

- three-layer 2048-dimensional projector;
- two-layer predictor with a 512-dimensional bottleneck;
- symmetric negative cosine similarity;
- stop-gradient on the target branch;
- SGD, momentum 0.9, weight decay 1e-4;
- learning rate 0.05 and global batch 32.

### MoCo v2

- 2048-to-128 projector;
- momentum key encoder with coefficient 0.999;
- queue length 4096 and temperature 0.2;
- symmetric InfoNCE;
- SGD, momentum 0.9, weight decay 1e-4;
- learning rate 0.03 and global batch 256.

### BYOL

- online and target encoders;
- 4096-dimensional hidden projector and 256-dimensional output;
- online predictor with the same bottleneck width;
- target momentum 0.996;
- SGD, momentum 0.9, weight decay 1e-4;
- learning rate 0.2 and global batch 256.

### FCMAE

- 60% spatial masking on a stride-32 grid;
- fully convolutional 1x1 decoder to normalized pixel patches;
- AdamW with learning rate 1.5e-4 and weight decay 0.05;
- global batch 64.

The FCMAE implementation uses dense masked inputs because the selected YOLO
modules do not provide sparse-convolution equivalents. Only the trained encoder
is retained after each SSL run.

## 4. Detector fine-tuning

Each detector/initialization pair is trained with five seeds. Baseline runs use
the same GRN architecture with random initialization. For SSL runs, every
encoder key and tensor shape must match before transfer. The neck and detection
head remain random. The entire network is trainable.

Fine-tuning uses 640-pixel images, 100 epochs, batch 16, requested initial
learning rate 0.01, the Ultralytics automatic optimizer, mixed precision, and
deterministic seed handling. The checkpoint with maximum validation AP at IoU
0.5 is retained for component evaluation.

## 5. Frozen feature extraction and linear probes

The crop classifier compares these frozen feature extractors:

- DINOv3 ViT-B/16;
- DINOv2 ViT-B/14;
- supervised ViT-B/16 trained on ImageNet-21k;
- supervised ConvNeXt-B trained on ImageNet-1k;
- supervised ResNet-50 trained on ImageNet-1k.

DINOv3 block 3, 6, 9, and 12 class tokens are also evaluated, along with their
concatenation. Feature dimensions are checked at runtime.

Training augmentation uses a random resized crop with scale 0.8–1.0,
horizontal and vertical flips, and ImageNet normalization. The validation
transform uses deterministic resize and the same normalization. The backbone
is always frozen, in evaluation mode, and executed without gradients in FP16.

One linear head is trained per type: GI has three outputs, PI has four, and CI
has four. AdamW uses learning rate 5e-4 and weight decay 0.01 for 50 epochs with
batch 64 and cosine scheduling. The final epoch is used without validation
checkpoint selection.

## 6. Two-stage inference

The detector runs at 640 pixels with confidence threshold 0.25, NMS IoU 0.70,
and a maximum of 300 boxes. Each box is expanded by 10%, clipped to image
boundaries, and discarded if either crop side is below 32 pixels. The crop is
resized to 224 pixels and routed to the linear head selected by predicted type.

Predictions are matched to ground truth greedily in descending confidence order
at IoU 0.5. Each ground-truth instance can be matched once. Error attribution
uses this order:

1. unmatched prediction: detection false positive;
2. unmatched target: detection miss;
3. matched box with incorrect type: routing error;
4. correct type with incorrect fine class: classification error;
5. otherwise: correct prediction.

Ground-truth-box/ground-truth-type and predicted-box/ground-truth-type controls
isolate classifier and localization effects from deployable routing.

## 7. Synthetic weather stress test

Fog, rain, and sun glare are applied independently to full images and crops.
The random seed is the CRC32 value of the relative path, ensuring that all model
seeds observe the same degraded image. The transformations do not alter box
geometry, so the original annotations remain valid.

Component evaluation is repeated for clean and degraded inputs using identical
checkpoints and inference thresholds. Generated images are written only to the
workspace.

