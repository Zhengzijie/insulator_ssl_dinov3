"""YOLO backbone construction, GRN insertion, and strict weight transfer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


MODEL_FILES = {
    "yolov8s": "yolov8s.yaml",
    "yolov10s": "yolov10s.yaml",
    "yolo12s": "yolo12s.yaml",
}
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


class GRN(nn.Module):
    """ConvNeXt V2 global response normalization for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


def register_grn() -> None:
    """Expose GRN in each namespace used by the Ultralytics YAML parser."""
    import ultralytics.nn as ultralytics_nn
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    ultralytics_nn.GRN = GRN
    modules.GRN = GRN
    tasks.GRN = GRN


def _make_divisible(value: float, divisor: int = 8) -> int:
    return int(math.ceil(value / divisor) * divisor)


def _scaled_channels(config: dict[str, Any], layer: list[Any]) -> int:
    base = int(layer[3][0])
    if config.get("scales"):
        _, width, max_channels = config["scales"][config["scale"]]
    else:
        width = config.get("width_multiple", 1.0)
        max_channels = float("inf")
    return _make_divisible(min(base, max_channels) * width, 8)


def _is_stride2(layer: list[Any]) -> bool:
    module, args = str(layer[2]), layer[3]
    return module in {"Conv", "SCDown", "ADown"} and len(args) >= 3 and int(args[2]) == 2


def _translate_ref(
    reference: int,
    old_index: int,
    output_index: dict[int, int],
    raw_index: dict[int, int],
) -> int:
    if reference == -1:
        return -1
    old_target = reference if reference >= 0 else old_index + reference
    if old_target < 0:
        return reference
    return output_index.get(old_target, raw_index[old_target])


def generate_grn_yaml(model_name: str, output_dir: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Insert GRN after the four backbone stages and rewrite layer references."""
    if model_name not in MODEL_FILES:
        raise KeyError(f"Unsupported model {model_name!r}; choose from {sorted(MODEL_FILES)}")
    register_grn()
    from ultralytics.nn.tasks import yaml_model_load

    config = copy.deepcopy(yaml_model_load(MODEL_FILES[model_name]))
    config["scale"] = "s"
    old_backbone = config["backbone"]
    old_head = config["head"]
    stage_ends = [
        index - 1
        for index, layer in enumerate(old_backbone)
        if index >= 3 and _is_stride2(layer)
    ]
    stage_ends.append(len(old_backbone) - 1)
    stage_ends = sorted(set(index for index in stage_ends if index >= 0))
    if len(stage_ends) != 4:
        raise RuntimeError(f"Expected four backbone stages for {model_name}; found {stage_ends}")

    new_backbone: list[list[Any]] = []
    raw_index: dict[int, int] = {}
    output_index: dict[int, int] = {}
    stage_channels: dict[int, int] = {}
    for old_index, original in enumerate(old_backbone):
        layer = copy.deepcopy(original)
        references = layer[0] if isinstance(layer[0], list) else [layer[0]]
        translated = [
            _translate_ref(int(reference), old_index, output_index, raw_index)
            for reference in references
        ]
        layer[0] = translated if isinstance(layer[0], list) else translated[0]
        raw_index[old_index] = len(new_backbone)
        new_backbone.append(layer)
        output_index[old_index] = raw_index[old_index]
        if old_index in stage_ends:
            channels = _scaled_channels(config, original)
            new_backbone.append([-1, 1, "GRN", [channels]])
            output_index[old_index] = len(new_backbone) - 1
            stage_channels[old_index] = channels

    new_head: list[list[Any]] = []
    old_backbone_length = len(old_backbone)
    offset = len(new_backbone) - old_backbone_length
    for head_position, original in enumerate(old_head):
        old_index = old_backbone_length + head_position
        layer = copy.deepcopy(original)
        references = layer[0] if isinstance(layer[0], list) else [layer[0]]
        translated = []
        for reference in references:
            reference = int(reference)
            if reference == -1:
                translated.append(-1)
                continue
            old_target = reference if reference >= 0 else old_index + reference
            translated.append(
                output_index[old_target]
                if old_target < old_backbone_length
                else old_target + offset
            )
        layer[0] = translated if isinstance(layer[0], list) else translated[0]
        new_head.append(layer)

    config["backbone"] = new_backbone
    config["head"] = new_head
    metadata = {
        "source_yaml": MODEL_FILES[model_name],
        "original_backbone_layers": old_backbone_length,
        "backbone_layers_with_grn": len(new_backbone),
        "stage_output_old_indices": stage_ends,
        "stage_output_new_indices": [output_index[index] for index in stage_ends],
        "stage_channels": stage_channels,
    }
    config["grn_metadata"] = metadata
    destination = (output_dir or GENERATED_DIR).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    yaml_path = destination / f"{model_name}_grn.yaml"
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if not yaml_path.exists() or yaml_path.read_text(encoding="utf-8") != rendered:
        yaml_path.write_text(rendered, encoding="utf-8")
    return yaml_path, metadata


def build_yolo(model_name: str, verbose: bool = True):
    """Construct a detector from YAML without loading pretrained weights."""
    register_grn()
    from ultralytics import YOLO

    yaml_path, metadata = generate_grn_yaml(model_name)
    wrapper = YOLO(str(yaml_path), verbose=verbose)
    wrapper.model.args["pretrained"] = False
    wrapper.model.grn_metadata = metadata
    return wrapper


def backbone_length(model: Any) -> int:
    core = model.model if hasattr(model, "model") and hasattr(model.model, "model") else model
    metadata = getattr(core, "grn_metadata", None)
    if metadata:
        return int(metadata["backbone_layers_with_grn"])
    return len(getattr(core, "yaml", {})["backbone"])


def extract_backbone(model: Any) -> nn.Sequential:
    """Return the complete YAML-defined backbone, including GRN modules."""
    core = model.model if hasattr(model, "model") and hasattr(model.model, "model") else model
    return nn.Sequential(*list(core.model[: backbone_length(model)]))


def load_backbone(model: Any, checkpoint_path: str | Path) -> dict[str, Any]:
    """Transfer an encoder only when all keys and tensor shapes match exactly."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("state_dict", checkpoint)
    target_module = extract_backbone(model)
    target = target_module.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    shape_mismatch = sorted(
        key for key in set(target) & set(source) if target[key].shape != source[key].shape
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "matched": len(target) - len(missing) - len(shape_mismatch),
        "target_keys": len(target),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }
    print("BACKBONE_LOAD " + json.dumps(report, sort_keys=True), flush=True)
    if missing or unexpected or shape_mismatch:
        raise RuntimeError("Backbone checkpoint is not an exact key-for-key match")
    target_module.load_state_dict(source, strict=True)
    return report


if __name__ == "__main__":
    reports = []
    for model_name in MODEL_FILES:
        path, metadata = generate_grn_yaml(model_name)
        reports.append({"model": model_name, "yaml": str(path), **metadata})
    print(json.dumps(reports, indent=2))

