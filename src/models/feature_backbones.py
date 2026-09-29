"""Frozen feature extractors used by the linear-probe training workflow."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


BACKBONES = (
    "dinov3",
    "dinov2",
    "vitb16_in21k",
    "convnextb_in1k",
    "resnet50_in1k",
    "dinov3_b3",
    "dinov3_b6",
    "dinov3_b9",
    "dinov3_b12",
    "dinov3_concat",
)

FEATURE_DIMS = {
    "dinov3": 768,
    "dinov2": 768,
    "vitb16_in21k": 768,
    "convnextb_in1k": 1024,
    "resnet50_in1k": 2048,
    "dinov3_b3": 768,
    "dinov3_b6": 768,
    "dinov3_b9": 768,
    "dinov3_b12": 768,
    "dinov3_concat": 3072,
}


class DINOv3Features(nn.Module):
    def __init__(self, model: nn.Module, variant: str):
        super().__init__()
        self.model = model
        self.variant = variant

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.variant == "dinov3":
            return self.model(images)
        outputs = self.model.get_intermediate_layers(
            images,
            n=[2, 5, 8, 11],
            return_class_token=True,
            norm=True,
        )
        class_tokens = [entry[1] for entry in outputs]
        if self.variant == "dinov3_concat":
            return torch.cat(class_tokens, dim=1)
        position = {
            "dinov3_b3": 0,
            "dinov3_b6": 1,
            "dinov3_b9": 2,
            "dinov3_b12": 3,
        }[self.variant]
        return class_tokens[position]


class DirectFeatures(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        if isinstance(output, dict):
            output = output.get("x_norm_clstoken", output.get("pooler_output"))
        if output.ndim > 2:
            output = output.flatten(2).mean(dim=2)
        return output


@dataclass
class LoadedBackbone:
    extractor: nn.Module
    feature_dim: int
    source: str


def _dinov3_compatibility_shim() -> None:
    """Support DINOv3 hub imports on older torch releases."""
    if not hasattr(torch.amp, "custom_fwd"):
        torch.amp.custom_fwd = lambda *args, **kwargs: (lambda function: function)  # type: ignore[attr-defined]
        torch.amp.custom_bwd = lambda *args, **kwargs: (lambda function: function)  # type: ignore[attr-defined]


def load_backbone(
    name: str,
    cache_dir: Path,
    dinov3_repo: Path | None = None,
    dinov3_weight: Path | None = None,
) -> LoadedBackbone:
    """Load and freeze one official pretrained feature extractor."""
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone {name!r}")
    cache_dir = cache_dir.expanduser().resolve()
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_dir / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if name.startswith("dinov3"):
        if dinov3_repo is None or dinov3_weight is None:
            raise ValueError("DINOv3 variants require --dinov3-repo and --dinov3-weight")
        repository = dinov3_repo.expanduser().resolve()
        weight = dinov3_weight.expanduser().resolve()
        if not (repository / "hubconf.py").is_file():
            raise FileNotFoundError(f"Invalid DINOv3 repository: {repository}")
        if not weight.is_file():
            raise FileNotFoundError(f"Missing DINOv3 checkpoint: {weight}")
        _dinov3_compatibility_shim()
        model = torch.hub.load(
            str(repository),
            "dinov3_vitb16",
            source="local",
            weights=str(weight),
        )
        extractor: nn.Module = DINOv3Features(model, name)
        source = "facebookresearch/dinov3 dinov3_vitb16"
    elif name == "dinov2":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        extractor = DirectFeatures(model)
        source = "facebookresearch/dinov2 dinov2_vitb14"
    else:
        import timm

        timm_name = {
            "vitb16_in21k": "vit_base_patch16_224.augreg_in21k",
            "convnextb_in1k": "convnext_base.fb_in1k",
            "resnet50_in1k": "resnet50.a1_in1k",
        }[name]
        model = timm.create_model(timm_name, pretrained=True, num_classes=0)
        extractor = DirectFeatures(model)
        source = f"timm {timm_name}"

    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    extractor.eval()
    return LoadedBackbone(extractor, FEATURE_DIMS[name], source)

