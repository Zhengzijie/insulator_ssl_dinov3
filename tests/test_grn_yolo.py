from __future__ import annotations

import torch
import pytest

from models.grn_yolo import GRN, build_yolo, extract_backbone, load_backbone


@pytest.mark.parametrize(
    ("model_name", "expected_layers"),
    (("yolov8s", 14), ("yolov10s", 15), ("yolo12s", 13)),
)
def test_backbone_construction_and_stride(model_name: str, expected_layers: int) -> None:
    wrapper = build_yolo(model_name, verbose=False)
    backbone = extract_backbone(wrapper).eval()
    assert len(backbone) == expected_layers
    assert sum(isinstance(module, GRN) for module in backbone) == 4
    with torch.no_grad():
        output = backbone(torch.zeros(1, 3, 224, 224))
    assert output.shape[-2:] == (7, 7)


@pytest.mark.parametrize("model_name", ("yolov8s", "yolov10s", "yolo12s"))
def test_strict_encoder_round_trip(model_name: str, tmp_path) -> None:
    source = build_yolo(model_name, verbose=False)
    checkpoint = tmp_path / f"{model_name}.pt"
    torch.save({"state_dict": extract_backbone(source).state_dict()}, checkpoint)
    target = build_yolo(model_name, verbose=False)
    report = load_backbone(target, checkpoint)
    assert report["matched"] == report["target_keys"]
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["shape_mismatch"] == []


def test_grn_is_identity_at_initialization() -> None:
    layer = GRN(8)
    sample = torch.randn(2, 8, 5, 5)
    assert torch.equal(layer(sample), sample)

