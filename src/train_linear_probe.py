#!/usr/bin/env python3
"""Train a linear head on top of a frozen pretrained feature extractor."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

from models.feature_backbones import BACKBONES, load_backbone


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_CLASSES = Path(__file__).resolve().parents[1] / "configs" / "classes.yaml"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


class CropDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_root: Path,
        classes: tuple[str, ...],
    ):
        self.rows = rows
        self.data_root = data_root
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                    ratio=(3 / 4, 4 / 3),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = Path(row["path"]).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_index[row["subclass"]]


def read_training_rows(
    index_csv: Path,
    type_name: str,
    classes: tuple[str, ...],
) -> list[dict[str, str]]:
    with index_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "type", "subclass", "split"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Index CSV is missing columns: {sorted(missing)}")
        rows = [
            row
            for row in reader
            if row["type"] == type_name and row["split"].lower() == "train"
        ]
    if not rows:
        raise RuntimeError(f"No training rows for type {type_name}")
    unknown = sorted({row["subclass"] for row in rows} - set(classes))
    if unknown:
        raise ValueError(f"Unknown subclasses for {type_name}: {unknown}")
    return rows


def worker_seed(_worker_id: int) -> None:
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)


def extract_features(extractor: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        return extractor(images).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--type", required=True, choices=("GI", "PI", "CI"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--index-csv", required=True, type=Path)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--dinov3-repo", type=Path)
    parser.add_argument("--dinov3-weight", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.device}")
    class_config = yaml.safe_load(args.classes.expanduser().resolve().read_text(encoding="utf-8"))
    classes = tuple(class_config[args.type])
    rows = read_training_rows(args.index_csv.expanduser().resolve(), args.type, classes)

    output_dir = args.output_dir.expanduser().resolve() / f"{args.backbone}_{args.type}_s{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last.pt"
    if checkpoint_path.exists() and not args.force:
        existing = torch.load(checkpoint_path, map_location="cpu")
        if int(existing.get("epoch", 0)) >= args.epochs:
            print(f"SKIP completed checkpoint: {checkpoint_path}")
            return

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        CropDataset(rows, args.data_root.expanduser().resolve(), classes),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    loaded = load_backbone(
        args.backbone,
        cache_dir=args.cache_dir,
        dinov3_repo=args.dinov3_repo,
        dinov3_weight=args.dinov3_weight,
    )
    extractor = loaded.extractor.to(device).eval()
    head = nn.Linear(loaded.feature_dim, len(classes)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 0

    if checkpoint_path.exists() and not args.force:
        state = torch.load(checkpoint_path, map_location="cpu")
        identity = (state.get("backbone"), state.get("type"), state.get("seed"))
        if identity != (args.backbone, args.type, args.seed):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
        head.load_state_dict(state["head"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        generator.set_state(state["loader_rng"])
        print(f"RESUME epoch={start_epoch} from {checkpoint_path}")

    for epoch in range(start_epoch, args.epochs):
        extractor.eval()
        head.train()
        running_loss = 0.0
        examples = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = head(extract_features(extractor, images))
            loss = nn.functional.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(labels)
            examples += len(labels)
        scheduler.step()
        torch.save(
            {
                "epoch": epoch + 1,
                "backbone": args.backbone,
                "backbone_source": loaded.source,
                "feature_dim": loaded.feature_dim,
                "type": args.type,
                "classes": classes,
                "seed": args.seed,
                "head": head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "loader_rng": generator.get_state(),
            },
            checkpoint_path,
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={running_loss / examples:.8f} "
            f"lr={optimizer.param_groups[0]['lr']:.8g}",
            flush=True,
        )
    print(f"SAVED final-epoch linear head: {checkpoint_path}")


if __name__ == "__main__":
    main()

