#!/usr/bin/env python3
"""Self-supervised pretraining for GRN-augmented YOLO backbones."""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from models.grn_yolo import build_yolo, extract_backbone


GLOBAL_BATCH = {"simsiam": 32, "mocov2": 256, "byol": 256, "fcmae": 64}


class TwoViews:
    def __init__(self, size: int = 224):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image), self.transform(image)


class SSLImages(Dataset):
    def __init__(self, data_root: Path, list_file: Path, pad_to: int):
        lines = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.paths = [
            candidate if (candidate := Path(line).expanduser()).is_absolute() else data_root / candidate
            for line in lines
        ]
        if not self.paths:
            raise RuntimeError(f"Empty SSL image list: {list_file}")
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Image from SSL list does not exist: {missing[0]}")
        self.real_length = len(self.paths)
        self.padded_length = math.ceil(self.real_length / pad_to) * pad_to
        self.augment = TwoViews(224)

    def __len__(self) -> int:
        return self.padded_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index % self.real_length]) as image:
            return self.augment(image.convert("RGB"))


class SimSiam(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048, bias=False),
            nn.BatchNorm1d(2048, affine=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images).mean(dim=(2, 3))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        z1, z2 = self.projector(self.encode(first)), self.projector(self.encode(second))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return -0.5 * (
            F.cosine_similarity(p1, z2.detach(), dim=1).mean()
            + F.cosine_similarity(p2, z1.detach(), dim=1).mean()
        )


class MoCoV2(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        queue_size: int = 4096,
        momentum: float = 0.999,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128),
        )
        self.key_encoder = copy.deepcopy(encoder)
        self.key_projector = copy.deepcopy(self.projector)
        for parameter in list(self.key_encoder.parameters()) + list(self.key_projector.parameters()):
            parameter.requires_grad_(False)
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("queue", F.normalize(torch.randn(128, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @staticmethod
    def pool(features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(2, 3))

    @torch.no_grad()
    def momentum_update(self) -> None:
        for query, key in zip(self.encoder.parameters(), self.key_encoder.parameters()):
            key.data.mul_(self.momentum).add_(query.data, alpha=1.0 - self.momentum)
        for query, key in zip(self.projector.parameters(), self.key_projector.parameters()):
            key.data.mul_(self.momentum).add_(query.data, alpha=1.0 - self.momentum)

    def contrastive_loss(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        positive = torch.einsum("nc,nc->n", query, key).unsqueeze(1)
        negative = torch.einsum("nc,ck->nk", query, self.queue.detach().clone())
        logits = torch.cat((positive, negative), dim=1) / self.temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, targets)

    @torch.no_grad()
    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor) -> None:
        keys = self.gather(keys)
        size = self.queue.shape[1]
        pointer = int(self.queue_ptr.item())
        if len(keys) >= size:
            self.queue.copy_(keys[-size:].T)
            self.queue_ptr.zero_()
            return
        end = pointer + len(keys)
        if end <= size:
            self.queue[:, pointer:end] = keys.T
        else:
            first = size - pointer
            self.queue[:, pointer:] = keys[:first].T
            self.queue[:, : end - size] = keys[first:].T
        self.queue_ptr[0] = end % size

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.projector(self.pool(self.encoder(first)))
        q2 = self.projector(self.pool(self.encoder(second)))
        with torch.no_grad():
            self.momentum_update()
            k1 = F.normalize(self.key_projector(self.pool(self.key_encoder(first))), dim=1)
            k2 = F.normalize(self.key_projector(self.pool(self.key_encoder(second))), dim=1)
        loss = 0.5 * (self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1))
        return loss, torch.cat((k1, k2), dim=0)


class BYOL(nn.Module):
    def __init__(self, encoder: nn.Module, input_dim: int, momentum: float = 0.996):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
        )
        self.predictor = nn.Sequential(
            nn.Linear(256, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256),
        )
        self.target_encoder = copy.deepcopy(encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for parameter in list(self.target_encoder.parameters()) + list(self.target_projector.parameters()):
            parameter.requires_grad_(False)
        self.momentum = momentum

    @staticmethod
    def pool(features: torch.Tensor) -> torch.Tensor:
        return features.mean(dim=(2, 3))

    @torch.no_grad()
    def momentum_update(self) -> None:
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(self.momentum).add_(online.data, alpha=1.0 - self.momentum)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        p1 = self.predictor(self.projector(self.pool(self.encoder(first))))
        p2 = self.predictor(self.projector(self.pool(self.encoder(second))))
        with torch.no_grad():
            self.momentum_update()
            z1 = self.target_projector(self.pool(self.target_encoder(first)))
            z2 = self.target_projector(self.pool(self.target_encoder(second)))
        return 0.5 * (
            (2 - 2 * F.cosine_similarity(p1, z2.detach(), dim=1)).mean()
            + (2 - 2 * F.cosine_similarity(p2, z1.detach(), dim=1)).mean()
        )


class FCMAE(nn.Module):
    """Dense masked-input FCMAE with a fully convolutional pixel decoder."""

    def __init__(
        self,
        encoder: nn.Module,
        input_dim: int,
        patch_size: int = 32,
        mask_ratio: float = 0.60,
    ):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.decoder = nn.Conv2d(input_dim, 3 * patch_size * patch_size, kernel_size=1)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        mean = patches.mean(dim=-1, keepdim=True)
        variance = patches.var(dim=-1, keepdim=True, unbiased=False)
        return (patches - mean) / torch.sqrt(variance + 1e-6)

    def forward(self, first: torch.Tensor, _second: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = first.shape
        grid_height, grid_width = height // self.patch_size, width // self.patch_size
        mask = torch.rand(batch, 1, grid_height, grid_width, device=first.device) < self.mask_ratio
        pixel_mask = F.interpolate(mask.float(), size=(height, width), mode="nearest")
        features = self.encoder(first * (1.0 - pixel_mask))
        prediction = self.decoder(features).flatten(2).transpose(1, 2)
        loss = (prediction - self.patchify(first)).pow(2).mean(dim=-1)
        return loss[mask.flatten(1)].mean()


def seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def init_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return world_size, rank, local_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(GLOBAL_BATCH))
    parser.add_argument("--model", required=True, choices=("yolov8s", "yolov10s", "yolo12s"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size, rank, local_rank = init_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")
    device = torch.device("cuda", local_rank)
    seed_everything(args.seed, rank)

    global_batch = GLOBAL_BATCH[args.method]
    micro_batch = min(args.micro_batch, global_batch // world_size)
    if micro_batch < 2 or global_batch % (micro_batch * world_size):
        raise ValueError(
            f"Global batch {global_batch} must be divisible by "
            f"micro_batch*world_size={micro_batch * world_size}"
        )
    accumulation = global_batch // (micro_batch * world_size)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = output_dir / f"{args.method}_{args.model}.pt"
    resume_checkpoint = output_dir / f".{args.method}_{args.model}.resume.pt"
    if final_checkpoint.exists() and not args.force:
        if rank == 0:
            print(f"SKIP completed checkpoint: {final_checkpoint}")
        if world_size > 1:
            dist.destroy_process_group()
        return

    dataset = SSLImages(
        args.data_root.expanduser().resolve(),
        args.image_list.expanduser().resolve(),
        pad_to=global_batch,
    )
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True)
        if world_size > 1
        else None
    )
    generator = torch.Generator().manual_seed(args.seed + rank)

    def worker_seed(worker_id: int) -> None:
        value = (args.seed + rank * 1000 + worker_id) % (2**32)
        random.seed(value)
        np.random.seed(value)

    loader = DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )

    detector = build_yolo(args.model, verbose=rank == 0)
    encoder = extract_backbone(detector).to(device)
    encoder.eval()
    with torch.no_grad():
        feature_dim = int(encoder(torch.zeros(1, 3, 224, 224, device=device)).shape[1])
    encoder.train()

    method_metadata: dict[str, object] = {}
    if args.method == "simsiam":
        model: nn.Module = SimSiam(encoder, feature_dim)
        learning_rate, weight_decay, optimizer_name = 0.05, 1e-4, "SGD"
    elif args.method == "mocov2":
        model = MoCoV2(encoder, feature_dim)
        learning_rate, weight_decay, optimizer_name = 0.03, 1e-4, "SGD"
    elif args.method == "byol":
        model = BYOL(encoder, feature_dim)
        learning_rate, weight_decay, optimizer_name = 0.2, 1e-4, "SGD"
        method_metadata = {"target_momentum": 0.996}
    else:
        model = FCMAE(encoder, feature_dim)
        learning_rate, weight_decay, optimizer_name = 1.5e-4, 0.05, "AdamW"
        method_metadata = {
            "mask_ratio": 0.60,
            "patch_size": 32,
            "sparse_convolution": False,
        }
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=True)

    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
    steps_per_epoch = len(loader) // accumulation
    if steps_per_epoch < 1:
        raise RuntimeError("The image list is too short for one optimizer step")
    total_steps = args.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 0.5
        * (1.0 + math.cos(math.pi * min(step, total_steps) / total_steps)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch = 0
    if resume_checkpoint.exists() and not args.force:
        state = torch.load(resume_checkpoint, map_location="cpu")
        unwrap(model).load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        if rank == 0:
            print(f"RESUME epoch={start_epoch} from {resume_checkpoint}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        running_loss = 0.0
        used_batches = steps_per_epoch * accumulation
        for batch_index, (first, second) in enumerate(loader):
            if batch_index >= used_batches:
                break
            first = first.to(device, non_blocking=True)
            second = second.to(device, non_blocking=True)
            synchronize = (batch_index + 1) % accumulation == 0
            context = nullcontext() if synchronize or not isinstance(model, DDP) else model.no_sync()
            with context:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(first, second)
                    loss, keys = output if args.method == "mocov2" else (output, None)
                    scaled_loss = loss / accumulation
                scaler.scale(scaled_loss).backward()
            if keys is not None:
                unwrap(model).enqueue(keys.detach())
            if synchronize:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += float(loss.detach())

        if world_size > 1:
            totals = torch.tensor([running_loss, used_batches], device=device)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            mean_loss = float(totals[0] / totals[1])
        else:
            mean_loss = running_loss / used_batches
        if rank == 0:
            print(
                f"epoch={epoch + 1}/{args.epochs} loss={mean_loss:.6f} "
                f"lr={scheduler.get_last_lr()[0]:.8f}",
                flush=True,
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": unwrap(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                resume_checkpoint,
            )
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        encoder_state = {
            key: value.detach().cpu()
            for key, value in unwrap(model).encoder.state_dict().items()
        }
        torch.save(
            {
                "state_dict": encoder_state,
                "method": args.method,
                "model": args.model,
                "seed": args.seed,
                "epochs": args.epochs,
                "input_size": 224,
                "effective_global_batch": global_batch,
                "micro_batch_per_gpu": micro_batch,
                "gradient_accumulation": accumulation,
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "scheduler": "cosine",
                "amp": True,
                **method_metadata,
            },
            final_checkpoint,
        )
        resume_checkpoint.unlink(missing_ok=True)
        print(f"SAVED encoder checkpoint: {final_checkpoint}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

