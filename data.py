from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 32,
        val_split: int = 5000,
        seed: int = 42,
        download: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.val_split = val_split
        self.seed = seed
        self.download = download

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        datasets.CIFAR10(self.data_dir, train=True, download=self.download)
        datasets.CIFAR10(self.data_dir, train=False, download=self.download)

    def setup(self, stage: str | None = None) -> None:
        train_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )

        if stage in (None, "fit"):
            train_base = datasets.CIFAR10(self.data_dir, train=True, transform=train_transform)
            val_base = datasets.CIFAR10(self.data_dir, train=True, transform=eval_transform)
            train_size = len(train_base) - self.val_split
            indices = torch.randperm(len(train_base), generator=torch.Generator().manual_seed(self.seed)).tolist()
            self.train_dataset = Subset(train_base, indices[:train_size])
            self.val_dataset = Subset(val_base, indices[train_size:])

        if stage in (None, "test"):
            self.test_dataset = datasets.CIFAR10(self.data_dir, train=False, transform=eval_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
