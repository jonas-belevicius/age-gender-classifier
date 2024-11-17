import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics import (
    MeanSquaredError,
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    R2Score,
    ExplainedVariance,
    MeanAbsoluteError,
)

import pytorch_lightning as pl

from typing import Optional, List
import time
import re

from PIL import Image


class AgeGenderDataset(Dataset):
    def __init__(self, image_paths: List[str], transform=None) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img_name = image_path.split("\\")[-1]
        image = Image.open(image_path).convert("RGB")

        match = re.match(r"^(\d+)_(\d+)_(\d+)", img_name)
        age_label = int(match.group(1))
        gender_label = int(match.group(2))

        if self.transform:
            image = self.transform(image)

        return image, age_label, gender_label


class AgeGenderModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model: nn.Module,
        lr: float = 1e-3,
        weights: Optional[List[float]] = None,
        dropout: Optional[float] = 0.3,
    ) -> None:
        """
        Initialize the AgeGenderModule.

        Args:

        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weights = weights
        self.dropout = dropout
        self.epoch_start_time: Optional[float] = None
        self.cuda = torch.device("cuda")

        self.train_gender_logits = torch.empty(0, device=self.cuda)
        self.train_gender_labels = torch.empty(
            0,
            dtype=torch.long,
            device=self.cuda,
        )

        self.val_gender_logits = torch.empty(0, device=self.cuda)
        self.val_gender_labels = torch.empty(
            0,
            dtype=torch.long,
            device=self.cuda,
        )

        self.criterion_age = nn.MSELoss()
        self.criterion_gender = nn.CrossEntropyLoss(weight=weights)

        self.log_sigma_age = nn.Parameter(torch.zeros(1))
        self.log_sigma_gender = nn.Parameter(torch.zeros(1))

        self.age_mse = MeanSquaredError()
        self.age_r2 = R2Score()
        self.age_mae = MeanAbsoluteError()
        self.age_ev = ExplainedVariance()

        self.gender_accuracy = Accuracy(task="binary")
        self.gender_precision = Precision(task="binary")
        self.gender_recall = Recall(task="binary")
        self.gender_f1 = F1Score(task="binary")
        self.gender_auroc = AUROC(
            task="binary", num_classes=num_classes, average="macro"
        )

        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.age_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(num_features, 1),
        )
        self.gender_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(num_features, num_classes),
        )

        for param in self.age_head.parameters():
            param.requires_grad = True
        for param in self.gender_head.parameters():
            param.requires_grad = True

    def model_grad(self, requires_grad: bool = False) -> None:
        """
        Sets the gradient computation for the base transformer model.

        Args:
            requires_grad (bool, optional): If False, disables gradient
            computation for the base model. Default is False.
        """
        for param in list(self.model.parameters()):
            param.requires_grad = requires_grad

    def unfreeze_top_layers(self) -> None:
        """
        Unfreezes the top `num_layers` of the base model for fine tuning.

        Args:
            num_layers (int): Number of layers from the top to unfreeze, the
            default is 2.
        """
        # layers = list(self.model.children())

        # for layer in layers[-num_layers:]:
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model(x)
        age_output = self.age_head(features).squeeze()
        gender_logits = self.gender_head(features)
        return age_output, gender_logits

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()
        print("Epoch start time:", self.epoch_start_time)

    def training_step(self, batch, batch_idx):
        x, age_labels, gender_labels = [item.to(self.cuda) for item in batch]

        age_output, gender_logits = self(x)
        # gender_logits = torch.argmax(gender_logits, dim=1)

        age_loss = self.criterion_age(age_output, age_labels.float())
        gender_loss = self.criterion_gender(gender_logits, gender_labels)

        sigma_age = torch.exp(self.log_sigma_age)
        sigma_gender = torch.exp(self.log_sigma_gender)

        self.train_gender_logits = (
            torch.cat([self.train_gender_logits, gender_logits.to(self.cuda)], dim=0)
            if self.train_gender_logits is not None
            else gender_logits
        )
        self.train_gender_labels = (
            torch.cat([self.train_gender_labels, gender_labels.to(self.cuda)], dim=0)
            if self.train_gender_labels is not None
            else gender_labels
        )

        loss = (
            (1 / sigma_age**2) * age_loss
            + (1 / sigma_gender**2) * gender_loss
            + torch.log(sigma_age * sigma_gender)
        )

        self.log(
            "train_age_loss",
            age_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train_gender_loss",
            gender_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "train_age_mse",
            self.age_mse(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_age_r2",
            self.age_r2(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_age_mae",
            self.age_mae(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_age_ev",
            self.age_ev(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train_gender_accuracy",
            self.gender_accuracy(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_gender_precision",
            self.gender_precision(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_gender_recall",
            self.gender_recall(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_gender_f1",
            self.gender_f1(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "train_gender_auroc",
        #     self.gender_auroc(
        #         torch.softmax(gender_logits, dim=1)[:, 1],
        #         gender_labels,
        #     ),
        #     logger=True,
        #     on_step=False,
        #     on_epoch=True,
        # )

        return loss

    def on_train_epoch_end(self) -> None:
        # gender_labels = torch.cat(self.train_gender_labels, dim=0)
        # gender_logits = torch.cat(self.train_gender_logits, dim=0)

        auroc = self.gender_auroc(
            torch.softmax(self.train_gender_logits, dim=1)[:, 1],
            self.train_gender_labels,
        )
        self.log(
            "train_total_gender_auroc",
            auroc,
            on_epoch=True,
            logger=True,
        )
        self.train_gender_logits = self.train_gender_logits.new_empty(0)
        self.train_gender_labels = self.train_gender_labels.new_empty(
            0, dtype=torch.long
        )

    def validation_step(self, batch, batch_idx):
        x, age_labels, gender_labels = [item.to(self.cuda) for item in batch]

        age_output, gender_logits = self(x)
        # gender_logits = torch.argmax(gender_logits, dim=1)

        age_loss = self.criterion_age(age_output, age_labels.float())
        gender_loss = self.criterion_gender(gender_logits, gender_labels)

        sigma_age = torch.exp(self.log_sigma_age)
        sigma_gender = torch.exp(self.log_sigma_gender)

        self.val_gender_logits = (
            torch.cat([self.val_gender_logits, gender_logits.to(self.cuda)], dim=0)
            if self.val_gender_logits is not None
            else gender_logits
        )
        self.val_gender_labels = (
            torch.cat([self.val_gender_labels, gender_labels.to(self.cuda)], dim=0)
            if self.val_gender_labels is not None
            else gender_labels
        )

        loss = (
            (1 / sigma_age**2) * age_loss
            + (1 / sigma_gender**2) * gender_loss
            + torch.log(sigma_age * sigma_gender)
        )

        self.log(
            "val_age_loss",
            age_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "val_gender_loss",
            gender_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

        self.log(
            "val_age_mse",
            self.age_mse(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_age_r2",
            self.age_r2(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_age_mae",
            self.age_mae(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_age_ev",
            self.age_ev(age_output, age_labels),
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "val_gender_accuracy",
            self.gender_accuracy(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_gender_precision",
            self.gender_precision(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_gender_recall",
            self.gender_recall(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_gender_f1",
            self.gender_f1(
                torch.argmax(gender_logits, dim=1),
                gender_labels,
            ),
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        # self.log(
        #     "train_gender_auroc",
        #     self.gender_auroc(
        #         torch.softmax(gender_logits, dim=1)[:, 1],
        #         gender_labels,
        #     ),
        #     logger=True,
        #     on_step=False,
        #     on_epoch=True,
        # )

        return loss

    def on_validation_epoch_end(self) -> None:
        # gender_labels = torch.cat(self.val_gender_labels, dim=0)
        # gender_logits = torch.cat(self.val_gender_logits, dim=0)

        auroc = self.gender_auroc(
            torch.softmax(self.val_gender_logits, dim=1)[:, 1],
            self.val_gender_labels,
        )
        self.log(
            "val_total_gender_auroc",
            auroc,
            on_epoch=True,
            logger=True,
        )

        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.log(
                "epoch_duration",
                epoch_duration / 60,
                on_epoch=True,
                logger=True,
            )
        else:
            print("Skipping epoch duration calculation for initial validation")

        self.val_gender_logits = self.val_gender_logits.new_empty(0)
        self.val_gender_labels = self.val_gender_labels.new_empty(0, dtype=torch.long)

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.lr,
            eps=1e-8,
            weight_decay=1e-4,
        )
        return [optimizer]

    def reinitialize_optimizer(self, lr):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = optim.AdamW(trainable_params, lr=lr, eps=1e-8)
        return optimizer

    # def train_dataloader(self):
    #     dataset = AgeGenderDataset(
    #         image_paths=train_images_paths,
    #         transform=train_tfms,
    #     )
    #     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    #     return loader

    # def val_dataloader(self):
    #     dataset = AgeGenderDataset(
    #         image_paths=val_images_paths,
    #         transform=val_tfms,
    #     )
    #     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    #     return loader


class AgeGenderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_images_paths,
        val_images_paths,
        train_tfms,
        val_tfms,
        batch_size,
    ):

        super().__init__()
        self.train_images_paths = train_images_paths
        self.val_images_paths = val_images_paths
        self.train_tfms = train_tfms
        self.val_tfms = val_tfms
        self.batch_size = batch_size

    def train_dataloader(self):
        train_dataset = AgeGenderDataset(
            image_paths=self.train_images_paths,
            transform=self.train_tfms,
        )

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        val_dataset = AgeGenderDataset(
            image_paths=self.val_images_paths,
            transform=self.val_tfms,
        )

        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
