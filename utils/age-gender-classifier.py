from torchvision.transforms import v2
import pytorch_lightning as pl
from typing import Optional
import torch.nn as nn

import torchvision
from torchvision.models import resnet18
from torchvision import transforms
from torchmetrics import MeanSquaredError, Accuracy, Precision, Recall, F1Score, AUROC
import torch.optim as optim
import time
import tqdm
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from torchvision.models import resnet18
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import Logger
from torchvision.models import ResNet18_Weights
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
utils_dir = os.path.join(project_dir, "utils")
sys.path.append(utils_dir)

from config import (
    DATA_DIR,
    TRAIN_VAL_DIR,
    TRAIN_EPOCHS,
    FINE_TUNE_EPOCHS,
    TRAIN_LR,
    FINE_TUNE_LR,
    BATCH_SIZE,
    NUM_CLASSES,
    CHECKPOINT_PATH,
    LOGS_PATH,
    LOGS_NAME,
    RANDOM_SEED,
)
from modules import AgeGenderDataModule, AgeGenderModule
from custom_functions import train_tune_save_model

if __name__ == "__main__":

    torchvision.disable_beta_transforms_warning()

    train_set_path = os.path.join(TRAIN_VAL_DIR, "train")
    val_set_path = os.path.join(TRAIN_VAL_DIR, "validation")

    train_images_paths = [
        os.path.join(train_set_path, img_name)
        for img_name in os.listdir(train_set_path)
    ]

    val_images_paths = [
        os.path.join(val_set_path, img_name) for img_name in os.listdir(val_set_path)
    ]

    early_stopping_callback = EarlyStopping(
        monitor="val_loss_epoch", patience=3, verbose=True, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH, filename="best_checkpoint-{epoch}-{val_loss_epoch}"
    )

    logger = CSVLogger(
        LOGS_PATH,
        name=LOGS_NAME,
        flush_logs_every_n_steps=10,
    )

    train_tfms = v2.Compose(
        [
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    age_gender_data_module = AgeGenderDataModule(
        train_images_paths, val_images_paths, train_tfms, val_tfms, BATCH_SIZE
    )

    age_gender_classifier = AgeGenderModule(
        num_classes=NUM_CLASSES, model=model, lr=TRAIN_LR, dropout=0.3
    )

    seed_everything(RANDOM_SEED, workers=True)

    model = model.to("cuda")

    train_tune_save_model(
        model=age_gender_classifier,
        data_module=age_gender_data_module,
        train_epochs=TRAIN_EPOCHS,
        fine_tune_epochs=FINE_TUNE_EPOCHS,
        train_lr=TRAIN_LR,
        fine_tune_lr=FINE_TUNE_EPOCHS,
        callbacks=[early_stopping_callback, checkpoint_callback],
        logger=logger,
        num_layers_unfreeze=3,
    )
