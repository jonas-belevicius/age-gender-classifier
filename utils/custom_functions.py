import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import Logger

from typing import List, Optional, Any, Dict
import os
import sys

project_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
utils_dir = os.path.join(project_dir, "utils")
sys.path.append(utils_dir)

from config import (
    TRAIN_EPOCHS,
    FINE_TUNE_EPOCHS,
    TRAIN_LR,
    FINE_TUNE_LR,
)


def train_model(
    model: LightningModule,
    data_module: LightningDataModule,
    callbacks: ModelCheckpoint,
    logger: Logger,
    epochs: int,
    lr: float,
) -> pl.Trainer:
    """
    Trains a PyTorch Lightning model using the provided data module, checkpoint
    callback, and logger.

    Args:
        model (LightningModule): The model to be trained.
        data_module (LightningDataModule): The data module that provides the
        training and validation data.
        checkpoint_callback (ModelCheckpoint): Callback to save model
        checkpoints during training.
        logger (Logger): Logger for tracking training progress and metrics.
        epochs (int): The number of training epochs.
        lr (float): The learning rate for the model.

    Returns:
        pl.Trainer: The PyTorch Lightning Trainer instance after training.
    """

    model.lr = lr
    model = model.to("cuda")
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=2,
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        enable_progress_bar=True,
        accumulate_grad_batches=2,
        precision=32,
    )

    trainer.fit(model, data_module)
    return trainer


def train_tune_save_model(
    model: LightningModule,
    data_module: LightningDataModule,
    callbacks: List,
    logger: Logger,
    train_epochs: int = TRAIN_EPOCHS,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
    train_lr: float = TRAIN_LR,
    fine_tune_lr: float = FINE_TUNE_LR,
    num_layers_unfreeze: int = 2,
) -> None:
    """
    Trains the classifier and fine-tunes it, saving checkpoints during both
    phases.

    Args:
        model (LightningModule): The model to be trained and fine-tuned.
        data_module (LightningDataModule): The data module providing training
        and validation data.
        checkpoint_callback (ModelCheckpoint): Callback to save model
        checkpoints.
        logger (Logger): Logger for tracking training progress and metrics.
        train_epochs (int, optional): Number of epochs for the initial training
        phase. Defaults to TRAIN_EPOCHS.
        fine_tune_epochs (int, optional): Number of epochs for the fine-tuning
        phase. Defaults to FINE_TUNE_EPOCHS.
        train_lr (float, optional): Learning rate for the initial training
        phase. Defaults to TRAIN_LR.
        fine_tune_lr (float, optional): Learning rate for the fine-tuning
        phase. Defaults to FINE_TUNE_LR.

    Returns:
        None
    """
    print("Training classifier:")
    train_model(
        model=model,
        data_module=data_module,
        callbacks=callbacks,
        logger=logger,
        epochs=train_epochs,
        lr=train_lr,
    )
    print("Classifier training finished, fine-tuning:")
    model.model_grad(requires_grad=False)
    model.unfreeze_top_layers()
    # model.reinitialize_optimizer(fine_tune_lr)
    train_model(
        model=model,
        data_module=data_module,
        callbacks=callbacks,
        logger=logger,
        epochs=fine_tune_epochs,
        lr=fine_tune_lr,
    )
