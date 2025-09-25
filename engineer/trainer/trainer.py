# Original code from: https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks

import datetime
import os
import subprocess
import time
import warnings
from types import SimpleNamespace
from typing import Any, Optional
from copy import deepcopy

import lightning.pytorch as pl
from lightning.pytorch import Trainer as PLTrainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger as PLLogger
from lightning.pytorch.loggers import CSVLogger
import torch
from torch import nn
from torch.utils.data import DataLoader

from .loggers import ConsoleLogger


def human_format(num: float):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def count_parameters(module: nn.Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_device(input, device, detach=True):
    if isinstance(input, torch.Tensor):
        input = input.to(device)
        if detach:
            input = input.detach()
        return input
    if isinstance(input, tuple):
        input = list(input)
    if isinstance(input, list):
        keys = range(len(input))
    elif isinstance(input, dict):
        keys = input.keys()
    else:
        raise ValueError(f"Unknown input type {type(input)}.")
    for k in keys:
        input[k] = to_device(input[k], device)
    return input


def run_bash_command(command: str) -> str:
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )

    if result.returncode == 0:
        output = result.stdout.strip()
        return output
    else:
        error = result.stderr.strip()
        raise RuntimeError(f"Error executing command: {error}")


def parse_time_components(time_string: str):
    days, hours, minutes, seconds = 0, 0, 0, 0

    # Splitting days if present.
    if "-" in time_string:
        try:
            days_str, time_string = time_string.split("-")
        except:
            raise ValueError(f"Invalid time format {time_string}.")
        days = int(days_str)

    # Splitting hours, minutes, and seconds.
    time_components = time_string.split(":")
    num_components = len(time_components)

    if num_components == 3:
        hours, minutes, seconds = map(int, time_components)
    elif num_components == 2:
        minutes, seconds = map(int, time_components)
    elif num_components == 1:
        seconds = int(time_components[0])
    else:
        raise ValueError(f"Invalid time format {time_string}.")

    return days, hours, minutes, seconds


def parse_slurm_time(time_string) -> datetime.timedelta:
    days, hours, minutes, seconds = parse_time_components(time_string)
    return datetime.timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def _parse_max_time(time):
    if time is None:
        return

    if time is None and "SLURM_JOB_ID" in os.environ:
        time = run_bash_command(
            "squeue -j $SLURM_JOB_ID -h --Format TimeLimit"
        ).splitlines()
        if len(time) > 1:
            warnings.warn(
                "More than one job found (array job?). Using the first one for setting the time limit."
            )
        time = time[0]

    max_time = parse_slurm_time(time)
    return max_time


class _LoggerAdapter(PLLogger):
    """Adapt an existing logger with `.log_metrics(...)` to Lightning's Logger API."""

    def __init__(self, base_logger: Any, name: str = "custom"):
        super().__init__()
        self._base = base_logger
        self._name = getattr(base_logger, "name", name)
        self._version = getattr(base_logger, "version", "0")

    @property
    def name(self) -> str:
        return str(self._name)

    @property
    def version(self) -> str:
        return str(self._version)

    @property
    def experiment(self):
        return getattr(self._base, "experiment", SimpleNamespace())

    def log_hyperparams(self, params):  # type: ignore[override]
        if hasattr(self._base, "log_hyperparams"):
            try:
                self._base.log_hyperparams(params)
            except Exception:
                pass

    def log_metrics(self, metrics, step: Optional[int] = None):  # type: ignore[override]
        if hasattr(self._base, "log_metrics"):
            try:
                self._base.log_metrics(metrics, step=step)
            except Exception:
                pass

    def save(self):  # type: ignore[override]
        if hasattr(self._base, "save"):
            try:
                self._base.save()
            except Exception:
                pass

    def finalize(self, status: str):  # type: ignore[override]
        if hasattr(self._base, "finalize"):
            try:
                self._base.finalize(status)
            except Exception:
                pass


class _WrappedLightningModule(pl.LightningModule):
    """Wrap a generic `nn.Module` with Lightning plumbing.

    Expects the wrapped model to implement `forward(batch, step)` and expose
    `train_metrics` and `test_metrics` objects compatible with `.update(...)`,
    `.compute()`, and `.reset()`.
    """

    def __init__(
        self,
        base_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        log_interval: int = 256,
        eval_test_during_val: bool = False,
    ):
        super().__init__()
        self.model = base_model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._log_interval = log_interval
        self._eval_test_during_val = bool(eval_test_during_val)

        # Separate metric collections for validation over different dataloaders
        # so we can compute and log val/ and test/ independently during validation.
        if self._eval_test_during_val and hasattr(self.model, "test_metrics") and self.model.test_metrics is not None:
            try:
                self._val_metrics_eval = deepcopy(self.model.test_metrics)
                self._test_metrics_eval = deepcopy(self.model.test_metrics)
            except Exception:
                self._val_metrics_eval = None
                self._test_metrics_eval = None
        else:
            self._val_metrics_eval = None
            self._test_metrics_eval = None

    def forward(self, batch: Any, step: int):
        return self.model(batch, step)

    def training_step(self, batch: Any, batch_idx: int):
        loss, outputs = self.model(batch, self.global_step)
        if torch.isnan(loss):
            raise ValueError("Loss is NaN.")

        if hasattr(self.model, "train_metrics") and self.model.train_metrics is not None:
            try:
                self.model.train_metrics.update(**outputs)
            except Exception:
                pass

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        if hasattr(self.model, "train_metrics") and self.model.train_metrics is not None:
            try:
                metrics = self.model.train_metrics.compute()
                self.model.train_metrics.reset()
                # add common scalars
                metrics = {f"train/{k}": v for k, v in metrics.items()}
                metrics["train/epoch"] = float(self.current_epoch)
                self.log_dict(metrics, on_epoch=True, prog_bar=False)
            except Exception:
                pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        _, outputs = self.model(batch, batch_idx)
        if self._eval_test_during_val:
            # dataloader_idx == 0 -> validation dataset, 1 -> test dataset (if provided)
            try:
                if dataloader_idx == 0 and self._val_metrics_eval is not None:
                    self._val_metrics_eval.update(**outputs)
                elif dataloader_idx == 1 and self._test_metrics_eval is not None:
                    self._test_metrics_eval.update(**outputs)
            except Exception:
                pass
        else:
            if hasattr(self.model, "test_metrics") and self.model.test_metrics is not None:
                try:
                    self.model.test_metrics.update(**outputs)
                except Exception:
                    pass

    def on_validation_epoch_end(self) -> None:
        # Log aggregated metrics for both val and test datasets after validation pass
        if self._eval_test_during_val:
            try:
                if self._val_metrics_eval is not None:
                    val_metrics = self._val_metrics_eval.compute()
                    self._val_metrics_eval.reset()
                    val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
                    self.log_dict(val_metrics, on_epoch=True, prog_bar=True)
            except Exception:
                pass
            try:
                if self._test_metrics_eval is not None:
                    test_metrics = self._test_metrics_eval.compute()
                    self._test_metrics_eval.reset()
                    test_metrics = {f"test_eval/{k}": v for k, v in test_metrics.items()}
                    self.log_dict(test_metrics, on_epoch=True, prog_bar=False)
            except Exception:
                pass
        else:
            if hasattr(self.model, "test_metrics") and self.model.test_metrics is not None:
                try:
                    metrics = self.model.test_metrics.compute()
                    self.model.test_metrics.reset()
                    metrics = {f"val/{k}": v for k, v in metrics.items()}
                    self.log_dict(metrics, on_epoch=True, prog_bar=True)
                except Exception:
                    pass

    def test_step(self, batch: Any, batch_idx: int):
        _, outputs = self.model(batch, batch_idx)
        if hasattr(self.model, "test_metrics") and self.model.test_metrics is not None:
            try:
                self.model.test_metrics.update(**outputs)
            except Exception:
                pass

    def on_test_epoch_end(self) -> None:
        if hasattr(self.model, "test_metrics") and self.model.test_metrics is not None:
            try:
                metrics = self.model.test_metrics.compute()
                self.model.test_metrics.reset()
                metrics = {f"test/{k}": v for k, v in metrics.items()}
                self.log_dict(metrics, on_epoch=True, prog_bar=True)
            except Exception:
                pass

    def configure_optimizers(self):
        if self._scheduler is None:
            return self._optimizer
        # Assume step-wise scheduler by default, matching previous behavior
        try:
            cfg = {"scheduler": self._scheduler, "interval": "step"}
            return {"optimizer": self._optimizer, "lr_scheduler": cfg}
        except Exception:
            return self._optimizer


class Trainer:
    """Thin wrapper that uses Lightning's Trainer under the hood."""

    def __init__(
        self,
        scheduler=None,
        logger: Any = None,
        max_steps: int = 0,
        max_time: str = None,
        limit_val_batches: int = float("inf"),
        val_check_interval: int = 1024,
        print_interval: int = 32,
        fast_dev_run: bool = False,
        wandb=None,
        callbacks=list(),
        log_interval=256,
        checkpoint=None,
        test_only=False,
        monitor_metric="val/loss",
        monitor_mode="min",
        eval_test_during_val: bool = False,
    ):
        # preserve original API (unused/kept for compatibility): print_interval, checkpoint, wandb
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.val_check_interval = val_check_interval
        self.limit_val_batches = limit_val_batches
        self.fast_dev_run = fast_dev_run
        self.test_only = test_only
        self.max_steps = max_steps
        self.max_time = _parse_max_time(max_time)
        self.eval_test_during_val = bool(eval_test_during_val)

        # Distributed strategy configuration (auto-detect GPUs)
        try:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            num_gpus = 0
        if num_gpus > 1:
            self.accelerator = "gpu"
            self.devices = num_gpus
            self.strategy = "ddp"
        elif num_gpus == 1:
            self.accelerator = "gpu"
            self.devices = 1
            self.strategy = "auto"
        else:
            self.accelerator = "cpu"
            self.devices = 1
            self.strategy = "auto"

        # callbacks: only keep Lightning-native callbacks
        self.callbacks: list[Callback] = [c for c in callbacks if isinstance(c, Callback)]
        # ensure a ModelCheckpoint exists monitoring validation loss
        has_ckpt = any(isinstance(c, ModelCheckpoint) for c in self.callbacks)
        if not has_ckpt:
            self.ckpt_callback = ModelCheckpoint(
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=1,
                save_last=True,
            )
            self.callbacks.append(self.ckpt_callback)
        else:
            self.ckpt_callback = next(c for c in self.callbacks if isinstance(c, ModelCheckpoint))

        # logger setup
        if logger is None:
            self.logger: Optional[PLLogger] = CSVLogger(save_dir=os.getcwd(), name="lightning_logs")
        elif isinstance(logger, PLLogger):
            self.logger = logger
        else:
            # adapt custom logger to Lightning
            self.logger = _LoggerAdapter(logger)

        self._pl_trainer: Optional[PLTrainer] = None

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        ckpt_path: Optional[str] = None,
    ):
        print("\nModel Summary\n---")
        print(model)
        print(f"Total parameters: {human_format(count_parameters(model))}\n")

        # Wrap generic model into LightningModule
        lightning_module = _WrappedLightningModule(
            base_model=model,
            optimizer=optimizer,
            scheduler=self.scheduler,
            log_interval=self.log_interval,
            eval_test_during_val=self.eval_test_during_val,
        )

        # Translate max_steps=0 (old semantics) to None for Lightning
        max_steps = None if self.max_steps in (0, None) else self.max_steps

        # Adjust logging cadence to avoid warnings when batches per epoch are small
        try:
            num_train_batches = len(train_loader)
        except Exception:
            num_train_batches = None
        effective_log_every_n_steps = self.log_interval
        if isinstance(num_train_batches, int) and num_train_batches > 0:
            effective_log_every_n_steps = max(1, min(self.log_interval, num_train_batches))

        self._pl_trainer = PLTrainer(
            logger=self.logger,
            callbacks=self.callbacks,
            max_steps=max_steps,
            max_time=self.max_time,
            val_check_interval=self.val_check_interval,
            check_val_every_n_epoch=None,
            log_every_n_steps=effective_log_every_n_steps,
            limit_val_batches=(
                1.0
                if (isinstance(self.limit_val_batches, float) and self.limit_val_batches == float("inf"))
                else self.limit_val_batches
            ),
            fast_dev_run=self.fast_dev_run,
            accelerator=self.accelerator,
            devices=self.devices,
            strategy=self.strategy,
        )

        if self.test_only:
            print("Testing mode.")
            self._pl_trainer.test(model=lightning_module, dataloaders=test_loader)
            return

        self._pl_trainer.fit(
            model=lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=(
                [val_loader, test_loader]
                if (self.eval_test_during_val and test_loader is not None)
                else val_loader
            ),
            ckpt_path=ckpt_path,
        )

        if test_loader is not None:
            # Prefer best checkpoint if available; otherwise, test current weights
            best_path = getattr(self.ckpt_callback, "best_model_path", None)
            print(best_path)
            if best_path and isinstance(best_path, str) and os.path.exists(best_path) and len(best_path) > 0:
                self._pl_trainer.test(ckpt_path=best_path, dataloaders=test_loader)
            else:
                self._pl_trainer.test(model=lightning_module, dataloaders=test_loader)

            print("Testing completed.")
            print(f"Best checkpoint: {best_path}")
