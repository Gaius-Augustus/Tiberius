import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import wandb
from hidten.config import ModelConfig
from pydantic import BaseModel
# from vipsania.train.callback import WandbLRUEigenvalues
from wandb.integration.keras import WandbMetricsLogger

from ..data import DatasetConfig, build_dataset
from ..model.base import Tiberius, TiberiusConfig
# from ..model.residual import ResidualTiberius, ResidualTiberiusConfig
from .callback import (AnnotationMetrics, AnnotationMetricsConfig,
                       WarmUpDecayFlatSchedule)
from .loss import CCE_F1_Loss, TwoStrandedAccuracy


class TrainerConfig(BaseModel):

    epochs: int
    train_steps: int
    val_steps: int

    lr: float = 1e-3
    weight_decay: float = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.999
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int | None = None

    lr_warmup: int | None = None
    """Number of epochs used for a cosine warmup."""
    lr_warmup_target: float = 1e-2
    """Learning rate after warmup."""
    lr_warmup_start: float = 1e-7
    """Learning rate at the start of training when using a warmup."""
    lr_decay: int = 100
    """Number of epochs used to decay to the default learning rate. Only
    applicable when `lr_warmup` is set."""
    lr_log_decay: int | None = None
    """Percentage of the total training steps used for an additional
    decay after the warmup. This decay brings the learning rate to zero
    with a given `lr_log_decay_velocity`."""
    lr_log_decay_velocity: float = 10.0
    """Velocity of the log decay that starts after the warmup decay."""

    use_cee: bool = True
    loss_f1_factor: float

    log_plot_freq: int = 25
    log_annotation_metrics: list[AnnotationMetricsConfig] = []


class Trainer:

    def __init__(
        self,
        config: TrainerConfig | Path | str,
        model_config: TiberiusConfig | Path | str,
        # model_config: TiberiusConfig | ResidualTiberiusConfig | Path | str,
        dataset_config: DatasetConfig | Path | str,
        checkpoints_dir: Path | str,
        jit_compile: bool = True,
        load: Path | str | None = None,
        verbose: bool = True,
        online: str | None = None,
    ) -> None:
        if not isinstance(model_config, ModelConfig):
            with open(Path(model_config).expanduser(), "r") as f:
                model_config = TiberiusConfig(**json.load(f))
        if isinstance(model_config, TiberiusConfig):
            self.model = Tiberius(**model_config.model_dump())
        # elif isinstance(model_config, ResidualTiberiusConfig):
        #     self.model = ResidualTiberius(**model_config.model_dump())
        self.model.build((None, None, 6))

        if not isinstance(config, TrainerConfig):
            with open(Path(config).expanduser(), "r") as f:
                config = TrainerConfig(**json.load(f))
        self.config = config

        if isinstance(dataset_config, (Path, str)):
            with open(dataset_config, "r") as f:
                dataset_config = DatasetConfig(**json.load(f))
        self.dataset_config = dataset_config
        self.train_dataset = build_dataset(
            self.dataset_config.train_paths,
            self.dataset_config,
        )
        self.val_dataset = None
        if self.dataset_config.validation_paths is not None:
            self.val_dataset = build_dataset(
                self.dataset_config.validation_paths,
                self.dataset_config,
            )

        self.jit_compile = jit_compile
        self.verbose = verbose
        self.online = online
        self.checkpoints_dir = Path(checkpoints_dir).expanduser()
        self.load = Path(load).expanduser() if load is not None else None
        self._path: Path | None = None

        if self.online is not None:
            entity, project = self.online.split("/")
            wandb.init(entity=entity, project=project)

    @property
    def path(self) -> Path:
        if self._path is not None:
            return self._path

        if self.online is not None:
            self._path = self.checkpoints_dir / wandb.run.id  # type: ignore
            self._path.mkdir()  # type: ignore
        else:
            version = max([0] +
                [int(f.stem[8:]) for f in self.checkpoints_dir.iterdir()
                 if f.stem.startswith("version_")]
            ) + 1
            self._path = self.checkpoints_dir / f"version_{version}"
            self._path.mkdir()
        return self._path  # type: ignore

    def log(self, config: dict[str, Any], name: str = "config.json") -> None:
        if (self.path / name).exists():
            raise FileExistsError(
                f"Configuration file {name!r} already exists in {self.path}."
            )
        with open(self.path / name, "w") as f:
            json.dump(config, f, indent=4)
        if self.online:
            wandb.config.update(config, allow_val_change=True)

    def compile(self) -> None:
        lr = self.config.lr
        if self.config.lr_warmup is not None:
            ws = int(self.config.lr_warmup * self.config.train_steps)
            ds = int(self.config.lr_decay * self.config.train_steps)
            if self.config.lr_log_decay is not None:
                ls = int(self.config.lr_log_decay * self.config.train_steps)
            else:
                ls = None
            lr = WarmUpDecayFlatSchedule(
                start_lr=self.config.lr_warmup_start,  # type: ignore
                base_lr=self.config.lr,  # type: ignore
                warmup_final_lr=self.config.lr_warmup_target,  # type: ignore
                warmup_steps=ws,  # type: ignore
                decay_steps=ds,  # type: ignore
                log_decay=ls,  # type: ignore
                log_velocity=self.config.lr_log_decay_velocity,  # type: ignore
            )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr,  # type: ignore
            weight_decay=self.config.weight_decay,
            beta_1=self.config.beta_1,
            beta_2=self.config.beta_2,
            clipnorm=self.config.gradient_clip_norm,
            gradient_accumulation_steps=
                self.config.gradient_accumulation_steps,
        )
        optimizer.exclude_from_weight_decay(
            var_names=["bias", "beta", "gamma"],
        )

        self.model.compile(
            loss=CCE_F1_Loss(
                f1_factor=self.config.loss_f1_factor,
                batch_size=self.dataset_config.batch_size,
                output_dim=self.dataset_config.output_size,
                use_cee=self.config.use_cee,
            ),
            optimizer=optimizer,  # type: ignore
            metrics=[TwoStrandedAccuracy()],
            jit_compile=self.jit_compile,  # type: ignore
        )
        if self.load is not None:
            self.model.load_weights(str(self.load))

    def get_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        callbacks = []

        if self.online is not None:
            wandbcallback = WandbMetricsLogger()
            wandb.config.update({"trainable_parameters": sum(
                np.prod(layer.shape)
                for layer in self.model.trainable_weights
            )})
            wandb.config.update({"non_trainable_parameters": sum(
                np.prod(layer.shape)
                for layer in self.model.non_trainable_weights
            )})
            callbacks.append(wandbcallback)

        save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            str(self.path / "best_val_loss.weights.h5"),
            monitor='val_loss',
            verbose=self.verbose,
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(save_best_model_callback)
        save_latest_model_callback = tf.keras.callbacks.ModelCheckpoint(
            str(self.path / "latest_checkpoint.weights.h5"),
            monitor='loss',
            verbose=self.verbose,
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(save_latest_model_callback)
        if self.online is not None:
            for lam in self.config.log_annotation_metrics:
                callbacks.append(AnnotationMetrics(
                    save_path=self.path,
                    **lam.model_dump(),
                ))
            # if (
            #     hasattr(self.model.config, "lru")
            #     and self.model.config.lru is not None
            # ):
            #     callbacks.append(WandbLRUEigenvalues(
            #         every_n_epochs=self.config.log_plot_freq,
            #     ))
        return callbacks

    def train(self) -> None:
        self.log({
            "model": self.model.config.model_dump(),
            "dataset": self.dataset_config.model_dump(),
            "trainer": self.config.model_dump(),
        }, name="config.json")
        self.model.fit(
            self.train_dataset,
            epochs=self.config.epochs,
            steps_per_epoch=self.config.train_steps,
            validation_data=self.val_dataset,
            validation_steps=self.config.val_steps,
            callbacks=self.get_callbacks(),
            verbose=0 if self.online is not None else 1,  # type: ignore
        )

        if self.online is not None:
            wandb.finish()
