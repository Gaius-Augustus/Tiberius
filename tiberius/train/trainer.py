import json
from pathlib import Path

import tensorflow as tf
from pydantic import BaseModel

from ..model.base import Tiberius, TiberiusConfig
from .callback import EpochSave, WarmupExponentialDecay, PrintLr
from .loss import CCE_F1_Loss


class TrainerConfig(BaseModel):

    epochs: int
    batch_size: int
    lr: float = 1e-4

    use_lr_scheduler: bool = False
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    decay_rate: float = 0.9
    steps_per_epoch: int

    loss_f1_factor: float

    model_save_dir: Path | str


class Trainer:

    def __init__(
        self,
        config: TrainerConfig | Path | str,
        model_config: TiberiusConfig | Path | str,
        dataset: tf.data.Dataset,
        val_data: tf.data.Dataset,
        load: Path | str | None = None,
    ) -> None:
        if not isinstance(model_config, TiberiusConfig):
            with open(Path(model_config).expanduser(), "r") as f:
                model_config = TiberiusConfig(**json.load(f))
        self.model = Tiberius(**model_config.model_dump())

        if not isinstance(config, TrainerConfig):
            with open(Path(config).expanduser(), "r") as f:
                config = TrainerConfig(**json.load(f))
        self.config = config

        self.load = Path(load).expanduser() if load is not None else None
        self.dataset = dataset
        self.val_data = val_data

    def compile(self) -> None:
        schedule = WarmupExponentialDecay(
            peak_lr=self.config.lr,
            warmup_epochs=self.config.warmup_epochs,
            decay_rate=self.config.decay_rate,
            min_lr=self.config.min_lr,
            steps_per_epoch=self.config.steps_per_epoch,
        )

        self.model.compile(
            loss=CCE_F1_Loss(
                f1_factor=self.config.loss_f1_factor,
                batch_size=self.config.batch_size,
                from_logits=True,
            ),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=schedule,
            ),
            metrics=["accuracy"],
        )
        if self.load is not None:
            self.model.load_weights(str(self.load))

    def get_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        model_dir = str(Path(self.config.model_save_dir).expanduser())
        return [
            EpochSave(model_dir),
            tf.keras.callbacks.CSVLogger(
                f'{model_dir}/training.log',
                append=True,
                separator=';',
            ),
            PrintLr(),
        ]

    def train(self) -> None:
        self.model.fit(
            self.dataset,
            epochs=self.config.epochs,
            steps_per_epoch=self.config.steps_per_epoch,
            validation_data=self.val_data,
            validation_batch_size=self.config.batch_size,
            callbacks=self.get_callbacks(),
        )
