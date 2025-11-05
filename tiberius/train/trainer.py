import json
from pathlib import Path

import tensorflow as tf
from hidten.config import ModelConfig
from pydantic import BaseModel

from ..data import DatasetConfig, build_dataset
from ..model.base import Tiberius, TiberiusConfig
from ..model.residual import ResidualTiberius, ResidualTiberiusConfig
from .callback import PrintLr, WarmupExponentialDecay
from .loss import CCE_F1_Loss


class TrainerConfig(BaseModel):

    epochs: int
    train_steps: int
    val_steps: int
    lr: float = 1e-4

    use_lr_scheduler: bool = False
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    decay_rate: float = 0.9

    use_cee: bool = True
    loss_f1_factor: float

    model_save_dir: Path | str


class Trainer:

    def __init__(
        self,
        config: TrainerConfig | Path | str,
        model_config: TiberiusConfig | ResidualTiberiusConfig | Path | str,
        dataset_config: DatasetConfig | Path | str,
        jit_compile: bool = True,
        load: Path | str | None = None,
        verbose: bool = True,
    ) -> None:
        if not isinstance(model_config, ModelConfig):
            with open(Path(model_config).expanduser(), "r") as f:
                model_config = TiberiusConfig(**json.load(f))
        if isinstance(model_config, TiberiusConfig):
            self.model = Tiberius(**model_config.model_dump())
        elif isinstance(model_config, ResidualTiberiusConfig):
            self.model = ResidualTiberius(**model_config.model_dump())
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
        self.load = Path(load).expanduser() if load is not None else None

    def compile(self) -> None:
        self.model.compile(
            loss=CCE_F1_Loss(
                f1_factor=self.config.loss_f1_factor,
                batch_size=self.dataset_config.batch_size,
                output_dim=self.dataset_config.output_size,
                use_cee=self.config.use_cee,
                from_logits=True,
            ),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.lr,
            ),  # type: ignore
            metrics=["accuracy"],
            jit_compile=self.jit_compile,  # type: ignore
        )
        if self.load is not None:
            self.model.load_weights(str(self.load))

    def get_callbacks(self) -> list[tf.keras.callbacks.Callback]:
        model_dir = Path(self.config.model_save_dir).expanduser()

        save_best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            str(model_dir / "best_val_loss.weights.h5"),
            monitor='val_loss',
            verbose=self.verbose,
            save_best_only=True,
            save_weights_only=True,
        )
        save_latest_model_callback = tf.keras.callbacks.ModelCheckpoint(
            str(model_dir / "latest_checkpoint.weights.h5"),
            monitor='loss',
            verbose=self.verbose,
            save_best_only=True,
            save_weights_only=True,
        )

        return [
            save_best_model_callback,
            save_latest_model_callback,
            tf.keras.callbacks.CSVLogger(
                f'{model_dir}/training.log',
                append=True,
                separator=';',
            ),
            PrintLr(),
        ]

    def train(self) -> None:
        self.model.fit(
            self.train_dataset,
            epochs=self.config.epochs,
            steps_per_epoch=self.config.train_steps,
            validation_data=self.val_dataset,
            validation_steps=self.config.val_steps,
            callbacks=self.get_callbacks(),
        )
