import math
from pathlib import Path
from typing import Literal

import bricks2marble as b2m
import tensorflow as tf
import wandb
from hidten.config import ModelConfig, with_config

from ..annotate import annotate_genome


class AnnotationMetricsConfig(ModelConfig):

    label: str | None = None

    fasta: Path | str
    reference: Path | str
    gffcompare: Path | str
    take_seqs: int | str | None = None
    T: int = 100_000
    B: int = 32
    split_seqname: bool = True

    use_head: int = 0
    use: Literal["VITERBI", "MEA"] = "VITERBI"
    N_token: Literal["track", "uniform"] = "track"
    repeats_input: Literal["track", "expand", "omit"] = "track"

    every_n_epochs: int = 1
    liberal: bool = False
    save_best_key: str | None = None
    jit_compile: bool = True


class AnnotationMetrics(tf.keras.callbacks.Callback):
    """Annotates the given FASTA and logs metrics for the comparison to
    the reference annotation.
    """

    def __init__(
        self,
        save_path: Path,
        **kwargs,
    ) -> None:
        self.config = AnnotationMetricsConfig(**kwargs)
        self.fasta = b2m.io.load_fasta(
            Path(self.config.fasta).expanduser(),
            T=self.config.T,
        )
        if (take_seqs := self.config.take_seqs) is not None:
            if isinstance(take_seqs, int): self.fasta = self.fasta[:take_seqs]
            else: self.fasta = b2m.struct.FASTA([self.fasta[take_seqs]])

        if self.config.save_best_key is not None:
            self.save_best_key = "annotation/" + self.config.save_best_key
            self.save_path = save_path / "best_annotation.weights.h5"
            self.best_metric = -1

        wandb.define_metric("annotation/*", step_metric="epoch/epoch")

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        if epoch % self.config.every_n_epochs != 0:
            return

        metrics = {}
        annotation = annotate_genome(
            self.model,  # type: ignore
            fasta=self.fasta,
            hmm_head=self.config.use_head,
            B=self.config.B,
            verbose=False,
            use=self.config.use,  # type: ignore
            liberal=self.config.liberal,
            N_token=self.config.N_token,
            repeats_input=self.config.repeats_input,
            jit_compile=self.config.jit_compile,
        )
        if self.config.split_seqname:
            annotation.rename(lambda x: x.split(" ")[0])
        comparison = b2m.tools.compare_gtf(
            annotation,
            self.config.reference,
            gffcompare=self.config.gffcompare,
        )
        for key, value in comparison:
            logkey = "annotation"
            if self.config.label is not None:
                logkey += f"/{self.config.label}"
            logkey += f"/{key}"
            for innerkey, innervalue in value:
                if innervalue is not None:
                    metrics[logkey+f"/{innerkey}"] = innervalue
                    if (
                        self.save_best_key is not None
                        and self.save_best_key == logkey
                        and self.best_metric < innervalue
                    ):
                        self.model.save_weights(
                            self.save_path,
                            overwrite=True,
                        )
                        tf.print(
                            f"Saving model in epoch {epoch}, "
                            f"as {self.save_best_key} improved from "
                            f"{self.best_metric} to {innervalue}."
                        )
                        self.best_metric = innervalue

            if (
                self.save_best_key is not None
                and self.save_best_key == logkey+"/F1"
                and self.best_metric < value.F1
            ):
                self.model.save_weights(
                    self.save_path,
                    overwrite=True,
                )
                tf.print(
                    f"Saving model in epoch {epoch}, "
                    f"as {self.save_best_key} improved from "
                    f"{self.best_metric} to {value.F1}."
                )
                self.best_metric = value.F1
            metrics[logkey+"/F1"] = value.F1
        wandb.log(metrics)


class WarmUpDecayFlatScheduleConfig(ModelConfig):

    start_lr: float
    base_lr: float
    warmup_final_lr: float
    warmup_steps: int
    decay_steps: int
    log_decay: int | None = None
    log_velocity: float = 5.0


@with_config(WarmUpDecayFlatScheduleConfig)  # type: ignore
class WarmUpDecayFlatSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule
):

    def __init__(self, **kwargs) -> None:
        self.config = WarmUpDecayFlatScheduleConfig(**kwargs)

    def __call__(self, step: int) -> float:
        step = tf.cast(step, tf.float32)
        start_lr = tf.cast(self.config.start_lr, tf.float32)
        warmup_final_lr = tf.cast(self.config.warmup_final_lr, tf.float32)
        final_lr = tf.cast(self.config.base_lr, tf.float32)
        warmup_steps = tf.cast(self.config.warmup_steps, tf.float32)
        decay_steps = tf.cast(self.config.decay_steps, tf.float32)
        if self.config.log_decay is not None:
            log_decay_steps = tf.cast(self.config.log_decay, tf.float32)
            log_velocity = tf.cast(self.config.log_velocity, tf.float32)

        def warmup_phase():
            progress = step / warmup_steps
            progress = 0.5 * (1 - tf.cos(math.pi * progress))
            return start_lr + (warmup_final_lr - start_lr) * progress

        def decay_phase():
            step_in_decay = step - warmup_steps
            progress = step_in_decay / decay_steps
            progress = 0.5 * (1 + tf.cos(math.pi * progress))
            return final_lr + (warmup_final_lr - final_lr) * progress

        def log_phase():
            step_in = step - warmup_steps - decay_steps
            progress = tf.clip_by_value(step_in / log_decay_steps, 0.0, 1.0)
            decay = tf.math.exp(-log_velocity * progress)
            return final_lr * decay

        def flat_phase():
            return final_lr

        return tf.case([
            (step < warmup_steps, warmup_phase),
            (step < warmup_steps + decay_steps, decay_phase),
        ], default=(
            flat_phase if self.config.log_decay is None else log_phase
        ))
