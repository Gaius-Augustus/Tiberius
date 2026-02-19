#!/usr/bin/env python3
# ==============================================================
# Authors: Lars Gabriel, Felix Becker
#
# Refactored training script:
# - single training function with selectable head: none | hmm | clamsa
# - epoch folder saving: epoch_XX/{weights.h5, model_config.json, model_layers.json}
# - never saves HMM weights (even if HMM head is used)
# - auto-resume epoch numbering without overwriting existing epochs
# ==============================================================

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam, SGD

import tiberius.parse_args as parse_args
import tiberius.models as models
from tiberius import DataGenerator
from tiberius.models import (
    Cast,
    EpochSave,  # not used anymore, kept for compatibility
    add_constant_hmm,
    add_hmm_layer,
    custom_cce_f1_loss,
    lstm_model,
    make_weighted_cce_loss,
)

# ----------------------------
# Strategy / hardware
# ----------------------------

gpus = tf.config.list_physical_devices("GPU")
strategy = tf.distribute.MirroredStrategy()


# ----------------------------
# Loading
# ----------------------------
def is_new_epoch_format(p: Path) -> bool:
    return p.is_dir() and (p / "weights.h5").exists()

def load_backbone_from_new_format(epoch_dir: Path, config: dict) -> tf.keras.Model:
    """
    Builds backbone from model_layers.json (preferred) and loads weights.h5.
    Falls back to rebuilding from config if model_layers.json is missing.
    """
    layers_json = epoch_dir / "model_layers.json"
    weights_h5  = epoch_dir / "weights.h5"

    if layers_json.exists():
        backbone = keras.models.model_from_json(
            layers_json.read_text(encoding="utf-8"),
            custom_objects={"Cast": Cast},
        )
    else:
        # fallback: rebuild from config (less robust if hyperparams changed)
        backbone = build_backbone(config, head="none")

    backbone.load_weights(str(weights_h5))
    return backbone

def load_any_model_or_epoch(path: str | None, config: dict) -> tf.keras.Model | None:
    if not path:
        return None
    p = Path(path)

    # NEW format: epoch directory
    if is_new_epoch_format(p):
        print(f"[load] Detected new epoch format folder: {p}")
        return load_backbone_from_new_format(p, config)

    # OLD / standard Keras model path (.keras / SavedModel)
    print(f"[load] Loading Keras model: {p}")
    return keras.models.load_model(
        str(p),
        custom_objects={
            "custom_cce_f1_loss": custom_cce_f1_loss(config.get("loss_f1_factor", 0.0), config["batch_size"]),
            "loss_": custom_cce_f1_loss(config.get("loss_f1_factor", 0.0), config["batch_size"]),
            "Cast": Cast,
        },
        compile=False,
    )
# ----------------------------
# LR schedule + debug callback
# ----------------------------

@tf.keras.utils.register_keras_serializable()
class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_epochs, decay_rate, min_lr, steps_per_epoch):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_epochs = warmup_epochs
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.steps_per_epoch = tf.constant(steps_per_epoch, dtype=tf.float32)

    def __call__(self, step):
        epoch_int = tf.cast(step, dtype=tf.float32) // self.steps_per_epoch
        epoch = tf.cast(epoch_int, tf.float32)

        lr = tf.cond(
            epoch < self.warmup_epochs,
            lambda: self.peak_lr * ((epoch + 1) / tf.cast(self.warmup_epochs, tf.float32)),
            lambda: tf.maximum(
                self.peak_lr * tf.pow(self.decay_rate, epoch - self.warmup_epochs + 1),
                self.min_lr,
            ),
        )
        return lr

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_epochs": self.warmup_epochs,
            "decay_rate": self.decay_rate,
            "min_lr": self.min_lr,
            "steps_per_epoch": int(self.steps_per_epoch.numpy()),
        }


class PrintLr(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = logs.get("lr")
        if lr is not None:
            print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}")
            return

        lr_t = self.model.optimizer.learning_rate
        if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr_t = lr_t(self.model.optimizer.iterations)
        print(f"Epoch {epoch+1}: Learning rate is {tf.keras.backend.get_value(lr_t):.6f}")


# ----------------------------
# Helpers: resume logic + saving
# ----------------------------

_EPOCH_DIR_RE = re.compile(r"^epoch_(\d+)$")


def _list_epoch_dirs(model_save_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    if not model_save_dir.exists():
        return out
    for p in model_save_dir.iterdir():
        if not p.is_dir():
            continue
        m = _EPOCH_DIR_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def find_resume_state(model_save_dir: Path) -> tuple[int, Path | None]:
    """
    Returns:
      next_epoch_index: int  (e.g. if epoch_00..epoch_12 exist -> returns 13)
      last_epoch_dir: Path|None (dir of latest epoch or None if none exist)
    """
    epoch_dirs = _list_epoch_dirs(model_save_dir)
    if not epoch_dirs:
        return 0, None
    last_idx, last_dir = epoch_dirs[-1]
    return last_idx + 1, last_dir


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class EpochFolderSaver(tf.keras.callbacks.Callback):
    """
    Saves per-epoch artifacts into:
      {model_save_dir}/epoch_XX/weights.h5           (backbone only)
      {model_save_dir}/epoch_XX/model_config.json   (full config dict)
      {model_save_dir}/epoch_XX/model_layers.json   (backbone model JSON)
    """
    def __init__(
        self,
        model_save_dir: Path,
        backbone_model: tf.keras.Model,
        config: dict[str, Any],
        start_epoch_index: int,
    ):
        super().__init__()
        self.model_save_dir = model_save_dir
        self.backbone_model = backbone_model
        self.config = config
        self.start_epoch_index = start_epoch_index

    def on_epoch_end(self, epoch, logs=None):
        # Keras passes `epoch` starting at `initial_epoch`, but it is still 0-based
        # relative to `initial_epoch`. We want absolute epoch index = initial_epoch + epoch
        abs_epoch = self.start_epoch_index + epoch

        epoch_dir = self.model_save_dir / f"epoch_{abs_epoch:02d}"
        ensure_dir(epoch_dir)

        # 1) backbone weights only
        weights_path = epoch_dir / "weights.h5"
        self.backbone_model.save_weights(str(weights_path))

        # 2) config as model_config.json
        with (epoch_dir / "model_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, sort_keys=True)

        # 3) backbone architecture JSON
        with (epoch_dir / "model_layers.json").open("w", encoding="utf-8") as f:
            f.write(self.backbone_model.to_json())


# ----------------------------
# Data helpers
# ----------------------------

def read_species(file_name: str) -> list[str]:
    with open(file_name, "r", encoding="utf-8") as f:
        species = f.read().strip().split("\n")
    return [s for s in species if s and s[0] != "#"]


def load_val_data(file, hmm_factor=1, output_size=7, clamsa=False, softmasking=True, oracle=False):
    data = np.load(file)
    x_val = data["array1"]
    y_val = data["array2"]
    if clamsa:
        clamsa_track = data["array3"]
    data.close()

    if not softmasking:
        x_val = x_val[:, :, :5]

    # output_size remapping (unchanged)
    if output_size == 5:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 5), np.float32)
        y_new[:, :, 0] = y_val[:, :, 0]
        y_new[:, :, 1] = np.sum(y_val[:, :, 1:4], axis=-1)
        y_new[:, :, 2] = np.sum(y_val[:, :, [4, 7, 10, 12]], axis=-1)
        y_new[:, :, 3] = np.sum(y_val[:, :, [5, 8, 13]], axis=-1)
        y_new[:, :, 4] = np.sum(y_val[:, :, [6, 9, 11, 14]], axis=-1)
        y_val = y_new
    elif output_size == 7:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 7), np.float32)
        y_new[:, :, :4] = y_val[:, :, :4]
        y_new[:, :, 4] = np.sum(y_val[:, :, [4, 7, 10, 12]], axis=-1)
        y_new[:, :, 5] = np.sum(y_val[:, :, [5, 8, 13]], axis=-1)
        y_new[:, :, 6] = np.sum(y_val[:, :, [6, 9, 11, 14]], axis=-1)
        y_val = y_new
    elif output_size == 3:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 3), np.float32)
        y_new[:, :, 0] = y_val[:, :, 0]
        y_new[:, :, 1] = np.sum(y_val[:, :, 1:4], axis=-1)
        y_new[:, :, 2] = np.sum(y_val[:, :, 4:], axis=-1)
        y_val = y_new
    elif output_size == 2:
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 2), np.float32)
        y_new[:, :, 0] = np.sum(y_val[:, :, :4], axis=-1)
        y_new[:, :, 1] = np.sum(y_val[:, :, 4:], axis=-1)
        y_val = y_new

    if hmm_factor:
        step_width = y_val.shape[1] // hmm_factor
        start = y_val[:, ::step_width, :]
        end = y_val[:, step_width - 1 :: step_width, :]
        hints = np.concatenate([start[:, :, tf.newaxis, :], end[:, :, tf.newaxis, :]], -2)
        return ([np.array(x_val), hints], np.array(y_val))

    if clamsa:
        return [[(x, c) for x, c in zip(x_val, clamsa_track)], y_val]

    return [[x_val, y_val], y_val.astype(np.float32)] if oracle else [x_val, y_val]


# ----------------------------
# Model build / compile
# ----------------------------

HeadType = Literal["none", "hmm", "clamsa"]


def build_optimizer(config: dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    if config.get("use_lr_scheduler", False):
        schedule = WarmupExponentialDecay(
            peak_lr=config["lr"],
            warmup_epochs=config.get("warmup", 1),
            decay_rate=config.get("lr_decay_rate", 0.9),
            min_lr=config.get("min_lr", 1e-6),
            steps_per_epoch=config["steps_per_epoch"],
        )
        return Adam(learning_rate=schedule)

    if config.get("sgd", False):
        return SGD(learning_rate=config["lr"])

    return Adam(learning_rate=config["lr"])


def build_loss_and_weights(config: dict[str, Any], head: HeadType):
    """
    Returns:
      loss, loss_weights
    Mirrors your prior behavior:
      - for HMM with multi_loss: [lstm_loss, hmm_loss], weights=[1, hmm_mul]
      - otherwise: use from_logits=True loss for HMM output
    """
    # Base loss (for non-HMM outputs or LSTM output in multi-loss)
    if config.get("loss_f1_factor"):
        base_loss = custom_cce_f1_loss(config["loss_f1_factor"], batch_size=config["batch_size"])
    elif config.get("loss_weights"):
        base_loss = make_weighted_cce_loss(config["loss_weights"], config["batch_size"])
    else:
        base_loss = tf.keras.losses.CategoricalCrossentropy()

    if config.get("output_size") == 1:
        base_loss = tf.keras.losses.BinaryCrossentropy()

    if head != "hmm":
        return base_loss, config.get("loss_weights") if config.get("loss_weights") else None

    # HMM head case:
    if config.get("multi_loss", False):
        hmm_loss = custom_cce_f1_loss(
            config.get("loss_f1_factor", 0.0),
            batch_size=config["batch_size"],
            from_logits=True,
        )
        return [base_loss, hmm_loss], [1, config.get("hmm_loss_weight_mul", 0.1)]

    # single-output HMM
    hmm_only_loss = custom_cce_f1_loss(
        config.get("loss_f1_factor", 0.0),
        batch_size=config["batch_size"],
        from_logits=True,
    )
    return hmm_only_loss, None


def load_any_model(path: str | None, config: dict[str, Any]) -> tf.keras.Model | None:
    if not path:
        return None
    # load without compile to allow hyperparameter changes
    return keras.models.load_model(
        path,
        custom_objects={
            "custom_cce_f1_loss": custom_cce_f1_loss(config.get("loss_f1_factor", 0.0), config["batch_size"]),
            "loss_": custom_cce_f1_loss(config.get("loss_f1_factor", 0.0), config["batch_size"]),
            "Cast": Cast,
        },
        compile=False,
    )


def build_backbone(config: dict[str, Any], head: HeadType) -> tf.keras.Model:
    """
    Builds the backbone network that we will:
      - compile/train (possibly with heads)
      - ALWAYS save weights for (never HMM)

    Behavior matches old code:
      - if config['oracle']: returns identity oracle model
      - else loads model_load
      - else creates new LSTM model (or clamsa models if head='clamsa')
    """
    if config.get("oracle", False):
        inputs = tf.keras.layers.Input(
            shape=(None, 6 if config.get("softmasking", True) else 5),
            name="main_input",
        )
        oracle_inputs = tf.keras.layers.Input(shape=(None, config["output_size"]), name="oracle_input")
        return tf.keras.Model(inputs=[inputs, oracle_inputs], outputs=oracle_inputs)

    # if --load points to epoch folder, this returns a backbone model already
    m = load_any_model_or_epoch(config.get("model_load"), config)
    if m is not None:
        return m

    # Highest priority: model_load (full model path)
    model_load = config.get("model_load")
    if model_load:
        m = load_any_model(model_load, config)
        if m is None:
            raise ValueError(f"Failed to load model from {model_load}")
        return m

    # Fresh build depends on head
    if head == "clamsa":
        # Keep same branching you had in train_clamsa
        if config.get("clamsa_with_lstm", True):
            relevant_keys = [
                "units", "filter_size", "kernel_size",
                "numb_conv", "numb_lstm", "dropout_rate",
                "pool_size", "stride", "lstm_mask", "clamsa",
                "output_size", "residual_conv", "softmasking",
                "clamsa_kernel", "lru_layer",
            ]
            relevant_args = {k: config[k] for k in relevant_keys if k in config}
            return models.lstm_model(**relevant_args)

        relevant_keys = ["output_size", "clamsa_kernel_size", "clamsa_emb_size"]
        relevant_args = {k: config[k] for k in relevant_keys if k in config}
        return models.clamsa_only_model(**relevant_args)

    # Default: LSTM backbone
    relevant_keys = [
        "units", "filter_size", "kernel_size",
        "numb_conv", "numb_lstm", "dropout_rate",
        "pool_size", "stride", "lstm_mask", "clamsa",
        "output_size", "residual_conv", "softmasking",
        "clamsa_kernel", "lru_layer",
    ]
    relevant_args = {k: config[k] for k in relevant_keys if k in config}
    return lstm_model(**relevant_args)


def attach_head(
    backbone: tf.keras.Model,
    config: dict[str, Any],
    head: HeadType,
) -> tf.keras.Model:
    """
    Returns the trainable model (may include extra layers).
    Note: we NEVER save weights of the HMM part; only backbone.
    """
    if head == "none":
        return backbone

    if head == "clamsa":
        # If you want the clamsa-only CNN and not the LSTM, you already built it in build_backbone.
        # For clamsa+hmm you previously used models.clamsa_hmm_model; here we keep head='hmm' for HMM case.
        return backbone

    if head != "hmm":
        raise ValueError(f"Unknown head: {head}")

    # Apply trainability choice to backbone layers
    for layer in backbone.layers:
        layer.trainable = bool(config.get("trainable_lstm", True))

    if config.get("constant_hmm", False):
        return add_constant_hmm(
            backbone,
            seq_len=config["sample_size"],
            batch_size=config["batch_size"],
            output_size=config["output_size"],
        )

    # Optionally seed HMM params from a previously trained HMM model
    gene_pred_layer = None
    model_load_hmm = config.get("model_load_hmm")
    if model_load_hmm:
        hmm_model = keras.models.load_model(
            model_load_hmm,
            custom_objects={
                "custom_cce_f1_loss": custom_cce_f1_loss(config.get("loss_f1_factor", 0.0), config["batch_size"]),
                "loss_": custom_cce_f1_loss(config.get("loss_f1_factor", 0.0), config["batch_size"]),
            },
            compile=False,
        )
        gene_pred_layer = hmm_model.layers[-3]

    return add_hmm_layer(
        backbone,
        gene_pred_layer,
        output_size=config["output_size"],
        num_hmm=config.get("num_hmm_layers", 1),
        hmm_factor=config["hmm_factor"],
        share_intron_parameters=config.get("hmm_share_intron_parameters", False),
        trainable_nucleotides_at_exons=config.get("hmm_nucleotides_at_exons", False),
        trainable_emissions=config.get("hmm_trainable_emissions", False),
        trainable_transitions=config.get("hmm_trainable_transitions", False),
        trainable_starting_distribution=config.get("hmm_trainable_starting_distribution", False),
        include_lstm_in_output=config.get("multi_loss", False),
    )


# ----------------------------
# Single training function
# ----------------------------

def train_model(
    dataset: tf.data.Dataset,
    model_save_dir: str | Path,
    config: dict[str, Any],
    *,
    head: HeadType,
    val_data=None,
) -> None:
    model_save_dir = Path(model_save_dir)
    ensure_dir(model_save_dir)

    # Detect existing epochs and pick next index; load last backbone weights if available
    next_epoch_idx, last_epoch_dir = find_resume_state(model_save_dir)

    with strategy.scope():
        optimizer = build_optimizer(config)
        backbone = build_backbone(config, head=head)

        # Resume backbone weights if possible
        if last_epoch_dir is not None:
            last_weights = last_epoch_dir / "weights.h5"
            if last_weights.exists():
                print(f"[resume] Loading backbone weights from: {last_weights}")
                backbone.load_weights(str(last_weights))
            else:
                print(f"[resume] Found last epoch dir {last_epoch_dir} but no weights.h5; starting fresh backbone.")

        # Attach head (may create new model object)
        model = attach_head(backbone, config, head=head)

        loss, loss_weights = build_loss_and_weights(config, head=head)
        model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"], loss_weights=loss_weights)

        model.summary()

        # Logging
        csv_logger = CSVLogger(str(model_save_dir / "training.log"), append=True, separator=";")

        # Save callback (NEW format)
        epoch_saver = EpochFolderSaver(
            model_save_dir=model_save_dir,
            backbone_model=backbone,            # <- never includes HMM
            config=config,
            start_epoch_index=next_epoch_idx,
        )

        callbacks = [epoch_saver, csv_logger]
        if config.get("use_lr_scheduler", False):
            callbacks.append(PrintLr())

        # Train (continue epoch numbering)
        model.fit(
            dataset,
            epochs=config["num_epochs"],
            initial_epoch=next_epoch_idx,
            validation_data=val_data,
            steps_per_epoch=config["steps_per_epoch"],
            validation_batch_size=config.get("batch_size"),
            callbacks=callbacks,
        )


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args.parseCmd()

    # NOTE: keeping your old w_size/batch defaults as-is
    w_size = 9999
    if w_size == 99999:
        batch_size = 96
    elif w_size == 50004:
        batch_size = 28
    elif w_size == 9999:
        batch_size = 512
    elif w_size == 29997:
        batch_size = 120 * 4
    else:
        batch_size = 512

    if args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "num_epochs": 2000,
            "steps_per_epoch": 5000,
            "threads": 96,
            "stride": 0,
            "units": 372,
            "filter_size": 128,
            "numb_lstm": 2,
            "numb_conv": 3,
            "dropout_rate": 0.0,
            "lstm_mask": False,
            "pool_size": 9,
            "lr": 1e-4,
            "warmup": 1,
            "min_lr": 1e-6,
            "lr_decay_rate": 0.9,
            "use_lr_scheduler": False,
            "batch_size": batch_size,
            "w_size": w_size,
            "filter": False,
            "trainable_lstm": True,
            "output_size": 15,
            "multi_loss": False,
            "hmm_factor": 99,
            "seq_weights": False,
            "softmasking": True,
            "residual_conv": True,
            "hmm_loss_weight_mul": 0.1,
            "hmm_dense": 32,
            "hmm_share_intron_parameters": False,
            "hmm_nucleotides_at_exons": False,
            "hmm_trainable_transitions": False,
            "hmm_trainable_starting_distribution": False,
            "hmm_trainable_emissions": False,
            "constant_hmm": False,
            "num_hmm_layers": 1,
            "clamsa": bool(args.clamsa),
            "clamsa_kernel_size": 7,
            "clamsa_emb_size": 32,
            "clamsa_with_lstm": True,
            "loss_f1_factor": 2.0,
            "sgd": False,
            "oracle": False,
            "lru_layer": False,
        }

    # Normalize paths / args
    config["model_load"] = os.path.abspath(args.load) if args.load else None
    config["model_save_dir"] = os.path.abspath(args.out)
    config["model_load_hmm"] = os.path.abspath(args.load_hmm) if args.load_hmm else None
    config["mask_tx_list_file"] = os.path.abspath(args.mask_tx_list) if args.mask_tx_list else None
    config["mask_flank"] = args.mask_flank if args.mask_flank else 100

    model_save_dir = Path(config["model_save_dir"])
    ensure_dir(model_save_dir)

    # Write a top-level config copy too (optional, but handy)
    with (model_save_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # Mask list
    mask_tx_list = read_species(config["mask_tx_list_file"]) if config.get("mask_tx_list_file") else []

    data_path = Path(args.data)
    ensure_dir(data_path)

    # Tfrecord paths
    species = read_species(str(args.train_species_file))
    file_paths = [str(data_path / f"{s}_{i}.tfrecords") for s in species for i in range(100)]

    # Dataset
    generator = DataGenerator(
        file_path=file_paths,
        batch_size=config["batch_size"],
        shuffle=True,
        repeat=True,
        filter=config["filter"],
        output_size=config["output_size"],
        hmm_factor=0,
        seq_weights=config["seq_weights"],
        softmasking=config["softmasking"],
        clamsa=config.get("clamsa", False),
        oracle=config.get("oracle", False),
        threads=config["threads"],
        tx_filter=mask_tx_list,
        tx_filter_region=config["mask_flank"],
    )
    dataset = generator.get_dataset()

    # Val data
    val_data = None
    if args.val_data:
        val_data = load_val_data(
            args.val_data,
            hmm_factor=0,
            output_size=config["output_size"],
            clamsa=config.get("clamsa", False),
            softmasking=config["softmasking"],
            oracle=config.get("oracle", False),
        )

    # Decide head
    if args.hmm:
        head: HeadType = "hmm"
    elif args.clamsa:
        head = "clamsa"
    else:
        head = "none"

    train_model(
        dataset=dataset,
        model_save_dir=model_save_dir,
        config=config,
        head=head,
        val_data=val_data,
    )


if __name__ == "__main__":
    main()
