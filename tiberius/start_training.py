#!/usr/bin/env python3
# ==============================================================
# Authors: Lars Gabriel
#
# Script for starting a Tiberius training
# ==============================================================
from __future__ import annotations
import argparse
import glob
import logging
from pathlib import Path
from typing import Sequence

import tensorflow as tf

from base import TiberiusConfig
from trainer import Trainer, TrainerConfig
from data_generator import DataGeneratorConfig, build_dataset
from config_utils import load_json, deep_update, log_diff

# logging
def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / "run.log", encoding="utf-8")],
    )
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

# helper
def expand_files(patterns: Sequence[str]) -> list[str]:
    files: list[str] = []
    for p in patterns:
        for part in [s for s in p.split(",") if s.strip()]:
            m = sorted(glob.glob(part))
            if not m and Path(part).exists():
                m = [part]
            files.extend(m)
    if not files:
        raise FileNotFoundError(f"No files found for: {patterns}")
    return files

def make_dataset(files: Sequence[str], batch_size: int, output_size: int,
                 *, shuffle: bool, repeat: bool, softmasking: bool) -> tf.data.Dataset:
    dl = DataLoaderConfig(
        files=list(files),
        batch_size=batch_size,
        shuffle=shuffle,
        repeat=repeat,
        output_size=output_size,
        softmasking=softmasking,
    )
    return build_dataset(dl)

# DEFAULT PARAMETER
MODEL_DEFAULTS = dict(
    units=372,
    filter_size=128,
    kernel_size=9,
    numb_conv=3,
    numb_lstm=2,
    pool_size=9,
    output_size=15,
    with_hmm=False,
    hmm_heads=1,
    hmm_reverse_strand=False,
    parallel_factor=1,
    initial_exon_len=100,
    initial_intron_len=10_000,
    initial_ir_len=10_000,
    intron_state_chain=1,
    train_transitions=True,
    train_start_dist=True,
    residual_conv=True,
    multi_loss=False,
)

TRAINER_DEFAULTS = dict(
    epochs=100,
    batch_size=500,
    steps_per_epoch=5000,
    lr=1e-4,
    use_lr_scheduler=False,
    warmup_epochs=1,
    min_lr=1e-4,
    decay_rate=0.9,
    loss_f1_factor=0.0,
    model_save_dir="./tiberius_training/",
)

DATAGENERATOR_DEFAULTS = dict(
    files="",
    batch_size=500,
    shuffle=True,
    repeat=True,
    output_size=15,
    input_size=6,
    clamsa=False,
    oracle=False,
    tx_filter=[],
    tx_filter_region=1000,
    seq_weights_window=250,
    seq_weights_value=100,
    compression="GZIP",
    shuffle_buffer=100,
)

# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser("Start a Tiberius training run")
    ap.add_argument("--train-files", required=True, nargs="+")
    # ap.add_argument("--val-files", required=True, nargs="+")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model_config", default=None, help="")
    ap.add_argument("--load-weights", default=None)
    # optional quick overrides for the most common arguments
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--steps-per-epoch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--softmasking", action="store_true", default=True)
    ap.add_argument("--no-softmasking", dest="softmasking", action="store_false")
    args = ap.parse_args()

    outdir = Path(args.output_dir).expanduser()
    setup_logging(outdir)
    log = logging.getLogger("train")

    train_files = expand_files(args.train_files)
    # val_files   = expand_files(args.val_files)
    log.info("Train files: %d | Val files: %d", len(train_files), len(val_files))

    # load json config
    model_cfg   = load_json(args.model_config)

    # overwrite cfg parameter with command line args
    if args.epochs is not None:
        model_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        model_cfg["batch_size"] = args.batch_size
    if args.steps_per_epoch is not None:
        model_cfg["steps_per_epoch"] = args.steps_per_epoch
    if args.lr is not None:
        model_cfg["lr"] = args.lr

    # merge input configs with default configs
    model_cfg_dict   = deep_update(MODEL_DEFAULTS, model_cfg)
    trainer_cfg_dict = deep_update(TRAINER_DEFAULTS, model_cfg)
    datagenerator_cfg_dict = deep_update(DATAGENERATOR_DEFAULTS, model_cfg)

    # Log differences
    log_diff(MODEL_DEFAULTS, model_cfg_dict, "model")
    log_diff(TRAINER_DEFAULTS, trainer_cfg_dict, "trainer")
    log_diff(DATAGENERATOR_DEFAULTS, datagenerator_cfg_dict, "datagenerator")

    # construct configs
    mcfg = TiberiusConfig(**model_cfg_dict)
    tcfg = TrainerConfig(**trainer_cfg_dict)
    dcfg = DataGeneratorConfig(**datagenerator_cfg_dict)

    # ensure save dir exists
    Path(tcfg.model_save_dir).expanduser().mkdir(parents=True, exist_ok=True)

    # 6) Dataset 
    train_ds = build_dataset(train_files, tcfg.batch_size, mcfg.output_size,
                            shuffle=True, repeat=True, softmasking=args.softmasking)
    # val_ds   = make_dataset(val_files,   tcfg.batch_size, mcfg.output_size,
    #                         shuffle=False, repeat=False, softmasking=args.softmasking)

    # Training with trainer
    trainer = Trainer(tcfg, mcfg, train_ds, val_ds, load=args.load_weights)
    trainer.compile()
    trainer.train()

if __name__ == "__main__":
    main()
