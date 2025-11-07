import argparse
from pathlib import Path

import tiberius

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"


def run(args: argparse.Namespace):
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    cm, cd, ct = tiberius.util.split_config(args.config)
    trainer = tiberius.Trainer(
        config=ct,
        model_config=cm,
        dataset_config=cd,
        checkpoints_dir=CHECKPOINTS_DIR,
        online="entity/project" if args.online else None,
    )
    trainer.compile()
    trainer.train()


def main():
    ap = argparse.ArgumentParser("Start a Tiberius training run")
    ap.add_argument(
        "config",
        help="tiberius configuration file (.json)",
        type=str,
    )
    ap.add_argument(
        "--online",
        action="store_true",
        help="use wandb to log training metrics online",
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
