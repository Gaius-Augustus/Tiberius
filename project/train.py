import argparse

import tiberius


def run(args: argparse.Namespace):
    cm, cd, ct = tiberius.util.split_config(args.config)
    trainer = tiberius.Trainer(
        config=ct,
        model_config=cm,
        dataset_config=cd,
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
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
