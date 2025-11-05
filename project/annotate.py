import argparse
from pathlib import Path

import tiberius
import bricks2marble as b2m


def run(args: argparse.Namespace):
    cm, cd, ct = tiberius.util.split_config(args.config)
    cm.lru.max_tree_depth = 16
    cm.hmm.parallel_factor = 250
    if isinstance(cm, tiberius.model.TiberiusConfig):
        model = tiberius.Tiberius(**cm.model_dump())
    elif isinstance(cm, tiberius.model.ResidualTiberiusConfig):
        model = tiberius.ResidualTiberius(**cm.model_dump())
    model.build((None, None, 6))

    fasta = b2m.io.load_fasta(Path(args.fasta).expanduser(), T=100_000)

    anno = tiberius.annotate_genome(
        model,
        fasta,
        B=32,
        jit_compile=True,
    )
    anno.to_gtf(args.o)


def main():
    ap = argparse.ArgumentParser("Start a Tiberius annotation")
    ap.add_argument(
        "config",
        help="tiberius configuration file (.json)",
        type=str,
    )
    ap.add_argument(
        "weights",
        help="tiberius weights path (.weights.h5)",
        type=str,
    )
    ap.add_argument(
        "fasta",
        help="fasta path (.fa)",
        type=str,
    )
    ap.add_argument(
        "-o",
        help="output path for the annotation file",
        type=str,
    )
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
