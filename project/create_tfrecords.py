import argparse


def main(args: argparse.Namespace):
    import tiberius

    tiberius.data.create_tfrecords(
        fasta=args.fasta,
        gtf=args.gtf,
        T=args.T,
        out_path=args.o,
        out_prefix=args.prefix,
        splits=args.split,
        verbose=args.verbose
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Tiberius - training data generation",
    )

    parser.add_argument(
        "fasta",
        type=str,
        help="genome sequence in FASTA format",
    )
    parser.add_argument(
        "gtf",
        type=str,
        help="annotation in GTF format",
    )
    parser.add_argument(
        "-T",
        type=int,
        help="chunk length of each sequence",
        default=9_999,
    )
    parser.add_argument(
        "-o",
        type=str,
        default="./tfrecords",
        help="directory for the output files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="prefix for the output files",
    )
    parser.add_argument(
        "--split",
        type=int,
        help="number of files to split the data into",
        default=100,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        help="verbosity level of the script; may be 0, 1 or 2",
        default=1,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
