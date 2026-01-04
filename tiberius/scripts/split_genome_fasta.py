#!/usr/bin/env python3
"""
Split a genome FASTA into multiple chunk FASTAs.

Rules:
  - A sequence (contig/chromosome) is never split across files.
  - Each output file aims to contain at least --min-size bases (default 20M).
  - A file may contain multiple sequences.
  - The total number of files is capped by --max-files.
  - If the genome is small or --max-files is too low, some files may have less
    than --min-size bases (especially the last one).

Usage:
    python split_genome_fasta.py \
        --genome genome.fa \
        --outdir chunks \
        --prefix chunk \
        --min-size 20000000 \
        --max-files 10
"""

import sys
import os
import argparse
import gzip
from math import ceil

def parse_args():
    ap = argparse.ArgumentParser(
        description="Split a genome FASTA into chunks without splitting contigs."
    )
    ap.add_argument("--genome", required=True,
                    help="Input genome FASTA file (plain or .gz).")
    ap.add_argument("--outdir", required=True,
                    help="Output directory for chunk FASTA files.")
    ap.add_argument("--prefix", default="genome_chunk",
                    help="Output file prefix (default: genome_chunk).")
    ap.add_argument("--min-size", type=int, default=20_000_000,
                    help="Minimum target number of bases per chunk (default: 20,000,000).")
    ap.add_argument("--max-files", type=int, required=True,
                    help="Maximum number of chunk files to create.")
    return ap.parse_args()


def open_maybe_gzip(path):
    """Open plain text or gzipped FASTA transparently."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def fasta_lengths(path):
    """Return list of (header_line, seq_name, length) for each sequence."""
    lengths = []
    with open_maybe_gzip(path) as fh:
        name = None
        header = None
        length = 0
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    lengths.append((header, name, length))
                header = line.rstrip("\n")
                # seq name = first token after '>'
                name = header[1:].split()[0]
                length = 0
            else:
                length += len(line.strip())
        if name is not None:
            lengths.append((header, name, length))
    return lengths


def make_groups(seq_info, min_size, max_files):
    """
    Group sequences into at most max_files groups.

    seq_info: list of (header, name, length)

    Returns: list of list of seq_names, one inner list per output file.
    """
    total_len = sum(l for _, _, l in seq_info)
    if not seq_info:
        return []

    # If the genome is smaller than min_size or max_files == 1, everything in one file
    if total_len <= min_size or max_files == 1:
        return [[name for _, name, _ in seq_info]]

    groups = []
    current_group = []
    current_len = 0
    remaining_seqs = len(seq_info)

    for idx, (header, name, length) in enumerate(seq_info):
        remaining_seqs -= 1
        current_group.append(name)
        current_len += length

        # remaining groups we are allowed to create after this one
        used_groups = len(groups) + 1  # including current
        remaining_groups_allowed = max_files - used_groups

        # remaining sequences must be at least remaining_groups_allowed
        # so that each future group can get at least one sequence
        can_split_here = remaining_groups_allowed > 0 and remaining_seqs >= remaining_groups_allowed

        # Close group if we've reached min_size and can still make more groups
        if current_len >= min_size: #and can_split_here:
            groups.append(current_group)
            current_group = []
            current_len = 0

    # Add leftover sequences to last group
    if current_group:
        groups.append(current_group)

    # If we somehow exceeded max_files (shouldn't happen with the logic above), merge last ones
    while len(groups) > max_files:
        # merge the last two groups
        last = groups.pop()
        groups[-1].extend(last)

    return groups


def write_chunks(genome_fa, seq_groups, outdir, prefix):
    """Write grouped sequences to FASTA chunk files.

    seq_groups: list of lists of seq_names per file.
    """
    os.makedirs(outdir, exist_ok=True)

    # Map seq_name -> group index
    seq_to_group = {}
    for gi, group in enumerate(seq_groups):
        for name in group:
            seq_to_group[name] = gi

    # Open one file handle per group
    handles = {}
    for gi in range(len(seq_groups)):
        fname = f"{prefix}_{gi+1:03d}.fa"
        fpath = os.path.join(outdir, fname)
        handles[gi] = open(fpath, "w", encoding="utf-8")

    # Stream through genome and dispatch sequences into the correct file
    with open_maybe_gzip(genome_fa) as fh:
        current_name = None
        current_group = None
        out = None
        for line in fh:
            if line.startswith(">"):
                # close previous (nothing to do for files, they stay open)
                header = line.rstrip("\n")
                current_name = header[1:].split()[0]
                current_group = seq_to_group.get(current_name)
                # If a sequence was not assigned (shouldn't happen), skip it
                out = handles.get(current_group)
                if out is not None:
                    out.write(header + "\n")
            else:
                if out is not None:
                    out.write(line)

    # Close all files
    for fh in handles.values():
        fh.close()


def main():
    args = parse_args()

    seq_info = fasta_lengths(args.genome)
    if not seq_info:
        sys.stderr.write("No sequences found in genome FASTA.\n")
        return 1

    seq_groups = make_groups(seq_info, args.min_size, args.max_files)

    write_chunks(args.genome, seq_groups, args.outdir, args.prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())
