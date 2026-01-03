#!/usr/bin/env python3
"""
Check whether transcripts end with a valid stop codon and have no in-frame internal stops.
Optionally output a filtered GFF containing only transcripts that pass this check.

Usage:
    # Just produce a TSV summary:
    python check_stop_codons.py input.gff genome.fa > stop_check.tsv

    # TSV + filtered GFF with only "good" transcripts:
    python check_stop_codons.py input.gff genome.fa \
        --write-passing-gff passing.gff > stop_check.tsv
"""

import sys
import argparse
from collections import defaultdict

STOP_CODONS = {"TAA", "TAG", "TGA"}


# -----------------------------------------------------------
# FASTA
# -----------------------------------------------------------
def parse_fasta(path):
    """
    Load a FASTA file into a dict: {seqid: sequence (uppercased, no gaps)}.
    """
    seqs = {}
    seq_id = None
    chunks = []

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    seqs[seq_id] = "".join(chunks).upper()
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if seq_id is not None:
            seqs[seq_id] = "".join(chunks).upper()
    return seqs


# -----------------------------------------------------------
# GFF parsing helpers
# -----------------------------------------------------------
def parse_attributes(attr_str):
    """
    Parse GFF3 attributes column into a dict.
    """
    attrs = {}
    for part in attr_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k] = v
    return attrs


def revcomp(seq):
    """
    Reverse-complement a DNA sequence.
    """
    t = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(t)[::-1]


def load_cds_from_gff(gff_path):
    """
    Read GFF file; return:
      transcripts[tid] = list of (seqid,start,end,strand)
      lines: list of raw lines (for rewriting GFF)
      parents: map from line index -> transcript ID(s) (Parent=)
    """
    transcripts = defaultdict(list)
    lines = []
    parents = {}  # line index -> set of transcript IDs

    with open(gff_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            lines.append(line)

            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 9:
                continue

            seqid, _, ftype, start, end, _, strand, _, attrs_str = cols
            attrs = parse_attributes(attrs_str)
            parent = attrs.get("Parent")
            if parent is None:
                continue

            # track which transcript(s) the line belongs to
            parents.setdefault(i, set()).add(parent)

            if ftype == "CDS":
                transcripts[parent].append(
                    (seqid, int(start), int(end), strand)
                )

    return transcripts, lines, parents


# -----------------------------------------------------------
# Build CDS sequence for a transcript
# -----------------------------------------------------------
def build_cds_sequence(segments, genome):
    """
    Build concatenated CDS sequence in 5'->3' transcript orientation.

    Returns: (seq, seqid, strand)
    """
    if not segments:
        return None, None, None

    seqid = segments[0][0]
    strand = segments[0][3]

    if seqid not in genome:
        return None, seqid, strand

    if strand == "+":
        segs_sorted = sorted(segments, key=lambda x: x[1])
    else:
        segs_sorted = sorted(segments, key=lambda x: x[1], reverse=False)

    chrom = genome[seqid]
    pieces = []
    for _, s, e, _ in segs_sorted:
        # GFF: 1-based inclusive; Python: 0-based, end-exclusive
        pieces.append(chrom[s - 1 : e])

    seq = "".join(pieces)
    if strand == "-":
        seq = revcomp(seq)

    return seq, seqid, strand


# -----------------------------------------------------------
# Main analysis
# -----------------------------------------------------------
def analyze_cds(cds_seq):
    """
    Given a CDS sequence, compute:
      - length
      - frame_ok
      - last_codon
      - has_valid_stop
      - has_internal_stop, n_internal_stops,
        first_internal_stop_pos_nt, first_internal_stop_codon
    """
    L = len(cds_seq)
    frame_ok = (L % 3 == 0 and L >= 3)
    last_codon = cds_seq[-3:] if L >= 3 else "NA"
    has_valid_stop = frame_ok and last_codon in STOP_CODONS

    has_internal = False
    n_internal = 0
    first_pos = "NA"
    first_codon = "NA"

    if frame_ok and L > 3:
        # iterate over all codons except the last one
        for i in range(0, L - 3, 3):
            codon = cds_seq[i : i + 3]
            if codon in STOP_CODONS:
                n_internal += 1
                if not has_internal:
                    has_internal = True
                    # CDS coordinate: 1-based nucleotide position of first base of codon
                    first_pos = i + 1
                    first_codon = codon

    return (L, frame_ok, last_codon, has_valid_stop,
            has_internal, n_internal, first_pos, first_codon)


def check_stop_codons_and_filter(gff_path, fasta_path, passing_gff_path, out_handle):

    genome = parse_fasta(fasta_path)
    transcripts, lines, parents = load_cds_from_gff(gff_path)

    # Results table header
    out_handle.write("\t".join(
        [
            "transcript_id",
            "seqid",
            "strand",
            "cds_len",
            "last_codon",
            "frame_ok",
            "has_valid_stop",
            "has_internal_stop",
            "n_internal_stops",
            "first_internal_stop_pos_nt",
            "first_internal_stop_codon",
        ]
    ) + "\n")

    passing_tids = set()

    for tid, segs in sorted(transcripts.items()):
        cds, seqid, strand = build_cds_sequence(segs, genome)

        if cds is None:
            out_handle.write(
                "\t".join(
                    [
                        tid,
                        seqid if seqid is not None else ".",
                        strand if strand is not None else ".",
                        "0",
                        "NA",
                        "False",
                        "False",
                        "False",
                        "0",
                        "NA",
                        "NA",
                    ]
                ) + "\n"
            )
            continue

        (L, frame_ok, last_codon, has_valid_stop,
         has_internal, n_internal, first_pos, first_codon) = analyze_cds(cds)

        out_handle.write(
            "\t".join(
                [
                    tid,
                    seqid,
                    strand,
                    str(L),
                    last_codon,
                    str(frame_ok),
                    str(has_valid_stop),
                    str(has_internal),
                    str(n_internal),
                    str(first_pos),
                    first_codon,
                ]
            ) + "\n"
        )

        # Define "passing" transcripts:
        #   - correct frame
        #   - valid terminal stop
        #   - no internal in-frame stops
        if frame_ok and has_valid_stop and not has_internal:
            passing_tids.add(tid)

    # Optional output filtered GFF with only passing transcripts
    if passing_gff_path:
        with open(passing_gff_path, "w", encoding="utf-8") as outgff:
            for idx, line in enumerate(lines):
                if line.startswith("#") or not line.strip():
                    outgff.write(line + "\n")
                    continue

                tids_here = parents.get(idx, set())
                if not tids_here:
                    continue

                if any(t in passing_tids for t in tids_here):
                    outgff.write(line + "\n")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=(
            "Check GFF CDS stop codons (terminal + in-frame internal). "
            "Optionally write a filtered GFF with only transcripts that "
            "end in a valid stop codon and have no internal in-frame stops."
        )
    )
    ap.add_argument("gff", help="Input GFF3 file with CDS features")
    ap.add_argument("genome", help="Genome FASTA file")
    ap.add_argument(
        "--write-passing-gff",
        help="Write a GFF including only transcripts that pass the stop-codon checks",
    )

    args = ap.parse_args()

    check_stop_codons_and_filter(
        args.gff,
        args.genome,
        args.write_passing_gff,
        sys.stdout,
    )


if __name__ == "__main__":
    main()
