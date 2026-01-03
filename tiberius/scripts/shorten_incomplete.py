#!/usr/bin/env python3
"""
Shorten incomplete ORFs in a TransDecoder peptide FASTA by trimming sequences to
the first methionine (M) and updating coordinates/length in the header.

Incomplete ORFs are those with "type:5prime_partial" or "type:internal" in the
FASTA description line.

Example
-------
$ shorten_incomplete_orfs.py candidates.pep -o shortened_candidates.pep
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Tuple

try:
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
except ImportError as e:
    sys.stderr.write(
        "ERROR: Biopython is required. Install with: pip install biopython\n"
    )
    raise

COORDS_RE = re.compile(r":(\d+)-(\d+)\(([+-])\)")
LEN_RE = re.compile(r"len:(\d+)")
INCOMPLETE_TYPES = ("type:5prime_partial", "type:internal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Shorten incomplete TransDecoder ORFs to first M and update headers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "transdecoder_pep",
        type=Path,
        help="Path to TransDecoder peptide FASTA (e.g. *transdecoder.pep)"
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("shortened_candidates.pep"),
        help="Output FASTA path",
    )
    p.add_argument(
        "--keep-non-incomplete",
        action="store_true",
        help="Also copy records that are not 5prime_partial/internal unchanged.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return p.parse_args()


def is_incomplete(record: SeqRecord) -> bool:
    desc = record.description or ""
    return any(t in desc for t in INCOMPLETE_TYPES)


def find_first_m_or_none(aa_seq: str) -> int:
    """Return 0-based index of first 'M' in aa_seq, or -1 if absent."""
    return aa_seq.find("M")


def shorten_record(record: SeqRecord) -> Tuple[SeqRecord, bool]:
    """
    If record is incomplete and coordinates are present, shorten to first 'M'
    and update header coords and len. Returns (record, written_flag).
    """
    if not is_incomplete(record):
        return record, False

    # Work on a copy to avoid mutating input iterator state unexpectedly
    rec = record[:]  # shallow copy of SeqRecord
    seq_str = str(rec.seq)

    # Strip a trailing stop if present for length calculation,
    # but preserve it in the written sequence if it existed after trimming.
    has_trailing_stop = seq_str.endswith("*")
    if has_trailing_stop:
        seq_core = seq_str[:-1]
    else:
        seq_core = seq_str

    m_pos = find_first_m_or_none(seq_core)
    if m_pos == -1:
        logging.debug("No 'M' found for %s; skipping.", rec.id)
        return rec, False

    # Shortened sequence (keep trailing '*' if it existed and remains after trim)
    shortened_core = seq_core[m_pos:]
    shortened_seq = shortened_core + ("*" if has_trailing_stop else "")
    rec.seq = rec.seq.__class__(shortened_seq)  # preserve Seq type

    # Update coordinates if present
    coords = COORDS_RE.search(rec.description)
    if not coords:
        logging.debug("No coords found in description for %s; skipping write.", rec.id)
        return rec, False

    old_start = int(coords.group(1))
    old_stop = int(coords.group(2))
    strand = coords.group(3)

    # Nucleotide offset corresponding to trimmed amino acids (3 nt per aa)
    nt_offset = m_pos * 3

    if strand == "+":
        new_start = old_start + nt_offset
        new_stop = old_stop
    else:
        # On the minus strand, 5' is the larger coordinate; trimming from 5' moves 'stop' inward.
        new_start = old_start
        new_stop = old_stop - nt_offset

    # Compute new peptide length excluding terminal '*' if present
    new_len = len(shortened_core)

    # Rewrite description: coords and len
    desc_updated = COORDS_RE.sub(f":{new_start}-{new_stop}({strand})", rec.description, count=1)
    if LEN_RE.search(desc_updated):
        desc_updated = LEN_RE.sub(f"len:{new_len}", desc_updated, count=1)
    else:
        # If len: not present, append it for convenience
        desc_updated = f"{desc_updated} len:{new_len}"

    rec.description = desc_updated
    return rec, True


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if not args.transdecoder_pep.exists():
        logging.error("Input file not found: %s", args.transdecoder_pep)
        return 2

    n_in = 0
    n_written = 0
    n_shortened = 0
    n_skipped_no_m = 0
    n_skipped_no_coords = 0
    n_copied_other = 0

    with args.output.open("w") as out_handle:
        for record in SeqIO.parse(str(args.transdecoder_pep), "fasta"):
            n_in += 1
            if is_incomplete(record):
                rec_short, ok = shorten_record(record)
                if ok:
                    SeqIO.write(rec_short, out_handle, "fasta")
                    n_written += 1
                    n_shortened += 1
                else:
                    # Determine reason for skip for stats
                    seq_str = str(record.seq)
                    core = seq_str[:-1] if seq_str.endswith("*") else seq_str
                    if "M" not in core:
                        n_skipped_no_m += 1
                    elif not COORDS_RE.search(record.description):
                        n_skipped_no_coords += 1
            else:
                if args.keep_non_incomplete:
                    SeqIO.write(record, out_handle, "fasta")
                    n_written += 1
                    n_copied_other += 1

    logging.info("Output written to: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
