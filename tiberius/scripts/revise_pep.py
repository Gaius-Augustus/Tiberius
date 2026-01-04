#!/usr/bin/env python3
"""
Compare DIAMOND results for normal vs. shortened TransDecoder ORFs to classify
CDS candidates, then produce a revised peptide FASTA that incorporates the
classifications and (when appropriate) the shortened sequences.

Two steps:
1) get_cds_classification: label each cdsID as "complete" or "incomplete"
   based on the merged DIAMOND hits from normal and shortened ORFs.
2) get_optimized_pep_file: write a revised FASTA where incomplete ORFs keep
   their original sequence, and complete/internal/5prime_partial are updated
   according to the classification and shortened entries.

Example
-------
$ revise_transdecoder_candidates.py \
    --diamond-normal normal.tsv \
    --diamond-short short.tsv \
    --transdecoder-pep candidates.pep \
    --shortened-pep shortened_candidates.pep \
    --revised-pep revised_candidates.pep \
    --classifications-json classifications.json
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
except ImportError:
    raise SystemExit("ERROR: Biopython is required. Install with: pip install biopython")

HEADER_LIST = [
    "cdsID",
    "proteinID",
    "percIdentMatches",
    "alignLength",
    "mismatches",
    "gapOpenings",
    "queryStart",
    "queryEnd",
    "targetStart",
    "targetEnd",
    "eValue",
    "bitScore",
]

INCOMPLETE_TYPES = {"type:5prime_partial", "type:internal"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify CDS with DIAMOND results and revise TransDecoder PEP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--diamond_normal", required=True, type=Path, help="DIAMOND TSV for normal ORFs")
    p.add_argument("--diamond_short", required=True, type=Path, help="DIAMOND TSV for shortened ORFs")
    p.add_argument("--transdecoder_pep", required=True, type=Path, help="Original TransDecoder PEP")
    p.add_argument("--shortened_pep", required=True, type=Path, help="Shortened PEP (from trimming to first M)")
    p.add_argument("--revised_pep", required=True, type=Path, help="Output: revised PEP")
    p.add_argument("--classifications_json", required=True, type=Path, help="Output: JSON with classifications")
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level",
    )
    return p.parse_args()


def _read_diamond(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=HEADER_LIST)
    # Ensure numeric columns are numeric
    numeric_cols = [c for c in HEADER_LIST if c not in ("cdsID", "proteinID")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def get_cds_classification(normal_tsv: Path, shortened_tsv: Path) -> Dict[str, str]:
    """
    Compare DIAMOND search results to classify codingseq candidates as complete or incomplete.

    Parameters
    ----------
    normal_tsv : Path
        DIAMOND results for the original TransDecoder output.
    shortened_tsv : Path
        DIAMOND results for the shortened ORFs.

    Returns
    -------
    dict
        {cdsID: "complete" | "incomplete"}
    """
    logging.info("Analyzing incomplete ORF predictions...")

    df_normal = _read_diamond(normal_tsv)
    df_short = _read_diamond(shortened_tsv)

    # Join on (cdsID, proteinID) to compare matched hits
    merged = pd.merge(
        df_short,
        df_normal,
        on=["cdsID", "proteinID"],
        suffixes=("_short", "_normal"),
        how="inner",
    )

    if merged.empty:
        logging.warning("Merged DIAMOND table is empty. No overlapping hits found.")
        return {}

    # Keep only columns we use
    cols_needed = [
        "cdsID",
        "proteinID",
        "queryStart_normal",
        "targetStart_normal",
        "percIdentMatches_normal",
        "bitScore_normal",
        "queryStart_short",
        "targetStart_short",
        "percIdentMatches_short",
        "bitScore_short",
    ]
    merged = merged[cols_needed].copy()

    # Support score:
    # (t_complete_start - t_incomplete_start) - (q_incomplete_start - 1) + log(aai_incomplete / aai_complete)^1000
    aai_incomp = merged["percIdentMatches_normal"].astype(float)
    aai_comp = merged["percIdentMatches_short"].astype(float)
    aai_comp_safe = aai_comp.replace(0.0, 1e-4)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        match_log = np.log(aai_incomp / aai_comp_safe)
        # preserve original logic: power of 1000 (may explode; users often want this exact behavior)
        match_component = np.power(match_log, 1000)

    support_score = (
        (merged["targetStart_short"] - merged["targetStart_normal"])
        - (merged["queryStart_normal"] - 1)
        + match_component
    )
    merged["supportScore"] = support_score

    # Best bit score per row across short/normal
    merged["bitScore_max"] = merged[["bitScore_short", "bitScore_normal"]].max(axis=1)

    # For each cdsID, look at top 25 hits by bitScore_max; if any supportScore > 0 => "incomplete"
    classifications: Dict[str, str] = {}
    for cds_id, group in merged.groupby("cdsID", sort=False):
        top = group.sort_values("bitScore_max", ascending=False).head(25)
        incomplete = (top["supportScore"] > 0).any()
        classifications[cds_id] = "incomplete" if incomplete else "complete"

    logging.info("CDS classification completed successfully. n=%d", len(classifications))
    return classifications


def _record_type(description: str):
    if "type:complete" in description:
        return "complete"
    if "type:5prime_partial" in description:
        return "5prime_partial"
    if "type:internal" in description:
        return "internal"
    if "type:3prime_partial" in description:
        return "3prime_partial"
    return None


def get_optimized_pep_file(
    normal_pep: Path,
    shortened_pep: Path,
    classifications: Dict[str, str],
    output_path: Path,
) -> Path:
    """
    Create a revised PEP file by incorporating classifications and shortened ORFs.

    Parameters
    ----------
    normal_pep : Path
        Original PEP file with all ORFs.
    shortened_pep : Path
        PEP file containing shortened ORFs.
    classifications : dict
        Mapping cdsID -> ("complete" | "incomplete").
    output_path : Path
        Where to write the revised PEP.

    Returns
    -------
    Path
        The path to the revised PEP.
    """
    # Ensure parent directory exists if provided
    if output_path.parent and str(output_path.parent) != "":
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Preload shortened ORFs
    shortened_pep_dict: Dict[str, tuple] = {
        rec.id: (rec.seq, rec.description) for rec in SeqIO.parse(str(shortened_pep), "fasta")
    }
    logging.info("Loaded %d shortened ORFs.", len(shortened_pep_dict))

    n_in = 0
    n_written = 0
    n_updated = 0
    n_copied = 0
    n_missing_short = 0

    with output_path.open("w") as out_handle:
        for record in SeqIO.parse(str(normal_pep), "fasta"):
            n_in += 1
            rec_type = _record_type(record.description) or "unknown"

            if rec_type in {"5prime_partial", "internal"}:
                cds_id = record.id
                if cds_id in classifications:
                    if classifications[cds_id] == "incomplete":
                        # Keep original record
                        SeqIO.write(record, out_handle, "fasta")
                        n_written += 1
                        n_copied += 1
                    else:
                        # Replace with shortened sequence + header fixups
                        if cds_id not in shortened_pep_dict:
                            # Fallback: write original if shortened not found
                            SeqIO.write(record, out_handle, "fasta")
                            n_written += 1
                            n_missing_short += 1
                            continue

                        seq, desc = shortened_pep_dict[cds_id]
                        record.seq = seq
                        if "type:5prime_partial" in desc:
                            desc = desc.replace("type:5prime_partial", "type:complete")
                        elif "type:internal" in desc:
                            desc = desc.replace("type:internal", "type:3prime_partial")
                        record.description = desc
                        SeqIO.write(record, out_handle, "fasta")
                        n_written += 1
                        n_updated += 1
                else:
                    # No classification: keep original
                    SeqIO.write(record, out_handle, "fasta")
                    n_written += 1
                    n_copied += 1
            else:
                # complete / 3prime_partial / unknown: write unchanged
                SeqIO.write(record, out_handle, "fasta")
                n_written += 1
                n_copied += 1

    logging.info("Revised PEP written: %s", output_path)
    logging.info("Input records:       %d", n_in)
    logging.info("Written records:     %d", n_written)
    logging.info("  Updated (short):   %d", n_updated)
    logging.info("  Copied unchanged:  %d", n_copied)
    if n_missing_short:
        logging.info("  Missing shortened: %d", n_missing_short)

    return output_path


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    for p in (
        args.diamond_normal,
        args.diamond_short,
        args.transdecoder_pep,
        args.shortened_pep,
    ):
        if not p.exists():
            logging.error("Required input not found: %s", p)
            return 2

    classes = get_cds_classification(args.diamond_normal, args.diamond_short)

    # Write classifications JSON
    args.classifications_json.parent.mkdir(parents=True, exist_ok=True)
    with args.classifications_json.open("w") as fh:
        json.dump(classes, fh)
    logging.info("Classifications JSON written: %s (n=%d)", args.classifications_json, len(classes))

    # Create revised PEP
    get_optimized_pep_file(
        args.transdecoder_pep,
        args.shortened_pep,
        classes,
        args.revised_pep,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
