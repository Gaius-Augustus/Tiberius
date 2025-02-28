#!/usr/bin/env python3

"""
Reads the transcript-level CSV produced by our GTF analysis script and provides summary statistics.
It prints out aggregated statistics (count, sum, mean, std, min, max) for each numeric column.
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Summarize transcript-level statistics from CSV.")
    parser.add_argument("-i", "--input_csv", required=True,
                        help="Path to the CSV file produced by the stats script.")
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input_csv)

    # Identify numeric columns (excluding transcript_id, gene_id)
    # Adjust this list if you changed your column names or structure.
    numeric_cols = [
        "coding_length", 
        "cds_count", 
        "overall_gc", 
        "cds_gc", 
        "has_canonical_start", 
        "has_inframe_stop", 
        "repeats_in_cds_fraction", 
        "repeats_in_cds_intron_fraction", 
        "canonical_splice_sites"
    ]
    d = df[df["has_inframe_stop"] == 1]
    print(len(d["gene_id"]))
    # Generate summary stats for these columns
    # The transpose (T) is to list columns in rows, which is often easier to read.
    summary_stats = df[numeric_cols].agg(["count", "sum", "mean", "std", "min", "max"]).T

    # Print a nice table
    print("Summary of transcript-level metrics:\n")
    print(summary_stats.to_string(float_format="%.4f"))
    print("\nDone.")

if __name__ == "__main__":
    main()
