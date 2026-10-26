#!/usr/bin/env python3
"""Summarize the spans.jsonl produced by `spliced_stop_hook.py`.

Usage:
    python scripts/diagnostics/analyze_spliced_stop_debug.py <debug_dir>
         [--max_examples N]           # how many examples per bucket to print (default 5)
         [--out summary.txt]          # also write the summary to this file

Classifies every detected span by:
    pre.has_inframe_stop  post.has_inframe_stop  bucket
         True                   True             TP_UNFIXED  (NSS failed to remove the stop)
         True                   False            TP_FIXED    (expected behavior)
         False                  True             FP_INTRODUCED  (reprediction CREATED an inframe stop)
         False                  False            FP_HARMLESS (over-triggered but no damage)

The FP_INTRODUCED bucket is the primary hypothesis for why NEW > OLD in
the benchmark. If it's non-empty, inspect the printed examples to see
what the standard-HMM CDS looked like vs the NSS-repredicted CDS.

The TP_UNFIXED bucket points at a potential HMM-level bug (restrictive
HMM should structurally forbid inframe stops in CDS).

A separate "oversized" tally counts spans where `patched=false` (span
length exceeds `seq_len`); those are left as-is by the pipeline and
contribute equally to NEW and OLD counts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


BUCKETS = ("TP_FIXED", "TP_UNFIXED", "FP_INTRODUCED", "FP_HARMLESS")


def classify(rec: dict) -> str:
    pre = rec["pre"]["has_inframe_stop"]
    post = rec["post"]["has_inframe_stop"]
    if pre and post:
        return "TP_UNFIXED"
    if pre and not post:
        return "TP_FIXED"
    if not pre and post:
        return "FP_INTRODUCED"
    return "FP_HARMLESS"


def format_example(rec: dict, max_bases: int = 180) -> str:
    pre = rec["pre"]
    post = rec["post"]

    def trim(s: str) -> str:
        return s if len(s) <= max_bases else s[:max_bases] + f"... ({len(s)} bases)"

    return (
        f"  call_idx={rec['call_idx']} strand={rec['strand']}  span=[{rec['tx_s']}, {rec['tx_e']})"
        f"  len={rec['span_len']}  patched={rec['patched']}\n"
        f"    pre : regions={pre['cds_regions']} stop_at={pre['stop_frame0_at']}\n"
        f"           bases={trim(pre['cds_bases'])}\n"
        f"           labels_rle(pre)={rec['pre_labels_rle']}\n"
        f"    post: regions={post['cds_regions']} stop_at={post['stop_frame0_at']}"
        f" (txs_in_span={post.get('num_transcripts_in_span', '?')})\n"
        f"           bases={trim(post['cds_bases'])}\n"
        f"           labels_rle(post)={rec['post_labels_rle']}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("debug_dir", help="dir containing spans.jsonl")
    ap.add_argument("--max_examples", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jsonl", default="spans.jsonl",
                    help="override jsonl filename (default: spans.jsonl)")
    args = ap.parse_args()

    log_path = Path(args.debug_dir) / args.jsonl
    if not log_path.exists():
        print(f"ERROR: no log at {log_path}", file=sys.stderr)
        return 1

    bucket_counts: Counter[str] = Counter()
    bucket_examples: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    oversized = 0
    total = 0
    strand_counts: Counter[str] = Counter()

    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            strand_counts[rec["strand"]] += 1
            if not rec["patched"]:
                oversized += 1
            bucket = classify(rec)
            bucket_counts[bucket] += 1
            if len(bucket_examples[bucket]) < args.max_examples:
                bucket_examples[bucket].append(rec)

    lines: list[str] = []
    lines.append(f"Total detected spliced-stop spans: {total}")
    lines.append(f"  + strand: {strand_counts['+']}   - strand: {strand_counts['-']}")
    lines.append(f"  oversized (not repredicted; both arrays identical here): {oversized}")
    lines.append("")
    lines.append("Classification (pre = standard-HMM, post = after NSS overwrite):")
    for b in BUCKETS:
        lines.append(f"  {b:<16s} {bucket_counts[b]}")
    lines.append("")

    tp = bucket_counts["TP_FIXED"] + bucket_counts["TP_UNFIXED"]
    fp = bucket_counts["FP_INTRODUCED"] + bucket_counts["FP_HARMLESS"]
    net = bucket_counts["FP_INTRODUCED"] - bucket_counts["TP_FIXED"]
    lines.append("Interpretation:")
    lines.append(
        f"  True positives (pre already had stop):   {tp}"
        f" -> fixed {bucket_counts['TP_FIXED']}, unfixed {bucket_counts['TP_UNFIXED']}"
    )
    lines.append(
        f"  False positives (pre was clean):          {fp}"
        f" -> introduced stop {bucket_counts['FP_INTRODUCED']},"
        f" harmless {bucket_counts['FP_HARMLESS']}"
    )
    lines.append(
        f"  Net effect on inframe-stop count:         "
        f"{'+' if net >= 0 else ''}{net}"
        "  (NSS introduces this many MORE inframe stops than it fixes)"
    )
    lines.append("")

    for b in BUCKETS:
        exs = bucket_examples[b]
        if not exs:
            continue
        lines.append(f"=== Examples: {b} ({bucket_counts[b]} total, showing {len(exs)}) ===")
        for rec in exs:
            lines.append(format_example(rec))

    out = "\n".join(lines)
    print(out)
    if args.out is not None:
        args.out.write_text(out)
        print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
