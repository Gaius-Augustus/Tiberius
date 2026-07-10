#!/usr/bin/env python3
"""Cap the number of transcripts per gene in a GFF3.

Streams a GFF3 on stdin, keeps at most --max transcripts per gene
(preferring the ones with the largest total CDS length), and writes
the reduced GFF3 to stdout. Non-feature lines (`##`, blank) are
preserved. Used before `gffread -y` in `PROTEIN_FROM_GFF` to work
around a gffread malloc() crash on genes with many isoforms; the
protein file is only consumed by DIAMOND species ranking, so one
representative per gene is enough.
"""
import argparse
import sys
from collections import defaultdict


def parse_attrs(attr_str):
    attrs = {}
    for part in attr_str.strip().split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        attrs[k.strip()] = v.strip()
    return attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("gff", nargs="?", default="-")
    args = ap.parse_args()

    src = sys.stdin if args.gff == "-" else open(args.gff, "r", encoding="utf-8")
    all_lines = src.readlines()
    if src is not sys.stdin:
        src.close()

    tx_parent = {}
    tx_cds_len = defaultdict(int)
    for line in all_lines:
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 9:
            continue
        ftype = cols[2]
        attrs = parse_attrs(cols[8])
        if ftype in {"mRNA", "transcript"}:
            tid = attrs.get("ID")
            parent = attrs.get("Parent", "").split(",")[0]
            if tid:
                tx_parent[tid] = parent
                tx_cds_len.setdefault(tid, 0)
        elif ftype == "CDS":
            parent = attrs.get("Parent", "").split(",")[0]
            if parent:
                tx_cds_len[parent] += int(cols[4]) - int(cols[3]) + 1

    by_gene = defaultdict(list)
    for tid, gid in tx_parent.items():
        by_gene[gid].append(tid)

    keep = set()
    for gid, tids in by_gene.items():
        tids_sorted = sorted(tids, key=lambda t: (-tx_cds_len[t], t))
        keep.update(tids_sorted[: args.max])

    for line in all_lines:
        if not line.strip() or line.startswith("#"):
            sys.stdout.write(line)
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) != 9:
            sys.stdout.write(line)
            continue
        ftype = cols[2]
        attrs = parse_attrs(cols[8])
        if ftype == "gene":
            sys.stdout.write(line)
        elif ftype in {"mRNA", "transcript"}:
            if attrs.get("ID") in keep:
                sys.stdout.write(line)
        else:
            parent = attrs.get("Parent", "").split(",")[0]
            if parent in keep:
                sys.stdout.write(line)


if __name__ == "__main__":
    main()
