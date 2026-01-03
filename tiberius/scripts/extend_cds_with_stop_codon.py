#!/usr/bin/env python3
"""
Extend CDS features to include stop codons and rebuild a clean GFF3.

Strategy:
    - Read input GFF3
    - For each transcript:
        * extend CDS with stop_codon coordinates
    - Build a NEW GFF3 that only contains:
        * gene  (spans all transcripts' CDS regions)
        * transcript  (replaces mRNA, spans all its CDS regions)
        * exon  (one per CDS, same coords as CDS)
        * CDS   (extended to include stop codon)
        * UTR   (five_prime_UTR / three_prime_UTR / UTR / 5UTR / 3UTR), if present

    - UTRs do NOT influence any logic (bounds, clustering, CDS extension).
    - If gene or transcript lines are missing, they are synthesized.
"""

import sys
import argparse
from collections import defaultdict

# ------------ Helpers ------------ #

def parse_attributes(attr_str: str) -> dict:
    """Parse GFF3 attributes column into a dict."""
    attrs = {}
    if not attr_str:
        return attrs
    for part in attr_str.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            attrs[key] = value
    return attrs


def attrs_to_str(attrs: dict) -> str:
    """Convert attributes dict back to a GFF3 attribute string.

    ID first, Parent second (if present), others sorted by key.
    """
    if not attrs:
        return "."
    items = []
    if "ID" in attrs:
        items.append(f"ID={attrs['ID']}")
    if "Parent" in attrs:
        items.append(f"Parent={attrs['Parent']}")
    for k in sorted(attrs.keys()):
        if k in {"ID", "Parent"}:
            continue
        items.append(f"{k}={attrs[k]}")
    return ";".join(items) if items else "."


# ------------ Data structures ------------ #

class Feature:
    """Simple container for a GFF feature."""
    __slots__ = (
        "seqid", "source", "type", "start", "end",
        "score", "strand", "phase", "attrs"
    )

    def __init__(self, seqid, source, ftype, start, end,
                 score, strand, phase, attrs):
        self.seqid = seqid
        self.source = source
        self.type = ftype
        self.start = int(start)
        self.end = int(end)
        self.score = score
        self.strand = strand
        self.phase = phase
        self.attrs = attrs

    def to_gff_row(self):
        return [
            self.seqid,
            self.source,
            self.type,
            str(self.start),
            str(self.end),
            self.score,
            self.strand,
            self.phase,
            attrs_to_str(self.attrs),
        ]


# ------------ Core logic ------------ #

def read_gff(gff_handle):
    """Parse GFF into gene, transcript, CDS, stop_codon and UTR structures."""
    genes = {}                        # gene_id -> Feature
    transcripts = {}                  # tx_id -> Feature (mRNA/transcript)
    cds_by_tx = defaultdict(list)     # tx_id -> [Feature]
    stops_by_tx = defaultdict(list)   # tx_id -> [Feature]
    utrs_by_tx = defaultdict(list)    # tx_id -> [Feature]
    gene_children = defaultdict(list) # gene_id -> [tx_id]

    UTR_TYPES = {
        "five_prime_UTR",
        "three_prime_UTR",
        "UTR",
        "5UTR",
        "3UTR",
    }

    for line in gff_handle:
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue

        cols = line.split("\t")
        if len(cols) != 9:
            # Skip malformed feature lines
            continue

        seqid, source, ftype, start, end, score, strand, phase, attr_str = cols
        attrs = parse_attributes(attr_str)
        fid = attrs.get("ID")
        parent = attrs.get("Parent")

        feat = Feature(seqid, source, ftype, start, end, score, strand, phase, attrs)

        if ftype == "gene" and fid:
            genes[fid] = feat
        elif ftype in {"mRNA", "transcript"} and fid:
            transcripts[fid] = feat
            if parent:
                gene_children[parent].append(fid)
        elif ftype == "CDS" and parent:
            cds_by_tx[parent].append(feat)
        elif ftype == "stop_codon" and parent:
            stops_by_tx[parent].append(feat)
        elif ftype in UTR_TYPES and parent:
            utrs_by_tx[parent].append(feat)
        # everything else is ignored in the rebuilt GFF

    return genes, transcripts, cds_by_tx, stops_by_tx, utrs_by_tx, gene_children


def extend_cds_with_stops(cds_by_tx, stops_by_tx):
    """Extend CDS segments to include stop codons for each transcript."""
    for tx_id, cds_list in cds_by_tx.items():
        if tx_id not in stops_by_tx:
            continue
        if not cds_list:
            continue
        stop_list = stops_by_tx[tx_id]
        # assume all CDS and stops share same strand/seqid
        strand = cds_list[0].strand

        if strand == "+":
            # choose stop with largest end as terminal stop
            chosen_stop = max(stop_list, key=lambda f: f.end)
            # extend CDS with largest end
            last_cds = max(cds_list, key=lambda f: f.end)
            if chosen_stop.end > last_cds.end:
                last_cds.end = chosen_stop.end
        elif strand == "-":
            # choose stop with smallest start as terminal stop
            chosen_stop = min(stop_list, key=lambda f: f.start)
            # extend CDS with smallest start
            first_cds = min(cds_list, key=lambda f: f.start)
            if chosen_stop.start < first_cds.start:
                first_cds.start = chosen_stop.start
        # if strand is '.', we don't do anything


def rebuild_gff(genes, transcripts, cds_by_tx, utrs_by_tx, gene_children, out_handle):
    """Build a new GFF3 using the extended CDS coordinates.

    Handles missing genes and/or transcripts by synthesizing them.

    IMPORTANT:
        - CDS is still used for exons and CDS features.
        - UTRs are preserved AND now contribute to gene/transcript spans.
    """
    # 1) compute CDS bounds per transcript
    tx_cds_bounds = {}  # tx_id -> (seqid, strand, min_start, max_end)
    for tx_id, cds_list in cds_by_tx.items():
        if not cds_list:
            continue
        # assume consistent seqid/strand
        seqid = cds_list[0].seqid
        strand = cds_list[0].strand
        starts = [f.start for f in cds_list]
        ends = [f.end for f in cds_list]
        tx_cds_bounds[tx_id] = (seqid, strand, min(starts), max(ends))

    # 1a) extend transcript bounds with UTR, if present
    #     (this will be used for gene and transcript coordinates only)
    tx_bounds = dict(tx_cds_bounds)  # start with CDS-only bounds
    for tx_id, utr_list in utrs_by_tx.items():
        if tx_id not in tx_bounds:
            # transcript with UTR but no CDS: original script ignored,
            # keep that behavior (we don't create genes/txs for them)
            continue
        seqid, strand, t_start, t_end = tx_bounds[tx_id]
        utr_starts = [f.start for f in utr_list]
        utr_ends = [f.end for f in utr_list]
        t_start = min(t_start, min(utr_starts))
        t_end = max(t_end, max(utr_ends))
        tx_bounds[tx_id] = (seqid, strand, t_start, t_end)

    # 1b) synthesize missing transcript features (CDS-only input)
    for tx_id, (seqid, strand, t_start, t_end) in tx_bounds.items():
        if tx_id not in transcripts:
            # Use first CDS feature as template for source, etc.
            cds0 = cds_by_tx[tx_id][0]
            attrs = {"ID": tx_id}
            transcripts[tx_id] = Feature(
                seqid=seqid,
                source=cds0.source,
                ftype="transcript",
                start=t_start,
                end=t_end,
                score=".",
                strand=strand,
                phase=".",
                attrs=attrs,
            )

    # 1c) build a complete gene_children map, synthesizing genes if missing
    complete_gene_children = defaultdict(list)

    # include existing gene->transcript links
    for gene_id, tx_ids in gene_children.items():
        for tx_id in tx_ids:
            if tx_id in tx_bounds:  # only keep transcripts w/ CDS (as before)
                complete_gene_children[gene_id].append(tx_id)

    # for each transcript with CDS, ensure it belongs to some gene
    for tx_id in tx_bounds:
        tx_feat = transcripts.get(tx_id)
        parent_gene = tx_feat.attrs.get("Parent") if tx_feat else None

        if parent_gene and parent_gene in genes:
            # transcript has a real gene parent
            if tx_id not in complete_gene_children[parent_gene]:
                complete_gene_children[parent_gene].append(tx_id)
        else:
            # no gene parent: synthesize one
            gene_id = f"{tx_id}.gene"
            if tx_id not in complete_gene_children[gene_id]:
                complete_gene_children[gene_id].append(tx_id)
            if gene_id not in genes:
                seqid, strand, t_start, t_end = tx_bounds[tx_id]
                attrs = {"ID": gene_id}
                genes[gene_id] = Feature(
                    seqid=seqid,
                    source="rebuild",
                    ftype="gene",
                    start=t_start,
                    end=t_end,
                    score=".",
                    strand=strand,
                    phase=".",
                    attrs=attrs,
                )

    # 2) compute gene bounds from their transcripts (now CDS+UTR span)
    gene_bounds = {}  # gene_id -> (seqid, strand, min_start, max_end)
    for gene_id, tx_ids in complete_gene_children.items():
        # restrict to transcripts that have CDS (as before)
        cds_tx_ids = [t for t in tx_ids if t in tx_bounds]
        if not cds_tx_ids:
            continue
        seqid, strand, g_start, g_end = None, None, None, None
        for t in cds_tx_ids:
            t_seqid, t_strand, t_start, t_end = tx_bounds[t]
            if seqid is None:
                seqid, strand = t_seqid, t_strand
                g_start, g_end = t_start, t_end
            else:
                g_start = min(g_start, t_start)
                g_end = max(g_end, t_end)
        gene_bounds[gene_id] = (seqid, strand, g_start, g_end)

    # 3) write GFF3
    out_handle.write("##gff-version 3\n")

    # sort genes for stable output
    def gene_sort_key(gid):
        if gid in gene_bounds:
            seqid, strand, start, end = gene_bounds[gid]
        else:
            seqid = genes[gid].seqid if gid in genes else "chr"
            start = genes[gid].start if gid in genes else 0
        return (seqid, start, gid)

    for gene_id in sorted(gene_bounds.keys(), key=gene_sort_key):
        if gene_id not in genes:
            continue

        gene_feat = genes[gene_id]
        g_seqid, g_strand, g_start, g_end = gene_bounds[gene_id]

        # update gene coords to CDS+UTR span
        gene_feat.seqid = g_seqid
        gene_feat.strand = g_strand
        gene_feat.start = g_start
        gene_feat.end = g_end

        # output gene
        out_handle.write("\t".join(gene_feat.to_gff_row()) + "\n")

        # transcripts under this gene that have CDS
        tx_ids = [t for t in complete_gene_children[gene_id] if t in tx_bounds]

        # sort transcripts by start
        tx_ids_sorted = sorted(
            tx_ids,
            key=lambda t: (tx_bounds[t][0], tx_bounds[t][2], t),
        )

        for tx_id in tx_ids_sorted:
            if tx_id not in transcripts:
                continue
            tx_feat = transcripts[tx_id]
            t_seqid, t_strand, t_start, t_end = tx_bounds[tx_id]

            # rename feature type to transcript
            tx_feat.type = "transcript"
            tx_feat.seqid = t_seqid
            tx_feat.strand = t_strand
            tx_feat.start = t_start   # CDS+UTR span
            tx_feat.end = t_end       # CDS+UTR span

            # ensure Parent is the gene
            tx_feat.attrs["Parent"] = gene_id

            out_handle.write("\t".join(tx_feat.to_gff_row()) + "\n")

            # CDS (and exon) features for this transcript
            cds_list = cds_by_tx[tx_id]
            # sort by genomic coord
            cds_list_sorted = sorted(cds_list, key=lambda f: (f.start, f.end))

            exon_counter = 1
            cds_counter = 1

            for cds_feat in cds_list_sorted:
                # exon (still exactly matching CDS)
                exon_attrs = {
                    "ID": f"{tx_id}.exon{exon_counter}",
                    "Parent": tx_id,
                }
                exon = Feature(
                    seqid=cds_feat.seqid,
                    source=cds_feat.source,
                    ftype="exon",
                    start=cds_feat.start,
                    end=cds_feat.end,
                    score=cds_feat.score,
                    strand=cds_feat.strand,
                    phase=".",  # exon has no phase
                    attrs=exon_attrs,
                )
                out_handle.write("\t".join(exon.to_gff_row()) + "\n")
                exon_counter += 1

                # CDS
                cds_attrs = dict(cds_feat.attrs)  # copy
                cds_attrs["Parent"] = tx_id
                cds_attrs.setdefault("ID", f"{tx_id}.cds{cds_counter}")
                cds_out = Feature(
                    seqid=cds_feat.seqid,
                    source=cds_feat.source,
                    ftype="CDS",
                    start=cds_feat.start,
                    end=cds_feat.end,
                    score=cds_feat.score,
                    strand=cds_feat.strand,
                    phase=cds_feat.phase,
                    attrs=cds_attrs,
                )
                out_handle.write("\t".join(cds_out.to_gff_row()) + "\n")
                cds_counter += 1

            # UTRs (preserved, now *also* covered by gene/tx coords)
            if tx_id in utrs_by_tx:
                utr_list = utrs_by_tx[tx_id]
                utr_list_sorted = sorted(utr_list, key=lambda f: (f.start, f.end))
                utr_counter = 1
                for utr_feat in utr_list_sorted:
                    utr_attrs = dict(utr_feat.attrs)
                    utr_attrs["Parent"] = tx_id
                    utr_attrs.setdefault("ID", f"{tx_id}.utr{utr_counter}")
                    utr_out = Feature(
                        seqid=utr_feat.seqid,
                        source=utr_feat.source,
                        ftype=utr_feat.type,
                        start=utr_feat.start,
                        end=utr_feat.end,
                        score=utr_feat.score,
                        strand=utr_feat.strand,
                        phase=".",  # UTR has no phase
                        attrs=utr_attrs,
                    )
                    out_handle.write("\t".join(utr_out.to_gff_row()) + "\n")
                    utr_counter += 1



# ------------ Main ------------ #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extend CDS with stop codons and rebuild a clean GFF3 containing "
            "gene/transcript/exon/CDS features, plus UTRs if present. "
            "Missing gene/transcript features are synthesized. "
            "UTRs are preserved but do not affect CDS-based logic."
        )
    )
    parser.add_argument(
        "gff",
        help="Input GFF3 file (use - for stdin)."
    )
    args = parser.parse_args()

    if args.gff == "-":
        gff_handle = sys.stdin
    else:
        gff_handle = open(args.gff, "r", encoding="utf-8")

    genes, transcripts, cds_by_tx, stops_by_tx, utrs_by_tx, gene_children = read_gff(gff_handle)

    if gff_handle is not sys.stdin:
        gff_handle.close()

    # extend CDS with stop codons
    extend_cds_with_stops(cds_by_tx, stops_by_tx)

    # rebuild GFF using only updated CDS for logic, but keeping UTRs
    rebuild_gff(genes, transcripts, cds_by_tx, utrs_by_tx, gene_children, sys.stdout)


if __name__ == "__main__":
    main()
