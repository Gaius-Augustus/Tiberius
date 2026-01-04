#!/usr/bin/env python3
"""
Merge multiple GTF/GFF files into a single GFF3 file.

Two modes:

1) Full merge (mode=full)
   - Take the union of all transcripts across all inputs.
   - Define each transcript by its exon structure (seqid, strand, exon coordinates).
   - Cluster overlapping transcripts into "genes" (loci) per seqid/strand.
   - Within each locus, collapse identical transcript structures.
   - Assign new gene IDs: gene_000001, gene_000002, ...
   - Assign new transcript IDs: gene_000001.t1, gene_000001.t2, ...

2) Priority merge (mode=priority)
   - One input file is chosen as priority via --priority-file.
   - Build genes from priority file transcripts (same clustering as in full mode).
   - Keep all these priority genes.
   - For transcripts from non-priority files:
        * Discard any transcript that overlaps any priority gene (same seqid & strand, any bp overlap).
        * Keep non-overlapping transcripts, build additional genes from them.
   - Collapse identical transcripts within each locus as in full mode.

UTR handling:
   - UTR features are parsed and carried over to output for any transcript that has them.
   - UTRs do NOT affect transcript merging/deduplication (structure key is exon-only).
   - If identical transcript structures exist in multiple inputs, the representative with the
     longer total UTR region is retained (ties keep the first encountered).
   - Output transcript/gene coordinates include UTRs when present (merge logic still does not).

Input:
    One or more GTF/GFF files (positional arguments). Each may be GTF or GFF3.
    We assume transcripts are defined by 'exon' and/or 'CDS' features.

Output:
    A GFF3 written to stdout (or redirected by the user). Source (column 2)
    preserves the originating annotation source for each gene and child feature.
"""

import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set


# ----------------- Attribute parsing ----------------- #

def parse_attributes_mixed(attr_str: str) -> Dict[str, str]:
    """
    Parse a GTF or GFF3 attributes column into a dict.

    Supports:
        - GFF3 style: key=value;key2=value2
        - GTF style:  key "value"; gene_id "X"; transcript_id "Y";
    """
    attrs: Dict[str, str] = {}
    if not attr_str or attr_str.strip() in {".", ""}:
        return attrs

    parts = [p for p in attr_str.strip().split(";") if p.strip()]
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" in part:  # GFF3-like
            key, value = part.split("=", 1)
            attrs[key.strip()] = value.strip()
        else:  # likely GTF-like: key "value"
            toks = part.split(None, 1)
            key = toks[0].strip()
            if len(toks) > 1:
                value = toks[1].strip().strip('"')
            else:
                value = ""
            attrs[key] = value
    return attrs


def attrs_to_str(attrs: Dict[str, str]) -> str:
    """Convert attributes dict back to a GFF3 attribute string."""
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


# ----------------- Interval helpers ----------------- #

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping/adjacent intervals. Input may be unsorted."""
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals_sorted[0]]
    for s, e in intervals_sorted[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def total_bp(intervals: List[Tuple[int, int]]) -> int:
    """Total basepairs covered by merged intervals."""
    return sum(e - s + 1 for s, e in merge_intervals(intervals))


# ----------------- Data structures ----------------- #

# Common UTR feature type names found in GFF3/GTF exports
UTR_TYPES = {
    "UTR",
    "five_prime_UTR",
    "three_prime_UTR",
    "5UTR",
    "3UTR",
    "5_prime_UTR",
    "3_prime_UTR",
}

@dataclass
class Transcript:
    """Transcript defined by its exon structure."""
    internal_id: str
    source_index: int
    seqid: str
    strand: str
    source_labels: Set[str] = field(default_factory=set)
    exons: List[Tuple[int, int]] = field(default_factory=list)
    cds: List[Tuple[int, int, str]] = field(default_factory=list)

    # store UTRs as (start, end, ftype)
    utr: List[Tuple[int, int, str]] = field(default_factory=list)

    @property
    def source(self) -> str:
        labels = sorted({s for s in self.source_labels if s})
        return ",".join(labels) if labels else "merge"

    @property
    def exon_intervals(self) -> List[Tuple[int, int]]:
        """Exon-like intervals used for structure/output. Falls back to CDS if no exons."""
        if self.exons:
            return self.exons
        return [(s, e) for s, e, _ in self.cds]

    @property
    def span(self) -> Tuple[int, int]:
        """
        (start, end) of the transcript span used for MERGE LOGIC.
        Intentionally ignores UTR so merging behavior is unchanged.
        """
        coords = self.exon_intervals
        starts = [s for s, _ in coords]
        ends = [e for _, e in coords]
        return min(starts), max(ends)

    @property
    def output_span(self) -> Tuple[int, int]:
        """
        (start, end) of transcript span used for OUTPUT.
        Includes UTR if present so UTR features fall within transcript coords.
        """
        starts = []
        ends = []
        # merge logic anchor: exons/CDS
        for s, e in self.exon_intervals:
            starts.append(s)
            ends.append(e)
        # output extension: UTR
        for s, e, _ in self.utr:
            starts.append(s)
            ends.append(e)
        return min(starts), max(ends)

    @property
    def struct_key(self) -> Tuple[str, str, Tuple[Tuple[int, int], ...]]:
        """
        Structure key for deduplication (unchanged conceptually):
        (seqid, strand, sorted_exons) -- UTR is intentionally excluded.
        """
        exons_sorted = tuple(sorted(self.exon_intervals))
        return self.seqid, self.strand, exons_sorted

    def utr_bp(self) -> int:
        """Total bp covered by UTR intervals (merged), regardless of UTR subtype."""
        return total_bp([(s, e) for s, e, _ in self.utr])


@dataclass
class GeneCluster:
    """Group of transcripts representing one merged gene locus."""
    gene_id: str
    seqid: str
    strand: str
    start: int
    end: int
    transcripts: List[Transcript] = field(default_factory=list)


# ----------------- Parsing inputs ----------------- #

def parse_transcripts_from_file(path: str, source_index: int) -> List[Transcript]:
    """
    Parse one GTF/GFF file and return a list of Transcript objects.
    Transcripts are defined by exon/CDS features. UTRs are optional and carried over.
    """
    tx_map: Dict[str, Transcript] = {}

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) != 9:
                continue
            seqid, source, ftype, start, end, score, strand, phase, attr_str = cols
            attrs = parse_attributes_mixed(attr_str)

            # We care about exons/CDS for structure, plus UTR for output
            if ftype not in {"exon", "CDS"} and ftype not in UTR_TYPES:
                continue

            # Get a transcript identifier
            tid: Optional[str] = (
                attrs.get("transcript_id")
                or attrs.get("transcriptId")
            )
            if not tid:
                parent = attrs.get("Parent")
                if parent:
                    tid = parent.split(",")[0].strip()

            if not tid:
                continue

            internal_id = f"{source_index}:{tid}"
            if internal_id not in tx_map:
                tx_map[internal_id] = Transcript(
                    internal_id=internal_id,
                    source_index=source_index,
                    seqid=seqid,
                    strand=strand,
                    source_labels={source},
                    exons=[],
                    cds=[],
                    utr=[],
                )

            tx = tx_map[internal_id]
            tx.source_labels.add(source)

            s_i = int(start)
            e_i = int(end)

            if ftype == "exon":
                tx.exons.append((s_i, e_i))
            elif ftype == "CDS":
                tx.cds.append((s_i, e_i, phase))
            else:
                # UTR types
                tx.utr.append((s_i, e_i, ftype))

    # Filter out transcripts with no exon/CDS signal (UTR alone does not define a transcript here)
    transcripts = [t for t in tx_map.values() if t.exons or t.cds]
    return transcripts


def parse_all_inputs(input_paths: List[str]) -> List[Transcript]:
    all_tx: List[Transcript] = []
    for idx, path in enumerate(input_paths):
        all_tx.extend(parse_transcripts_from_file(path, source_index=idx))
    return all_tx


# ----------------- Gene building / clustering ----------------- #

def _prefer_tx_by_utr(existing: Transcript, candidate: Transcript) -> Transcript:
    """
    Given two transcripts with identical struct_key, return the one to keep.
    Preference: larger total UTR bp (merged). If tie, keep existing.
    """
    ex_utr = existing.utr_bp()
    ca_utr = candidate.utr_bp()
    if ca_utr > ex_utr:
        return candidate
    return existing


def build_gene_clusters_from_transcripts(transcripts: List[Transcript]) -> List[GeneCluster]:
    """
    Cluster transcripts into genes by overlap, using tx.span (exon/CDS only).
    UTRs do not affect overlap clustering nor struct_key.
    """
    clusters: List[GeneCluster] = []
    by_chr_strand: Dict[Tuple[str, str], List[Transcript]] = defaultdict(list)
    for tx in transcripts:
        by_chr_strand[(tx.seqid, tx.strand)].append(tx)

    for (seqid, strand), tx_list in by_chr_strand.items():
        tx_list_sorted = sorted(tx_list, key=lambda t: t.span[0])

        current_group: List[Transcript] = []
        current_start = None
        current_end = None

        for tx in tx_list_sorted:
            t_start, t_end = tx.span
            if not current_group:
                current_group = [tx]
                current_start, current_end = t_start, t_end
            else:
                if t_start <= current_end:
                    current_group.append(tx)
                    current_end = max(current_end, t_end)
                else:
                    clusters.append(
                        GeneCluster(
                            gene_id="",
                            seqid=seqid,
                            strand=strand,
                            start=current_start,
                            end=current_end,
                            transcripts=current_group,
                        )
                    )
                    current_group = [tx]
                    current_start, current_end = t_start, t_end

        if current_group:
            clusters.append(
                GeneCluster(
                    gene_id="",
                    seqid=seqid,
                    strand=strand,
                    start=current_start,
                    end=current_end,
                    transcripts=current_group,
                )
            )

    # Deduplicate identical transcripts within each cluster
    # If duplicates, keep the one with longer UTR region.
    for cluster in clusters:
        seen_structs: Dict[Tuple[str, str, Tuple[Tuple[int, int], ...]], Transcript] = {}
        for tx in cluster.transcripts:
            key = tx.struct_key
            if key not in seen_structs:
                seen_structs[key] = tx
            else:
                seen_structs[key] = _prefer_tx_by_utr(seen_structs[key], tx)
        # preserve deterministic order by span start then internal_id
        cluster.transcripts = sorted(seen_structs.values(), key=lambda t: (t.span[0], t.internal_id))

    return clusters


# ----------------- Priority overlap filtering ----------------- #

def build_priority_interval_index(priority_clusters: List[GeneCluster]):
    """
    Build an index of priority gene intervals for overlap checks.
    Intervals are based on cluster start/end, which are based on merge logic spans (UTR ignored).
    """
    idx: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    for gc in priority_clusters:
        idx[(gc.seqid, gc.strand)].append((gc.start, gc.end))
    for key in idx:
        idx[key].sort(key=lambda x: x[0])
    return idx


def transcript_overlaps_priority(
    tx: Transcript,
    interval_index: Dict[Tuple[str, str], List[Tuple[int, int]]]
) -> bool:
    """Return True if transcript overlaps any priority gene (merge logic span; UTR ignored)."""
    t_start, t_end = tx.span
    intervals = interval_index.get((tx.seqid, tx.strand))
    if not intervals:
        return False
    for g_start, g_end in intervals:
        if g_start > t_end:
            break
        if g_end >= t_start and g_start <= t_end:
            return True
    return False


# ----------------- Output GFF3 ----------------- #

def write_clusters_as_gff3(clusters: List[GeneCluster], out_handle):
    """
    Write gene clusters as a GFF3 to out_handle.

    Features:
        - gene
        - transcript
        - exon
        - UTR (and/or five_prime_UTR / three_prime_UTR, as in input)
        - CDS
    """
    out_handle.write("##gff-version 3\n")

    clusters_sorted = sorted(clusters, key=lambda gc: (gc.seqid, gc.start, gc.strand))
    gene_counter = 1

    for gc in clusters_sorted:
        gc.gene_id = f"gene_{gene_counter:06d}"
        gene_counter += 1

        # Update gene span for OUTPUT to include UTRs where present
        # (merge logic previously used tx.span; here we use tx.output_span)
        starts = [tx.output_span[0] for tx in gc.transcripts]
        ends = [tx.output_span[1] for tx in gc.transcripts]
        gc.start = min(starts)
        gc.end = max(ends)

        gene_sources = sorted({src for tx in gc.transcripts for src in tx.source.split(",") if src})
        gene_source = ",".join(gene_sources) if gene_sources else "merge"

        gene_attrs = {"ID": gc.gene_id}
        out_handle.write("\t".join([
            gc.seqid, gene_source, "gene",
            str(gc.start), str(gc.end),
            ".", gc.strand, ".",
            attrs_to_str(gene_attrs),
        ]) + "\n")

        # Sort transcripts by merge span start for stable order
        txs_sorted = sorted(gc.transcripts, key=lambda t: (t.span[0], t.internal_id))

        tx_counter = 1
        for tx in txs_sorted:
            tx_id = f"{gc.gene_id}.t{tx_counter}"
            tx_counter += 1

            t_start, t_end = tx.output_span  # output includes UTR

            out_handle.write("\t".join([
                tx.seqid, tx.source, "transcript",
                str(t_start), str(t_end),
                ".", tx.strand, ".",
                attrs_to_str({"ID": tx_id, "Parent": gc.gene_id}),
            ]) + "\n")

            # Exons: based on exon_intervals (structure)
            exons_sorted = sorted(tx.exon_intervals)
            for i, (s, e) in enumerate(exons_sorted, start=1):
                out_handle.write("\t".join([
                    tx.seqid, tx.source, "exon",
                    str(s), str(e),
                    ".", tx.strand, ".",
                    attrs_to_str({"ID": f"{tx_id}.exon{i}", "Parent": tx_id}),
                ]) + "\n")

            # UTRs: preserve input feature type
            if tx.utr:
                utr_sorted = sorted(tx.utr, key=lambda x: (x[0], x[1], x[2]))
                for i, (s, e, utr_type) in enumerate(utr_sorted, start=1):
                    out_handle.write("\t".join([
                        tx.seqid, tx.source, utr_type,
                        str(s), str(e),
                        ".", tx.strand, ".",
                        attrs_to_str({"ID": f"{tx_id}.utr{i}", "Parent": tx_id}),
                    ]) + "\n")

            # CDS
            if tx.cds:
                cds_sorted = sorted(tx.cds, key=lambda item: item[0])
                for i, (s, e, phase) in enumerate(cds_sorted, start=1):
                    out_handle.write("\t".join([
                        tx.seqid, tx.source, "CDS",
                        str(s), str(e),
                        ".", tx.strand, phase if phase else ".",
                        attrs_to_str({"ID": f"{tx_id}.cds{i}", "Parent": tx_id}),
                    ]) + "\n")


# ----------------- Main logic ----------------- #

def main():
    ap = argparse.ArgumentParser(description="Merge GTF/GFF annotations into a single GFF3 file.")
    ap.add_argument(
        "--mode",
        choices=["full", "priority"],
        required=True,
        help="Merge mode: 'full' for union; 'priority' to respect one priority file."
    )
    ap.add_argument(
        "--priority-file",
        help="Path to the priority annotation file (required for --mode priority)."
    )
    ap.add_argument("inputs", nargs="+", help="Input GTF/GFF files.")
    args = ap.parse_args()

    if args.mode == "priority":
        if not args.priority_file:
            ap.error("--priority-file is required when --mode priority is used.")
        if args.priority_file not in args.inputs:
            ap.error("--priority-file must be one of the input files.")

    transcripts = parse_all_inputs(args.inputs)
    if not transcripts:
        sys.stderr.write("No transcripts (exon/CDS features) found in inputs.\n")
        return 1

    if args.mode == "full":
        clusters = build_gene_clusters_from_transcripts(transcripts)
        write_clusters_as_gff3(clusters, sys.stdout)
    else:
        priority_index = args.inputs.index(args.priority_file)

        priority_tx = [t for t in transcripts if t.source_index == priority_index]
        other_tx = [t for t in transcripts if t.source_index != priority_index]

        priority_clusters = build_gene_clusters_from_transcripts(priority_tx)
        interval_index = build_priority_interval_index(priority_clusters)

        kept_other_tx = [t for t in other_tx if not transcript_overlaps_priority(t, interval_index)]
        other_clusters = build_gene_clusters_from_transcripts(kept_other_tx)

        all_clusters = priority_clusters + other_clusters
        write_clusters_as_gff3(all_clusters, sys.stdout)

    return 0


if __name__ == "__main__":
    sys.exit(main())
