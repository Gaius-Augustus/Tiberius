#!/usr/bin/env python3

"""
Example script to parse a GTF with CDS and intron features, and compute transcript-level metrics:
  - coding region length
  - CDS count
  - overall GC content (CDS+introns)
  - CDS-only GC content
  - check if the annotated CDS begins with a canonical start codon (ATG)
  - check for in-frame stop codons (internal)
  - fraction of repeat content in coding region, inferred from lowercase (soft-masked) sequence
  - fraction of repeat content in CDS+intron, inferred from lowercase (soft-masked) sequence
  - check if donor/acceptor sites (GT/AG for typical eukaryotes) are canonical
Outputs a CSV with columns:
  transcript_id, gene_id, coding_length, cds_count, overall_gc, cds_gc,
  has_canonical_start, has_inframe_stop, repeats_in_cds_fraction,
  repeats_in_cds_intron_fraction, canonical_splice_sites
"""

import sys, re
import csv
from collections import defaultdict
from Bio import SeqIO
from Bio.Seq import Seq
import pybedtools

# -------------------------------------------------------------
# 1. Parse command-line arguments
# -------------------------------------------------------------
import argparse
parser = argparse.ArgumentParser(description="Compute various transcript features from a GTF (CDS + introns) and a soft-masked FASTA.")
parser.add_argument("-g", "--gtf", required=True, help="GTF file containing transcript, CDS, intron features.")
parser.add_argument("-f", "--fasta", required=True, 
                    help="Genome FASTA file (soft-masked: repeats are in lowercase).")
parser.add_argument("-o", "--out", required=True, help="Output CSV file.")
args = parser.parse_args()

# -------------------------------------------------------------
# 2. Load the (soft-masked) genome
# -------------------------------------------------------------
# In a soft-masked reference, repeats are lowercased. 
# We'll leverage that to compute repeat fraction by counting lowercase bases.
seq_dict = SeqIO.to_dict(SeqIO.parse(args.fasta, "fasta"))

# -------------------------------------------------------------
# 3. Convert GTF to BedTool objects for easier interval handling
# -------------------------------------------------------------
gtf_bed = pybedtools.BedTool(args.gtf)

# -------------------------------------------------------------
# 4. Parse GTF features, grouping by transcript
# -------------------------------------------------------------
transcript_cds_intervals = defaultdict(list)      # transcript_id -> list of CDS intervals
transcript_intron_intervals = defaultdict(list)   # transcript_id -> list of intron intervals
transcript_strand = {}                             # transcript_id -> "+" or "-"
transcript_chrom = {}                              # transcript_id -> chromosome
transcript_gene = {}                               # transcript_id -> gene_id

for feature in gtf_bed:
    # The feature is a BedTool interval. 
    # The GTF "type" is typically in feature.fields[2] => e.g. "CDS" or "intron"
    ftype = feature.fields[2]

    # Parse the last column (attributes) to extract transcript_id/gene_id
    attr_field = feature.fields[8]
    attrs = {}
    for attr_chunk in attr_field.strip().split(";"):
        attr_chunk = attr_chunk.strip()
        if not attr_chunk:
            continue
        key_val = attr_chunk.replace('"', '').split()
        if len(key_val) >= 2:
            key = key_val[0]
            val = key_val[1]
            attrs[key] = val

    transcript_id = attrs.get("transcript_id", None)
    gene_id = attrs.get("gene_id", None)
    if not transcript_id:
        continue

    if gene_id:
        transcript_gene[transcript_id] = gene_id

    transcript_strand[transcript_id] = feature.strand
    transcript_chrom[transcript_id] = feature.chrom

    if ftype == "CDS":
        transcript_cds_intervals[transcript_id].append(feature)
    elif ftype == "intron":
        transcript_intron_intervals[transcript_id].append(feature)

# -------------------------------------------------------------
# 5. Define helper functions
# -------------------------------------------------------------
def gc_content(seq):
    """Compute GC fraction of a sequence string (case-insensitive)."""
    seq = seq.upper()
    gc_count = seq.count('G') + seq.count('C')
    return float(gc_count) / len(seq) if len(seq) > 0 else 0.0

def fraction_lowercase(seq):
    """Return fraction of bases that are lowercase in seq (soft-masked repeats)."""
    if len(seq) == 0:
        return 0.0
    return sum(ch.islower() for ch in seq) / len(seq)

def get_sequence(chrom, start, end, seq_lookup, strand='+'):
    """
    Retrieve the sequence [start, end) from seq_lookup (dict of SeqRecords).
    Return as a Bio.Seq (uppercase or lowercase). 
    If strand == '-', return reverse complement.
    """
    # Note: pybedtools intervals are typically 0-based
    ref_seq = seq_lookup[chrom].seq[start:end]  # slice    
    return ref_seq if strand == '+' else ref_seq.reverse_complement()

def check_start_codon(seq):
    """Check if the provided coding sequence (DNA) begins with the canonical start codon ATG."""
    return seq.upper().startswith("ATG")

def find_inframe_stop(seq):
    """
    Check if there's an in-frame stop codon *before* the final codon.
    We expect the final codon might be the actual stop. 
    Typical stops: TAA, TAG, TGA.
    """
    seq = seq.upper()
    stops = {"TAA", "TAG", "TGA"}
    # if len(seq) % 3 > 0:
    #     return False
    # Iterate in steps of 3, ignoring the last codon:
    for i in range(0, len(seq) - 4, 3):
        codon = seq[i:i+3]
        if codon in stops:
            return True
    return False

def check_splice_sites(intron_intervals, seq_lookup):
    """
    For each intron, check if the first 2 bases (donor) and last 2 bases (acceptor)
    are canonical GT/AG. On the '-' strand, the returned seq is already reversed
    (because get_sequence does strand adjustment).
    Return True if all introns appear canonical, False otherwise.
    """
    if not intron_intervals:
        # No introns => treat as True (no introns to check)
        return True

    for intron in intron_intervals:
        chrom = intron.chrom
        start = intron.start
        end   = intron.end
        strand = intron.strand

        # Donor site => [start, start+2)
        donor_seq = get_sequence(chrom, start, start+2, seq_lookup, strand).upper()
        # Acceptor site => [end-2, end)
        acceptor_seq = get_sequence(chrom, end-2, end, seq_lookup, strand).upper()

        # On the plus strand, we want GT at donor, AG at acceptor. 
        # On the minus strand, get_sequence returns the reversed bases, 
        # so we check the same "GT/AG".
        if not (donor_seq == "GT" and acceptor_seq == "AG"):
            return False

    return True

# -------------------------------------------------------------
# 6. Main loop: compute metrics
# -------------------------------------------------------------
results = []

for tx_id in transcript_cds_intervals.keys():

    gene_id = transcript_gene.get(tx_id, "NA")
    strand  = transcript_strand.get(tx_id, "+")
    chrom   = transcript_chrom.get(tx_id, "chrNA")

    cds_list = transcript_cds_intervals[tx_id]
    intron_list = transcript_intron_intervals.get(tx_id, [])

    # Sort intervals
    cds_list_sorted = sorted(cds_list, key=lambda x: x.start)
    intron_list_sorted = sorted(intron_list, key=lambda x: x.start)

    # ---------------------------------------------------------
    # 6.1. coding region length & gather CDS sequence
    # ---------------------------------------------------------
    coding_length = 0
    cds_seq_parts = []
    for iv in cds_list_sorted:
        seg_len = iv.end - iv.start
        coding_length += seg_len
        seg_seq = get_sequence(iv.chrom, iv.start, iv.end, seq_dict)
        cds_seq_parts.append(str(seg_seq))
    full_cds_seq = "".join(cds_seq_parts)
    if strand == '-':
        full_cds_seq = str(Seq(full_cds_seq).reverse_complement())
        if not full_cds_seq[:3].upper() == 'ATG':
            print(full_cds_seq, tx_id, iv.strand)
    # ---------------------------------------------------------
    # 6.2. CDS count
    # ---------------------------------------------------------
    cds_count = len(cds_list_sorted)

    # ---------------------------------------------------------
    # 6.3. Overall GC content (CDS + introns)
    # ---------------------------------------------------------
    # Build combined intervals for CDS+intron
    all_features = cds_list_sorted + intron_list_sorted
    all_features_sorted = sorted(all_features, key=lambda x: (x.start, x.end))

    total_seq_parts = []
    for iv in all_features_sorted:
        seg_seq = get_sequence(iv.chrom, iv.start, iv.end, seq_dict)
        total_seq_parts.append(str(seg_seq))
    total_seq_str = "".join(total_seq_parts)
    if strand == '-':
        total_seq_str = str(Seq(total_seq_str).reverse_complement())
    overall_gc = gc_content(full_cds_seq)

    # ---------------------------------------------------------
    # 6.4. CDS-only GC content
    # ---------------------------------------------------------
    cds_gc = gc_content(full_cds_seq)

    # ---------------------------------------------------------
    # 6.5. Check if CDS begins with canonical start (ATG)
    # ---------------------------------------------------------
    has_canonical_start = check_start_codon(full_cds_seq)

    # ---------------------------------------------------------
    # 6.6. Check for in-frame stop codons
    # ---------------------------------------------------------
    has_inframe_stop = find_inframe_stop(full_cds_seq)

    # ---------------------------------------------------------
    # 6.7. Fraction of repeat content in coding region
    #      (fraction of lowercase in CDS seq)
    # ---------------------------------------------------------
    # In the soft-masked genome, repeats are lowercased
    cds_repeat_fraction = fraction_lowercase(full_cds_seq)

    # ---------------------------------------------------------
    # 6.8. Fraction of repeat content in CDS + intron
    # ---------------------------------------------------------
    total_repeat_fraction = fraction_lowercase(total_seq_str)

    # ---------------------------------------------------------
    # 6.9. Check canonical splice sites (GT/AG)
    # ---------------------------------------------------------
    is_canonical_splice = check_splice_sites(intron_list_sorted, seq_dict)

    # Store result
    results.append([
        tx_id,
        gene_id,
        chrom, 
        strand,
        cds_list_sorted[0].start,
        cds_list_sorted[-1].end,
        coding_length,
        cds_count,
        f"{overall_gc:.4f}",
        f"{cds_gc:.4f}",
        1 if has_canonical_start else 0,
        1 if has_inframe_stop else 0,
        f"{cds_repeat_fraction:.4f}",
        f"{total_repeat_fraction:.4f}",
        1 if is_canonical_splice else 0
    ])

# -------------------------------------------------------------
# 7. Write results to CSV
# -------------------------------------------------------------
with open(args.out, "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    header = [
        "transcript_id", "gene_id", 'seq_name', 'strand', 
        'start', 'end', 
        "coding_length", "cds_count",
        "overall_gc", "cds_gc",
        "has_canonical_start", "has_inframe_stop",
        "repeats_in_cds_fraction", "repeats_in_cds_intron_fraction",
        "canonical_splice_sites"
    ]
    writer.writerow(header)
    for row in results:
        writer.writerow(row)

print(f"Done. Wrote {len(results)} transcripts to {args.out}")
