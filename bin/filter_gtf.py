# ==============================================================
# Authors: Lars Gabriel
#
# Merge a number of GTF files and filter their transcripts.
# 
# Transformers 4.31.0
# ==============================================================

import sys, os, re, json, sys, csv
from Bio import SeqIO
from Bio.Seq import Seq
from BCBio import GFF
from genome_anno import Anno
import re
import argparse

# Function to assemble transcript taking strand into account
def assemble_transcript(exons, sequence, strand):
    parts = []
    exons.sort(reverse=strand=='-')
    for exon in exons:
        exon_seq = sequence.seq[exon[0]-1:exon[1]]
        if strand == '-':
            exon_seq = exon_seq.reverse_complement()
        parts.append(str(exon_seq))  # Convert Seq object to string here

    coding_seq = Seq("".join(parts))
    if len(coding_seq)%3==0:
        prot_seq = coding_seq.translate()
        if prot_seq[-1] == '*':
            return coding_seq, prot_seq
    return None, None

# Check for in-frame stop codons
def check_in_frame_stop_codons(seq):
    return '*' in seq[:-1]

def main():
    args = parseCmd()
    # File paths
    fasta_file = args.fasta
    gtf_files = args.gtf.split(',')
    output_gtf = args.out
    
    anno_outp = Anno('', f'anno')

    # Load the genome sequence from the FASTA file
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
    
    # Load GFF file    
    for i, gtf in enumerate(gtf_files):
        anno_inp = Anno(gtf, f'anno{i}')
        anno_inp.addGtf()
        anno_inp.norm_tx_format()

        out_tx = {}
        for tx_id, tx in anno_inp.transcripts.items():
            exons = tx.get_type_coords('CDS', frame=False)
            filt=False
            
            # filter out tx with inframe stop codons
            if args.filter_inframestop:
                coding_seq, prot_seq = assemble_transcript(exons, genome[tx.chr], tx.strand )
                if not coding_seq or check_in_frame_stop_codons(prot_seq):
                    filt = True
            # filter out transcripts with cds len shorter than args.filter_short
            if not filt and tx.get_cds_len() < args.filter_short:
                filt = True
                
            if not filt:
                out_tx[tx_id] = tx
        # print(len(out_tx))
        anno_outp.add_transcripts(out_tx, f'anno{i}')
        
    anno_outp.find_genes()
    anno_outp.rename_tx_ids()
    anno_outp.write_anno(output_gtf)
    
    prot_seq_out = ""
    coding_seq_out = ""
    if args.protseq or args.codingseq:
        for tx_id, tx in anno_outp.transcripts.items():
            exons = tx.get_type_coords('CDS', frame=False)
            coding_seq, prot_seq = assemble_transcript(exons, genome[tx.chr], tx.strand)
            if args.codingseq:
                coding_seq_out +=f">{tx_id}\n{coding_seq}\n"
            if args.protseq:
                prot_seq_out +=f">{tx_id}\n{prot_seq}\n"
    
    if args.codingseq:
        with open(args.codingseq, 'w+') as f:
            f.write(coding_seq_out.strip())
    if args.protseq:
        with open(args.protseq, 'w+') as f:
            f.write(prot_seq_out.strip())
        
def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(
        description="""Filter GTF file
    """)
    parser.add_argument('-g', '--gtf', type=str, required=True, default='',
        help='List of input gtf file.')
    parser.add_argument('-f', '--fasta', type=str, required=True, default='',
        help='Genomic sequences in FASTA format.')
    parser.add_argument('-o', '--out', type=str, required=True, default='',
        help='Output file of filtered gene structures.')
    parser.add_argument('--codingseq', type=str, required=False, default='',
        help='Output file for coding sequences.')
    parser.add_argument('-p', '--protseq', type=str, required=False, default='',
        help='Output file for protein sequences.')
    parser.add_argument('-I', '--filter_inframestop', action='store_true',
        help='Filter out transcripts with in-frame stop codons.')
    parser.add_argument('-S', '--filter_short', type=int, default='',
        help='Filter out transcripts with in-frame stop codons.')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
