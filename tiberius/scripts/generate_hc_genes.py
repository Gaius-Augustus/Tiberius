#!/usr/bin/env python3
import argparse
from hc_module import (
    getting_hc_supported_by_proteins, getting_hc_supported_by_intrinsic,
    choose_one_isoform, from_pep_file_to_gff3, from_transcript_to_genome)

ap = argparse.ArgumentParser()
ap.add_argument("--diamond_revised", required=True)
ap.add_argument("--revised_pep", required=True)
ap.add_argument("--proteins", required=True)
ap.add_argument("--stringtie_gtf", required=True)
ap.add_argument("--stringtie_gff3", required=True)
ap.add_argument("--transcripts_fasta", required=True)
ap.add_argument("--miniprot_gff", required=True)
ap.add_argument("--transdecoder_util", required=True)
ap.add_argument("--bedtools_path", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--min_cds_len", type=int, default=250)
ap.add_argument("--score_threshold", type=float, default=50.0)


args = ap.parse_args()
hc_pep = getting_hc_supported_by_proteins(args.diamond_revised, args.revised_pep, args.proteins, 
                                          f"{args.outdir}/hc_genes.pep")

hc_pep, lc_genes = getting_hc_supported_by_intrinsic(
    hc_pep, args.stringtie_gtf, args.stringtie_gff3, args.transcripts_fasta,
    args.miniprot_gff, args.transdecoder_util, args.bedtools_path,
    output_dir=args.outdir, min_cds_len=args.min_cds_len, score_threshold=args.score_threshold
)


hc_single_pep = choose_one_isoform(hc_pep, f"{args.outdir}/hc_one_isoform.pep")
hc_single_gff = from_pep_file_to_gff3(hc_single_pep, args.stringtie_gtf, f"{args.outdir}/hc_one_isoform.gff3")
from_transcript_to_genome(hc_single_gff, args.stringtie_gff3, 
                args.transcripts_fasta, f"{args.outdir}/training.gff", args.transdecoder_util)
