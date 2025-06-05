#!/usr/bin/env python
import sys
import argparse

def parse_attributes(attr_str):
    """
    Parse the attribute column from a GTF/GFF3 file.
    Handles GTF style (key "value"; ...) and GFF3 style (key=value;...).
    """
    attrs = {}
    attr_str = attr_str.strip()
    if not attr_str:
        return attrs
    # If it contains an '=' and no quotes, assume GFF3 format.
    if "=" in attr_str and ('"' not in attr_str):
        parts = attr_str.split(";")
        for part in parts:
            part = part.strip()
            if part:
                if "=" in part:
                    key, value = part.split("=", 1)
                    attrs[key] = value
    else:
        # Assume GTF format: key "value"; key "value";
        parts = attr_str.split(";")
        for part in parts:
            part = part.strip()
            if part:
                if " " in part:
                    key, value = part.split(" ", 1)
                    value = value.strip('"')
                    attrs[key] = value
    return attrs

def main():
    parser = argparse.ArgumentParser(
        description="Select transcript with longest CDS from a GTF/GFF3 file and output as a GTF file with gene and transcript feature lines added."
    )
    parser.add_argument("input_file", help="Input GTF or GFF3 file")
    args = parser.parse_args()
    
    # Dictionary to hold transcript data:
    # key: transcript_id, value: dict with keys 'gene_id', 'cds_length', and 'lines' (list of tuples (line_number, line))
    transcripts = {}
    header_lines = []  # Save header lines (comments) to print at the top
    
    with open(args.input_file, "r") as infile:
        for lineno, line in enumerate(infile):
            line = line.rstrip("\n")
            if line.startswith("#"):
                header_lines.append(line)
                continue
            
            fields = line.split("\t")
            if len(fields) < 9:
                continue  # skip malformed lines
            seqid, source, feature, start, end, score, strand, phase, attributes_str = fields
            attrs = parse_attributes(attributes_str)
            
            # Determine transcript id.
            transcript_id = None
            if "transcript_id" in attrs:
                # Typical for GTF
                transcript_id = attrs["transcript_id"]
            elif feature in ["mRNA", "transcript"]:
                # In GFF3, the transcript record itself often has an ID attribute.
                transcript_id = attrs.get("ID", None)
            else:
                # For CDS/exon in GFF3 the 'Parent' attribute usually holds the transcript id.
                transcript_id = attrs.get("Parent", None)
                if transcript_id and "," in transcript_id:
                    # If there are multiple parents, take the first one.
                    transcript_id = transcript_id.split(",")[0]
            
            # Determine gene id (if available)
            gene_id = None
            if "gene_id" in attrs:
                gene_id = attrs["gene_id"]
            elif "gene" in attrs:
                gene_id = attrs["gene"]
            # If still missing, we can optionally set gene_id to transcript_id (treating each transcript as its own gene)
            if gene_id is None:
                gene_id = transcript_id
            
            # If we couldn't determine a transcript id, skip this feature.
            if transcript_id is None:
                continue
            
            # Initialize dictionary entry for this transcript if needed.
            if transcript_id not in transcripts:
                transcripts[transcript_id] = {"gene_id": gene_id, "cds_length": 0, "lines": []}
            else:
                if transcripts[transcript_id]["gene_id"] is None and gene_id is not None:
                    transcripts[transcript_id]["gene_id"] = gene_id
            
            # If this feature is a CDS, add its length.
            if feature == "CDS":
                try:
                    cds_len = int(end) - int(start) + 1
                except ValueError:
                    cds_len = 0
                transcripts[transcript_id]["cds_length"] += cds_len
            
            # Save the line along with its original line number.
            transcripts[transcript_id]["lines"].append((lineno, line))
    
    # Group transcripts by gene and select the one with the longest CDS for each gene.
    # Map: gene_id -> (transcript_id, cds_length)
    selected_transcripts = {}
    for tid, data in transcripts.items():
        gene = data["gene_id"]
        if gene not in selected_transcripts:
            selected_transcripts[gene] = (tid, data["cds_length"])
        else:
            if data["cds_length"] > selected_transcripts[gene][1]:
                selected_transcripts[gene] = (tid, data["cds_length"])
    
    # Prepare output lines.
    # For each selected transcript, compute overall coordinates and add gene and transcript lines.
    output_lines = []
    for gene, (tid, cds_length) in selected_transcripts.items():
        transcript_data = transcripts[tid]
        # Sort transcript lines by original order.
        feature_lines = sorted(transcript_data["lines"], key=lambda x: x[0])
        min_start = None
        max_end = None
        seqid = None
        source = None
        strand = None
        
        # Determine the transcript's genomic span using all feature lines.
        for _, line in feature_lines:
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            try:
                start = int(fields[3])
                end = int(fields[4])
            except ValueError:
                continue
            if min_start is None or start < min_start:
                min_start = start
            if max_end is None or end > max_end:
                max_end = end
            if seqid is None:
                seqid = fields[0]
            if source is None:
                source = fields[1]
            if strand is None:
                strand = fields[6]
        # If we could not compute coordinates, skip this transcript.
        if min_start is None or max_end is None:
            continue
        
        # Build new gene and transcript feature lines in GTF format.
        # GTF fields: seqid, source, feature, start, end, score, strand, phase, attributes
        gene_attributes = f'gene_id "{gene}";'
        transcript_attributes = f'gene_id "{gene}"; transcript_id "{tid}";'
        gene_line = "\t".join([seqid, source, "gene", str(min_start), str(max_end), ".", strand, ".", gene_attributes])
        transcript_line = "\t".join([seqid, source, "transcript", str(min_start), str(max_end), ".", strand, ".", transcript_attributes])
        
        # Insert the gene and transcript lines before the original transcript features.
        # We assign them artificial line numbers so that they appear before the first original line.
        first_lineno = feature_lines[0][0]
        output_lines.append((first_lineno - 0.2, gene_line))
        output_lines.append((first_lineno - 0.1, transcript_line))
        for ln, l in feature_lines:
            output_lines.append((ln, l))
    
    # Sort all output lines by their assigned order.
    output_lines.sort(key=lambda x: x[0])
    
    # Print header lines, then output lines.
    for h in header_lines:
        print(h)
    for _, line in output_lines:
        print(line)

if __name__ == "__main__":
    main()
