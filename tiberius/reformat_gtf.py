#!/usr/bin/env python3
import sys
import argparse
import re
import os

def parse_attributes_gtf(attr_str):
    """
    Parse a GTF attribute string into a dict.
    Expected format: key "value"; key "value";
    """
    attrs = {}
    # split on semicolons then extract key and value
    for attr in attr_str.strip().split(";"):
        attr = attr.strip()
        if not attr:
            continue
        m = re.match(r'(\S+)\s+"([^"]+)"', attr)
        if m:
            key, value = m.groups()
            attrs[key] = value
    return attrs

def format_attributes_gtf(attrs):
    """
    Format a dictionary of attributes into a GTF attribute string.
    """
    parts = []
    for key, value in attrs.items():
        parts.append(f'{key} "{value}";')
    return " ".join(parts)

def parse_attributes_gff3(attr_str):
    """
    Parse a GFF3 attribute string into a dict.
    Expected format: key=value;key=value;...
    """
    attrs = {}
    for attr in attr_str.strip().split(";"):
        attr = attr.strip()
        if not attr:
            continue
        if "=" in attr:
            key, value = attr.split("=", 1)
            attrs[key] = value
    return attrs

def format_attributes_gff3(attrs):
    """
    Format a dictionary of attributes into a GFF3 attribute string.
    """
    parts = []
    for key, value in attrs.items():
        parts.append(f"{key}={value}")
    return ";".join(parts)

def process_gtf(infile, prefix, outfile):
    """
    Process a GTF file.
    Updates the gene_id and transcript_id in the attributes to:
      gene_id -> <prefix>gene<number>
      transcript_id -> <prefix>gene<number>.t<t_number>
    """
    gene_counter = 1
    gene_map = {}  # old_gene_id -> new_gene_id
    transcript_map = {}  # old_transcript_id -> new_transcript_id
    gene_transcript_counter = {}  # old_gene_id -> next transcript number

    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                fout.write(line)
                continue
            attr_str = fields[8]
            attrs = parse_attributes_gtf(attr_str)
            # Update gene_id if present.
            if "gene_id" in attrs:
                old_gene = attrs["gene_id"]
                if old_gene not in gene_map:
                    new_gene = f"{prefix}gene{gene_counter}"
                    gene_map[old_gene] = new_gene
                    gene_transcript_counter[old_gene] = 1
                    gene_counter += 1
                else:
                    new_gene = gene_map[old_gene]
                attrs["gene_id"] = new_gene
            # Update transcript_id if present.
            if "transcript_id" in attrs:
                old_transcript = attrs["transcript_id"]
                if old_transcript not in transcript_map:
                    # Use the transcript counter for this gene
                    transcript_num = gene_transcript_counter[old_gene]
                    new_transcript = f"{new_gene}.t{transcript_num}"
                    transcript_map[old_transcript] = new_transcript
                    gene_transcript_counter[old_gene] += 1
                else:
                    new_transcript = transcript_map[old_transcript]
                attrs["transcript_id"] = new_transcript

            fields[8] = format_attributes_gtf(attrs)
            fout.write("\t".join(fields) + "\n")

def process_gff3(infile, prefix, outfile):
    """
    Process a GFF3 file.
    For gene features (feature == "gene"), update the ID to <prefix>gene<number>.
    For transcript features (feature == "mRNA" or "transcript"), update the ID to
      <prefix>gene<number>.t<t_number> and update Parent accordingly.
    Other features that have a Parent attribute are updated if their parent's ID has been changed.
    """
    gene_counter = 1
    gene_map = {}  # old gene id -> new gene id
    transcript_map = {}  # old transcript id -> new transcript id
    gene_transcript_counter = {}  # old gene id -> next transcript number

    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith("#") or not line.strip():
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                fout.write(line)
                continue
            feature = fields[2].lower()
            attr_str = fields[8]
            attrs = parse_attributes_gff3(attr_str)

            if feature == "gene":
                old_gene = attrs.get("ID")
                if old_gene:
                    if old_gene not in gene_map:
                        new_gene = f"{prefix}gene{gene_counter}"
                        gene_map[old_gene] = new_gene
                        gene_transcript_counter[old_gene] = 1
                        gene_counter += 1
                    else:
                        new_gene = gene_map[old_gene]
                    attrs["ID"] = new_gene

            elif feature in ["mrna", "transcript"]:
                # In GFF3, a transcript feature has an ID and a Parent (gene)
                old_transcript = attrs.get("ID")
                parent = attrs.get("Parent")
                if parent:
                    # Assume a single parent for simplicity
                    old_gene = parent.split(",")[0]
                    if old_gene not in gene_map:
                        new_gene = f"{prefix}gene{gene_counter}"
                        gene_map[old_gene] = new_gene
                        gene_transcript_counter[old_gene] = 1
                        gene_counter += 1
                    else:
                        new_gene = gene_map[old_gene]
                    if old_transcript not in transcript_map:
                        transcript_num = gene_transcript_counter[old_gene]
                        new_transcript = f"{new_gene}.t{transcript_num}"
                        transcript_map[old_transcript] = new_transcript
                        gene_transcript_counter[old_gene] += 1
                    else:
                        new_transcript = transcript_map[old_transcript]
                    attrs["ID"] = new_transcript
                    # Update Parent to the new gene id
                    attrs["Parent"] = new_gene

            else:
                # For other features, if a Parent attribute is present, update it if needed.
                if "Parent" in attrs:
                    new_parents = []
                    for pid in attrs["Parent"].split(","):
                        if pid in transcript_map:
                            new_parents.append(transcript_map[pid])
                        elif pid in gene_map:
                            new_parents.append(gene_map[pid])
                        else:
                            new_parents.append(pid)
                    attrs["Parent"] = ",".join(new_parents)

            fields[8] = format_attributes_gff3(attrs)
            fout.write("\t".join(fields) + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Reformat transcript (and gene) IDs in a GTF/GFF file to a new naming scheme."
    )
    parser.add_argument("--input", required=True, help="Input GTF or GFF file.")
    parser.add_argument("--prefix", required=True, help="Prefix for the new IDs.")
    parser.add_argument("--out", required=True, help="Output file with reformatted IDs.")

    args = parser.parse_args()
    infile = args.input
    prefix = args.prefix
    outfile = args.out

    # Determine file type from extension (default to GTF)
    _, ext = os.path.splitext(infile)
    ext = ext.lower()
    if ext in [".gff", ".gff3"]:
        process_gff3(infile, prefix, outfile)
    else:
        process_gtf(infile, prefix, outfile)

if __name__ == "__main__":
    main()
