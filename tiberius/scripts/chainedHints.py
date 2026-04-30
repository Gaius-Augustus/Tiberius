#!/usr/bin/env python3
# ==============================================================
# Emit per-chain hints whose genomic positions pass the
# high-confidence filter in hc.gff. Each output line keeps the
# alignment-chain identity from the boundary-scorer output
# (Parent= and prot=), and an extra chain_id=<Parent>_<prot>
# attribute is appended for convenience.
#
# Group the output by chain_id to obtain non-contradicting hint
# sets per alignment chain (and therefore per locus, when the
# input is miniprot_representatives.gff).
# ==============================================================


import argparse
import csv
import re
import sys


HINT_TYPES = ("intron", "start_codon", "stop_codon", "cds")


def signature(row):
    return (row[0], row[2].lower(), row[3], row[4], row[6])


def extractAttribute(attrs, feature):
    m = re.search(feature + r'=([^;]+)', attrs)
    return m.group(1) if m else None


def loadHcSignatures(hcFile):
    sigs = set()
    with open(hcFile) as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) != 9:
                continue
            if row[2].lower() not in HINT_TYPES:
                continue
            sigs.add(signature(row))
    return sigs


def emitChainedHints(chainsFile, hcSigs, out):
    with open(chainsFile) as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) != 9:
                continue
            if row[2].lower() not in HINT_TYPES:
                continue
            if signature(row) not in hcSigs:
                continue

            parent = extractAttribute(row[8], "Parent")
            prot = extractAttribute(row[8], "prot")
            if parent and prot:
                chain_id = f'{parent}_{prot}'
            else:
                chain_id = parent or prot or "."

            row[1] = "miniprothint"
            attrs = row[8].rstrip().rstrip(';')
            row[8] = f'{attrs};chain_id={chain_id};' if attrs else f'chain_id={chain_id};'
            out.write("\t".join(row) + "\n")


def main():
    args = parseCmd()
    hcSigs = loadHcSignatures(args.hc)
    if args.output:
        with open(args.output, "w") as out:
            emitChainedHints(args.chains, hcSigs, out)
    else:
        emitChainedHints(args.chains, hcSigs, sys.stdout)


def parseCmd():
    parser = argparse.ArgumentParser(
        description='Emit per-chain hints whose positions pass the '
                    'high-confidence filter in hc.gff. Each output line '
                    'carries a chain_id=<Parent>_<prot> attribute; group '
                    'by chain_id to obtain non-contradicting hint sets '
                    'per alignment chain.')
    parser.add_argument('hc', metavar='hc.gff',
                        help='High-confidence hint set produced by miniprothint.')
    parser.add_argument('chains', metavar='miniprot_representatives.gff',
                        help='Per-chain boundary-scorer GFF — typically '
                             'miniprot_representatives.gff, or '
                             'miniprot_parsed.gff for all chains.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file. Stdout if omitted.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
