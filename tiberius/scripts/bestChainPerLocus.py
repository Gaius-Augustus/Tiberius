#!/usr/bin/env python3
# ==============================================================
# Reduce a chain-tagged hint GFF (output of chainedHints.py) to a
# single chain per gene locus by selecting the best-scoring chain.
#
# Loci are defined as the connected components of chain spans that
# overlap on the same seqid and strand. The chain score is the sum
# of eScore over CDS, start_codon and stop_codon hints (introns
# are excluded to avoid double-counting exon boundaries via
# LeScore/ReScore). Ties are broken by hint count, then chain_id.
# ==============================================================


import argparse
import csv
import re
import sys
from collections import defaultdict


def extractAttribute(attrs, feature):
    m = re.search(feature + r'=([^;]+)', attrs)
    return m.group(1) if m else None


def loadChains(inputFile):
    chains = defaultdict(lambda: {
        'rows': [], 'seqid': None, 'strand': None,
        'start': None, 'end': None, 'score': 0.0, 'nHints': 0,
    })
    with open(inputFile) as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) != 9:
                continue
            cid = extractAttribute(row[8], 'chain_id')
            if cid is None:
                continue
            c = chains[cid]
            c['rows'].append(row)
            c['nHints'] += 1

            start, end = int(row[3]), int(row[4])
            if c['seqid'] is None:
                c['seqid'], c['strand'] = row[0], row[6]
                c['start'], c['end'] = start, end
            else:
                if start < c['start']:
                    c['start'] = start
                if end > c['end']:
                    c['end'] = end

            if row[2].lower() != 'intron':
                e = extractAttribute(row[8], 'eScore')
                if e is not None:
                    c['score'] += float(e)
    return chains


def clusterLoci(chains):
    items = sorted(chains.items(),
                   key=lambda kv: (kv[1]['seqid'], kv[1]['strand'],
                                   kv[1]['start'], kv[1]['end']))
    loci, current, curEnd, curKey = [], [], None, None
    for cid, c in items:
        key = (c['seqid'], c['strand'])
        if key != curKey or c['start'] > curEnd:
            if current:
                loci.append(current)
            current, curKey, curEnd = [cid], key, c['end']
        else:
            current.append(cid)
            if c['end'] > curEnd:
                curEnd = c['end']
    if current:
        loci.append(current)
    return loci


def pickBest(locus, chains):
    return max(locus, key=lambda cid: (chains[cid]['score'],
                                       chains[cid]['nHints'], cid))


def main():
    args = parseCmd()
    chains = loadChains(args.input)
    if not chains:
        sys.stderr.write("warning: no chain_id attributes found in input.\n")
        return

    loci = clusterLoci(chains)
    kept = [pickBest(locus, chains) for locus in loci]
    keptSet = set(kept)

    out = open(args.output, 'w') if args.output else sys.stdout
    try:
        for cid in sorted(keptSet, key=lambda k: (chains[k]['seqid'],
                                                  chains[k]['start'],
                                                  chains[k]['end'])):
            for row in chains[cid]['rows']:
                out.write("\t".join(row) + "\n")
    finally:
        if args.output:
            out.close()

    if args.report:
        with open(args.report, 'w') as r:
            r.write("locus\tseqid\tstrand\tstart\tend\tn_chains\t"
                    "kept_chain_id\tkept_score\tkept_n_hints\n")
            for i, locus in enumerate(loci):
                best = pickBest(locus, chains)
                lo = min(chains[c]['start'] for c in locus)
                hi = max(chains[c]['end'] for c in locus)
                seq = chains[best]['seqid']
                strand = chains[best]['strand']
                r.write(f"locus_{i+1}\t{seq}\t{strand}\t{lo}\t{hi}\t"
                        f"{len(locus)}\t{best}\t"
                        f"{chains[best]['score']:.2f}\t"
                        f"{chains[best]['nHints']}\n")


def parseCmd():
    parser = argparse.ArgumentParser(
        description='Reduce a chain-tagged hint GFF to one chain (isoform) '
                    'per gene locus by selecting the best-scoring chain. '
                    'Loci are connected components of chain spans that '
                    'overlap on the same seqid and strand. Chain score is '
                    'the sum of eScore over CDS/start/stop hints.')
    parser.add_argument('input', metavar='hc_chained.gff',
                        help='Chain-tagged hint GFF from chainedHints.py.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output GFF. Stdout if omitted.')
    parser.add_argument('--report', default=None,
                        help='Optional TSV with per-locus selection summary.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
