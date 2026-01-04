#!/usr/bin/env python3
# ==============================================================
# Tomas Bruna
# Copyright 2022, Georgia Institute of Technology, USA
#
# Remove predictions which are in conflict with ProtHint. Specifically, remove
# predictions which have an intron/start/stop not supported by any
# ProtHint alignment and in conflict with the top chain.
# ==============================================================


import argparse
import csv
import re
import tempfile
import os
import EvidencePipeline.predictionAnalysis as analysis


class Feature():
    def __init__(self, start, end, strand, ID=None):
        self.start = start
        self.end = end
        self.ID = ID
        self.strand = strand

    def __gt__(self, f2):
        return self.start > f2.start


class Chain():
    def __init__(self, ID, strand):
        self.ID = ID
        self.strand = strand
        self.exons = []
        self.introns = []

    def addFeature(self, row):
        if row[2] == "CDS":
            self.exons.append(Feature(int(row[3]), int(row[4]), self.strand))
        elif row[2].lower() == "intron":
            self.introns.append(Feature(int(row[3]), int(row[4]), self.strand))

    def sortFeatures(self):
        self.exons.sort()
        self.introns.sort()


def extractFeatureGtf(text, feature):
    regex = feature + ' "([^"]+)"'
    return re.search(regex, text).groups()[0]


def extractFeatureGff(text, feature):
    regex = feature + '=([^;]+);'
    result = re.search(regex, text)
    if result:
        return result.groups()[0]
    else:
        return None


def temp(prefix, suffix):
    tmp = tempfile.NamedTemporaryFile("w", delete=False,
                                      dir=".",
                                      prefix=prefix, suffix=suffix)
    return tmp


def loadChains(spaln, unsupportedIDs):
    chains = {}
    for row in csv.reader(open(spaln), delimiter='\t'):
        if len(row) != 9:
            continue
        if extractFeatureGff(row[8], "topProt") != "TRUE":
            continue

        ID = extractFeatureGff(row[8], "seed_gene_id")
        if ID not in unsupportedIDs:
            continue
        if ID not in chains:
            chains[ID] = Chain(ID, row[6])
        chains[ID].addFeature(row)

    for chain in chains.values():
        chain.sortFeatures()

    return chains


def getUnsupportedParts(pred, prothint):
    unsupported = temp("unsupported", ".gtf").name
    introns = []
    starts = []
    stops = []
    unsupportedIDs = set()

    predAnalysis = analysis.PredictionAnalysis(pred, prothint)
    predAnalysis.saveWithMissingSupport(unsupported)

    for row in csv.reader(open(unsupported), delimiter='\t'):
        if len(row) != 9:
            continue
        if row[2] == "CDS":
            continue
        if extractFeatureGtf(row[8], "supported") == "False":
            ID = extractFeatureGtf(row[8], "gene_id")
            unsupportedIDs.add(ID)
            if row[2].lower() == "intron":
                introns.append(Feature(int(row[3]), int(row[4]), row[6], ID))
            if row[2].lower() == "start_codon":
                starts.append(Feature(int(row[3]), int(row[4]), row[6], ID))
            if row[2].lower() == "stop_codon":
                stops.append(Feature(int(row[3]), int(row[4]), row[6], ID))
    os.remove(unsupported)
    return introns, starts, stops, unsupportedIDs


def overlaps(queryList, target, startFlag=False, margin=0):
    # Check whether the target feature overlaps any feature in the queryList

    i = 0
    while i < len(queryList) and queryList[i].end < target.start:
        i += 1

    if i == len(queryList):
        # Ran out of exons, none satisfied the condition
        return False

    if queryList[i].start <= target.end:
        # The opposite condition is already guaranteed by the while loop
        if startFlag:
            # Check whether the overlap coincides with the feature start. Start
            # codons can be overlapped by CDS like this.
            if target.strand == "+":
                if queryList[i].start != target.start:
                    return True
            else:
                if queryList[i].end != target.end:
                    return True
        else:
            if target.end - queryList[i].start > margin and \
               queryList[i].end - target.start > margin:
                return True

    return False


def getConflictingStarts(starts, chains):
    conflicts = set()
    for start in starts:
        if start.ID in chains and overlaps(chains[start.ID].exons,
                                           start, True):
            conflicts.add(start.ID)
        if start.ID in chains and overlaps(chains[start.ID].introns,
                                           start):
            conflicts.add(start.ID)
    return conflicts


def getConflictingStops(stops, chains):
    conflicts = set()
    for stop in stops:
        if stop.ID in chains and overlaps(chains[stop.ID].exons,
                                          stop):
            conflicts.add(stop.ID)
        if stop.ID in chains and overlaps(chains[stop.ID].introns,
                                          stop):
            conflicts.add(stop.ID)
    return conflicts


def getConflictingIntrons(introns, chains, intronMargin):
    conflicts = set()
    for intron in introns:
        if intron.ID in chains and overlaps(chains[intron.ID].exons,
                                            intron, margin=intronMargin):
            conflicts.add(intron.ID)
    return conflicts


def remove(pred, spaln, intronMargin, outFile):
    introns, starts, stops, unsupportedIDs = getUnsupportedParts(pred, spaln)
    chains = loadChains(spaln, unsupportedIDs)
    conflicting = getConflictingStarts(starts, chains)
    conflicting.update(getConflictingStops(stops, chains))
    conflicting.update(getConflictingIntrons(introns, chains, intronMargin))

    output = open(outFile, "w")
    for row in csv.reader(open(pred), delimiter='\t'):
        ID = extractFeatureGtf(row[8], "gene_id")
        if ID not in conflicting:
            output.write("\t".join(row) + "\n")
    output.close()


def main():
    args = parseCmd()
    remove(args.pred, args.spaln, args.intronMargin, args.outFile)


def parseCmd():

    parser = argparse.ArgumentParser(description='Remove predictions which \
        are in conflict with ProtHint. Specifically, remove predictions which \
        have an intron/start/stop not supported by any ProtHint \
        alignment and in conflict with the top chain.')

    parser.add_argument('pred', metavar='pred.gtf', type=str,
                        help='Predictions to filter.')

    parser.add_argument('spaln', metavar='spaln.gff', type=str,
                        help='Processed Spaln alignments (from ProtHint).')

    parser.add_argument('outFile', type=str, help='Save output here.')

    parser.add_argument('--intronMargin', type=int, default=100,
                        help='Allowed this big intron overlap. Default = 100')

    return parser.parse_args()


if __name__ == '__main__':
    main()