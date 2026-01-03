#!/usr/bin/env python3
import sys
from collections import defaultdict

if len(sys.argv) < 2:
    sys.stderr.write(f"Usage: {sys.argv[0]} diamond_hits.tsv [TOP_N]\n")
    sys.exit(1)

in_path = sys.argv[1]
TOP_N = int(sys.argv[2]) if len(sys.argv) > 2 else 8

# thresholds (tune if you want)
MAX_EVALUE = 1e-5
MIN_PIDENT = 30.0
MIN_QCOV   = 0.5   # coverage = aligned_length / qlen

def get_species_id(sseqid: str) -> str:
    """
    Extract species ID from OrthoDB-style sseqid.
    Example: '101020_0:000003' -> '101020'
    Adjust if your header format is different.
    """
    head = sseqid.split()[0]
    return head.split("_", 1)[0]

# best hit *per query* (over all species)
# q -> (best_score, best_species)
best_hit_for_query = {}

with open(in_path, "r") as fh:
    for line in fh:
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        if len(cols) < 8:
            continue

        qseqid, sseqid = cols[0], cols[1]
        pident = float(cols[2])
        alen   = float(cols[3])
        evalue = float(cols[4])
        bits   = float(cols[5])
        qlen   = float(cols[6])
        slen   = float(cols[7])

        if evalue > MAX_EVALUE:
            continue
        if pident < MIN_PIDENT:
            continue
        if qlen <= 0:
            continue

        qcov = alen / qlen
        if qcov < MIN_QCOV:
            continue

        species = get_species_id(sseqid)
        score   = bits * qcov   # combined strength

        if qseqid not in best_hit_for_query or score > best_hit_for_query[qseqid][0]:
            best_hit_for_query[qseqid] = (score, species)

# aggregate per species from the best hits
species_votes = defaultdict(int)
species_score = defaultdict(float)

for q, (score, sp) in best_hit_for_query.items():
    species_votes[sp] += 1
    species_score[sp] += score

ranking = []
for sp in species_votes:
    ranking.append((sp, species_votes[sp], species_score[sp]))

# sort: 1) #queries, 2) total score
ranking.sort(key=lambda x: (-x[1], -x[2]))

with open("top_species.txt", "w") as out_sp:
    for i, (sp, n_q, tot) in enumerate(ranking):
        mark = "TOP" if i < TOP_N else ""
        sys.stdout.write(f"{i+1}\t{sp}\tqueries={n_q}\ttotal_score={tot:.1f}\t{mark}\n")
        if i < TOP_N:
            out_sp.write(sp + "\n")

sys.stderr.write(f"Wrote top {TOP_N} species to top_species.txt\n")
