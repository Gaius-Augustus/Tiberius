# ==============================================================
# Authors: Lars Gabriel
#
# Read intron / start / stop hints from a GFF-like file and convert
# intron hints into the (interior, left_border, right_border) channels
# expected by the HMM's ``intron_hint_emitter`` (bricks2marble).
# ==============================================================

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

_FEATURE_ALIASES = {
    "intron": "intron",
    "start": "start",
    "start_codon": "start",
    "stop": "stop",
    "stop_codon": "stop",
}


@dataclass(frozen=True)
class Hint:
    seqname: str
    feature: str
    start: int
    end: int
    strand: str
    score: float = 0.0
    al_score: float = 0.0
    prots: frozenset[str] = frozenset()
    attrs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def length(self) -> int:
        return self.end - self.start


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        if value == ".":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_attrs(attr_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for chunk in attr_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        elif " " in chunk:
            key, value = chunk.split(" ", 1)
            value = value.strip('"')
        else:
            key, value = chunk, ""
        out[key.strip()] = value.strip()
    return out


def load_hints(path: str) -> Dict[str, List[Hint]]:
    """
    Read a GFF-like hints file and return ``{seqname: [Hint, ...]}``.

    Coordinates are converted to 0-based half-open ``[start, end)``.
    Only ``intron`` / ``start[_codon]`` / ``stop[_codon]`` features and
    ``+`` / ``-`` strands are kept. The HMM-channel pipeline currently
    only consumes intron hints; start/stop entries are loaded but
    ignored downstream.
    """
    hints: Dict[str, List[Hint]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            seqname, _src, feature, start, end, score, strand, _frame = parts[:8]
            attrs = parts[8] if len(parts) > 8 else ""

            feature = _FEATURE_ALIASES.get(feature.lower())
            if feature is None or strand not in ("+", "-"):
                continue
            try:
                s = int(start) - 1
                e = int(end)
            except ValueError:
                continue
            if e <= s:
                continue
            attr_map = _parse_attrs(attrs)
            prots_raw = attr_map.get("prots", "")
            prots = frozenset(x for x in prots_raw.split(",") if x)
            hints[seqname].append(
                Hint(
                    seqname=seqname,
                    feature=feature,
                    start=s,
                    end=e,
                    strand=strand,
                    score=_safe_float(score, 0.0),
                    al_score=_safe_float(attr_map.get("al_score"), 0.0),
                    prots=prots,
                    attrs=tuple(sorted(attr_map.items())),
                )
            )
    return dict(hints)


def count_intron_hints(hints: Dict[str, List[Hint]]) -> int:
    if not hints:
        return 0
    return sum(1 for hs in hints.values() for h in hs if h.feature == "intron")


def build_intron_hint_channels(
    fasta,
    hints: Dict[str, List[Hint]],
    strand: str,
) -> np.ndarray:
    """
    Build the ``(interior, left_border, right_border)`` channels expected
    by the HMM's ``intron_hint_emitter`` for one strand.

    Output shape: ``(N_total, T, 3)`` where ``N_total`` is the sum of
    chunk counts ``seq.N`` over all sequences in ``fasta`` and ``T`` is
    the chunk length of the fasta. The three channels are mutually
    exclusive at every position (zero everywhere outside intron hints).

    Frames:
        ``strand == '+'``: genomic chunk frame
            ``chunk = pos // T``, ``pos_in_chunk = pos % T``.
            ``left_border`` is the donor (low genomic end of the intron),
            ``right_border`` is the acceptor (high genomic end).
        ``strand == '-'``: bwd-chunk frame used by Tiberius after
            ``x_one_hot_bwd[:, ::-1, :]``:
            ``chunk = pos // T``, ``pos_in_chunk = T - 1 - (pos % T)``.
            On the minus strand the donor sits at the high genomic end,
            which becomes the lowest position in bwd-chunk order, so it
            is still emitted as ``left_border``; the acceptor at the
            low genomic end becomes ``right_border``.

    Only intron hints with the requested ``strand`` are emitted. Introns
    shorter than 3 bp or that do not lie inside the sequence range are
    skipped.
    """
    if strand not in ("+", "-"):
        raise ValueError(f"Bad strand argument: {strand!r}")

    arrays: List[np.ndarray] = []
    for seq in fasta:
        N, T = seq.N, seq.T
        ih = np.zeros((N, T, 3), dtype=np.float32)
        for h in hints.get(seq.name, ()) if hints else ():
            if h.feature != "intron" or h.strand != strand:
                continue
            a = h.start - seq.start
            b = h.end - seq.start
            if a < 0 or b > seq.size or b - a < 3:
                continue

            positions = np.arange(a, b)
            chunks = positions // T
            mask = (chunks >= 0) & (chunks < N)
            if not mask.any():
                continue
            positions = positions[mask]
            chunks = chunks[mask]
            local = positions % T
            if strand == "+":
                poss = local
                donor_p = a
                acceptor_p = b - 1
            else:
                poss = T - 1 - local
                donor_p = b - 1
                acceptor_p = a

            ih[chunks, poss, :] = 0.0
            ih[chunks, poss, 0] = 1.0
            donor_mask = positions == donor_p
            if donor_mask.any():
                dc, dp = chunks[donor_mask][0], poss[donor_mask][0]
                ih[dc, dp, :] = 0.0
                ih[dc, dp, 1] = 1.0
            acceptor_mask = positions == acceptor_p
            if acceptor_mask.any():
                ac, ap = chunks[acceptor_mask][0], poss[acceptor_mask][0]
                ih[ac, ap, :] = 0.0
                ih[ac, ap, 2] = 1.0
        arrays.append(ih)

    if not arrays:
        return np.zeros((0, 0, 3), dtype=np.float32)
    return np.concatenate(arrays, axis=0)
