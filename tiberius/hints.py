# ==============================================================
# Authors: Lars Gabriel
#
# Read hints (introns, start/stop codons), select consistent
# per-locus chains, and apply them to LSTM class probabilities
# before the HMM stage.
# ==============================================================

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

I_INNER = (1, 2, 3)
E0, E1, E2 = 4, 5, 6
START = 7
EI = (8, 9, 10)
IE = (11, 12, 13)
STOP = 14

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

    @property
    def span(self) -> Tuple[int, int]:
        return self.start, self.end


@dataclass
class HintChain:
    seqname: str
    strand: str
    support_id: Optional[str]
    start_hint: Optional[Hint]
    stop_hint: Optional[Hint]
    introns: List[Hint]
    score: float
    anchored: bool
    hints: List[Hint]

    @property
    def span(self) -> Tuple[int, int]:
        starts = [h.start for h in self.hints]
        ends = [h.end for h in self.hints]
        return min(starts), max(ends)


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        if value == ".":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_attrs(attr_text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for field in attr_text.split(";"):
        field = field.strip()
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
        elif " " in field:
            key, value = field.split(" ", 1)
            value = value.strip('"')
        else:
            key, value = field, ""
        out[key.strip()] = value.strip()
    return out


def load_hints(path: str) -> Dict[str, List[Hint]]:
    """
    Read a GFF-like hints file.

    Returns:
        dict[str, list[Hint]]
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


def _tx_interval(h: Hint, strand: str) -> Tuple[int, int]:
    """
    Transcript-oriented interval.

    For '+' this is just genomic [start, end).
    For '-' we mirror the coordinates so transcript order becomes ascending.
    """
    if strand == "+":
        return h.start, h.end
    return -h.end, -h.start


def _hint_weight(h: Hint) -> float:
    """
    Heuristic score for choosing among competing hints/chains.
    """
    support_bonus = np.log1p(len(h.prots))
    return float(h.score) + 10.0 * float(h.al_score) + float(support_bonus)


def _cluster_loci(hints: Sequence[Hint], max_gap: int = 20000) -> List[List[Hint]]:
    """
    Group nearby hints into coarse loci.
    """
    if not hints:
        return []

    ordered = sorted(hints, key=lambda h: h.start)
    loci: List[List[Hint]] = []

    cur = [ordered[0]]
    cur_end = ordered[0].end
    for h in ordered[1:]:
        if h.start <= cur_end + max_gap:
            cur.append(h)
            cur_end = max(cur_end, h.end)
        else:
            loci.append(cur)
            cur = [h]
            cur_end = h.end
    loci.append(cur)
    return loci


def _introns_between(
    introns: Sequence[Hint],
    start_hint: Optional[Hint],
    stop_hint: Optional[Hint],
    strand: str,
) -> List[Hint]:
    out: List[Hint] = []
    start_tx = _tx_interval(start_hint, strand)[1] if start_hint is not None else None
    stop_tx = _tx_interval(stop_hint, strand)[0] if stop_hint is not None else None

    for intron in introns:
        i_s, i_e = _tx_interval(intron, strand)
        if start_tx is not None and i_s < start_tx:
            continue
        if stop_tx is not None and i_e > stop_tx:
            continue
        out.append(intron)

    return out


def _weighted_interval_scheduling_mod3(
    hints: Sequence[Hint],
    strand: str,
    required_mod: Optional[int] = None,
) -> List[Hint]:
    """
    Best non-overlapping set of intervals.
    If required_mod is not None, the sum of selected hint lengths must satisfy:
        sum(lengths) % 3 == required_mod
    """
    if not hints:
        return []

    ordered = sorted(hints, key=lambda h: (_tx_interval(h, strand)[1], _tx_interval(h, strand)[0]))
    tx = [_tx_interval(h, strand) for h in ordered]
    starts = [t[0] for t in tx]
    ends = [t[1] for t in tx]
    weights = [_hint_weight(h) for h in ordered]
    mods = [h.length % 3 for h in ordered]

    prev = [-1] * len(ordered)
    for j in range(len(ordered)):
        i = j - 1
        while i >= 0:
            if ends[i] <= starts[j]:
                prev[j] = i
                break
            i -= 1

    neg_inf = -1e30
    dp = [[neg_inf] * 3 for _ in range(len(ordered))]
    take = [[False] * 3 for _ in range(len(ordered))]
    back_mod = [[0] * 3 for _ in range(len(ordered))]

    for j in range(len(ordered)):
        for r in range(3):
            excl = dp[j - 1][r] if j > 0 else (0.0 if r == 0 else neg_inf)

            incl_best = neg_inf
            incl_prev_r = 0
            for prev_r in range(3):
                base = dp[prev[j]][prev_r] if prev[j] >= 0 else (0.0 if prev_r == 0 else neg_inf)
                if base <= neg_inf / 2:
                    continue
                new_r = (prev_r + mods[j]) % 3
                if new_r != r:
                    continue
                cand = base + weights[j]
                if cand > incl_best:
                    incl_best = cand
                    incl_prev_r = prev_r

            if incl_best > excl:
                dp[j][r] = incl_best
                take[j][r] = True
                back_mod[j][r] = incl_prev_r
            else:
                dp[j][r] = excl
                take[j][r] = False
                back_mod[j][r] = r

    if required_mod is None:
        final_r = max(range(3), key=lambda r: dp[-1][r])
    else:
        final_r = required_mod
        if dp[-1][final_r] <= neg_inf / 2:
            return []

    chosen: List[Hint] = []
    j = len(ordered) - 1
    r = final_r
    while j >= 0:
        if take[j][r]:
            chosen.append(ordered[j])
            prev_r = back_mod[j][r]
            j = prev[j]
            r = prev_r
        else:
            j -= 1

    chosen.reverse()
    return chosen


def _full_chain_cds_len(
    start_hint: Hint,
    stop_hint: Hint,
    introns: Sequence[Hint],
    strand: str,
) -> int:
    start_tx = _tx_interval(start_hint, strand)[0]
    stop_tx = _tx_interval(stop_hint, strand)[1]
    intron_len = sum(h.length for h in introns)
    return (stop_tx - start_tx) - intron_len


def _chain_is_frame_consistent(
    start_hint: Optional[Hint],
    stop_hint: Optional[Hint],
    introns: Sequence[Hint],
    strand: str,
) -> bool:
    """
    Check whether the chain is compatible with a frame-consistent CDS.
    """
    if start_hint is None:
        return False

    ordered_introns = sorted(introns, key=lambda h: _tx_interval(h, strand)[0])

    cursor = _tx_interval(start_hint, strand)[0]
    for intron in ordered_introns:
        i_s, i_e = _tx_interval(intron, strand)
        exon_len = i_s - cursor
        if exon_len <= 0:
            return False
        cursor = i_e

    if stop_hint is not None:
        cds_len = _full_chain_cds_len(start_hint, stop_hint, ordered_introns, strand)
        if cds_len <= 0 or cds_len % 3 != 0:
            return False

    return True


def _chain_score(
    start_hint: Optional[Hint],
    stop_hint: Optional[Hint],
    introns: Sequence[Hint],
) -> float:
    score = 0.0
    if start_hint is not None:
        score += _hint_weight(start_hint) + 5.0
    if stop_hint is not None:
        score += _hint_weight(stop_hint) + 5.0
    score += sum(_hint_weight(h) for h in introns)
    return score


def _build_best_chain_for_support(
    locus_hints: Sequence[Hint],
    support_id: str,
    strand: str,
    require_anchored: bool = True,
) -> Optional[HintChain]:
    """
    Build the best compatible chain for one support/protein ID inside one locus.

    Notes:
    - Because the hint file is merged evidence, support_id is only a heuristic.
    - We require a start for exact phase propagation.
    """
    subset = [h for h in locus_hints if h.strand == strand and support_id in h.prots]
    if not subset:
        return None

    starts = [h for h in subset if h.feature == "start"]
    stops = [h for h in subset if h.feature == "stop"]
    introns = [h for h in subset if h.feature == "intron"]

    candidates: List[HintChain] = []

    if starts:
        for start_hint in starts:
            start_tx = _tx_interval(start_hint, strand)[0]

            if stops:
                for stop_hint in stops:
                    stop_tx = _tx_interval(stop_hint, strand)[0]
                    if start_tx >= stop_tx:
                        continue

                    inner_introns = _introns_between(introns, start_hint, stop_hint, strand)

                    # Need:
                    # (stop_tx_end - start_tx_start - sum(intron_lengths)) % 3 == 0
                    # -> sum(intron_lengths) % 3 == span_mod
                    span_mod = (_tx_interval(stop_hint, strand)[1] - _tx_interval(start_hint, strand)[0]) % 3
                    best_introns = _weighted_interval_scheduling_mod3(
                        inner_introns,
                        strand=strand,
                        required_mod=span_mod,
                    )

                    if not _chain_is_frame_consistent(start_hint, stop_hint, best_introns, strand):
                        continue

                    hints = [start_hint, *best_introns, stop_hint]
                    candidates.append(
                        HintChain(
                            seqname=start_hint.seqname,
                            strand=strand,
                            support_id=support_id,
                            start_hint=start_hint,
                            stop_hint=stop_hint,
                            introns=best_introns,
                            score=_chain_score(start_hint, stop_hint, best_introns),
                            anchored=True,
                            hints=hints,
                        )
                    )
            else:
                downstream_introns = _introns_between(introns, start_hint, None, strand)
                best_introns = _weighted_interval_scheduling_mod3(
                    downstream_introns,
                    strand=strand,
                    required_mod=None,
                )

                if not _chain_is_frame_consistent(start_hint, None, best_introns, strand):
                    continue

                hints = [start_hint, *best_introns]
                candidates.append(
                    HintChain(
                        seqname=start_hint.seqname,
                        strand=strand,
                        support_id=support_id,
                        start_hint=start_hint,
                        stop_hint=None,
                        introns=best_introns,
                        score=_chain_score(start_hint, None, best_introns),
                        anchored=True,
                        hints=hints,
                    )
                )

    if not candidates and not require_anchored and introns:
        best_introns = _weighted_interval_scheduling_mod3(
            introns,
            strand=strand,
            required_mod=None,
        )
        if best_introns:
            candidates.append(
                HintChain(
                    seqname=best_introns[0].seqname,
                    strand=strand,
                    support_id=support_id,
                    start_hint=None,
                    stop_hint=None,
                    introns=best_introns,
                    score=_chain_score(None, None, best_introns),
                    anchored=False,
                    hints=list(best_introns),
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda ch: (ch.score, ch.anchored, len(ch.introns), ch.stop_hint is not None),
        reverse=True,
    )
    return candidates[0]


def _chains_overlap(a: HintChain, b: HintChain) -> bool:
    a_s, a_e = a.span
    b_s, b_e = b.span
    return not (a_e <= b_s or b_e <= a_s)


def select_consistent_hint_chains(
    hints: Dict[str, List[Hint]],
    max_locus_gap: int = 20000,
    max_chains_per_locus: int = 1,
    min_chain_score: float = 0.0,
    require_anchored: bool = True,
) -> Dict[str, List[HintChain]]:
    """
    Convert a merged hint pool into selected compatible chains.

    Default behavior is conservative:
    - one chain per locus
    - require start anchoring
    """
    out: Dict[str, List[HintChain]] = defaultdict(list)

    for seqname, seq_hints in hints.items():
        for strand in ("+", "-"):
            strand_hints = [h for h in seq_hints if h.strand == strand]
            if not strand_hints:
                continue

            for locus in _cluster_loci(strand_hints, max_gap=max_locus_gap):
                support_ids: Set[str] = set()
                for h in locus:
                    support_ids.update(h.prots)

                candidates: List[HintChain] = []
                for support_id in support_ids:
                    chain = _build_best_chain_for_support(
                        locus,
                        support_id=support_id,
                        strand=strand,
                        require_anchored=require_anchored,
                    )
                    if chain is not None and chain.score >= min_chain_score:
                        candidates.append(chain)

                candidates.sort(
                    key=lambda ch: (ch.score, ch.anchored, len(ch.introns), ch.stop_hint is not None),
                    reverse=True,
                )

                selected: List[HintChain] = []
                for chain in candidates:
                    if len(selected) >= max_chains_per_locus:
                        break
                    if any(_chains_overlap(chain, prev) for prev in selected):
                        continue
                    selected.append(chain)

                out[seqname].extend(selected)

    return dict(out)


def _emit_intron_ambiguous(
    gpos: List[int],
    classes: List[Tuple[int, ...]],
    start: int,
    end: int,
    strand: str,
) -> None:
    """
    Fallback for unanchored introns: still phase-ambiguous.
    """
    L = end - start
    if L < 6:
        for p in range(start, end):
            gpos.append(p)
            classes.append(I_INNER + EI + IE)
        return

    if strand == "+":
        donor = range(start, start + 3)
        acceptor = range(end - 3, end)
    else:
        # On the minus strand the donor is at the genomic high end.
        donor = range(end - 3, end)
        acceptor = range(start, start + 3)

    donor_set = set(donor)
    acceptor_set = set(acceptor)

    for p in donor:
        gpos.append(p)
        classes.append(EI)
    for p in range(start, end):
        if p in donor_set or p in acceptor_set:
            continue
        gpos.append(p)
        classes.append(I_INNER)
    for p in acceptor:
        gpos.append(p)
        classes.append(IE)


def _emit_intron_exact(
    gpos: List[int],
    classes: List[Tuple[int, ...]],
    start: int,
    end: int,
    phase: int,
    strand: str,
) -> None:
    """
    Emit one exact intron phase, not all three.
    """
    if phase not in (0, 1, 2):
        raise ValueError(f"Invalid intron phase: {phase}")

    L = end - start
    donor_cls = (EI[phase],)
    inner_cls = (I_INNER[phase],)
    acceptor_cls = (IE[phase],)

    if L < 6:
        # Too short to cleanly split donor/body/acceptor.
        # Still keep it exact-phase, but allow any intron-related state of this phase.
        for p in range(start, end):
            gpos.append(p)
            classes.append(donor_cls + inner_cls + acceptor_cls)
        return

    if strand == "+":
        donor = range(start, start + 3)
        acceptor = range(end - 3, end)
    else:
        donor = range(end - 3, end)
        acceptor = range(start, start + 3)

    donor_set = set(donor)
    acceptor_set = set(acceptor)

    for p in donor:
        gpos.append(p)
        classes.append(donor_cls)
    for p in range(start, end):
        if p in donor_set or p in acceptor_set:
            continue
        gpos.append(p)
        classes.append(inner_cls)
    for p in acceptor:
        gpos.append(p)
        classes.append(acceptor_cls)


def _emit_codon(
    gpos: List[int],
    classes: List[Tuple[int, ...]],
    start: int,
    end: int,
    kind: str,
    strand: str,
) -> None:
    if end - start < 3:
        return

    if kind == "start":
        cls_seq = ((START,), (E1,), (E2,))
    else:
        cls_seq = ((E0,), (E1,), (STOP,))

    if strand == "+":
        seq_pos = (start, start + 1, start + 2)
    else:
        seq_pos = (end - 1, end - 2, end - 3)

    for p, c in zip(seq_pos, cls_seq):
        gpos.append(p)
        classes.append(c)


def _emit_chain_constraints(chain: HintChain) -> Tuple[List[int], List[Tuple[int, ...]]]:
    """
    Convert a selected chain into exact per-position state constraints.
    """
    gpos: List[int] = []
    classes: List[Tuple[int, ...]] = []

    if chain.start_hint is not None:
        _emit_codon(
            gpos,
            classes,
            chain.start_hint.start,
            chain.start_hint.end,
            "start",
            chain.strand,
        )

    ordered_introns = sorted(chain.introns, key=lambda h: _tx_interval(h, chain.strand)[0])

    if chain.start_hint is not None:
        # Exact phase propagation from the start codon.
        cursor = _tx_interval(chain.start_hint, chain.strand)[0]
        for intron in ordered_introns:
            intron_tx_start, intron_tx_end = _tx_interval(intron, chain.strand)
            exon_len = intron_tx_start - cursor
            phase = exon_len % 3
            _emit_intron_exact(
                gpos,
                classes,
                intron.start,
                intron.end,
                phase=phase,
                strand=chain.strand,
            )
            cursor = intron_tx_end
    else:
        # Unanchored fallback: keep these soft/ambiguous.
        for intron in ordered_introns:
            _emit_intron_ambiguous(
                gpos,
                classes,
                intron.start,
                intron.end,
                strand=chain.strand,
            )

    if chain.stop_hint is not None:
        _emit_codon(
            gpos,
            classes,
            chain.stop_hint.start,
            chain.stop_hint.end,
            "stop",
            chain.strand,
        )

    return gpos, classes


def _looks_like_chain_dict(hints) -> bool:
    if not hints:
        return False
    first_value = next(iter(hints.values()))
    return bool(first_value) and isinstance(first_value[0], HintChain)


def apply_hints(
    lstm_out: np.ndarray,
    fasta,
    hints,
    weight: float,
    strand: str,
    mode: str = "soft",
    max_locus_gap: int = 20000,
    max_chains_per_locus: int = 1,
    min_chain_score: float = 0.0,
    require_anchored: bool = True,
) -> np.ndarray:
    """
    Apply selected hint chains to LSTM outputs.

    Args:
        lstm_out: shape (N, T, C), post-softmax probabilities.
        fasta: chunked fasta object used to produce lstm_out.
        hints:
            either raw output of load_hints()
            or output of select_consistent_hint_chains()
        weight:
            multiplicative factor for allowed classes.
        strand:
            '+' or '-'
        mode:
            'soft' -> multiply allowed classes by weight
            'hard' -> zero disallowed classes at constrained positions,
                      then optionally weight the allowed ones
        max_locus_gap, max_chains_per_locus, min_chain_score, require_anchored:
            passed to select_consistent_hint_chains if raw hints are given.
    """
    if hints is None or weight <= 0.0:
        return lstm_out
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")

    if _looks_like_chain_dict(hints):
        chains = hints
    else:
        chains = select_consistent_hint_chains(
            hints,
            max_locus_gap=max_locus_gap,
            max_chains_per_locus=max_chains_per_locus,
            min_chain_score=min_chain_score,
            require_anchored=require_anchored,
        )

    if not chains:
        return lstm_out

    N, T, C = lstm_out.shape
    chunk_offset = 0
    allowed_by_pos: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

    for seq in fasta:
        n_chunks = seq.N
        seq_start = seq.start
        seq_end = seq.start + seq.size

        for chain in chains.get(seq.name, ()):
            if chain.strand != strand:
                continue

            gpos_buf, cls_buf = _emit_chain_constraints(chain)
            for gpos, allowed_classes in zip(gpos_buf, cls_buf):
                if gpos < seq_start or gpos >= seq_end:
                    continue

                rel = gpos - seq_start
                chunk_local = rel // T
                pos_local = rel % T
                if strand == "-":
                    pos_local = T - 1 - pos_local

                if chunk_local < 0 or chunk_local >= n_chunks:
                    continue

                allowed_by_pos[(chunk_offset + int(chunk_local), int(pos_local))].update(allowed_classes)

        chunk_offset += n_chunks

    if not allowed_by_pos:
        return lstm_out

    for (chunk_idx, pos_idx), allowed in allowed_by_pos.items():
        allowed_idx = np.fromiter(sorted(allowed), dtype=np.int64)

        if mode == "soft":
            if weight != 1.0:
                lstm_out[chunk_idx, pos_idx, allowed_idx] *= np.float32(weight)
        else:
            mask = np.zeros(C, dtype=bool)
            mask[allowed_idx] = True
            lstm_out[chunk_idx, pos_idx, ~mask] = np.float32(0.0)
            if weight != 1.0:
                lstm_out[chunk_idx, pos_idx, allowed_idx] *= np.float32(weight)

        denom = lstm_out[chunk_idx, pos_idx].sum(dtype=np.float32)
        if denom <= 1e-12:
            if mode == "hard":
                lstm_out[chunk_idx, pos_idx] = np.float32(0.0)
                lstm_out[chunk_idx, pos_idx, allowed_idx] = np.float32(1.0 / len(allowed_idx))
            continue

        lstm_out[chunk_idx, pos_idx] /= np.float32(denom)

    return lstm_out