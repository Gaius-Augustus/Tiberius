"""Instrumentation hook for the NSS spliced-stop reprediction path.

Importing this module monkey-patches
`tiberius.eval_model_class.PredictionGTF._patch_spliced_stops` so that
every call records a per-span JSONL entry describing the state before
and after the NSS overwrite. Enabled by setting
`TIBERIUS_SPLICED_STOP_DEBUG_DIR` to the output directory; otherwise the
patch is a transparent passthrough.

Per-span record (one JSONL line per detected spliced-stop span):

{
    "call_idx":    int,                # monotonically increasing per Python process
    "strand":      "+" | "-",
    "tx_s":        int,                # flat start of the transcript in this batch
    "tx_e":        int,                # flat end
    "span_len":    int,                # tx_e - tx_s
    "patched":     bool,               # false iff span_len > seq_len (oversized, left alone)
    "pre": {                           # standard-HMM annotation of this span
        "cds_regions": [[start, end], ...],  # in flat batch coords
        "cds_bases":   "ATG...",             # concatenated CDS, strand-oriented
        "frame0":      [codon_code, ...],    # non-overlapping step-3 codons
        "stop_frame0_at": [codon_idx, ...],  # which frame-0 codons are stops (excl. terminal)
        "has_inframe_stop": bool,
    },
    "post": {                          # same, but on the labels after NSS overwrite
        "cds_regions": [...],
        "cds_bases":   "...",
        "frame0":      [...],
        "stop_frame0_at": [...],
        "has_inframe_stop": bool,
        "num_transcripts_in_span": int,   # how many IR-bounded txs fall in [tx_s, tx_e]
        "span_label_sequence_summary": [[state, length], ...],  # run-length of label states
    }
}

Classification derived in the analyzer:
    pre.has_inframe_stop  post.has_inframe_stop
         True                   True     -> TP, NSS failed to fix (bug: NSS HMM / remap)
         True                   False    -> TP, fixed (expected)
         False                  True     -> FP, reprediction INTRODUCED a stop (hypothesis for NEW>OLD)
         False                  False    -> FP, harmless

Oversized (patched=False) spans are left with their original labels and
contribute whatever the standard HMM produced; both runs count them.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import numpy as np

from bricks2marble.struct.fasta import complement, nucleotides_to_kmers
from bricks2marble.tools.annotate import _split_regions, _transcripts_from_regions

from tiberius import eval_model_class as _emc
from tiberius.spliced_stop import (
    STOP_CODON_CODES,
    find_spliced_stop_transcripts,
)


_DEBUG_DIR = os.environ.get("TIBERIUS_SPLICED_STOP_DEBUG_DIR")
_CALL_IDX = 0
_NUC_CHARS = "ACGTN"


def _nuc_to_str(nuc_arr: np.ndarray) -> str:
    arr = np.asarray(nuc_arr, dtype=np.int64).copy()
    arr[arr > 4] -= 5
    out = []
    for c in arr.tolist():
        out.append(_NUC_CHARS[c] if 0 <= c < 5 else "?")
    return "".join(out)


def _cds_info(cds_regions, flat_nuc: np.ndarray, strand: str) -> dict[str, Any]:
    if not cds_regions:
        return {
            "cds_regions": [],
            "cds_bases": "",
            "frame0": [],
            "stop_frame0_at": [],
            "has_inframe_stop": False,
        }
    cds_nuc = np.concatenate(
        [flat_nuc[r.start:r.end] for r in cds_regions]
    ).astype(np.int64)
    if strand == "-":
        cds_nuc = complement(cds_nuc.copy(), reverse=True)
    base_str = _nuc_to_str(cds_nuc)
    if cds_nuc.size < 6:
        return {
            "cds_regions": [[int(r.start), int(r.end)] for r in cds_regions],
            "cds_bases": base_str,
            "frame0": [],
            "stop_frame0_at": [],
            "has_inframe_stop": False,
        }
    kmers = nucleotides_to_kmers(cds_nuc.copy()).tolist()
    stops = set(STOP_CODON_CODES.tolist())
    stop_at = [i for i, k in enumerate(kmers[:-1]) if k in stops]
    return {
        "cds_regions": [[int(r.start), int(r.end)] for r in cds_regions],
        "cds_bases": base_str,
        "frame0": [int(k) for k in kmers],
        "stop_frame0_at": [int(i) for i in stop_at],
        "has_inframe_stop": len(stop_at) > 0,
    }


def _find_pre_transcript(flat_labels: np.ndarray, tx_s: int, tx_e: int, strand: str):
    """Return CDS region list for the IR-bounded transcript at exactly
    [tx_s, tx_e). Matches how `find_spliced_stop_transcripts` emitted the
    span, so the lookup is unambiguous."""
    regions = _split_regions(flat_labels, strand=strand)
    for tx_regions in _transcripts_from_regions(regions):
        if int(tx_regions[0].start) == int(tx_s) and int(tx_regions[-1].end) == int(tx_e):
            return [r for r in tx_regions if r.name == "CDS"]
    return []


def _collect_post_transcripts(flat_labels: np.ndarray, tx_s: int, tx_e: int, strand: str):
    """Return every IR-bounded transcript whose span overlaps [tx_s, tx_e)."""
    regions = _split_regions(flat_labels, strand=strand)
    out = []
    for tx_regions in _transcripts_from_regions(regions):
        s = int(tx_regions[0].start)
        e = int(tx_regions[-1].end)
        if e > tx_s and s < tx_e:
            out.append([r for r in tx_regions if r.name == "CDS"])
    return out


def _label_runlength(labels_1d: np.ndarray, limit: int = 32):
    """Compact (state, run_len) list for a span's labels. Truncates to
    `limit` runs so we don't bloat the JSONL with giant genes."""
    arr = np.asarray(labels_1d).ravel()
    if arr.size == 0:
        return []
    change = np.where(np.diff(arr) != 0)[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [arr.size]))
    runs = [[int(arr[s]), int(e - s)] for s, e in zip(starts, ends)]
    if len(runs) > limit:
        runs = runs[:limit] + [[-1, int(sum(e - s for s, e in zip(starts[limit:], ends[limit:])))]]
    return runs


_ORIG_PATCH = _emc.PredictionGTF._patch_spliced_stops


def _patched(self, labels, nuc, x_one_hot, lstm_out, strand, hmm_frame_reversed=False):
    global _CALL_IDX
    if _DEBUG_DIR is None or labels is None or labels.size == 0:
        return _ORIG_PATCH(
            self, labels, nuc, x_one_hot, lstm_out, strand, hmm_frame_reversed,
        )

    out_dir = pathlib.Path(_DEBUG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "spans.jsonl"

    call_idx = _CALL_IDX
    _CALL_IDX += 1

    seq_T = int(getattr(self, "seq_len", labels.shape[-1]))

    # Snapshot pre-patch state.
    flat_labels_pre = labels.reshape(-1).copy()
    flat_nuc = nuc.reshape(-1)
    spans = find_spliced_stop_transcripts(labels, nuc, strand=strand)
    pre_entries = []
    for span in spans:
        pre_cds = _find_pre_transcript(flat_labels_pre, span.start, span.end, strand)
        pre_entries.append({
            "span": (int(span.start), int(span.end)),
            "pre_info": _cds_info(pre_cds, flat_nuc, strand),
            "pre_labels_rle": _label_runlength(
                flat_labels_pre[span.start:span.end]
            ),
        })

    # Delegate: this mutates `labels` in place for small/big spans.
    _ORIG_PATCH(self, labels, nuc, x_one_hot, lstm_out, strand, hmm_frame_reversed)

    flat_labels_post = labels.reshape(-1)
    with log_path.open("a") as fh:
        for entry in pre_entries:
            tx_s, tx_e = entry["span"]
            span_len = tx_e - tx_s
            patched = span_len <= seq_T  # oversized spans are left untouched
            post_txs = _collect_post_transcripts(flat_labels_post, tx_s, tx_e, strand)
            post_combined_cds = [cds for tx in post_txs for cds in tx]
            post_info = _cds_info(post_combined_cds, flat_nuc, strand)
            post_info["num_transcripts_in_span"] = len(post_txs)
            record = {
                "call_idx": call_idx,
                "strand": strand,
                "tx_s": tx_s,
                "tx_e": tx_e,
                "span_len": span_len,
                "patched": bool(patched),
                "pre": entry["pre_info"],
                "pre_labels_rle": entry["pre_labels_rle"],
                "post": post_info,
                "post_labels_rle": _label_runlength(
                    flat_labels_post[tx_s:tx_e]
                ),
            }
            fh.write(json.dumps(record) + "\n")


_emc.PredictionGTF._patch_spliced_stops = _patched
