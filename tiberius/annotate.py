from functools import partial
from typing import Literal

import bricks2marble as b2m
import numpy as np
import tensorflow as tf
from hidten import HMMMode
from vipsania.xai.evaluate import predict_sequence


def _fix_intron_state_chain_labels(a: np.ndarray, isc: int) -> np.ndarray:
    mask = np.logical_and(0 < a, a <= 3*isc)
    a[mask] = ((a[mask] - 1) // isc) + 1
    a[a > 3*isc] = a[a > 3*isc] - 3*(isc-1)
    return a


def _evaluate(
    fasta: b2m.struct.FASTA,
    model: tf.keras.Model,
    B: int = 1,
    hmm_head: int = 0,
    use: Literal["VITERBI", "MEA"] = "VITERBI",
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    jit_compile: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.set_inference(True, mode=HMMMode[use])
    labels = predict_sequence(
        model,
        fasta,
        B=B,
        return_batched=True,
        N_token=N_token,
        masked=False,
        repeats_input=repeats_input,
        jit_compile=jit_compile,
    ).numpy()  # type: ignore
    model.set_inference(False)
    H = labels.shape[2] // 2
    labels_f = labels[:, :, hmm_head]
    labels_b = labels[:, :, H+hmm_head]
    if (isc := model.config.hmm.intron_state_chain) > 1:
        labels_f = _fix_intron_state_chain_labels(labels_f, isc=isc)
        labels_b = _fix_intron_state_chain_labels(labels_b, isc=isc)
    return labels_f, labels_b


def annotate_genome(
    model: tf.keras.Model,
    fasta: b2m.struct.FASTA,
    hmm_head: int = 0,
    B: int = 1,
    verbose: bool = True,
    use: Literal["VITERBI", "MEA"] = "VITERBI",
    repredict_exon_at_boundary: int | None = None,
    liberal: bool = True,
    N_token: Literal["track", "uniform"] = "track",
    repeats_input: Literal["track", "expand", "omit"] = "track",
    jit_compile: bool = False,
) -> b2m.struct.Annotation:
    annotation = b2m.tools.GTF_from_model(
        fasta,
        predict_func=partial(_evaluate,
            model=model,
            B=B,
            hmm_head=hmm_head,
            use=use,
            N_token=N_token,
            repeats_input=repeats_input,
            jit_compile=jit_compile,
        ),
        verbose=verbose,
        repredict_exon_at_boundary=repredict_exon_at_boundary,
        liberal=liberal,
    )
    return annotation
