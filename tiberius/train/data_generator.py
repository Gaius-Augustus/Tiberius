# ==============================================================
# Authors: Lars Gabriel
#
# Datagenerator for loading training examples from tfrecords
# ==============================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Union, Optional, Tuple, Dict, Any

import tensorflow as tf

# ───────────────────────── config ─────────────────────────
@dataclass(frozen=True)
class DataGeneratorConfig:
    files: Union[str, Sequence[str]]             # list of paths
    batch_size: int = 500
    shuffle: bool = True
    repeat: bool = True
    output_size: int = 15
    hmm_factor: Optional[int] = None
    input_size: int = 6
    clamsa: bool = False
    oracle: bool = False
    tx_filter: Sequence[bytes] = ()
    tx_filter_region: int = 1000
    seq_weights_window: int = 250                # r
    seq_weights_value: float = 100.0             # w

    compression: Optional[str] = "GZIP"
    shuffle_buffer: int = 100
    num_parallel_calls: Any = tf.data.AUTOTUNE

    # default value: serialized empty [0,3] string tensor
    @property
    def empty_tx_serial(self) -> tf.Tensor:
        return tf.io.serialize_tensor(tf.constant([], shape=[0, 3], dtype=tf.string))



# ─────────────────────── feature specs ───────────────────────
def _feature_spec(clamsa: bool) -> Dict[str, tf.io.FixedLenFeature]:
    spec = {
        "input":  tf.io.FixedLenFeature([], tf.string),
        "output": tf.io.FixedLenFeature([], tf.string),
        "tx_ids": tf.io.FixedLenFeature([], tf.string),  # will fill default in parse
    }
    if clamsa:
        spec["clamsa"] = tf.io.FixedLenFeature([], tf.string)
    return spec


# ───────────────────── vectorized parse ─────────────────────
def _parse_batch(examples: tf.Tensor, cfg: DataLoaderConfig) -> Dict[str, tf.Tensor]:
    feats = tf.io.parse_example(examples, _feature_spec(cfg.clamsa))

    # ensure a default for missing tx_ids
    tx_ids_serial = tf.where(
        tf.equal(feats["tx_ids"], ""),
        tf.fill(tf.shape(feats["tx_ids"]), tf.cast(cfg.empty_tx_serial, tf.string)),
        feats["tx_ids"],
    )

    x = tf.io.parse_tensor(feats["input"], out_type=tf.int32)    # [B, T, C]
    y = tf.io.parse_tensor(feats["output"], out_type=tf.int32)   # [B, T, K]
    t = tf.io.parse_tensor(tx_ids_serial, out_type=tf.string)    # [B, N, 3] or [B, 0, 3]

    out: Dict[str, tf.Tensor] = {"x": x, "y": y, "t": t}
    if cfg.clamsa:
        out["clamsa"] = tf.io.parse_tensor(feats["clamsa"], out_type=tf.float64)  # or float32
    return out


# ─────────────────── label reformat (vector) ───────────────────
def _reformat_labels(y: tf.Tensor, output_size: int) -> tf.Tensor:
    # y: [B, T, C]
    C = tf.shape(y)[-1]
    def eq(v): return tf.equal(C, v)

    def id_(): return tf.cast(y, tf.float32)

    def from7():
        def to5():
            return tf.concat([y[:, :, :1],
                              tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                              y[:, :, 4:]], axis=-1)
        def to3():
            return tf.concat([y[:, :, :1],
                              tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                              tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True)], axis=-1)
        return tf.switch_case(
            tf.cast(output_size, tf.int32),
            branch_fns={
                3: to3,
                5: to5,
                7: id_,
            },
            default=id_
        )

    def from15():
        idx_1 = [4, 7, 10, 12]
        idx_2 = [5, 8, 13]
        idx_3 = [6, 9, 11, 14]
        g = lambda idx: tf.reduce_sum(tf.gather(y, idx, axis=-1), axis=-1, keepdims=True)
        def to3():
            return tf.concat([y[:, :, :1],
                              tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                              tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True)], axis=-1)
        def to5():
            return tf.concat([y[:, :, :1],
                              tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                              g(idx_1), g(idx_2), g(idx_3)], axis=-1)
        def to7():
            return tf.concat([y[:, :, :4], g(idx_1), g(idx_2), g(idx_3)], axis=-1)
        def to2():
            return tf.concat([tf.reduce_sum(y[:, :, :4], axis=-1, keepdims=True),
                              tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True)], axis=-1)
        def to4():
            return tf.concat([tf.reduce_sum(y[:, :, :4], axis=-1, keepdims=True),
                              g(idx_1), g(idx_2), g(idx_3)], axis=-1)
        return tf.switch_case(
            tf.cast(output_size, tf.int32),
            branch_fns={2: to2, 3: to3, 4: to4, 5: to5, 7: to7, 15: id_},
            default=id_
        )

    return tf.cast(tf.case([(eq(7), from7), (eq(15), from15)], default=id_), tf.float32)


# ─────────────── tx mask (cond, not where!) ───────────────
def _seq_mask(seq_len: tf.Tensor, transcripts: tf.Tensor,
              tx_filter: tf.Tensor, r_f: int, r_u: int) -> tf.Tensor:
    # transcripts: [N,3] strings (tx_id, start, end) — may be N=0
    has_tx = tf.greater(tf.shape(transcripts)[0], 0)

    def _compute():
        tx_ids = transcripts[:, 0]
        starts = tf.strings.to_number(transcripts[:, 1], out_type=tf.int32)
        ends   = tf.strings.to_number(transcripts[:, 2], out_type=tf.int32)

        is_f = tf.reduce_any(
            tf.equal(tx_ids[:, None], tx_filter[None, :]),
            axis=1
        )
        st_f = tf.boolean_mask(starts, is_f); en_f = tf.boolean_mask(ends, is_f)
        st_u = tf.boolean_mask(starts, ~is_f); en_u = tf.boolean_mask(ends, ~is_f)

        st_f_exp = tf.clip_by_value(st_f - r_f, 0, seq_len)
        en_f_exp = tf.clip_by_value(en_f + r_f, 0, seq_len)
        st_u_exp = tf.clip_by_value(st_u - r_u, 0, seq_len)
        en_u_exp = tf.clip_by_value(en_u + r_u, 0, seq_len)

        pos = tf.range(seq_len, dtype=tf.int32)[:, None]
        hit_f = tf.reduce_any((pos >= st_f_exp[None, :]) & (pos < en_f_exp[None, :]), axis=1)
        hit_u = tf.reduce_any((pos >= st_u_exp[None, :]) & (pos < en_u_exp[None, :]), axis=1)
        hit_core = tf.reduce_any((pos >= st_f[None, :]) & (pos < en_f[None, :]), axis=1)

        mask_dual = tf.where(hit_u, tf.ones_like(hit_u, tf.float32),
                             tf.where(hit_f, tf.zeros_like(hit_f, tf.float32),
                                      tf.ones_like(hit_f, tf.float32)))
        return tf.where(hit_core, tf.zeros_like(hit_core, tf.float32), mask_dual)

    return tf.cond(has_tx, _compute, lambda: tf.ones([seq_len], tf.float32))


# ───────────────────── seq-weighting (optional) ─────────────────────
def _border_weights(y: tf.Tensor, r: int, w: float, output_size: int) -> tf.Tensor:
    # y: [T, K] one-hot
    simp = tf.argmax(y, axis=-1)  # [T]
    if output_size == 5:
        simp = tf.where(simp < 2, 0, 1)
    elif output_size == 7:
        simp = tf.where(simp < 4, 0, 1)
    weights = tf.ones_like(simp, tf.float32)
    diffs = tf.concat([tf.zeros([1], simp.dtype), tf.experimental.numpy.diff(simp)], axis=0)
    idx = tf.where(diffs != 0)[:, 0]  # [M]

    def body(i, wv):
        t = idx[i]
        start = tf.maximum(0, t - r)
        end = tf.minimum(tf.shape(wv)[0], t + r)
        upd_idx = tf.range(start, end, dtype=tf.int32)
        return i + 1, tf.tensor_scatter_nd_update(wv, upd_idx[:, None], tf.fill([tf.shape(upd_idx)[0]], tf.cast(w, tf.float32)))

    _, weights = tf.while_loop(lambda i, _: i < tf.shape(idx)[0], body, [0, weights])
    return weights  # [T]


# ───────────────────── preprocess (batched) ─────────────────────
def _preprocess_batched(ex: Dict[str, tf.Tensor], cfg: DataLoaderConfig):
    x = tf.reshape(ex["x"], [tf.shape(ex["x"])[0], -1, tf.shape(ex["x"])[-1]])       # [B, T, C]
    y = tf.reshape(ex["y"], [tf.shape(ex["y"])[0], -1, tf.shape(ex["y"])[-1]])       # [B, T, K]
    t = ex["t"]                                                                       # [B, N, 3] strings

    if not cfg.softmasking:
        x = x[:, :, :5]

    y = _reformat_labels(y, cfg.output_size)  # [B, T, O]
    B = tf.shape(y)[0]
    T = tf.shape(y)[1]

    # Build inputs/labels depending on mode
    if cfg.hmm_factor:
        step = tf.maximum(1, T // cfg.hmm_factor)
        start = y[:, ::step, :]
        end   = y[:, step - 1::step, :]
        hints = tf.stack([start, end], axis=-2)   # [B, N, 2, O]
        X = (x, hints)
        Y = (y, y)
    elif cfg.clamsa:
        X = (x, tf.cast(ex["clamsa"], tf.float32))
        Y = y
    elif cfg.oracle:
        X = (x, y)
        Y = y
    else:
        X = x
        Y = y

    # Weights (per example independently)
    def per_example_weights(i):
        tx_i = t[i]  # [Ni,3]
        w = _seq_mask(seq_len=T, transcripts=tx_i,
                      tx_filter=tf.constant(cfg.tx_filter, tf.string),
                      r_f=cfg.tx_filter_region // 2, r_u=cfg.tx_filter_region)
        if cfg.seq_weights_window > 0:
            # combine: take max of structural weights and border emphasis
            bw = _border_weights(y[i], cfg.seq_weights_window, cfg.seq_weights_value, cfg.output_size)
            w = tf.maximum(w, bw)
        return w

    w = tf.map_fn(lambda i: per_example_weights(i),
                  elems=tf.range(B), fn_output_signature=tf.float32)  # [B, T]
    w = w[..., None]  # [B, T, 1]

    return X, Y, w


# ───────────────────── dataset builder ─────────────────────
def build_dataset(cfg: DataLoaderConfig) -> tf.data.Dataset:
    paths = cfg.files if isinstance(cfg.files, (list, tuple)) else [cfg.files]
    files_ds = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths)) if len(paths) > 1 else tf.data.Dataset.from_tensor_slices(paths)

    ds = files_ds.interleave(
        lambda p: tf.data.TFRecordDataset(p, compression_type=cfg.compression),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=cfg.num_parallel_calls,
        deterministic=not cfg.shuffle,
    )

    if cfg.shuffle:
        ds = ds.shuffle(cfg.shuffle_buffer)

    if cfg.repeat:
        ds = ds.repeat()

    # Batch BEFORE parse to use parse_example
    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    ds = ds.map(lambda batch: _parse_batch(batch, cfg), num_parallel_calls=cfg.num_parallel_calls)

    ds = ds.map(lambda ex: _preprocess_batched(ex, cfg), num_parallel_calls=cfg.num_parallel_calls)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
