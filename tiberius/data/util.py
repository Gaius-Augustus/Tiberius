from pathlib import Path
from typing import TYPE_CHECKING

import tensorflow as tf

if TYPE_CHECKING:
    from .dataset import DatasetConfig


def expand_path(path: Path) -> list[Path]:
    """Expands path components that contain an asterisk to any possible
    existing paths.
    """
    total_paths = [Path()]
    for component in path.expanduser().parts:
        new_paths = []
        for i in range(len(total_paths)):
            if "*" in component:
                new_paths.extend([p for p in total_paths[i].glob(component)])
            else:
                new_paths.append(total_paths[i] / component)
        total_paths = new_paths
    return total_paths


def _feature_spec(clamsa: bool) -> dict[str, tf.io.FixedLenFeature]:
    spec = {
        "input":  tf.io.FixedLenFeature([], tf.string),
        "output": tf.io.FixedLenFeature([], tf.string),
    }
    if clamsa:
        spec["clamsa"] = tf.io.FixedLenFeature([], tf.string)
    return spec


def _parse_batch(
    examples: tf.Tensor,
    cfg: "DatasetConfig",
) -> dict[str, tf.Tensor]:
    feats = tf.io.parse_example(examples, _feature_spec(cfg.clamsa))

    # [B, T, C]
    x = tf.io.parse_tensor(feats["input"], out_type=tf.float32)
    # [B, T, K]
    y = tf.io.parse_tensor(feats["output"], out_type=tf.float32)

    out: dict[str, tf.Tensor] = {"x": x, "y": y}
    if cfg.clamsa:
        out["clamsa"] = tf.io.parse_tensor(
            feats["clamsa"],
            out_type=tf.float64,
        )  # or float32
    return out


def _reformat_labels(y: tf.Tensor, output_size: int) -> tf.Tensor:
    # y: [B, T, C]
    C = tf.shape(y)[-1]
    def eq(v): return tf.equal(C, v)

    def id_(): return tf.cast(y, tf.float32)

    def from7():
        def to5():
            return tf.cast(tf.concat([
                y[:, :, :1],
                tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                y[:, :, 4:],
            ], axis=-1), dtype=tf.float32)
        def to3():
            return tf.cast(tf.concat([
                y[:, :, :1],
                tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True),
            ], axis=-1), dtype=tf.float32)
        return tf.case([
            (tf.equal(output_size, 3), to3),
            (tf.equal(output_size, 5), to5),
            (tf.equal(output_size, 7), id_),
        ], default=id_)

    def from15():
        idx_1 = [4, 7, 10, 12]
        idx_2 = [5, 8, 13]
        idx_3 = [6, 9, 11, 14]
        g = lambda idx: tf.reduce_sum(
            tf.gather(y, idx, axis=-1),
            axis=-1,
            keepdims=True,
        )
        def to3():
            return tf.cast(tf.concat([
                y[:, :, :1],
                tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True),
            ], axis=-1), dtype=tf.float32)
        def to5():
            return tf.cast(tf.concat([
                y[:, :, :1],
                tf.reduce_sum(y[:, :, 1:4], axis=-1, keepdims=True),
                g(idx_1), g(idx_2), g(idx_3),
            ], axis=-1), dtype=tf.float32)
        def to7():
            return tf.cast(tf.concat([
                y[:, :, :4],
                g(idx_1), g(idx_2), g(idx_3)
            ], axis=-1), dtype=tf.float32)
        def to2():
            return tf.cast(tf.concat([
                tf.reduce_sum(y[:, :, :4], axis=-1, keepdims=True),
                tf.reduce_sum(y[:, :, 4:], axis=-1, keepdims=True),
            ], axis=-1), dtype=tf.float32)
        def to4():
            return tf.cast(tf.concat([
                tf.reduce_sum(y[:, :, :4], axis=-1, keepdims=True),
                g(idx_1), g(idx_2), g(idx_3),
            ], axis=-1), dtype=tf.float32)
        return tf.case([
            (tf.equal(output_size, 2), to2),
            (tf.equal(output_size, 3), to3),
            (tf.equal(output_size, 4), to4),
            (tf.equal(output_size, 5), to5),
            (tf.equal(output_size, 7), to7),
            (tf.equal(output_size, 15), id_),
        ], default=id_)

    return tf.cast(
        tf.case([(eq(7), from7), (eq(15), from15)], default=id_),
        tf.float32,
    )


def _seq_mask(
    seq_len: tf.Tensor,
    transcripts: tf.Tensor,
    tx_filter: tf.Tensor,
    r_f: int,
    r_u: int,
) -> tf.Tensor:
    # transcripts: [N,3] strings (tx_id, start, end) — may be N=0
    has_tx = tf.greater(tf.shape(transcripts)[0], 0)

    def _compute():
        tx_ids = transcripts[:, 0]
        starts = tf.strings.to_number(transcripts[:, 1], out_type=tf.int32)
        ends   = tf.strings.to_number(transcripts[:, 2], out_type=tf.int32)

        is_f = tf.reduce_any(
            tf.equal(tx_ids[:, None], tx_filter[None, :]),
            axis=1,
        )
        st_f = tf.boolean_mask(starts, is_f)
        en_f = tf.boolean_mask(ends, is_f)
        st_u = tf.boolean_mask(starts, ~is_f)
        en_u = tf.boolean_mask(ends, ~is_f)

        st_f_exp = tf.clip_by_value(st_f - r_f, 0, seq_len)
        en_f_exp = tf.clip_by_value(en_f + r_f, 0, seq_len)
        st_u_exp = tf.clip_by_value(st_u - r_u, 0, seq_len)
        en_u_exp = tf.clip_by_value(en_u + r_u, 0, seq_len)

        pos = tf.range(seq_len, dtype=tf.int32)[:, None]
        hit_f = tf.reduce_any(
            (pos >= st_f_exp[None, :]) & (pos < en_f_exp[None, :]),
            axis=1,
        )
        hit_u = tf.reduce_any(
            (pos >= st_u_exp[None, :]) & (pos < en_u_exp[None, :]),
            axis=1,
        )
        hit_core = tf.reduce_any(
            (pos >= st_f[None, :]) & (pos < en_f[None, :]),
            axis=1,
        )

        mask_dual = tf.where(
            hit_u,
            tf.ones_like(hit_u, tf.float32),
            tf.where(
                hit_f,
                tf.zeros_like(hit_f, tf.float32),
                tf.ones_like(hit_f, tf.float32),
            ),
        )
        return tf.where(
            hit_core,
            tf.zeros_like(hit_core, tf.float32),
            mask_dual,
        )

    return tf.cond(has_tx, _compute, lambda: tf.ones([seq_len], tf.float32))


def _border_weights(
    y: tf.Tensor,
    r: int,
    w: float,
    output_size: int,
) -> tf.Tensor:
    simp = tf.argmax(y, axis=-1)  # [T]
    if output_size == 5:
        simp = tf.where(simp < 2, 0, 1)
    elif output_size == 7:
        simp = tf.where(simp < 4, 0, 1)
    weights = tf.ones_like(simp, tf.float32)
    diffs = tf.concat([
        tf.zeros([1], simp.dtype),
        tf.experimental.numpy.diff(simp),
    ], axis=0)
    idx = tf.cast(tf.where(diffs != 0)[:, 0], tf.int32)  # [M]

    def body(i, wv):
        t = idx[i]
        start = tf.maximum(0, t - r)
        end = tf.minimum(tf.shape(wv)[0], t + r)
        upd_idx = tf.range(start, end, dtype=tf.int32)
        return (
            i + 1,
            tf.tensor_scatter_nd_update(
                wv,
                upd_idx[:, None],
                tf.fill([tf.shape(upd_idx)[0]], tf.cast(w, tf.float32)),
            )
        )

    _, weights = tf.while_loop(
        lambda i, _: i < tf.shape(idx)[0],
        body,
        [0, weights],
    )
    return weights  # [T]


def _preprocess_batched(ex: dict[str, tf.Tensor], cfg: "DatasetConfig"):
    # [B, T, C]
    x = tf.reshape(ex["x"], [tf.shape(ex["x"])[0], tf.shape(ex["x"])[-1]])
    # [B, T, K]
    y = tf.reshape(ex["y"], [tf.shape(ex["y"])[0], tf.shape(ex["y"])[-1]])

    if not cfg.softmasking:
        x = x[..., :5]

    # TODO: currently not working
    # y = _reformat_labels(y, cfg.output_size)  # [B, T, O]
    y = tf.cast(y, tf.float32)
    y.set_shape((cfg.T, cfg.output_size))
    x = tf.cast(x, tf.float32)
    x.set_shape((cfg.T, cfg.input_size))

    if cfg.clamsa:
        X = (x, tf.cast(ex["clamsa"], tf.float32))
        Y = y
    elif cfg.oracle:
        X = (x, y)
        Y = y
    else:
        X = x
        Y = y

    # TODO: sample_weights
    # # Weights (per example independently)
    # def per_example_weights(i):
    #     tx_i = t[i]  # [Ni,3]
    #     tx_i.set_shape([None, 3])
    #     w = _seq_mask(seq_len=T, transcripts=tx_i,
    #                   tx_filter=tf.constant(cfg.tx_filter, tf.string),
    #                   r_f=cfg.tx_filter_region // 2, r_u=cfg.tx_filter_region)
    #     if cfg.seq_weights_window > 0:
    #         # combine: take max of structural weights and border emphasis
    #         bw = _border_weights(
    #             y[i],
    #             cfg.seq_weights_window,
    #             cfg.seq_weights_value,
    #             cfg.output_size,
    #         )
    #         w = tf.maximum(w, bw)
    #     return w

    # w = tf.map_fn(
    #     lambda i: per_example_weights(i),
    #     elems=tf.range(B),
    #     fn_output_signature=tf.float32,
    # )  # [B, T]
    # w = w[..., None]  # [B, T, 1]

    return X, Y, None#w
