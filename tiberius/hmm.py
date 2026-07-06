import copy
import math

import tensorflow as tf
from bricks2marble.tf import AnnotationHMM
from hidten import HMMMode


def compute_parallel_factor(seq_len: int) -> int:
    """Return a divisor of `seq_len` closest to sqrt(seq_len).

    The HMM's parallel scan requires `seq_len % parallel_factor == 0`;
    this picks a value that both satisfies the constraint and minimises
    the size of the intermediate reshaped tensor
    (seq_len // parallel_factor).
    """
    sqrt_n = int(math.sqrt(seq_len))
    for i in range(0, seq_len - sqrt_n + 1):
        if seq_len % (sqrt_n - i) == 0:
            return sqrt_n - i
        if seq_len % (sqrt_n + i) == 0:
            return sqrt_n + i
    return sqrt_n


def sanitize_parallel_factor(
    requested: int, seq_len: int, name: str = "parallel_factor",
) -> int:
    """If `requested` doesn't divide `seq_len`, return a nearby divisor
    and print a warning. Otherwise return `requested` unchanged."""
    if seq_len is None or seq_len <= 0 or requested <= 0:
        return requested
    if seq_len % requested == 0:
        return requested
    adjusted = compute_parallel_factor(seq_len)
    print(
        f"[hmm] {name}={requested} does not divide seq_len={seq_len}; "
        f"using {adjusted} instead (closest divisor to sqrt(seq_len))."
    )
    return adjusted


class HMMBlock(AnnotationHMM):
    def __init__(
        self,
        mode: HMMMode,
        parallel: int,
        training: bool,
        emitter_epsilon: float = 0.01,
        initial_exon_len: int = 200,
        initial_intron_len: int = 4500,
        initial_ir_len: int = 10000,
        train_emitter: bool= False,
        transitioner_share_frames: bool = False,
        transitioner_share_noncoding: bool = False,
        train_transitions: bool = False,
        train_start_dist: bool = False

    ) -> None:
        self.mode = mode
        self.parallel = parallel
        self.training = training
        super().__init__(
            use_reverse_strand=False,
            emitter_eye=emitter_epsilon,
            train_emitter=train_emitter,
            initial_exon_len=initial_exon_len,
            initial_intron_len=initial_intron_len,
            initial_ir_len=initial_ir_len,
            transitioner_share_frames=transitioner_share_frames,
            transitioner_share_noncoding=transitioner_share_noncoding,
            train_transitions=train_transitions,
            train_start_dist=train_start_dist
        )

    def call(self, x, nuc):
        return super().call(
            x,
            nuc,
            mode=self.mode,
            parallel=self.parallel,
            training=self.training,
        )


# Default HMM config for TrainableHMMHead. Mirrors the vipsania config in
# vipsania/vipsania/model/hmm.py, but with use_reverse_strand=False since
# Tiberius predicts on a single strand only.
_DEFAULT_HMM_CFG: dict = {
    "start_codons": [["ATG", 1.0]],
    "stop_codons": [["TAG", 0.34], ["TAA", 0.33], ["TGA", 0.33]],
    "intron_begin_pattern": [["NGT", 0.99], ["NGC", 0.01]],
    "intron_end_pattern": [["AGN", 1.0]],
    "heads": 1,
    "dropout_heads": 0.0,
    "compute_heads_sequentially": False,
    "use_reverse_strand": False,
    "emitter_share_noncoding": True,
    "emitter_share_frames": True,
    "emitter_sigmoid_activation": False,
    "emitter_eye": None,
    "emitter_prior": None,
    "train_emitter": True,
    "repeats_emitter": None,
    "intron_state_chain": 2,
    "intron_chain_starts": True,
    "intron_chain_skips": True,
    "intron_chain_loop": False,
    "intron_chain_transitions": False,
    "initial_exon_len": 1000,
    "initial_intron_len": [100, 10000],
    "initial_ir_len": 10000,
    "transitions_model3": False,
    "train_transitions": True,
    "train_start_dist": True,
    "transitioner_share_noncoding": False,
    "transitioner_share_frames": True,
    "transition_scorer": None,
    "uniform_N": False,
    "nudge_IR": 0.0,
    "nudge_repeats_noncoding": 0.0,
    "intron_regularization": 0.0,
    "ir_intron_ratio_regularization": 0.0,
    "repeats_at_borders": None,
}


def default_trainable_hmm_config() -> dict:
    return copy.deepcopy(_DEFAULT_HMM_CFG)


@tf.keras.utils.register_keras_serializable(package="tiberius")
class TrainableHMMHead(tf.keras.layers.Layer):
    """Trainable-HMM head for Tiberius, adapted from vipsania's HMMBlock.

    Signal path (single-strand):
        x   -- (B, T, D_in) pre-emission stream from backbone
        nuc -- (B, T, 5)    one-hot A,C,G,T,N (no softmask column)

        1. Optional LayerNorm on x.
        2. Dense projection to `embed` dim with softmax activation:
           acts as the state-emission distribution over `embed` bins.
        3. AnnotationHMM.call(x_emb, nuc, mode=POSTERIOR) with all HMM
           parameters (emitter, transitions, start dist) trainable.
        4. Posterior (B, T, H, n_states) reshaped to (B, T, H*n_states).
        5. Conv1D (kernel=readout_conv_kernel) or Dense projection to
           `readout_units` (default = D_in), returning logit-like output
           consumed by categorical cross-entropy with from_logits=True.

    Serializable via Keras get_config/from_config so full-model save/load
    round-trips the head together with the backbone.
    """

    def __init__(
        self,
        hmm_config: dict | None = None,
        embed: int | None = 160,
        embed_norm: str | None = "layer",
        embed_bias: bool = False,
        embed_activation: str | None = "softmax",
        dropout: float = 0.0,
        readout_units: int | None = None,
        readout_type: str = "conv",
        readout_conv_kernel: int = 9,
        readout_norm: str | None = None,
        readout_bias: bool = False,
        pre_readout_activation: str | None = None,
        readout_activation: str | None = None,
        parallel_factor: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        cfg = default_trainable_hmm_config()
        if hmm_config is not None:
            cfg.update(hmm_config)
        cfg["use_reverse_strand"] = False
        self.hmm_config = cfg

        self.embed = embed
        self.embed_norm = embed_norm
        self.embed_bias = embed_bias
        self.embed_activation = embed_activation
        self.dropout_rate = dropout
        self.readout_units_arg = readout_units
        self.readout_type = readout_type
        self.readout_conv_kernel = readout_conv_kernel
        self.readout_norm_name = readout_norm
        self.readout_bias = readout_bias
        self.pre_readout_activation_name = pre_readout_activation
        self.readout_activation_name = readout_activation
        self.parallel_factor = parallel_factor

        self._n_states = (
            12 + 3 * self.hmm_config["intron_state_chain"]
        )
        self._heads = self.hmm_config["heads"]

    def _make_norm(self, name: str, kind: str | None):
        if kind is None:
            return None
        if kind == "layer":
            return tf.keras.layers.LayerNormalization(name=name)
        if kind == "batch":
            return tf.keras.layers.BatchNormalization(name=name)
        if kind == "RMS":
            return tf.keras.layers.LayerNormalization(rms_scaling=True, name=name)
        raise ValueError(f"Unknown norm kind: {kind}")

    def build(self, input_shape):
        # input_shape corresponds to x's shape (last dim = D_in)
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple, tf.TensorShape)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        d_in = int(x_shape[-1])

        self.embed_norm_layer = self._make_norm("embed_norm", self.embed_norm)

        if self.embed is not None:
            self.embedding_layer = tf.keras.layers.Dense(
                self.embed,
                use_bias=self.embed_bias,
                activation=self.embed_activation,
                name="embed",
            )
        else:
            self.embedding_layer = None

        self.hmmlayer = AnnotationHMM(**self.hmm_config)

        self.dropout_layer = (
            tf.keras.layers.Dropout(self.dropout_rate)
            if self.dropout_rate > 0 else None
        )

        readout_units = (
            self.readout_units_arg if self.readout_units_arg is not None
            else d_in
        )
        self._readout_units = readout_units

        self.readout_norm_layer = self._make_norm(
            "readout_norm", self.readout_norm_name,
        )
        self.pre_readout_activation = (
            tf.keras.activations.get(self.pre_readout_activation_name)
            if self.pre_readout_activation_name is not None else None
        )
        if self.readout_type == "dense":
            self.readout_layer = tf.keras.layers.Dense(
                readout_units,
                use_bias=self.readout_bias,
                activation=self.readout_activation_name,
                name="readout",
            )
        elif self.readout_type == "conv":
            self.readout_layer = tf.keras.layers.Conv1D(
                readout_units,
                self.readout_conv_kernel,
                padding="same",
                use_bias=self.readout_bias,
                activation=self.readout_activation_name,
                name="readout",
            )
        else:
            raise ValueError(f"Unknown readout_type: {self.readout_type}")

        # We do not call the sub-layers' build() explicitly; Keras will
        # build them lazily on the first forward pass. `super().build` at
        # the end so that the base Layer marks this layer as built.
        super().build(input_shape)

    def call(self, x, nuc, training: bool = False):
        h = x
        if self.embed_norm_layer is not None:
            h = self.embed_norm_layer(h, training=training)
        if self.embedding_layer is not None:
            h = self.embedding_layer(h)

        posterior = self.hmmlayer(
            h,
            nuc,
            mode=HMMMode.POSTERIOR,
            parallel=self.parallel_factor,
            training=training,
        )
        # posterior: (B, T, H, n_states)
        shp = tf.shape(posterior)
        y = tf.reshape(posterior, (shp[0], shp[1], shp[2] * shp[3]))

        if self.dropout_layer is not None:
            y = self.dropout_layer(y, training=training)
        if self.readout_norm_layer is not None:
            y = self.readout_norm_layer(y, training=training)
        if self.pre_readout_activation is not None:
            y = self.pre_readout_activation(y)
        y = self.readout_layer(y)
        return y

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 \
                and isinstance(input_shape[0], (list, tuple, tf.TensorShape)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        readout_units = (
            self.readout_units_arg if self.readout_units_arg is not None
            else int(x_shape[-1])
        )
        return tuple(x_shape[:-1]) + (readout_units,)

    def get_config(self):
        base = super().get_config()
        base.update({
            "hmm_config": self.hmm_config,
            "embed": self.embed,
            "embed_norm": self.embed_norm,
            "embed_bias": self.embed_bias,
            "embed_activation": self.embed_activation,
            "dropout": self.dropout_rate,
            "readout_units": self.readout_units_arg,
            "readout_type": self.readout_type,
            "readout_conv_kernel": self.readout_conv_kernel,
            "readout_norm": self.readout_norm_name,
            "readout_bias": self.readout_bias,
            "pre_readout_activation": self.pre_readout_activation_name,
            "readout_activation": self.readout_activation_name,
            "parallel_factor": self.parallel_factor,
        })
        return base
