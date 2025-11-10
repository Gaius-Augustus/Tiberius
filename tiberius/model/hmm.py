from typing import Literal

import tensorflow as tf
from bricks2marble.tf import AnnotationHMM
from hidten.config import ModelConfig, with_config
from hidten.hmm import HMMMode


class HMMBlockConfig(ModelConfig):

    heads: int
    parallel_factor: int
    use_reverse_strand: bool = False
    compute_heads_sequentially: bool = False

    embed: int | None = None
    in_activation: Literal["softmax", "sigmoid"] = "softmax"
    emitter_sigmoid_activation: bool = False
    share_noncoding_params: bool = False

    intron_state_chain: int = 1
    intron_chain_skips: bool = False
    intron_chain_loop: bool = False
    initial_exon_len: int | float | None = None
    initial_intron_len: int | float | list[float | int] | None = None
    initial_ir_len: int | float | None = None
    train_transitioner: bool = True
    uniform_N: bool = False

    dropout_heads: float = 0.0
    nudge_IR: float = 0.0
    nudge_repeats_noncoding: float = 0.0
    intron_regularization: float = 0.0


@with_config(HMMBlockConfig)
class HMMBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = HMMBlockConfig(**kwargs)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        if self.config.embed is not None:
            self.input_proj = tf.keras.layers.Dense(
                self.config.embed,
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    mean=0.0,
                    stddev=0.01,
                ),  # type: ignore
                use_bias=False,
                activation=self.config.in_activation,
            )
            self.input_proj.build(input_shape)

        self.hmmlayer = AnnotationHMM(
            heads=self.config.heads,
            use_reverse_strand=self.config.use_reverse_strand,
            compute_heads_sequentially=self.config.compute_heads_sequentially,
            start_codons=[("ATG", 1.)],
            stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
            intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.01)],
            intron_end_pattern=[("AGN", 1.)],
            initial_exon_len=self.config.initial_exon_len,
            initial_intron_len=self.config.initial_intron_len,
            initial_ir_len=self.config.initial_ir_len,
            train_transitioner=self.config.train_transitioner,
            share_noncoding_params=self.config.share_noncoding_params,
            parallel_factor=self.config.parallel_factor,
            dropout_heads=self.config.dropout_heads,
            intron_state_chain=self.config.intron_state_chain,
            intron_chain_skips=self.config.intron_chain_skips,
            intron_chain_loop=self.config.intron_chain_loop,
            emitter_sigmoid_activation=self.config.emitter_sigmoid_activation,
            uniform_N=self.config.uniform_N,
            nudge_IR=self.config.nudge_IR,
            nudge_repeats_noncoding=self.config.nudge_repeats_noncoding,
            intron_regularization=self.config.intron_regularization,
        )  # type: ignore
        self.hmmlayer.build(input_shape[:-1] + (
            self.config.embed if self.config.embed is not None
            else input_shape[-1],
        ))

    def call(
        self,
        x: tf.Tensor,
        nuc: tf.Tensor,
        mode: HMMMode = HMMMode.POSTERIOR,
        training: bool = False,
    ) -> tf.Tensor:
        if self.config.embed is not None:
            x = self.input_proj(x)  # type: ignore

        posterior = self.hmmlayer(
            x, nuc,
            mode=mode,
            parallel=self.config.parallel_factor,
            training=training,
        )  # type: ignore
        if mode != HMMMode.POSTERIOR:
            return posterior  # type: ignore

        x = tf.reshape(posterior, tf.concat((
            tf.shape(posterior)[:2],  # type: ignore
            (tf.shape(posterior)[2] * tf.shape(posterior)[3], )  # type: ignore
        ), 0))
        return x

    def compute_output_shape(
        self,
        input_shape: tuple[int | None, ...],
    ) -> tuple[int | None, ...]:
        layer_out = self.hmmlayer.compute_output_shape(
            input_shape[:-1] + (self.config.embed, )
        )
        return input_shape[:-1] + (
            layer_out[-2] * layer_out[-1],  # type: ignore
        )
