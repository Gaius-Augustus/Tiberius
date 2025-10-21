from typing import Literal

import tensorflow as tf
from hidten.config import ModelConfig, with_config
from hidten.hmm import HMMMode
from vipsania.model.ffn import GLUFFN, MLP, FFNConfig
from vipsania.model.lru import BidirectionalLRU, BidirectionalLRUConfig

from .hmm import HMMBlock, HMMBlockConfig
from .util import extract_nucleotides


class ResidualTiberiusConfig(ModelConfig):

    d_hidden: int
    n_layers: int
    output_size: int = 15
    layernorm: bool = True
    residual: bool = True

    lru: BidirectionalLRUConfig | None = None
    ffn_type: Literal["mlp", "glu"] = "glu"
    ffn: FFNConfig

    multi_loss: bool = False
    residual_conv: bool = True

    hmm: HMMBlockConfig | None = None

    model_config = {"frozen": True}


@with_config(ResidualTiberiusConfig)
class ResidualTiberius(tf.keras.Model):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = ResidualTiberiusConfig(**kwargs)

        self.embedding = tf.keras.layers.Dense(
            self.config.d_hidden,
            activation="softmax",
        )

        self.layernorms = []
        if self.config.layernorm:
            for _ in range(self.config.n_layers+1):
                self.layernorms.append(tf.keras.layers.LayerNormalization())

        self.lrus = []
        self.lru_norms = []
        self.mlps = []
        for _ in range(self.config.n_layers):
            if self.config.lru is not None:
                self.lru_norms.append(
                    tf.keras.layers.LayerNormalization()
                )
                self.lrus.append(
                    BidirectionalLRU(**self.config.lru.model_dump() | {
                        "funnel_directions": self.config.d_hidden,
                    })
                )
            match self.config.ffn_type:
                case "mlp":
                    self.mlps.append(MLP(**self.config.ffn.model_dump() | {
                        "d_out": self.config.d_hidden,
                    }))
                case "glu":
                    self.mlps.append(GLUFFN(**self.config.ffn.model_dump() | {
                        "d_out": self.config.d_hidden,
                    }))

        if self.config.hmm is not None:
            self.hmm = HMMBlock(**self.config.hmm.model_dump())

        self.unembedding = tf.keras.layers.Dense(self.config.output_size)
        self._hmm_inference: None | HMMMode = None

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        BT = input_shape[:-1]
        d_in = input_shape[-1]

        self.embedding.build(BT + (d_in, ))

        if self.config.layernorm:
            for l in range(self.config.n_layers+1):
                self.layernorms[l].build(BT + (self.config.d_hidden, ))

        for l in range(self.config.n_layers):
            if self.config.lru is not None:
                self.lru_norms[l].build(BT + (self.config.d_hidden, ))
                self.lrus[l].build(BT + (self.config.d_hidden, ))
            self.mlps[l].build(BT + (self.config.d_hidden, ))

        if self.config.hmm is not None:
            self.hmm.build(BT + (self.config.d_hidden, ))
            hmm_out = self.hmm.compute_output_shape(
                BT + (self.config.d_hidden, )
            )[-1]
        else:
            hmm_out = self.config.d_hidden

        self.unembedding.build(BT + (hmm_out, ))

    def set_inference(
        self,
        inference: bool = False,
        mode: HMMMode = HMMMode.VITERBI,
    ) -> None:
        """Sets the model to inference mode. The output of the model
        will then be the Viterbi (or MEA) sequence of states of the
        HMM in the model.
        """
        self._hmm_inference = None if not inference else mode

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        if self.config.hmm is not None:
            nuc = extract_nucleotides(x)

        x = self.embedding(x)  # type: ignore

        for l in range(self.config.n_layers):
            if self.config.lru is not None:
                y = self.lru_norms[l](x)
                y = self.lrus[l](y, training=training)
                if self.config.residual: x = x + y
                else: x = y

            y = x if not self.config.layernorm else self.layernorms[l](x)
            y = self.mlps[l](y, training=training)
            if self.config.residual: x = x + y
            else: x = y

        x = x if not self.config.layernorm else self.layernorms[-1](x)
        if self.config.hmm is not None:
            x = self.hmm(
                x, nuc,
                mode=HMMMode.POSTERIOR,
            )  # type: ignore
            if self._hmm_inference is not None:
                return self.hmm(
                    x, nuc,
                    mode=self._hmm_inference,
                )  # type: ignore

        x = self.unembedding(x)  # type: ignore
        return x
