from typing import Any, Literal

import tensorflow as tf
from bricks2marble.tf import AnnotationHMM, AnnotationHMMConfig
from hidten.config import with_config
from hidten.hmm import HMMMode


class HMMBlockConfig(AnnotationHMMConfig):

    embed: int | None = None
    in_activation: Literal["softmax", "sigmoid"] = "softmax"

    def parent_config(self) -> dict[str, Any]:
        parent_fields = AnnotationHMMConfig.model_fields.keys()
        return {
            k: v for k, v in self.model_dump().items()
            if k in parent_fields
        }


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

        self.hmmlayer = AnnotationHMM(**self.config.parent_config())
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
