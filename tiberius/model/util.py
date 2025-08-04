import tensorflow as tf
from hidten.config import ModelConfig, with_config


def extract_nucleotides(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(
        x[0][..., :5] if isinstance(x, list) else x[..., :5],
        tf.float32,
    )


class LSTMInferenceLossConfig(ModelConfig):

    pool_size: int
    output_size: int


@with_config(LSTMInferenceLossConfig)
class LSTMInferenceLoss(tf.keras.Layer):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = LSTMInferenceLossConfig(**kwargs)
        self.dense = tf.keras.layers.Dense(
            self.config.pool_size*self.config.output_size,
        )

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.dense.build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B = tf.shape(x)[0]
        y = self.dense(x)
        y = tf.reshape(y, (B, -1, self.config.output_size))
        self.add_loss(tf.nn.softmax(y, axis=-1))
        return x
