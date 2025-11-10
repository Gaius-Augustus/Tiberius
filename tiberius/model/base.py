import tensorflow as tf
from hidten.config import ModelConfig, with_config
from hidten.hmm import HMMMode

from .hmm import HMMBlock, HMMBlockConfig
from .util import extract_nucleotides


class TiberiusConfig(ModelConfig):

    n_conv: int = 3
    filter_size: int = 128
    kernel_size: int = 9

    pool_size: int = 9
    n_lstm: int = 2
    units: int = 372

    output_size: int = 30

    hmm: HMMBlockConfig | None = None

    model_config = {"frozen": True, "extra": "forbid"}


@with_config(TiberiusConfig)
class Tiberius(tf.keras.Model):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = TiberiusConfig(**kwargs)

        self.conv = [
            tf.keras.layers.Conv1D(
                self.config.filter_size,
                3 if i == 0 else self.config.kernel_size,
                padding="same",
                activation="relu",
                name="initial_conv" if i == 0 else f"conv_{i+1}",
            )
            for i in range(self.config.n_conv)
        ]
        self.norms = [
            tf.keras.layers.LayerNormalization(
                name=f"layer_normalization{i+1}",
            )
            for i in range(self.config.n_conv-1)
        ]
        self.dense_to_lstm = tf.keras.layers.Dense(
            2*self.config.units,
            name="pre_lstm_dense",
        )
        self.lstm = [
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                self.config.units,
                return_sequences=True,
            ), name=f'biLSTM_{i+1}')
            for i in range(self.config.n_lstm)
        ]

        self.out_dense_1 = tf.keras.layers.Dense(
            self.config.pool_size*2*self.config.output_size,
            activation="relu",
        )
        self.out_dense_2 = tf.keras.layers.Dense(
            self.config.output_size,
        )

        if self.config.hmm is not None:
            self.hmm = HMMBlock(**self.config.hmm.model_dump())
            self.hmm_dense = tf.keras.layers.Dense(self.config.output_size)

        self._hmm_inference = None

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        d_in = input_shape[-1]

        self.conv[0].build(input_shape)
        for i in range(self.config.n_conv-1):
            self.norms[i].build(input_shape[:-1] + (self.config.filter_size, ))
            self.conv[i+1].build(
                input_shape[:-1] + (self.config.filter_size, )
            )

        self.d_out_cnn = self.config.pool_size * (
            self.config.filter_size + d_in
        )
        self.dense_to_lstm.build(input_shape[:-1] + (self.d_out_cnn, ))

        for i in range(self.config.n_lstm):
            self.lstm[i].build(input_shape[:-1] + (2*self.config.units, ))

        self.out_dense_1.build(input_shape[:-1] + (2*self.config.units, ))
        self.out_dense_2.build(input_shape[:-1] + (
            2*self.config.output_size + self.config.filter_size,
        ))

        if self.config.hmm is not None:
            self.hmm.build(input_shape[:-1] + (self.config.output_size, ))
            hmm_out = self.hmm.compute_output_shape(input_shape[:-1] + (
                self.config.output_size,
            ))
            self.hmm_dense.build(input_shape[:-1] + (hmm_out[-1], ))

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

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B, T, _ = tf.unstack(tf.shape(x))

        if self.config.hmm is not None:
            nuc = extract_nucleotides(x)
        y = self.conv[0](x)
        for i in range(self.config.n_conv-1):
            y = self.norms[i](y)
            y = self.conv[i+1](y)
        x = tf.concat([x, y], axis=-1)
        x = tf.reshape(x, (B, -1, self.d_out_cnn))

        x = self.dense_to_lstm(x)
        for i in range(self.config.n_lstm):
            x = self.lstm[i](x)

        x = self.out_dense_1(x)
        x = tf.reshape(x, (B, T, 2*self.config.output_size))
        x = tf.concat([x, y], axis=-1)
        x = self.out_dense_2(x)

        if self.config.hmm is not None:
            x = tf.nn.softmax(x, axis=-1)
            if self._hmm_inference is not None:
                return self.hmm(
                    x, nuc,
                    mode=self._hmm_inference,
                )  # type: ignore
            x = self.hmm(
                x, nuc,
                mode=HMMMode.POSTERIOR,
            )  # type: ignore
            x = self.hmm_dense(x)
        return x
