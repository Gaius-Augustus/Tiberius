import tensorflow as tf
from bricks2marble.tf import AnnotationHMM
from hidten.config import ModelConfig, with_config
from hidten.hmm import HMMMode

from .util import LSTMInferenceLoss, extract_nucleotides


class TiberiusConfig(ModelConfig):

    units: int = 372
    filter_size: int = 128
    kernel_size: int = 9
    numb_conv: int = 3
    numb_lstm: int = 2
    pool_size: int = 9
    output_size: int = 15

    multi_loss: bool = False
    residual_conv: bool = True

    with_hmm: bool = False
    hmm_heads: int = 1
    hmm_reverse_strand: bool = False
    parallel_factor: int = 1

    initial_exon_len: int = 100
    initial_intron_len: int = 10000
    initial_ir_len: int = 10000
    intron_state_chain: int = 1
    train_transitions: bool = False
    train_start_dist: bool = False
    share_noncoding_params: bool = False

    model_config = {"frozen": True}


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
                name="initial_conv" if i == 0  else f"conv_{i+1}",
            )
            for i in range(self.config.numb_conv)
        ]
        self.norms = [
            tf.keras.layers.LayerNormalization(
                name=f"layer_normalization{i+1}",
            )
            for i in range(self.config.numb_conv)
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
            for i in range(self.config.numb_lstm)
        ]

        if self.config.residual_conv:
            self.out_dense_1 = tf.keras.layers.Dense(
                self.config.pool_size*30,
                activation="relu",
            )
            self.out_dense_2 = tf.keras.layers.Dense(
                self.config.output_size,
            )
        else:
            self.out_dense = tf.keras.layers.Dense(
                self.config.pool_size*self.config.output_size,
            )

        if self.config.multi_loss:
            self.lstm_loss = [
                LSTMInferenceLoss(
                    pool_size=self.config.pool_size,
                    output_size=self.config.output_size,
                )
                for _ in range(self.config.numb_lstm - 1)
            ]

        if self.config.with_hmm:
            self.hmm = AnnotationHMM(
                heads=self.config.hmm_heads,
                use_reverse_strand=self.config.hmm_reverse_strand,
                parallel_factor=self.config.parallel_factor,
                initial_exon_len=self.config.initial_exon_len,
                initial_intron_len=self.config.initial_intron_len,
                initial_ir_len=self.config.initial_ir_len,
                intron_state_chain=self.config.intron_state_chain,
                train_transitions=self.config.train_transitions,
                train_start_dist=self.config.train_start_dist,
                share_noncoding_params=self.config.share_noncoding_params,
            )
            self.hmm_dense = tf.keras.layers.Dense(15, activation="softmax")

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        self.d_in = input_shape[-1]

        self.conv[0].build(input_shape)
        for i in range(self.config.numb_conv-1):
            self.norms[i].build(input_shape[:-1] + (self.config.filter_size, ))
            self.conv[i+1].build(
                input_shape[:-1] + (self.config.filter_size, )
            )

        self.d_out_cnn = self.config.pool_size * (
            self.config.filter_size + self.d_in
        )
        self.dense_to_lstm.build(input_shape[:-1] + (self.d_out_cnn, ))

        for i in range(self.config.numb_lstm):
            self.lstm[i].build(input_shape[:-1] + (2*self.config.units, ))

        if self.config.residual_conv:
            self.out_dense_1.build(input_shape[:-1] + (2*self.config.units, ))
            self.out_dense_2.build(
                input_shape[:-1] + (30+self.config.filter_size, )
            )
        else:
            self.out_dense.build(input_shape[:-1] + (2*self.config.units, ))

        if self.config.with_hmm:
            self.hmm.build(input_shape[:-1] + (15, ))
            self.hmm.hmm.mode = HMMMode.POSTERIOR
            self.hmm.hmm.parallel_factor = self.config.parallel_factor
            hmm_out = self.hmm.compute_output_shape(input_shape[:-1]+(15, ))
            self.hmm_dense.build(input_shape[:-1]+(hmm_out[-2]*hmm_out[-1], ))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B, T, _ = tf.unstack(tf.shape(x))

        if self.config.with_hmm:
            nuc = extract_nucleotides(x)
        y = self.conv[0](x)
        for i in range(self.config.numb_conv-1):
            y = self.norms[i](y)
            y = self.conv[i+1](y)
        x = tf.concat([x, y], axis=-1)
        x = tf.reshape(x, (B, -1, self.d_out_cnn))

        x = self.dense_to_lstm(x)
        for i in range(self.config.numb_lstm):
            x = self.lstm[i](x)
            if self.config.multi_loss and i < self.config.numb_lstm-1:
                self.lstm_loss[i](x)

        if self.config.residual_conv:
            x = self.out_dense_1(x)
            x = tf.reshape(x, (B, -1, 30))
            x = tf.concat([x, y], axis=-1)
            x = self.out_dense_2(x)
        else:
            x = self.out_dense(x)
            x = tf.reshape(x, (B, -1, self.config.output_size))

        x = tf.nn.softmax(x)
        if self.config.with_hmm:
            x = self.hmm(x, nuc)
            x = self.hmm_dense(tf.reshape(x, (B, T, -1)))
        return x
