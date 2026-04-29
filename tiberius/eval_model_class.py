# ==============================================================
# Authors: Lars Gabriel
#
# Class handling the prediction and evaluation for a single species
# ==============================================================

import sys, json, os, sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tiberius.models import (custom_cce_f1_loss, lstm_model, Cast)
from hidten import HMMMode
from tiberius.hmm import HMMBlock
from tiberius.hints import apply_hints
import bricks2marble as b2m
import math

def compute_parallel_factor(seq_len):
        sqrt_n = int(math.sqrt(seq_len))
        for i in range(0, seq_len - sqrt_n + 1):
            if seq_len % (sqrt_n-i) == 0:
                return sqrt_n-i
            if seq_len % (sqrt_n+i) == 0:
                return sqrt_n+i
        return sqrt_n

class PredictionGTF:
    """Class for generating GTF predictions based on a model's output.

        Attributes:
            model_path (str): Path to the pre-trained model.
            seq_len (int): Length of the sequences to process.
            batch_size (int): Batch size for prediction.
            hmm (bool): Flag to indicate whether to use HMM for prediction.
            model (keras.Model): Loaded Keras model for predictions.
    """
    def __init__(self, model_path='', model_path_old='', model_path_lstm_old='',
                 seq_len=500004, batch_size=200,
                 hmm=False, hmm_emitter_epsilon=0,
                 hmm_initial_exon_len=200,
                hmm_initial_intron_len=4500,
                hmm_initial_ir_len=10000,
                model_path_hmm='',
                 temp_dir='', num_hmm=1,
                 hmm_factor=None,
                 annot_path='', genome_path='', genome=None, softmask=False,
                 parallel_factor=1,
                 lstm_cfg='',
                 hints=None, hint_weight=1.0,):
        """
        Arguments:
            - model_path (str): Path to the main model file that includes a HMM layer.
            - model_path_old (str): Path to full model with HMM, old version.
            - model_path_lstm_old (str): Path to LSTM model without HMM, old version.
            - seq_len (int): The sequence length to be used for prediction.
            - batch_size (int): The size of the batches to be used.
            - hmm (bool): A flag to indicate whether Hidden Markov Models (HMM) should be used. Defaults to False.
            - hmm_emitter_epsilon (float): A small deviation from the identity matrix of the emitter of the HMM.
            - model_path_hmm (str): Path to the HMM model file.
            - temp_dir (str): Temporary directory path for intermediate files.
            - num_hmm (int): Number of HMMs to be used.
            - hmm_factor: Parallelization factor of HMM (deprecated, remove in a later version)
            - transformer (bool): A flag to indicate whether a transformer model should be used. (depprecated!)
            - trans_lstm (bool): A flag indicating whether a transform-LSTM hybrid model should be used. (depprecated!)
            - annot_path (str): Path to the reference annotation file (GTF).
            - genome_path (str): Path to the genome file (Fasta).
            - genome: dictionary of SeqRecords, (overriding) alternative to genome_path
            - softmask (bool): Whether to use softmasking.
            - parallel_factor (int): The parallel factor used for Viterbi.
            - lstm_cfg (str): path to lstm cfg to load weights instead of the whole model
            - hints (dict | None): Optional dict produced by ``tiberius.hints.load_hints``
              mapping sequence names to ``(feature, start, end, strand)`` entries.
              When provided alongside ``hint_weight != 1``, the LSTM class
              probabilities are biased at the hint positions before HMM decoding.
            - hint_weight (float): Multiplicative weight applied to the boosted
              classes at hint positions (followed by per-position renormalization).
              ``1.0`` disables weighting.
        """
        self.model_path = model_path
        self.model_path_old = model_path_old
        self.model_path_lstm_old = model_path_lstm_old
        self.seq_len = seq_len
        # self.adapted_seq_len = seq_len
        self.batch_size = batch_size
        self.adapted_batch_size = batch_size # can be increased if chunksize is reduced
        self.annot_path = annot_path
        self.genome_path = genome_path
        self.genome = genome
        self.softmask = softmask
        self.hmm = hmm
        self.hmm_emitter_epsilon = hmm_emitter_epsilon
        self.hmm_initial_exon_len=hmm_initial_exon_len
        self.hmm_initial_intron_len=hmm_initial_intron_len
        self.hmm_initial_ir_len=hmm_initial_ir_len
        self.model = None
        self.model_path_hmm = model_path_hmm
        self.fasta_seq_lens = {}
        self.num_hmm = num_hmm
        self.hmm_factor = hmm_factor
        self.lstm_cfg = lstm_cfg
        if temp_dir and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir
        self.lstm_pred = None
        self.parallel_factor = parallel_factor
        self.lstm_model = None
        self.inp_size = 15
        self.hints = hints if hints else None
        self.hint_weight = hint_weight


    def load_model(self, summary=True):
        """Loads the model from the given model path.

        Args:
            summary (bool, optional): If True, prints the model summary. Defaults to True.
        """
        self.custom_obj = {
            'custom_cce_f1_loss': custom_cce_f1_loss(2, self.adapted_batch_size),
            'loss_': custom_cce_f1_loss(2, self.adapted_batch_size),
            "Cast": Cast,
            "HMMBlock": HMMBlock(
                parallel=self.parallel_factor,
                mode=HMMMode.POSTERIOR,
                training=True,
                emitter_epsilon=self.hmm_emitter_epsilon,
                initial_exon_len=self.hmm_initial_exon_len,
                initial_intron_len=self.hmm_initial_intron_len,
                initial_ir_len=self.hmm_initial_ir_len,
            )
            }
        if self.model_path:
            try:
                if self.lstm_cfg:
                    with open(self.lstm_cfg, 'r') as f:
                        config = json.load(f)
                else:
                    with open(f"{self.model_path}/model_config.json", 'r') as f:
                        config = json.load(f)
            except Exception as e:
                print(f"Error could not find config of the model. It should be located at {self.model_path}/model_config.json: {e}", file=sys.stderr)
                sys.exit(1)
            relevant_keys = ['units', 'filter_size', 'kernel_size',
                'numb_conv', 'numb_lstm', 'dropout_rate',
                'pool_size', 'lstm_mask', 'clamsa',
                'output_size', 'residual_conv',
                'clamsa_kernel', 'lru_layer']
            relevant_args = {key: config[key] for key in relevant_keys if key in config}

            if "inp_size" in config:
                self.softmask = config["inp_size"]==6
            elif "softmasking" in config:
                self.softmask = config["softmasking"]

            self.lstm_model = lstm_model(**relevant_args, softmasking=self.softmask)

            weights_h5  = f"{self.model_path}/weights.h5"
            if not os.path.exists(weights_h5):
                weights_h5 = f"{self.model_path}/model.weights.h5"
            self.lstm_model.load_weights(weights_h5)

            if self.model_path_hmm:
                model_hmm = keras.models.load_model(
                        self.model_path_hmm,
                        custom_objects=self.custom_obj
                                )
                self.gene_pred_hmm_layer = model_hmm.get_layer('gene_pred_hmm_layer')
                self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
                self.gene_pred_hmm_layer.cell.recurrent_init()
            elif 'hmm' in config and config["hmm"]:
                try:
                    self.gene_pred_hmm_layer = self.lstm_model.get_layer('gene_pred_hmm_layer')
                except ValueError as e:
                    self.gene_pred_hmm_layer = self.lstm_model.layers[-1]
                try:
                    lstm_output=self.lstm_model.get_layer('out').output
                except ValueError as e:
                    lstm_output=self.lstm_model.get_layer('lstm_out').output
                self.lstm_model = Model(
                                inputs=self.lstm_model.input,
                                outputs=lstm_output
                                )
            else:
                self.make_default_hmm(inp_size=self.lstm_model.output_shape[-1])
        # loading full models for training or old models
        elif self.model_path_lstm_old:
            self.lstm_model = keras.models.load_model(self.model_path_lstm_old,
                    custom_objects=self.custom_obj,
                    compile=False,
                    )
            self.make_default_hmm(inp_size=self.lstm_model.output.shape[-1])
        elif self.model_path_old:

            self.model = keras.models.load_model(
                    self.model_path_old,
                    custom_objects=self.custom_obj,
                    compile=False,
                    )
            try:
                lstm_output=self.model.get_layer('out').output
            except ValueError as e:
                lstm_output=self.model.get_layer('lstm_out').output
            self.lstm_model = Model(
                            inputs=self.model.input,
                            outputs=lstm_output
                            )
            self.gene_pred_hmm_layer = self.model.get_layer('gene_pred_hmm_layer')
            if self.parallel_factor is not None:
                self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
            print(f"Running gene pred hmm layer with parallel factor {self.gene_pred_hmm_layer.parallel_factor}", file=sys.stderr)
            self.gene_pred_hmm_layer.cell.recurrent_init()
        self.inp_size = self.lstm_model.output_shape[-1]
        if summary:
            self.lstm_model.summary()


    def adapt_batch_size(self, adapted_chunksize):
        """Adapts the batch size based on the chunk size.
        """
        old_adapted_batch_size = self.adapted_batch_size
        self.adapted_batch_size = self.batch_size * self.seq_len // adapted_chunksize
        # round down to nearest power of 2
        self.adapted_batch_size = 2**int(np.log2(self.adapted_batch_size))
        if self.adapted_batch_size != old_adapted_batch_size:
            self.parallel_factor = compute_parallel_factor(adapted_chunksize)
            self.make_default_hmm(self.inp_size)


    def load_clamsa_data(self, clamsa_prefix, seq_names, strand='', chunk_len=None, pad=False):
        clamsa_chunks = []
        for seq_name in seq_names:
            if not os.path.exists(f'{clamsa_prefix}{seq_name}.npy'):
                print(f'CLAMSA PATH {clamsa_prefix}{seq_name}.npy does not exist!')
            clamsa_array = np.load(f'{clamsa_prefix}{seq_name}.npy')
            numb_chunks = clamsa_array.shape[0] // chunk_len
            clamsa_array_new = clamsa_array[:numb_chunks*chunk_len].reshape(numb_chunks, chunk_len, 4)
            clamsa_chunks.append(clamsa_array_new)
            last_chunksize = clamsa_array.shape[0]%chunk_len
            if pad and last_chunksize > 0:
                padding = np.zeros((1,chunk_len, 4),dtype=np.uint8)
                padding[0,0:last_chunksize] = clamsa_array[-last_chunksize:]
                clamsa_chunks.append(padding)

        clamsa_chunks = np.concatenate(clamsa_chunks, axis=0)
        if strand == '-':
            clamsa_chunks = clamsa_chunks[::-1,::-1, [1,0,3,2]]
        return clamsa_chunks

    def predict_function(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:

        # fwd prediction
        x_one_hot_fwd = fasta.one_hot(
                pad_index = 4,
                repeats = "track" if self.softmask else "omit",
                N = "track",
                dtype = np.float32,
            )
        lstm_out_fwd = self.lstm_prediction(x_one_hot_fwd)
        if self.hints and self.hint_weight != 1.0:
            apply_hints(lstm_out_fwd, fasta, self.hints, self.hint_weight, "+")

        hmm_out_fwd = self.hmm_prediction(
            x_one_hot_fwd, lstm_out_fwd,
        )

        # bwd prediction
        fasta_bwd = fasta.complement()
        x_one_hot_bwd = fasta_bwd.one_hot(
            pad_index = 4,
            repeats = "track" if self.softmask else "omit",
            N = "track",
            dtype = np.float32,
        )
        x_one_hot_bwd = x_one_hot_bwd[:, ::-1, :]
        lstm_out_bwd = self.lstm_prediction(x_one_hot_bwd)
        if self.hints and self.hint_weight != 1.0:
            apply_hints(lstm_out_bwd, fasta, self.hints, self.hint_weight, "-")

        hmm_out_bwd = self.hmm_prediction(
            x_one_hot_bwd, lstm_out_bwd,
        )

        hmm_out_bwd = hmm_out_bwd[:,::-1]
        return hmm_out_fwd, hmm_out_bwd

    def repredict_function(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        # fwd prediction
        indices_fwd = np.where(np.isin(fasta.evidence[:,0], [0, 2]))[0]
        hmm_out_fwd_expand = np.empty((fasta.N, fasta.T), dtype=np.int32)
        if indices_fwd.size > 0:
            x_one_hot_fwd = fasta.one_hot(
                    pad_index = 4,
                    repeats = "track" if self.softmask else "omit",
                    N = "track",
                    dtype = np.float32,
                )[indices_fwd]
            lstm_out_fwd = self.lstm_prediction(x_one_hot_fwd)


            hmm_out_fwd = self.hmm_prediction(
                x_one_hot_fwd, lstm_out_fwd,
            )
            hmm_out_fwd_expand[indices_fwd] = hmm_out_fwd


        # bwd prediction
        indices_bwd = np.where(np.isin(fasta.evidence[:,0], [1, 2]))[0]
        hmm_out_bwd_expand = np.empty((fasta.N,fasta.T), dtype=np.int32)
        if indices_bwd.size > 0:
            fasta_bwd = fasta.complement()
            x_one_hot_bwd = fasta_bwd.one_hot(
                pad_index = 4,
                repeats = "track" if self.softmask else "omit",
                N = "track",
                dtype = np.float32,
            )[indices_bwd]
            x_one_hot_bwd = x_one_hot_bwd[:, ::-1, :]
            lstm_out_bwd = self.lstm_prediction(x_one_hot_bwd)

            hmm_out_bwd = self.hmm_prediction(
                x_one_hot_bwd, lstm_out_bwd,
            )

            hmm_out_bwd = hmm_out_bwd[:,::-1]
            hmm_out_bwd_expand[indices_bwd] = hmm_out_bwd
        return hmm_out_fwd_expand, hmm_out_bwd_expand

    def get_predictions(
            self,
            fasta: b2m.struct.Fasta,
            clamsa_inp=None,
            starting_tx_id: int = 0,
            complement: bool = False
    ) -> b2m.struct.Annotation:
        annotation = b2m.tools.GTF_from_model(
            fasta,
            predict_func=self.get_predict_fun(complement),
            repredict_exon_at_boundary=None,
            liberal=True,
            starting_tx_id=starting_tx_id,
        )
        return annotation


    def predict_lstm_batch(self, batch):
        def _is_cudnn_lstm_not_supported(err: BaseException) -> bool:
            msg = str(err)
            return (
                "CUDNN_STATUS_NOT_SUPPORTED" in msg
                or "CudnnRNNV3" in msg
                or "cudnnSetRNNDataDescriptor" in msg
            )
        try:
            return self.lstm_model.predict_on_batch(batch)
        except (tf.errors.OpError, tf.errors.InternalError, RuntimeError) as e:
            if _is_cudnn_lstm_not_supported(e):
                print(
                    f"""\nERROR: cuDNN failed at a prediction step. \n
                    This is a known issue with TensorFlow. Please use a \n
                    sequence length <= 500004 (--seq_len).""",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    def lstm_prediction(self, inp_chunks, clamsa_inp=None, batch_size=None):
        """Generates predictions using a LSTM model.

        Arguments:
            inp_ids (np.array): The input IDs for the transformer model, expected to be in a numpy array format.
            clamsa_inp (np.array): Optional clamsa input with same size as inp_chunks
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.

        Returns:
            lstm_predictions (np.array or list of np.array): The predictions generated by the LSTM model.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size
        num_batches = inp_chunks.shape[0] // batch_size
        lstm_predictions = []

        # decriptive error message when there is an input embedding dim mismatch
        # due to softmasking mismatch between training and inference
        expected_input_shape = self.lstm_model.input_shape
        actual_input_shape = inp_chunks.shape

        if expected_input_shape[-1] != actual_input_shape[-1]:
            error_msg = (
                f"Input shape mismatch: Model expects input with {expected_input_shape[-1]} features, "
                f"but received {actual_input_shape[-1]} features.\n\n"
            )
            if expected_input_shape[-1] == 6 and actual_input_shape[-1] == 5:
                error_msg += (
                    "This appears to be a softmasking compatibility issue.\n"
                    "The model was trained with softmasking enabled, but inference is running without softmasking.\n"
                    "SOLUTION: Remove the '--no_softmasking' flag from your command, or use a model trained without softmasking.\n"
                )
            elif expected_input_shape[-1] == 5 and actual_input_shape[-1] == 6:
                error_msg += (
                    "This appears to be a softmasking compatibility issue.\n"
                    "The model was trained without softmasking, but inference is running with softmasking enabled.\n"
                    "SOLUTION: Add the '--no_softmasking' flag to your command, or use a model trained with softmasking.\n"
                )
            else:
                error_msg += (
                "Please check that your model and input data are compatible.\n"
                )

            raise ValueError(error_msg)

        if inp_chunks.shape[0] % batch_size > 0:
            num_batches += 1
        for i in range(num_batches):
            start_pos = i * batch_size
            end_pos = (i+1) * batch_size
            if clamsa_inp is not None:
                y = self.predict_lstm_batch([
                    inp_chunks[start_pos:end_pos],
                    clamsa_inp[start_pos:end_pos]
                ])
            else:
                y = self.predict_lstm_batch(inp_chunks[start_pos:end_pos])
            if len(y.shape) == 1:
                y = np.expand_dims(y,0)
            lstm_predictions.append(y)
        lstm_predictions = np.concatenate(lstm_predictions, axis=0)
        return np.array(lstm_predictions)

    def hmm_prediction(self, nuc_seq, lstm_predictions,batch_size=None):
        """Generates predictions using a HMM model and the viterbi algorithm.

        Arguments:
            nuc_seq (np.array): One hot encoded representation of the input nucleotide sequence.
            lstm_predictions (np.array): Class label predictions from a LSTM model
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.

        Returns:
            HMM predictions (np.array or list of np.array): The predictions generated by the HMM model.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size
        num_batches = nuc_seq.shape[0] // batch_size
        hmm_predictions = []

        if nuc_seq.shape[0] % batch_size > 0:
            num_batches += 1
        for i in range(num_batches):
            start_pos = i * batch_size
            end_pos = (i+1) * batch_size
            y_hmm = self.predict_vit(nuc_seq[start_pos:end_pos],
                lstm_predictions[start_pos:end_pos]).numpy().squeeze()
            if len(y_hmm.shape) == 1:
                y_hmm = np.expand_dims(y_hmm,0)
            hmm_predictions.append(y_hmm)
        hmm_predictions = np.concatenate(hmm_predictions, axis=0)
        return hmm_predictions


    @tf.function
    def predict_vit(self, x, y_lstm):
        """Perform prediction using the Viterbi algorithm on the output of an LSTM model.

        This method applies the Viterbi algorithm to the sequence probabilities output by
        the LSTM model to find the most likely sequence of hidden states.

        Args:
            x (tf.Tensor): Input sequence tensor for which the predictions are to be made.
            y_lstm (np.array): LSTM predictions used as input for viterbi.

        Returns:
            tf.Tensor: The predicted state sequence tensor after applying Viterbi decoding.
        """
        if self.lstm_model and self.hmm:
            nuc = Cast()(x)
            if y_lstm.ndim == 2:
                y_lstm = y_lstm[np.newaxis, :, :]
            y_vit = self.gene_pred_hmm_layer(y_lstm, nuc)
        else:
            nuc = tf.cast(x[:,:,:5], tf.float32)
            y_vit = self.gene_pred_hmm_layer(y_lstm, nuc)
        return y_vit


    def make_default_hmm(self, inp_size=15):
        self.gene_pred_hmm_layer = HMMBlock(
            parallel=self.parallel_factor,
            mode=HMMMode.VITERBI,
            training=False,
            emitter_epsilon=self.hmm_emitter_epsilon,
            initial_exon_len=self.hmm_initial_exon_len,
            initial_intron_len=self.hmm_initial_intron_len,
            initial_ir_len=self.hmm_initial_ir_len,
        )
        self.gene_pred_hmm_layer.build((self.adapted_batch_size, self.seq_len, inp_size))
