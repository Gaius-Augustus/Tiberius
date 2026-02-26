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
from tiberius import (custom_cce_f1_loss, lstm_model, Cast)
from hidten import HMMMode
from tiberius.hmm import HMMBlock
import bricks2marble as b2m
import math


from typing import Optional
from typing import Literal, Optional
from pathlib import Path



try:
    import pyBigWig  # type: ignore
except ImportError as e:
    raise ImportError(
        "pyBigWig is required to write BigWig without UCSC tools.\n"
        "Install: conda install -c bioconda pybigwig  (or pip install pybigwig)"
    ) from e


def write_bigwig_from_lstm(
    lstm_pred: np.ndarray,  # (Nchunks, chunk_len, 15)
    out_bw: str | Path,
    chrom: str,
    chrom_size: int,
    *,
    true_len: Optional[int] = None,  # trim padded tail; if None uses N*T
    value: Literal["gene_score", "cds_sum", "intron_sum", "non_intergenic", "state_max"] = "gene_score",
    intron_weight: float = 0.5,
    smooth_win: int = 1,     # smoothing disabled for state_max
    decimals: int = 5,
    compress_runs: bool = True,
) -> None:
    """
    Write a BigWig track for JBrowse2 directly using pyBigWig (no UCSC binaries).

    - Coordinates are 0-based half-open in BigWig.
    - `chrom` must match the reference sequence name in your JBrowse assembly config.
    - `chrom_size` should be the contig length.
    - `true_len` can be used to drop padded tail when your last chunk is padded.

    `value` options:
      - cds_sum        : sum p(states 4..14)
      - intron_sum     : sum p(states 1..3)
      - non_intergenic : 1 - p(state 0)
      - gene_score     : cds_sum + intron_weight * intron_sum
      - state_max      : argmax state (0..14) as float track (best debug view)
    """

    out_bw = Path(out_bw)

    if lstm_pred.ndim != 3 or lstm_pred.shape[-1] != 15:
        raise ValueError(f"Expected lstm_pred shape (N,T,15), got {lstm_pred.shape}")

    N, T, K = lstm_pred.shape
    flat = lstm_pred.reshape(-1, K)  # (N*T, 15)

    if true_len is None:
        true_len = flat.shape[0]
    true_len = int(true_len)
    if not (0 <= true_len <= flat.shape[0]):
        raise ValueError(f"true_len={true_len} out of range [0, {flat.shape[0]}]")

    flat = flat[:true_len]

    # --- choose signal ---
    if value == "cds_sum":
        y = flat[:, 4:15].sum(axis=1)
    elif value == "intron_sum":
        y = flat[:, 1:4].sum(axis=1)
    elif value == "non_intergenic":
        y = 1.0 - flat[:, 0]
    elif value == "intergenic":
        y = flat[:, 0]
    elif value == "gene_score":
        cds = flat[:, 4:15].sum(axis=1)
        intr = flat[:, 1:4].sum(axis=1)
        y = cds + float(intron_weight) * intr
    elif value == "state_max":
        y = np.argmax(flat, axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown value={value}")

    y = y.astype(np.float32)

    # --- optional smoothing (not meaningful for state_max) ---
    if smooth_win and smooth_win > 1 and value != "state_max":
        win = int(smooth_win)
        pad_l = win // 2
        pad_r = win - 1 - pad_l
        yp = np.pad(y, (pad_l, pad_r), mode="edge")
        c = np.cumsum(yp, dtype=np.float64)
        y = ((c[win:] - c[:-win]) / float(win)).astype(np.float32)

    # enforce chrom_size consistency
    if true_len > chrom_size:
        raise ValueError(f"true_len ({true_len}) > chrom_size ({chrom_size}).")
    if chrom_size <= 0:
        raise ValueError("chrom_size must be > 0")

    # --- build intervals ---
    if compress_runs:
        # Run-length encode (huge size win for state_max, also helps probs)
        starts = []
        ends = []
        vals = []

        if true_len == 0:
            starts_arr = np.array([], dtype=np.int64)
            ends_arr = np.array([], dtype=np.int64)
            vals_arr = np.array([], dtype=np.float32)
        else:
            s = 0
            cur = float(y[0])
            for i in range(1, true_len):
                v = float(y[i])
                if v != cur:
                    starts.append(s)
                    ends.append(i)
                    vals.append(round(cur, decimals))
                    s = i
                    cur = v
            starts.append(s)
            ends.append(true_len)
            vals.append(round(cur, decimals))

            starts_arr = np.asarray(starts, dtype=np.int64)
            ends_arr = np.asarray(ends, dtype=np.int64)
            vals_arr = np.asarray(vals, dtype=np.float32)
    else:
        starts_arr = np.arange(true_len, dtype=np.int64)
        ends_arr = starts_arr + 1
        vals_arr = np.round(y, decimals).astype(np.float32)

    # --- write bigwig ---
    # pyBigWig requires a header with (chrom, size)
    bw = pyBigWig.open(str(out_bw), "w")
    try:
        bw.addHeader([(chrom, int(chrom_size))])
        if starts_arr.size:
            bw.addEntries(
                [chrom] * int(starts_arr.size),
                starts_arr.tolist(),
                ends=ends_arr.tolist(),
                values=vals_arr.tolist(),
            )
    finally:
        bw.close()


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
                 hmm=False, hmm_emitter_epsilon=0, model_path_hmm='',
                 temp_dir='', num_hmm=1,
                 hmm_factor=None,
                 annot_path='', genome_path='', genome=None, softmask=True,
                 strand='+', parallel_factor=1, oracle=False,
                 lstm_cfg='',):
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
            - genome_path (str): Path to the genome file (FASTA).
            - genome: dictionary of SeqRecords, (overriding) alternative to genome_path
            - softmask (bool): Whether to use softmasking.
            - strand (str): Indicates the strand ('+' for positive, '-' for negative).
            - parallel_factor (int): The parallel factor used for Viterbi.
            - lstm_cfg (str): path to lstm cfg to load weights instead of the whole model
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
        self.strand=strand
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
            elif config["hmm"]:
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
        if strand is None:
            strand = self.strand

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

    def predict_function_old_test(
                        self,
                        fasta: b2m.struct.FASTA,
                        lstm_repred_fwd: np.ndarray | None = None,
                        lstm_repred_bwd: np.ndarray | None = None,
                ) -> tuple[np.ndarray, np.ndarray]:

        x_one_hot = fasta.one_hot(
            pad_index = 4,
            repeats = "track" if self.softmask else "omit",
            N = "track",
            dtype = np.float32,
        )


        if lstm_repred_fwd is None:
            orig_shape = x_one_hot.shape
            x_one_hot = x_one_hot.reshape(-1, 500_004, orig_shape[-1])

            lstm_out_fwd = self.lstm_prediction(x_one_hot, batch_size=self.adapted_batch_size)

            x_one_hot = x_one_hot.reshape(*orig_shape)
            lstm_out_fwd = lstm_out_fwd.reshape(orig_shape[0], orig_shape[1], lstm_out_fwd.shape[-1])
        else:
            lstm_out_fwd = lstm_repred_fwd

        # Get bw tracks
        # write_bigwig_from_lstm(lstm_out_fwd, "lstm_fwd_cds.bw", chrom_size=50000400,chrom="chr12",
        #                 value="cds_sum" )
        # write_bigwig_from_lstm(lstm_out_fwd, "lstm_fwd_intron.bw", chrom_size=50000400,chrom="chr12",
        #                 value="intron_sum" )
        # write_bigwig_from_lstm(lstm_out_fwd, "lstm_fwd_ir.bw", chrom_size=50000400,chrom="chr12",
        #                 value="intergenic" )


        print(x_one_hot.shape, lstm_out_fwd.shape, file=sys.stderr)
        hmm_out_fwd = self.hmm_prediction(
            x_one_hot, lstm_out_fwd,
        )


        # Reprdict on specific section
        # orig_shape = hmm_out_fwd.shape
        # x_one_hot = x_one_hot.reshape(-1,x_one_hot.shape[-1])
        # lstm_out_fwd = lstm_out_fwd.reshape(-1,lstm_out_fwd.shape[-1])
        # hmm_out_fwd = hmm_out_fwd.reshape(-1)
        # print(orig_shape, file=sys.stderr)
        # re_x = x_one_hot[1_900_000:2_900_008].reshape(2,500_004,-1)
        # re_l = lstm_out_fwd[1_900_000:2_900_008].reshape(2,500_004,-1)

        # # re_l[0] *= 3
        # # re_l[4:] /= 2
        # re_h = self.hmm_prediction(
        #     re_x, re_l,
        # )


        # hmm_out_fwd[1_900_000:2_900_008] = re_h.reshape(-1)

        # hmm_out_fwd = hmm_out_fwd.reshape(orig_shape)

        fasta_bwd = fasta.complement(in_place=False)
        x_one_hot_bwd = fasta_bwd.one_hot(
            pad_index = 4,
            repeats = "track" if self.softmask else "omit",
            N = "track",
            dtype = np.float32,
        )

        if lstm_repred_bwd is None:
            orig_shape = x_one_hot_bwd.shape
            x_one_hot_bwd = x_one_hot_bwd[:, ::-1]
            x_one_hot_bwd = x_one_hot_bwd.reshape(-1, 500_004, orig_shape[-1])

            lstm_out_bwd = self.lstm_prediction(x_one_hot_bwd, batch_size=self.adapted_batch_size)

            x_one_hot_bwd = x_one_hot_bwd.reshape(*orig_shape)
            lstm_out_bwd = lstm_out_bwd.reshape(orig_shape[0], orig_shape[1], lstm_out_bwd.shape[-1])
        else:
            lstm_out_bwd = lstm_repred_bwd


        hmm_out_bwd = self.hmm_prediction(
            x_one_hot_bwd, lstm_out_bwd
        )

        return hmm_out_fwd, hmm_out_bwd[:,::-1]#, lstm_out_fwd, lstm_out_bwd[:,::-1]


    def predict_function(
                self,
                fasta: b2m.struct.FASTA
            ) -> tuple[np.ndarray, np.ndarray]:

        x_one_hot = fasta.one_hot(
            pad_index = 4,
            repeats = "track" if self.softmask else "omit",
            N = "track",
            dtype = np.float32,
        )

        lstm_out_fwd = self.lstm_prediction(x_one_hot)

        # # Get bw tracks
        # write_bigwig_from_lstm(lstm_out_fwd, "lstm_fwd_cds_lstmtrain.bw", chrom_size=50000400,chrom="Chr1",
        #                 value="cds_sum" )
        # write_bigwig_from_lstm(lstm_out_fwd, "lstm_fwd_intron_lstmtrain.bw", chrom_size=50000400,chrom="Chr1",
        #                 value="intron_sum" )
        # write_bigwig_from_lstm(lstm_out_fwd, "lstm_fwd_ir_lstmtrain.bw", chrom_size=50000400,chrom="Chr1",
        #                 value="intergenic" )
        # exit()

        hmm_out_fwd = self.hmm_prediction(
            x_one_hot, lstm_out_fwd,
        )

        # fasta_bwd = fasta.complement(in_place=False)
        # x_one_hot_bwd = fasta_bwd.one_hot(
        #     pad_index = 4,
        #     repeats = "track" if self.softmask else "omit",
        #     N = "track",
        #     dtype = np.float32,
        # )
        # x_one_hot_bwd = x_one_hot_bwd[:, ::-1, :]
        # lstm_out_bwd = self.lstm_prediction(x_one_hot_bwd)

        # hmm_out_bwd = self.hmm_prediction(
        #     x_one_hot_bwd, lstm_out_bwd
        # )
        return hmm_out_fwd, None# hmm_out_bwd[:,::-1]



    def get_predictions(
            self,
            fasta: b2m.struct.FASTA,
            clamsa_inp=None,
            starting_tx_id: int = 0,
    ) -> b2m.struct.Annotation:
        annotation = b2m.tools.GTF_from_model(
            fasta,
            predict_func=self.predict_function,
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


    def lstm_prediction(self, inp_chunks, clamsa_inp=None, trans_emb=None, batch_size=None):
        """Generates predictions using a LSTM model.

        Arguments:
            inp_ids (np.array): The input IDs for the transformer model, expected to be in a numpy array format.
            clamsa_inp (np.array): Optional clamsa input with same size as inp_chunks
            trans_emb (np.array):  Optional transformer emb used as input with same size as inp_chunks
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
        return lstm_predictions

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
        print("BB", batch_size, file=sys.stderr)
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

    def hmm_predictions_filtered(self, inp_chunks, lstm_predictions, batch_size=None):
        """Generates predictions using a HMM model and the viterbi algorithm.
        It first analyzes the class probabilities from LSTM predictions over
        windows of 200 base pairs (bp) in length. The HMM makes predictions on
        an example only if there's at least one window where the average class
        probability for the CDS class is 0.8 or higher. If no such window exists,
        the HMM will skip making predictions for that example, and all positions
        within it are labeled as intergenic region.

        Arguments:
            inp_chunks (np.array): One hot encoded representation of the input nucleotide sequence.
            lstm_predictions (np.array): Class label predictions from a LSTM model

        Returns:
            HMM predictions (np.array or list of np.array): The predictions generated by the HMM model.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size

        hmm_predictions = []

        if self.hmm_factor > 1:
            inp_chunks = inp_chunks.reshape((inp_chunks.shape[0]*self.hmm_factor, inp_chunks.shape[1]//self.hmm_factor, -1))
            lstm_predictions[0] = lstm_predictions[0].reshape((lstm_predictions[0].shape[0]*self.hmm_factor, lstm_predictions[0].shape[1]//self.hmm_factor, -1))
            lstm_predictions[1] = lstm_predictions[1].reshape((lstm_predictions[1].shape[0]*self.hmm_factor, lstm_predictions[1].shape[1]//self.hmm_factor, -1))

        batch_i = []
        hmm_predictions = np.zeros((inp_chunks.shape[0],inp_chunks.shape[1]), int)
        for i in range(inp_chunks.shape[0]):
            coding_prob = np.min(lstm_predictions[i,:,0]) < 0.7
            if True or coding_prob:
                batch_i += [i]
            else:
                hmm_predictions[i] = lstm_predictions[i].argmax(-1)

            if len(batch_i) == batch_size*self.hmm_factor or i == inp_chunks.shape[0]-1:
                y_hmm = self.predict_vit(
                    inp_chunks[batch_i],
                    lstm_predictions[batch_i],
                ).numpy().squeeze()
                if len(y_hmm.shape) == 1:
                    y_hmm = np.expand_dims(y_hmm,0)
                for j1, j2 in enumerate(batch_i):
                    hmm_predictions[j2] = y_hmm[j1]
                batch_i = []

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
            if y_lstm.shape[-1] ==7:
                new_y_lstm = tf.concat([y_lstm[:,:,0:1],
                        tf.reduce_sum(y_lstm[:,:,1:4], axis=-1, keepdims=True),
                        y_lstm[:,:,4:]], axis=-1)
            elif y_lstm.shape[-1] == 15:
                # A = make_aggregation_matrix(k=1)
                # new_y_lstm = tf.matmul(y_lstm, A)
                new_y_lstm = y_lstm
            elif y_lstm.shape[-1] == 2:
                new_y_lstm = tf.concat([
                    y_lstm[:,:,:1]/2,
                    y_lstm[:,:,:1]/2,
                    y_lstm[:,:,1:]/3,
                    y_lstm[:,:,1:]/3,
                    y_lstm[:,:,1:]/3], axis=-1)
            elif y_lstm.shape[-1] == 3:
                new_y_lstm = tf.concat([
                    y_lstm[:,:,:1],
                    y_lstm[:,:,1:2],
                    y_lstm[:,:,2:]/3,
                    y_lstm[:,:,2:]/3,
                    y_lstm[:,:,2:]/3], axis=-1)
            else:
                new_y_lstm = y_lstm
            nuc = Cast()(x)
            y_vit = self.gene_pred_hmm_layer(new_y_lstm, nuc)
        else:
            nuc = tf.cast(x[:,:,:5], tf.float32)
            y_vit = self.gene_pred_hmm_layer(y_lstm, nuc)
        return y_vit


    def make_default_hmm(self, inp_size=15):
        self.gene_pred_hmm_layer = HMMBlock(
            parallel=self.parallel_factor,
            mode=HMMMode.VITERBI,
            training=False,
            emitter_epsilon=0.01,
        )
        self.gene_pred_hmm_layer.build((self.adapted_batch_size, self.seq_len, inp_size))
