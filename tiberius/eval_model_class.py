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
from tiberius.models import (custom_cce_f1_loss, build_backbone_from_config, Cast,
                             add_hmm_new_layer)
from hidten import HMMMode
from tiberius.hmm import HMMBlock, TrainableHMMHead, fix_intron_state_chain_labels
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
            - genome_path (str): Path to the genome file (Fasta).
            - genome: dictionary of SeqRecords, (overriding) alternative to genome_path
            - softmask (bool): Whether to use softmasking.
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
        self.both_strands = False
        # Optional cache of the last preHMM (LSTM) softmax output, populated
        # inside predict_function when cache_softmax=True. Holds one entry per
        # group as processed by bricks2marble.tools.annotate.annotate_genome,
        # so downstream postprocess sees exactly the softmax that produced the
        # HMM labels for the group it is called on.
        self.cache_softmax = False
        self.last_softmax_fwd: np.ndarray | None = None
        self.last_softmax_bwd: np.ndarray | None = None
        # Optional callback fired inside predict_function immediately after
        # the LSTM softmax is computed for a group. Signature:
        # ``callback(fasta, softmax_fwd, softmax_bwd)`` where both arrays are
        # shaped ``(N_chunks, T, 15)`` and already indexed in forward-strand
        # genome coordinates. Used for the streaming BigWig output.
        self.softmax_callback = None


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
            ),
            "TrainableHMMHead": TrainableHMMHead,
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
            if "inp_size" in config:
                self.softmask = config["inp_size"]==6
            elif "softmasking" in config:
                self.softmask = config["softmasking"]

            # Chunk `BorderGatedSelfAttention` layers to avoid O(L^2) OOM at
            # inference on long sequences. Injects `attn_chunk_size` into
            # the config *before* the backbone is built, because a Keras
            # functional model traces each layer's `call` exactly once at
            # build time -- mutating `chunk_size` post-hoc has no effect on
            # the traced graph. Prefers an explicit config key, else uses
            # the training-time pooled length (w_size / pool_size), else
            # falls back to a safe default. No-op for archs without a
            # `BorderGatedSelfAttention` layer since the arg is ignored.
            if config.get("arch") == "border_gated_attn_lstm" \
                    and config.get("attn_chunk_size") is None:
                w_size = config.get("w_size")
                pool_size = config.get("pool_size", 9)
                if w_size and pool_size:
                    config["attn_chunk_size"] = w_size // pool_size
                else:
                    config["attn_chunk_size"] = 1200
                print(f"Set attn_chunk_size={config['attn_chunk_size']} for "
                      f"long-sequence inference.", file=sys.stderr)

            self.lstm_model = build_backbone_from_config(config, softmasking=self.softmask)

            weights_h5  = f"{self.model_path}/weights.h5"
            if not os.path.exists(weights_h5):
                weights_h5 = f"{self.model_path}/model.weights.h5"

            # Trainable-HMM head: weights.h5 stores backbone + HMM head,
            # so attach the head *before* loading weights, and expose the
            # HMM layer for downstream inference.
            if config.get("head") == "hmm_new":
                hmm_new_cfg = config.get("hmm_new_config") or {}
                # Pick a parallel_factor tuned to the inference sequence
                # length rather than reusing the training default: at T=500k
                # a factor of 1 makes the scan tree ~19 levels deep and
                # blows up memory. compute_parallel_factor gives the
                # closest divisor to sqrt(seq_len).
                if self.parallel_factor and self.parallel_factor > 1:
                    inference_parallel = self.parallel_factor
                else:
                    inference_parallel = compute_parallel_factor(self.seq_len)
                    self.parallel_factor = inference_parallel
                both_strands = config.get("both_strands", False)
                # Backbone outputs 30 for both_strands; HMM head operates on
                # per-strand size of 15.
                hmm_output_size = config["output_size"] // 2 if both_strands else config["output_size"]
                full_model = add_hmm_new_layer(
                    self.lstm_model,
                    output_size=hmm_output_size,
                    hmm_config=hmm_new_cfg,
                    embed=config.get("hmm_new_embed", 160),
                    embed_norm=config.get("hmm_new_embed_norm", "layer"),
                    embed_activation=config.get("hmm_new_embed_activation", "softmax"),
                    readout_type=config.get("hmm_new_readout_type", "conv"),
                    readout_conv_kernel=config.get("hmm_new_readout_conv_kernel", 9),
                    parallel_factor=inference_parallel,
                    residual_from_input=config.get("hmm_new_residual", True),
                    both_strands=both_strands,
                )
                full_model.load_weights(weights_h5)
                self.trainable_hmm_head = full_model.get_layer("trainable_hmm_head")
                # Follow vipsania's inference contract: flip the HMM into
                # VITERBI mode so it returns integer state labels
                # (B, T, H) instead of a (B, T, H*S) posterior we then
                # softmax over 15 classes. Saves ~15x memory downstream.
                self.trainable_hmm_head.set_mode(HMMMode.VITERBI)
                self._trainable_hmm_isc = self.trainable_hmm_head.intron_state_chain
                self.both_strands = both_strands
                self.inp_size = self.lstm_model.output_shape[-1]
                if summary:
                    self.lstm_model.summary()
                    full_model.summary()
                return

            self.lstm_model.load_weights(weights_h5)

            # hmm_middle_residual_stream: the backbone already contains a
            # TrainableHMMHead at 'hmm_middle_head'. For inference we:
            #   1. Extract a sub-model that ends at 'hmm_in_softmax'
            #      (the pre-HMM class probabilities).  This becomes lstm_model
            #      so lstm_prediction returns the inputs the HMM expects.
            #   2. Switch the baked-in HMM head to VITERBI mode and expose it
            #      as self.trainable_hmm_head so predict_vit's existing
            #      `use_trainable` path drives it -- no new HMM is added.
            if config.get("arch") == "hmm_middle_residual_stream":
                full_backbone = self.lstm_model
                if self.parallel_factor and self.parallel_factor > 1:
                    inference_parallel = self.parallel_factor
                else:
                    inference_parallel = compute_parallel_factor(self.seq_len)
                    self.parallel_factor = inference_parallel
                # Extract HMM head before narrowing lstm_model.
                hmm_head = full_backbone.get_layer('hmm_middle_head')
                hmm_head.parallel_factor = inference_parallel
                hmm_head.set_mode(HMMMode.VITERBI)
                self.trainable_hmm_head = hmm_head
                self._trainable_hmm_isc = hmm_head.intron_state_chain
                # Narrow to the pre-HMM sub-model (shares layer weights).
                self.lstm_model = Model(
                    inputs=full_backbone.inputs,
                    outputs=full_backbone.get_layer('hmm_in_softmax').output,
                )
                self.inp_size = self.lstm_model.output_shape[-1]
                if summary:
                    self.lstm_model.summary()
                return

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
                out_dim = self.lstm_model.output_shape[-1]
                if config.get("both_strands", False):
                    # Backbone outputs 30; HMM operates on each strand's 15-class slice.
                    self.both_strands = True
                    out_dim = out_dim // 2
                self.make_default_hmm(inp_size=out_dim)
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
            # Trainable HMM head (hmm_new): update in-place so its scan
            # tree matches the current chunksize. For old checkpoints
            # without this head, fall through to make_default_hmm.
            if getattr(self, "trainable_hmm_head", None) is not None:
                self.trainable_hmm_head.parallel_factor = self.parallel_factor
            else:
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

        if self.both_strands:
            return self._predict_function_dual_strand(fasta)

        # fwd prediction
        x_one_hot_fwd = fasta.one_hot(
                pad_index = 4,
                repeats = "track" if self.softmask else "omit",
                N = "track",
                dtype = np.float32,
            )
        lstm_out_fwd = self.lstm_prediction(x_one_hot_fwd)

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

        hmm_out_bwd = self.hmm_prediction(
            x_one_hot_bwd, lstm_out_bwd,
        )

        hmm_out_bwd = hmm_out_bwd[:,::-1]

        if self.cache_softmax or self.softmax_callback is not None:
            # bwd lstm was computed on reverse-complemented chunks that were
            # additionally flipped along the time axis; flip back so both
            # arrays are indexed in forward-strand coordinates. These arrays
            # are the exact preHMM emission distribution that the HMM has
            # just consumed above.
            sm_fwd = np.asarray(lstm_out_fwd)
            sm_bwd = np.asarray(lstm_out_bwd)[:, ::-1, :]
            if self.cache_softmax:
                self.last_softmax_fwd = sm_fwd
                self.last_softmax_bwd = sm_bwd
            if self.softmax_callback is not None:
                self.softmax_callback(fasta, sm_fwd, sm_bwd)

        return hmm_out_fwd, hmm_out_bwd

    def repredict_function(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:

        if self.both_strands:
            return self._repredict_function_dual_strand(fasta)

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

    def _predict_function_dual_strand(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        """Single-pass dual-strand prediction.

        Dispatches to the trainable HMM head path (hmm_new) when available,
        otherwise uses the backbone-only path that splits the 30-dim output.
        """
        if getattr(self, "trainable_hmm_head", None) is not None:
            return self._predict_dual_strand_hmm_new(fasta)
        return self._predict_dual_strand_backbone(fasta)

    def _predict_dual_strand_hmm_new(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        """hmm_new + both_strands: single forward pass through backbone + HMM.

        The HMM head with use_reverse_strand=True handles both strands in one
        Viterbi pass and returns labels (B, T, 2) where [..., 0] = fwd,
        [..., 1] = rev, both already in genomic forward-strand coordinates.
        """
        x = fasta.one_hot(
            pad_index=4,
            repeats="track" if self.softmask else "omit",
            N="track",
            dtype=np.float32,
        )
        lstm_out = self.lstm_prediction(x)
        hmm_fwd, hmm_bwd = self.hmm_prediction_dual(x, lstm_out)

        if self.cache_softmax or self.softmax_callback is not None:
            # lstm_out has shape (N, T, 30): logits for fwd [:15] and rev [15:],
            # both in genomic forward order (the HMM handles RC internally).
            logits = np.asarray(lstm_out)
            sm_fwd = tf.nn.softmax(logits[:, :, :15]).numpy()
            sm_bwd = tf.nn.softmax(logits[:, :, 15:]).numpy()
            if self.cache_softmax:
                self.last_softmax_fwd = sm_fwd
                self.last_softmax_bwd = sm_bwd
            if self.softmax_callback is not None:
                self.softmax_callback(fasta, sm_fwd, sm_bwd)

        return hmm_fwd, hmm_bwd

    def _predict_dual_strand_backbone(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        """Backbone-only both_strands: single forward pass, split 30-dim output.

        The backbone is trained with both strands in genomic forward order:
          lstm_out[:, t, :15] = fwd-strand logits at genomic position t
          lstm_out[:, t, 15:] = rev-strand logits at genomic position t

        The backward HMM expects 5'→3' (minus-strand) order, so the rev slice
        is time-flipped before being fed to the HMM.  The HMM output is then
        flipped back to genomic forward-strand coordinates.
        RC nucleotides are passed to the bwd HMM for splice-site scoring.
        """
        repeats = "track" if self.softmask else "omit"
        x_fwd = fasta.one_hot(pad_index=4, repeats=repeats, N="track", dtype=np.float32)
        lstm_out = self.lstm_prediction(x_fwd)      # (N, T, 30)

        out_fwd = lstm_out[:, :, :15]               # genomic forward order
        out_rev = lstm_out[:, ::-1, 15:]            # flip to 5'→3' order for backward HMM

        x_bwd = fasta.complement().one_hot(
            pad_index=4, repeats=repeats, N="track", dtype=np.float32,
        )[:, ::-1, :]                               # RC + time-reverse → 5'→3' order

        hmm_fwd = self.hmm_prediction(x_fwd, out_fwd)
        hmm_bwd = self.hmm_prediction(x_bwd, out_rev)[:, ::-1]   # flip to genomic order

        if self.cache_softmax or self.softmax_callback is not None:
            sm_fwd = tf.nn.softmax(out_fwd).numpy()
            # out_rev is in 5'→3' order; flip to genomic order for caching
            sm_bwd = tf.nn.softmax(out_rev).numpy()[:, ::-1, :]
            if self.cache_softmax:
                self.last_softmax_fwd = sm_fwd
                self.last_softmax_bwd = sm_bwd
            if self.softmax_callback is not None:
                self.softmax_callback(fasta, sm_fwd, sm_bwd)

        return hmm_fwd, hmm_bwd

    def _repredict_function_dual_strand(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        """Evidence-filtered dual-strand prediction. Dispatches by head type."""
        if getattr(self, "trainable_hmm_head", None) is not None:
            return self._repredict_dual_strand_hmm_new(fasta)
        return self._repredict_dual_strand_backbone(fasta)

    def _repredict_dual_strand_hmm_new(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        """hmm_new + both_strands evidence-filtered repredict."""
        indices = np.where(np.isin(fasta.evidence[:, 0], [0, 1, 2]))[0]
        hmm_fwd_expand = np.empty((fasta.N, fasta.T), dtype=np.int32)
        hmm_bwd_expand = np.empty((fasta.N, fasta.T), dtype=np.int32)
        if indices.size > 0:
            x = fasta.one_hot(
                pad_index=4,
                repeats="track" if self.softmask else "omit",
                N="track",
                dtype=np.float32,
            )[indices]
            lstm_out = self.lstm_prediction(x)
            hmm_fwd, hmm_bwd = self.hmm_prediction_dual(x, lstm_out)
            hmm_fwd_expand[indices] = hmm_fwd
            hmm_bwd_expand[indices] = hmm_bwd
        return hmm_fwd_expand, hmm_bwd_expand

    def _repredict_dual_strand_backbone(
            self,
            fasta: b2m.struct.Fasta
        ) -> tuple[np.ndarray, np.ndarray]:
        """Backbone-only both_strands evidence-filtered repredict."""
        indices = np.where(np.isin(fasta.evidence[:, 0], [0, 1, 2]))[0]
        hmm_fwd_expand = np.empty((fasta.N, fasta.T), dtype=np.int32)
        hmm_bwd_expand = np.empty((fasta.N, fasta.T), dtype=np.int32)
        if indices.size > 0:
            repeats = "track" if self.softmask else "omit"
            x_fwd = fasta.one_hot(
                pad_index=4, repeats=repeats, N="track", dtype=np.float32,
            )[indices]
            lstm_out = self.lstm_prediction(x_fwd)          # (k, T, 30)

            out_fwd = lstm_out[:, :, :15]
            out_rev = lstm_out[:, ::-1, 15:]                # flip to 5'→3' for backward HMM

            x_bwd = fasta.complement().one_hot(
                pad_index=4, repeats=repeats, N="track", dtype=np.float32,
            )[indices][:, ::-1, :]

            hmm_fwd_expand[indices] = self.hmm_prediction(x_fwd, out_fwd)
            hmm_bwd_expand[indices] = self.hmm_prediction(x_bwd, out_rev)[:, ::-1]  # flip to genomic
        return hmm_fwd_expand, hmm_bwd_expand

    def hmm_prediction_dual(self, nuc_seq, lstm_predictions, batch_size=None):
        """Batched Viterbi for both strands simultaneously via hmm_new.

        Returns (hmm_fwd, hmm_bwd) each shaped (N, T) with intron-chain labels
        folded back to the 15-class Tiberius layout when intron_state_chain > 1.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size
        n = nuc_seq.shape[0]
        num_batches = (n + batch_size - 1) // batch_size
        hmm_fwd_all, hmm_bwd_all = [], []
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            y_fwd, y_bwd = self.predict_vit_dual(
                nuc_seq[s:e], lstm_predictions[s:e]
            )
            y_fwd = y_fwd.numpy().squeeze()
            y_bwd = y_bwd.numpy().squeeze()
            if y_fwd.ndim == 1:
                y_fwd = np.expand_dims(y_fwd, 0)
            if y_bwd.ndim == 1:
                y_bwd = np.expand_dims(y_bwd, 0)
            hmm_fwd_all.append(y_fwd)
            hmm_bwd_all.append(y_bwd)
        hmm_fwd = np.concatenate(hmm_fwd_all, axis=0)
        hmm_bwd = np.concatenate(hmm_bwd_all, axis=0)
        isc = getattr(self, "_trainable_hmm_isc", 1)
        if isc and isc > 1:
            hmm_fwd = fix_intron_state_chain_labels(hmm_fwd, isc)
            hmm_bwd = fix_intron_state_chain_labels(hmm_bwd, isc)
        return hmm_fwd, hmm_bwd

    @tf.function
    def predict_vit_dual(self, x, y_lstm):
        """Viterbi for both strands in a single HMM pass.

        Requires self.trainable_hmm_head in VITERBI mode with
        use_reverse_strand=True (set by add_hmm_new_layer when both_strands=True).
        Returns (y_fwd, y_bwd) each (B, T) int32 in genomic forward-strand order.
        The HMM un-reverses the minus-strand output internally.
        """
        nuc = tf.cast(x[:, :, :5], tf.float32)
        labels = self.trainable_hmm_head(y_lstm, nuc, training=False)
        return tf.cast(labels[..., 0], tf.int32), tf.cast(labels[..., 1], tf.int32)

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
                y_hmm = np.expand_dims(y_hmm, 0)
            hmm_predictions.append(y_hmm)
        hmm_predictions = np.concatenate(hmm_predictions, axis=0)
        # Fold intron-chain labels back to the base 15-class Tiberius
        # layout when running the trainable head with intron_state_chain>1.
        # (See vipsania.annotate._fix_intron_state_chain_labels.)
        isc = getattr(self, "_trainable_hmm_isc", 1)
        if isc and isc > 1:
            hmm_predictions = fix_intron_state_chain_labels(
                hmm_predictions, isc,
            )
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
        # Prefer the trainable HMM head when it exists (hmm_new
        # checkpoint). In VITERBI mode (set at load time), the head
        # returns integer state labels of shape (B, T, H); we take the
        # first head so the shape matches the old HMM's (B, T) output.
        # The intron-chain fold to Tiberius's 15-class layout happens
        # on the numpy side in hmm_prediction so the fix runs after the
        # @tf.function trace, matching vipsania's approach.
        use_trainable = getattr(self, "trainable_hmm_head", None) is not None

        if self.lstm_model and self.hmm:
            nuc = Cast()(x)
            if y_lstm.ndim == 2:
                y_lstm = y_lstm[np.newaxis, :, :]
            if use_trainable:
                labels = self.trainable_hmm_head(y_lstm, nuc, training=False)
                y_vit = tf.cast(labels[..., 0], tf.int32)
            else:
                y_vit = self.gene_pred_hmm_layer(y_lstm, nuc)
        else:
            nuc = tf.cast(x[:,:,:5], tf.float32)
            if use_trainable:
                labels = self.trainable_hmm_head(y_lstm, nuc, training=False)
                y_vit = tf.cast(labels[..., 0], tf.int32)
            else:
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
