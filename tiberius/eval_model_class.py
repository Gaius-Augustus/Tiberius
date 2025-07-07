# ==============================================================
# Authors: Lars Gabriel
#
# Class handling the prediction and evaluation for a single species
# ==============================================================

import sys, json, os, re, csv, time
import subprocess as sp
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from learnMSA.msa_hmm.Viterbi import viterbi
from tensorflow.keras.models import Model
from tiberius import (GenePredHMMLayer, 
                    make_5_class_emission_kernel, 
                    make_aggregation_matrix, 
                    make_15_class_emission_kernel,
                    GenomeSequences,
                    GeneStructure, Anno,
                    custom_cce_f1_loss, lstm_model, Cast)

tf.config.optimizer.set_jit(False)

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
                 hmm=False,  model_path_hmm='', 
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
        self.batch_size = batch_size
        self.adapted_batch_size = batch_size # can be increased if chunksize is reduced
        self.annot_path = annot_path
        self.genome_path = genome_path
        self.genome = genome
        self.softmask = softmask
        self.hmm = hmm
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
    
    def reduce_label(self, arr, num_hmm=1):
        """Reduces intron and exon labels.
        
        Args:
            arr (np.ndarray): Array containing the labels to be reduced.
            
        Returns:
            np.ndarray: Array with reduced label space.
        """ 
        A = make_aggregation_matrix(k=num_hmm)
        new_arr = A.argmax(1)[arr]
        new_arr[(new_arr > 1)] = 2
        return new_arr
    
    def load_model(self, summary=True):
        """Loads the model from the given model path.
        
        Args:
            summary (bool, optional): If True, prints the model summary. Defaults to True.
        """
        if self.model_path:
            try:
                if self.lstm_cfg:
                    with open(self.lstm_cfg, 'r') as f:
                        config = json.load(f)
                else:
                    with open(f"{self.model_path}/model_config.json", 'r') as f:
                        config = json.load(f)
            except Exception as e:
                print(f"Error could not find config of the model. It should be located at {self.model_path}/model_config.json: {e}")
                sys.exit(1)
            relevant_keys = ['units', 'filter_size', 'kernel_size', 
                'numb_conv', 'numb_lstm', 'dropout_rate', 
                'pool_size', 'stride', 'lstm_mask', 'clamsa',
                'output_size', 'residual_conv', 'softmasking',
                'clamsa_kernel', 'lru_layer']
            relevant_args = {key: config[key] for key in relevant_keys if key in config}
            self.lstm_model = lstm_model(**relevant_args)
            if Path(f"{self.model_path}/weights.h5").exists():
                self.lstm_model.load_weights(f"{self.model_path}/weights.h5")
            elif Path(f"{self.model_path}/model.weights.h5").exists():
                self.lstm_model.load_weights(f"{self.model_path}/model.weights.h5")
            else:
                print(f"Could not find weights for model {self.model_path}. "
                      "Please make sure that either weights.h5 or model.weights.h5 is present in your model savepoint.")
                sys.exit(1)

            if self.model_path_hmm:
                model_hmm = keras.models.load_model(self.model_path_hmm, 
                                                    custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.adapted_batch_size),
                                                            'loss_': custom_cce_f1_loss(2, self.adapted_batch_size)})
                self.gene_pred_hmm_layer = model_hmm.get_layer('gene_pred_hmm_layer')
                self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
                self.gene_pred_hmm_layer.cell.recurrent_init()
            else:
                self.make_default_hmm(inp_size=self.lstm_model.output_shape[-1])
        # loading full models for training or old models
        elif self.model_path_lstm_old:
            self.lstm_model = keras.models.load_model(self.model_path_lstm_old, 
                    custom_objects={
                    'custom_cce_f1_loss': custom_cce_f1_loss(2, self.adapted_batch_size),
                    'loss_': custom_cce_f1_loss(2, self.adapted_batch_size),
                    "Cast": Cast}, 
                    compile=False,
                    )
            self.make_default_hmm(inp_size=self.lstm_model.output.shape[-1])
        elif self.model_path_old:
            self.model = keras.models.load_model(self.model_path_old, 
                    custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.adapted_batch_size),
                        'loss_': custom_cce_f1_loss(2, self.adapted_batch_size),
                        "Cast": Cast})
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
            print(f"Running gene pred hmm layer with parallel factor {self.gene_pred_hmm_layer.parallel_factor}")
            self.gene_pred_hmm_layer.cell.recurrent_init()
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
            # print(f"Adapted batch size to {self.adapted_batch_size} using chunksize {adapted_chunksize}")
            self.load_model(summary=False)

    def init_fasta(self,  genome_path=None, chunk_len=None, min_seq_len=0):
        if genome_path is None:
            genome_path = self.genome_path
        if chunk_len is None:
            chunk_len = self.seq_len
        if (self.genome):
            fasta = GenomeSequences(genome=self.genome, chunksize=chunk_len, 
                overlap=0, min_seq_len=min_seq_len)
        else:
            fasta = GenomeSequences(fasta_file=genome_path, chunksize=chunk_len, 
                overlap=0, min_seq_len=min_seq_len)
        return fasta
    
    def load_genome_data(self, fasta_object, seq_names, strand='', softmask=True):
        if strand is None:
            strand = self.strand
        
        fasta_object.encode_sequences(seq=seq_names)

        f_chunk, coords, adapted_chunksize = fasta_object.get_flat_chunks(strand=strand, coords=True, 
                                                       sequence_names=seq_names, adapt_chunksize=True, 
                                                       parallel_factor = self.parallel_factor)
        if not softmask:
            f_chunk = f_chunk[:,:,:5]
        return f_chunk, coords, adapted_chunksize
     
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
    
    def load_inp_data(self, annot_path=None, genome_path=None,
                      chunk_coords=True, softmask=True, 
                      chunk_len=None, use_file=True, strand=None, clamsa_path=None,
                     pad=True):
        """Loads input data, encodes genome sequences, and gets reference annotations.
        
        Args:
            annot_path (str): Path to the annotation file.
            genome_path (str): Path to the genome fasta file.
            overlap_size (int, optional): Size of the overlap between chunks. Defaults to 0.
            strand (str, optional): Strand to get the sequences from. Can be '+' or '-'. Defaults to '+'.
            chunk_coords (bool, optional): If True, returns chunk coordinates.
            softmask (bool): Adds softmask track to input.
            chunk_len (int): Sequence length of training examples, if it differs from self.seq_len.
            use_file (bool): Load data from file or save it to a file
            strand (str): Indicates the strand ('+' for positive, '-' for negative). 
            clamsa_path (str): Clamsa file for additional clamsa track
        
        Returns:
            Tuple[np.ndarray, np.ndarray, ...]: Output arrays with input data, labels, and coordinates
        """
        if annot_path is None:
            annot_path = self.annot_path
        if genome_path is None:
            genome_path = self.genome_path
        if chunk_len is None:
            chunk_len = self.seq_len
        if strand is None:
            strand = self.strand
        if use_file and self.temp_dir and os.path.exists(f'{self.temp_dir}/input.npz'):
            data = np.load(f'{self.temp_dir}/input.npz')
            f_chunk = data['array1']
            r_chunk = data['array2']
            if len(data) > 2:
                coords = data['array3']
                return f_chunk, r_chunk, coords
            return f_chunk, r_chunk
        
        if (self.genome):
            fasta = GenomeSequences(genome=self.genome, chunksize=chunk_len, overlap=0)
        else:
            fasta = GenomeSequences(fasta_file=genome_path, chunksize=chunk_len, overlap=0)
        
        fasta.encode_sequences() 
        f_chunk, coords, _ = fasta.get_flat_chunks(strand=strand, coords=chunk_coords, pad=pad)
        seq_len = [len(s) for s in fasta.sequences]
        self.fasta_seq_lens = dict(zip(fasta.sequence_names, seq_len))
        # load clamsa data
        if clamsa_path:
            clamsa_chunks = []
            for seq_name, s_len in self.fasta_seq_lens.items():
                if not os.path.exists(f'{clamsa_path}{seq_name}.npy'):
                    print(f'CLAMSA PATH {clamsa_path}{seq_name}.npy does not exist!')
                clamsa_array_load = np.load(f'{clamsa_path}{seq_name}.npy')
                numb_chunks = s_len // chunk_len
                clamsa_array = clamsa_array_load[:numb_chunks*chunk_len].reshape(numb_chunks, chunk_len, 4)
                clamsa_chunks.append(clamsa_array)
                last_chunksize = clamsa_array_load.shape[0]-(numb_chunks*chunk_len)
                if pad and last_chunksize>0:
                    padding = np.zeros((1, chunk_len, 4),dtype=np.uint8)                
                    padding[0,0:last_chunksize] = clamsa_array_load[-last_chunksize:]
                    clamsa_chunks.append(padding)
            clamsa_chunks = np.concatenate(clamsa_chunks, axis=0)
            if strand == '-':
                clamsa_chunks = clamsa_chunks[::-1,::-1, [1,0,3,2]]
        if not softmask:
            f_chunk[:,:,5] = 0
        del fasta.sequences
        del fasta.one_hot_encoded
        
        if annot_path:
            ref_anno = GeneStructure(annot_path, 
                                chunksize=chunk_len, 
                                overlap=0)    

            ref_anno.translate_to_one_hot_hmm(fasta.sequence_names, 
                                    seq_len, transition=True)
        
            r_chunk = ref_anno.get_flat_chunks_hmm(
                fasta.sequence_names, strand=strand, coords=False)
            outp = [f_chunk, r_chunk, coords]
        else:
            outp = [f_chunk, np.array([]), coords]
        
        if clamsa_path:
            outp.append(clamsa_chunks)
        
        return outp
   
    def lstm_prediction(self, inp_chunks, clamsa_inp=None, trans_emb=None, save=True, batch_size=None):    
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
    
        print('### LSTM prediction')
        if save and self.temp_dir and os.path.exists(f'{self.temp_dir}/lstm_predictions.npz'):
            lstm_predictions = np.load(f'{self.temp_dir}/lstm_predictions.npz')
            lstm_predictions = lstm_predictions['array1']
            return lstm_predictions
        
        if inp_chunks.shape[0] % batch_size > 0:
            num_batches += 1
        for i in range(num_batches):
            start_pos = i * batch_size
            end_pos = (i+1) * batch_size
            if clamsa_inp is not None:
                y = self.lstm_model.predict_on_batch([
                    inp_chunks[start_pos:end_pos],
                    clamsa_inp[start_pos:end_pos]
                ])           
            else:
                y = self.lstm_model(inp_chunks[start_pos:end_pos])
            if len(y.shape) == 1:
                y = np.expand_dims(y,0)
            lstm_predictions.append(y)        
        lstm_predictions = np.concatenate(lstm_predictions, axis=0)
        if save and self.temp_dir:            
            np.savez(f'{self.temp_dir}/lstm_predictions.npz', array1=lstm_predictions)
        return lstm_predictions
    
    def hmm_prediction(self, nuc_seq, lstm_predictions, save=True, batch_size=None):
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
        print('### HMM Viterbi')
        if save and self.temp_dir and os.path.exists(f'{self.temp_dir}/hmm_predictions.npy'):
            hmm_predictions = np.load(f'{self.temp_dir}/hmm_predictions.npy')
            return hmm_predictions
        
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
            #print(len(hmm_predictions), '/', num_batches, file=sys.stderr)  
        hmm_predictions = np.concatenate(hmm_predictions, axis=0)
        if save and self.temp_dir:
            np.save(f'{self.temp_dir}/hmm_predictions.npy', hmm_predictions)
        return hmm_predictions
    
    def hmm_predictions_filtered(self, inp_chunks, lstm_predictions, save=True, batch_size=None):
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
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.

        Returns:
            HMM predictions (np.array or list of np.array): The predictions generated by the HMM model.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size
            
        print('### HMM Viterbi')
        
        def sliding_window_avg(array, window_size):
            shape = array.shape[:-2] + (array.shape[-2] - window_size + 1, window_size) + array.shape[-1:]
            strides = array.strides[:-2] + (array.strides[-2], array.strides[-2]) + array.strides[-1:]
            windows = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
            return np.nanmean(windows, axis=-2)
        
        hmm_predictions = []
        
        if save and self.temp_dir and os.path.exists(f'{self.temp_dir}/hmm_predictions.npy'):
            hmm_predictions = np.load(f'{self.temp_dir}/hmm_predictions.npy')
            return hmm_predictions
        
        if self.hmm_factor > 1:
            inp_chunks = inp_chunks.reshape((inp_chunks.shape[0]*self.hmm_factor, inp_chunks.shape[1]//self.hmm_factor, -1))
            lstm_predictions[0] = lstm_predictions[0].reshape((lstm_predictions[0].shape[0]*self.hmm_factor, lstm_predictions[0].shape[1]//self.hmm_factor, -1))
            lstm_predictions[1] = lstm_predictions[1].reshape((lstm_predictions[1].shape[0]*self.hmm_factor, lstm_predictions[1].shape[1]//self.hmm_factor, -1))
            
        batch_i = []
        hmm_predictions = np.zeros((inp_chunks.shape[0],inp_chunks.shape[1]), int)
        for i in range(inp_chunks.shape[0]):
            slide_mean = 0
            if slide_mean < 0.8:
                batch_i += [i]
            else:
                hmm_predictions[i] = lstm_predictions[0][i].argmax(-1)

            if len(batch_i) == batch_size*self.hmm_factor or i == inp_chunks.shape[0]-1:                
                y_hmm = self.predict_vit(inp_chunks[batch_i], 
                        lstm_predictions[batch_i]).numpy().squeeze()
                if len(y_hmm.shape) == 1:
                    y_hmm = np.expand_dims(y_hmm,0)
                for j1, j2 in enumerate(batch_i):
                    hmm_predictions[j2] = y_hmm[j1]
                batch_i = []
        
        if save and self.temp_dir:
            np.save(f'{self.temp_dir}/hmm_predictions.npy', hmm_predictions)
        return hmm_predictions
    
    def get_predictions(self, inp_chunks, clamsa_inp=None, hmm_filter=False, 
                save=True, encoding_layer_oracle=None, batch_size=None):
        """Gets predictions for input chunks.

        Args:
            inp_chunks (np.ndarray): Input chunks for which to get predictions.
            clamsa_inp (np.array): Optional clamsa input with same size as inp_chunks.
            hmm_filter (bool): Use faster hmm_filter method for HMM prediction.
            save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.       
            encoding_layer_oracle (bool): Can be used to skip the encoding layer and use the provided predictions. Use for debugging.     

        Returns:
            np.ndarray: HMM predictions for all chunks.
        """
        if not batch_size:
            batch_size = self.adapted_batch_size
            
        start_time = time.time()
        if encoding_layer_oracle is not None:
            encoding_layer_pred = encoding_layer_oracle
        else:
            # LSTM prediction
            encoding_layer_pred = self.lstm_prediction(inp_chunks, clamsa_inp=clamsa_inp, save=save,
                                                      batch_size=batch_size)
        
        self.lstm_pred = encoding_layer_pred
        lstm_end = time.time()
        duration = lstm_end - start_time
        print(f"LSTM took {duration/60:.4f} minutes to execute.")
        if not self.hmm:   
            encoding_layer_pred = np.argmax(encoding_layer_pred, axis=-1)
            return encoding_layer_pred
        
        if hmm_filter:
            hmm_predictions = self.hmm_predictions_filtered(inp_chunks, encoding_layer_pred, save=save,
                                                      batch_size=batch_size)
        else:
            hmm_predictions = self.hmm_prediction(inp_chunks, encoding_layer_pred, save=save,
                                                      batch_size=batch_size)
        hmm_end = time.time()
        duration = hmm_end - lstm_end
        print(f"HMM took {duration/60:.4f} minutes to execute.")
        return hmm_predictions
            
    def get_tp_fn_fp (self, predictions, true_labels):
        """Calculates true positives, false positives, 
        and false negatives for given predictions and true labels.

        Args:
            predictions (np.ndarray): Array of predicted labels.
            true_labels (np.ndarray): Array of true labels.

        Returns:
            dict: A dictionary with counts of true positives, 
                    false positives, and false negatives for each class.
        """
        classes = [0, 1, 2]
        metrics = {c: {"TP": 0, "FP": 0, "FN": 0} for c in classes + ['all']}
        for c in classes:
            tp_mask = (predictions == c) & (true_labels == c)
            fp_mask = (predictions == c) & (true_labels != c)
            fn_mask = (predictions != c) & (true_labels == c)

            metrics[c]["TP"] = np.sum(tp_mask)
            metrics[c]["FP"] = np.sum(fp_mask)
            metrics[c]["FN"] = np.sum(fn_mask)
            for k in ["TP", "FP", "FN"]:
                metrics['all'][k] += metrics[c][k]
        
        return metrics

    def calculate_metrics(self, data_dict):
        """Calculates precision, recall, and F1 score from 
        a dictionary of TP, FP, and FN values.

        Args:
            data_dict (dict): A dictionary containing 
                'TP', 'FP', 'FN' keys with their counts as values.

        Returns:
            dict: A dictionary containing 'precision', 
                'recall', and 'F1' keys with their calculated values.
        """
        metrics_dict = {}

        for key, values in data_dict.items():
            tp = values['TP']
            fp = values['FP']
            fn = values['FN']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_dict[key] = {
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
        return metrics_dict

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
            y_vit = self.gene_pred_hmm_layer.viterbi(new_y_lstm, nucleotides=nuc)
        else:
            nuc = tf.cast(x[:,:,:5], tf.float32)
            y_vit = self.gene_pred_hmm_layer.viterbi(y_lstm, nucleotides=nuc)
        return y_vit
        
    def merge_re_prediction(self, all_tx, new_tx, breakpoint): 
        """Merges two sets of transcript predictions (`all_tx` and `new_tx`) at a specified breakpoint.

        This function integrates predictions from two different prediction sets by considering their overlaps and the
        specified breakpoint. It aims to create a combined prediction that respects the continuity of transcripts across
        the breakpoint, favoring the retention of longer transcripts or more accurate predictions based on the overlap
        analysis.

        Arguments:
            all_tx (list of tuples): The list of all current transcript predictions before the breakpoint. Each element in
                                       the list is a tuple representing a transcript with its start and end positions.
            new_tx (list of tuples): The list of new transcript predictions that may overlap with `all_tx` at the breakpoint.
            breakpoint (int): The position in the sequence where the division between the old and new predictions is made.

        Returns:
            list of tuples: The merged list of transcript predictions, considering the breakpoint and overlaps between
                          `all_tx` and `new_tx`.

        The merging process follows these rules:
        - If one of the prediction sets is empty, it returns the concatenation of both.
        - If the breakpoint is in the intergenic region (outside the range of any transcripts in both sets), it merges
          the predictions without overlapping transcripts.
        - If the breakpoint indicates overlapping regions but no direct overlap between transcripts, it concatenates
          the predictions up to and from the breakpoint.
        - If there's an overlap and one of the transcripts surrounding the breakpoint is larger, the larger transcript
          is preferred in the merged output.
        """
        # print(all_tx, new_tx, breakpoint)
        overlap1 = 0
        for i, tx in enumerate(all_tx):
            if breakpoint < tx[0][1]:
                break
            overlap1 = i
        overlap2 = 0
        for i, tx in enumerate(new_tx):
            if breakpoint < tx[0][1]:
                break
            overlap2 = i

        if not all_tx or not new_tx:
            # no tx in one of the predictions
            return all_tx + new_tx
        elif breakpoint > all_tx[overlap1][-1][2] and breakpoint > new_tx[overlap2][-1][2]:
            # breakpoint already in intergenic region of both sets
            return all_tx[:overlap1+1] + new_tx[overlap2+1:]
        elif all_tx[overlap1][-1][2] < new_tx[overlap2][0][1]:
            # breakpoint in one of the transcripts but they don't overlap
            return all_tx[:overlap1+1] + new_tx[overlap2:]
        elif all_tx[overlap1][-1][2] - all_tx[overlap1][0][1] > new_tx[overlap2][-1][2] - new_tx[overlap2][0][1]:
            # tx from all_tx is larger so keep it instead of the one from new_tx
            return all_tx[:overlap1+1] + new_tx[overlap2+1:]
        else:
            # tx from new_tx is larger so keep it instead of the one from all_tx
            return all_tx[:overlap1] + new_tx[overlap2:]
        
    def get_tx_from_range(self, range_):
        """
        Extracts transcript regions from a given tuples of class ranges. It extracts for each transcript
        the regions of their CDS. Additionally, it reports fragmented txs add the start or end of the 
        input ranges.

        Parameters:
        - range_ (list of tuples/lists): A sequence of regions, where each region is represented as a tuple or list.
                                         The first element of each tuple/list indicates the type of the region ('intergenic'
                                         or otherwise), and the subsequent elements provide additional details about the region.

        Returns:
        - initial_tx (list): List of exon ranges of the first fragmented transcript.
        - txs (list of lists): List of transcripts with their CDS ranges.
        - current_tx (list): Last fragmented Transcript
        """        
        
        txs = []
        current_tx = []
        initial_tx = []

        for region in range_:
            if region[0] == 'intergenic':
                if current_tx:
                    txs.append(current_tx)
                    current_tx = []
            else:
                current_tx.append(region)                
        if range_[0][0] != 'intergenic' and txs:
            initial_tx = txs[0]
            txs = txs[1:]
        return initial_tx, txs, current_tx
        
    def create_gtf(self, y_label, coords, f_chunks, out_file='', clamsa_inp=None, 
                   strand='+', correct_y_label=None, anno=None, tx_id=0,
                  filt=True):
        """Create a GTF file with the gene annotations from predictions.
        
        This method translates HMM predictions into a GTF format which contains the annotations
        of the genomic features such as CDS (Coding Sequences) and introns. For regions where 
        predictions from consecutive chunks don't align (or are both IR) the method performs 
        re-predictions. This is done by concatenating the input data of the misaligned chunks 
        and generating a new prediction for this combined segment, aiming for a consistent 
        annotation across breakpoints.
        
        Args:
            y_label (np.ndarray): The array of encoded labels predicted by the model.
            coords (np.ndarray): The array of genomic coordinates: [seq_name, strand, chunk_start, chunk_end].
            out_file (str): Path to the output GTF file to which the annotations will be written.
            f_chunks (np.array): One hot encoded nucleotide sequence.
            clamsa_inp (np.array): Optional clamsa input with same size as inp_chunks.
            correct_y_label (np.array): Correct y_label for debugging.
        """
        batch_size = self.adapted_batch_size//2

        # revert data if - strand
        if strand == '-':
            y_label = y_label[::-1, ::-1]
            f_chunks = f_chunks[::-1]
            coords = coords[::-1]
        
        # coordinates of ranges connected gene features in y_label
        ranges = {}
        # Anno object for gtf file
        if not anno:
            anno = Anno(out_file, f'anno')        
        
        re_pred_inp = []
        re_clamsa_inp = []
        # index of previous element in coords]
        re_pred_index = []
        re_correct_y_label = []
        
        for i in range(y_label.shape[0]-1):
            # add overlap of i-th and i+1-th chunk to repred 
            # if they are from the same seq and if at least 
            # one of the borders is not IR
            if coords[i][0] == coords[i+1][0] \
                and not (y_label[i,-1] ==  y_label[i+1,0]):
                    if coords[i][1] == '+':
                        re_pred_inp.append(
                            np.concatenate([f_chunks[i],
                                            f_chunks[i+1]], axis=0))
                        if clamsa_inp is not None:
                            re_clamsa_inp.append(
                                np.concatenate([clamsa_inp[i],
                                                clamsa_inp[i+1]], axis=0))
                        if correct_y_label is not None:
                            re_correct_y_label.append(
                                np.concatenate([correct_y_label[i],
                                                correct_y_label[i+1]], axis=0))
                    else:
                        re_pred_inp.append(
                            np.concatenate([f_chunks[i+1],
                                            f_chunks[i]], axis=0))
                        if clamsa_inp is not None:
                            re_clamsa_inp.append(
                                np.concatenate([clamsa_inp[i+1],
                                                clamsa_inp[i]], axis=0))
                        if correct_y_label is not None:
                            re_correct_y_label.append(
                                np.concatenate([correct_y_label[i+1],
                                                correct_y_label[i]], axis=0))
                    re_pred_index.append(i)                        
        
        re_pred_inp = np.array(re_pred_inp) 
        re_clamsa_inp = np.array(re_clamsa_inp)
        re_correct_y_label = np.array(re_correct_y_label)
        # get new predictions for overlap regions
        if re_pred_inp.any():
            if clamsa_inp is not None:
                re_pred = self.get_predictions(re_pred_inp, 
                                               clamsa_inp=re_clamsa_inp, save=False, batch_size=batch_size,
                                               encoding_layer_oracle=re_correct_y_label if correct_y_label is not None else None)
            else:
                re_pred = self.get_predictions(re_pred_inp, save=False, batch_size=batch_size,
                                               encoding_layer_oracle=re_correct_y_label if correct_y_label is not None else None)
        
        current_re_index = -1
        re_txs = None
        end_fragment = []
        
        for i, (y, c) in enumerate(zip(y_label, coords)):
            y_ranges = self.get_ranges(y, c[2])
            is_ir = 'intergenic' in [r[0] for r in y_ranges]       
            coord_diff = 0 if i == 0 else int(c[3])-int(coords[i-1][2])
            start_fragment, txs, new_end_fragment = self.get_tx_from_range(y_ranges)
            
            # new seq
            if c[0] not in ranges:
                ranges[c[0]] = []
            
            # if the start of the first fragmented tx matches the fragment from the last chunk
            # combine them
            if not re_txs and is_ir and end_fragment and start_fragment and y_label[i-1,-1] == y_label[i,0]:
                end_fragment[-1][2] = start_fragment[0][2]
                end_fragment += start_fragment[1:]
                ranges[c[0]] += [end_fragment]
            if is_ir and txs:
                if re_txs:
                    ranges[c[0]] = self.merge_re_prediction(ranges[c[0]], txs, c[2] + coord_diff//2)
                else:
                    ranges[c[0]] += txs
            if is_ir:
                end_fragment = new_end_fragment
            re_txs = None
            if re_pred_index and i == re_pred_index[0]:
                re_pred_index.pop(0)
                c_re = c[:2] + [c[2], c[3] + coord_diff]
                current_re = re_pred[0]
                re_pred = re_pred[1:]
                if c_re[1] == '-':
                    current_re = current_re[::-1]  
                re_ranges = self.get_ranges(current_re, c_re[2])       
                start_fragment, re_txs, new_end_fragment = self.get_tx_from_range(re_ranges)
                if not is_ir and end_fragment and start_fragment \
                    and y_label[i-1,-1] == current_re[0]:
                    end_fragment[-1][2] = start_fragment[0][2]
                    end_fragment += start_fragment[1:]
                    ranges[c[0]] += [end_fragment]
                if re_txs:
                    ranges[c[0]] = self.merge_re_prediction(
                        ranges[c[0]], re_txs, c_re[2] + coord_diff//2)                
                
                end_fragment = new_end_fragment
            
        for seq in ranges:
            new_tx = False
            phase = -1
            for tx in ranges[seq]:                
                tx_id += 1
                t_id = f'g{tx_id}.t1'
                g_id = f'g{tx_id}'
                phase = 0
                anno.transcript_update(t_id, g_id, seq, strand)
                anno.genes_update(g_id, t_id)  
                for r in tx:
                    line = [seq, 'Tiberius', r[0], r[1], r[2], '.', strand, phase, 
                           f'gene_id "{g_id}"; transcript_id "{t_id}";']
                    anno.transcripts[t_id].add_line(line)
                    if r[0] == 'CDS':
                        phase = (3 - (r[2] - r[1] + 1 - phase)%3)%3
        
        remove_tx = []                
        for tx in anno.transcripts.values():
            tx.check_splits()
            if filt and tx.get_cds_len() < 201:
                remove_tx.append(tx.id)
            else:                
                tx.redo_phase()
            
        for tx in remove_tx:
            anno.transcripts.pop(tx)

        if out_file:
            anno.norm_tx_format()
            anno.find_genes()
            anno.write_anno(out_file)
            
        return anno, tx_id
    
    def create_gtf_single_batch(self, nuc_seq, lstm_predictions, coords, out_file, strand='+'):        
        anno = Anno(out_file, f'anno')
        tx_id = 0
        inp_nuc = [[nuc_seq[0]]]
        inp_lstm = [[lstm_predictions[0]]]
        seq_names = [coords[0][0]]
        
        for i in range(1, nuc_seq.shape[0]-1):
            if not coords[i][0] == coords[i-1][0]:
                inp_nuc.append([])
                inp_lstm.append([])
                seq_names.append(coords[i][0])
            inp_nuc[-1].append(nuc_seq[i])
            inp_lstm[-1].append(lstm_predictions[i])
            
        for i in range(len(inp_nuc)):
            y_hmm = self.predict_vit(
                    np.expand_dims(np.concatenate(inp_nuc[i], axis=0), 0),
                    np.expand_dims(np.concatenate(inp_lstm[i], axis=0), 0),
                    ).numpy().squeeze()
            y_ranges = self.get_ranges(y_hmm, 1)
            start_fragment, txs, end_fragment = self.get_tx_from_range(y_ranges)            
            phase = -1
            for tx in txs:                
                tx_id += 1
                t_id = f'g{tx_id}.t1'
                g_id = f'g{tx_id}'
                phase = 0
                anno.transcript_update(t_id, g_id, seq_names[i], strand)
                anno.genes_update(g_id, t_id)  
                for r in tx:
                    line = [seq_names[i], 'Tiberius', r[0], r[1], r[2], '.', strand, phase, 
                           f'gene_id "{g_id}"; transcript_id "{t_id}";']
                    anno.transcripts[t_id].add_line(line)
                    if r[0] == 'CDS':
                        phase = (3 - (r[2] - r[1] + 1 - phase)%3)%3            
            
        remove_tx = []                
        for tx in anno.transcripts.values():
            if 'CDS' not in tx.transcript_lines:
                continue
            tx.check_splits()
            tx.redo_phase()
            if tx.get_cds_len() < 201:
                remove_tx.append(tx.id)

        for tx in remove_tx:
            anno.transcripts.pop(tx)

        anno.norm_tx_format()
        anno.find_genes()
        anno.write_anno(out_file)
    
    
    def get_ranges(self, encoded_labels, offset=0):
        """Obtain the genomic feature ranges from encoded labels.
        
        This method processes an array of encoded labels to identify continuous ranges of the same label
        and categorizes them into genomic features such as intergenic regions, introns, and CDS.
        
        Args:
            encoded_labels (Iterable[int]): Encoded labels representing genomic features.
            
        Returns:
            List[Tuple[str, int, int]]: A list of tuples where each tuple contains the feature type
            as a string and the start and end points as integers.
        """
        arr = np.array(encoded_labels)
        
#         if arr.max() > 3:
        arr = self.reduce_label(arr, self.num_hmm)
            
        # Find where the array changes
        change_points = np.where(np.diff(arr) != 0)[0]        

        # Start points are one position after each change point
        start_points = np.insert(change_points + 1, 0, 0)

        # End points are the change points
        end_points = np.append(change_points, arr.size - 1)

        features = ["intergenic", "intron", "CDS"]
        ranges = [[features[arr[start]], start+offset, end+offset] for start, end in zip(start_points, end_points)]

        return ranges


    def make_default_hmm(self, inp_size=5):
        self.gene_pred_hmm_layer = GenePredHMMLayer(
            parallel_factor=self.parallel_factor
        )
        self.gene_pred_hmm_layer.build([self.adapted_batch_size, self.seq_len, inp_size])
