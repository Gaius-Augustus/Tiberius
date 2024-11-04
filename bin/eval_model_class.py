# ==============================================================
# Authors: Lars Gabriel
#
# Class handling the prediction and evaluation for a single species
# 
# 
# ==============================================================

import sys, json, os, re, sys, csv, time
from genome_fasta import GenomeSequences
from annotation_gtf import GeneStructure
import subprocess as sp
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from learnMSA.msa_hmm.Viterbi import viterbi
from tensorflow.keras.models import Model
from genome_anno import Anno
from models import custom_cce_f1_loss, lstm_model
from gene_pred_hmm import class3_emission_matrix, GenePredHMMLayer, make_5_class_emission_kernel, make_aggregation_matrix, make_15_class_emission_kernel
from learnMSA.msa_hmm.Initializers import ConstantInitializer
# from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFEsmForMaskedLM
from tensorflow.keras.layers import (Conv1D, SimpleRNN, Conv1DTranspose, LSTM, GRU, Dense, Bidirectional, Dropout, Activation, Input, BatchNormalization, LSTM, Reshape, Embedding, Add, LayerNormalization,
                                    AveragePooling1D)

class PredictionGTF:
    """Class for generating GTF predictions based on a model's output.
    """
    """
        Attributes:
            model_path (str): Path to the pre-trained model.
            seq_len (int): Length of the sequences to process.
            batch_size (int): Batch size for prediction.        
            hmm (bool): Flag to indicate whether to use HMM for prediction.
            model (keras.Model): Loaded Keras model for predictions.
            model_path_lstm (str): Path to the LSTM model if HMM is used.
        """
    def __init__(self, model_path='', seq_len=500004, batch_size=200, 
                 hmm=False, model_path_lstm='', model_path_hmm='', 
                 temp_dir='', emb=False, num_hmm=1,
                 hmm_factor=None, 
                 # transformer=False, trans_lstm=False, 
                 annot_path='', genome_path='', softmask=True,
                strand='+', parallel_factor=1, oracle=False,
                lstm_cfg='',):
        """
        Arguments:
            - model_path (str): Path to the main model file that includes a HMM layer.
            - seq_len (int): The sequence length to be used for prediction.
            - batch_size (int): The size of the batches to be used.
            - hmm (bool): A flag to indicate whether Hidden Markov Models (HMM) should be used. Defaults to False.
            - model_path_lstm (str): Path to the LSTM model file. A default HMM will then be used except when model_path_hmm is provided.
            - model_path_hmm (str): Path to the HMM model file.
            - temp_dir (str): Temporary directory path for intermediate files. 
            - emb (bool): A flag to indicate whether HMM embedding should be used. 
            - num_hmm (int): Number of HMMs to be used.
            - hmm_factor: Parallelization factor of HMM (deprecated, remove in a later version)
            - transformer (bool): A flag to indicate whether a transformer model should be used. (depprecated!)
            - trans_lstm (bool): A flag indicating whether a transform-LSTM hybrid model should be used. (depprecated!)
            - annot_path (str): Path to the reference annotation file (GTF).
            - genome_path (str): Path to the genome file (FASTA). 
            - softmask (bool): Whether to use softmasking. 
            - strand (str): Indicates the strand ('+' for positive, '-' for negative).
            - parallel_factor (int): The parallel factor used for Viterbi.
            - lstm_cfg (str): path to lstm cfg to load weights instead of the whole model
        """
        self.model_path = model_path 
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.annot_path = annot_path
        self.genome_path = genome_path
        self.softmask = softmask
        self.hmm = hmm   
        self.emb = emb
        self.strand=strand
        # self.transformer = transformer
        self.model = None
        self.model_path_lstm = model_path_lstm
        self.model_path_hmm = model_path_hmm
        self.fasta_seq_lens = {}
        self.num_hmm = num_hmm
        self.hmm_factor = hmm_factor
        # self.trans_lstm = trans_lstm
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
        if self.hmm and self.model_path_lstm:
            # only the lstm model is provided, use the default HMM Layer
            # if self.transformer or self.trans_lstm:
            #     # lstm model includes transformer
            #     lstm_model_full = keras.models.load_model(self.model_path_lstm, 
            #                               custom_objects={'TFEsmForMaskedLM': TFEsmForMaskedLM})
            #     self.trans_model = self.transformer_model(self.seq_len, lstm_model_full)
            #     self.lstm_model = self.trans_lstm_model(lstm_model_full)
            # elif self.lstm_cfg:
            if self.lstm_cfg:
                 with open(self.lstm_cfg, 'r') as f:
                    config = json.load(f)
                 relevant_keys = ['units', 'filter_size', 'kernel_size', 
                     'numb_conv', 'numb_lstm', 'dropout_rate', 
                     'pool_size', 'stride', 'lstm_mask', 'clamsa',
                     'output_size', 'residual_conv', 'softmasking',
                    'clamsa_kernel', 'lru_layer']
                 relevant_args = {key: config[key] for key in relevant_keys if key in config}
                 self.lstm_model = lstm_model(**relevant_args)
                 self.lstm_model.load_weights(self.model_path_lstm + '/variables/variables')
            else:
                self.lstm_model = keras.models.load_model(self.model_path_lstm, 
                                          custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.batch_size),
                                         'loss_': custom_cce_f1_loss(2, self.batch_size)})
            if self.model_path_hmm:
                model_hmm = keras.models.load_model(self.model_path_hmm, 
                                                    custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.batch_size),
                                                            'loss_': custom_cce_f1_loss(2, self.batch_size)})
                self.gene_pred_hmm_layer = model_hmm.get_layer('gene_pred_hmm_layer')
                self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
                self.gene_pred_hmm_layer.cell.recurrent_init() 
            else:
                self.make_default_hmm(inp_size=self.lstm_model.output.shape[-1])
                # self.make_default_hmm()
            if summary:
                self.lstm_model.summary()
        elif self.model_path_lstm:
            self.lstm_model = keras.models.load_model(self.model_path_lstm, 
                                                        custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.batch_size),
                                                         'loss_': custom_cce_f1_loss(2, self.batch_size)})
        elif self.model_path and self.emb:
            # load LSTM+HMM model and extract LSTM part
            self.model = keras.models.load_model(self.model_path,
                                                custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.batch_size),
                                                            'loss_': custom_cce_f1_loss(2, self.batch_size)})
            if summary:
                self.model.summary()
            # if len(self.model.input) == 2:
            #     inp = self.model.input[0]
            # else:
            #     inp = self.model.input
            self.lstm_model = Model(
                    inputs=self.model.input, 
                    outputs=[self.model.get_layer('lstm_out').output,
                            self.model.get_layer('layer_normalization_hmm').output]
                    )
            self.gene_pred_hmm_layer = self.model.get_layer('gene_pred_hmm_layer')
            if self.parallel_factor is not None:
                self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
            self.gene_pred_hmm_layer.cell.recurrent_init()            
        elif self.model_path:
            if False:
                self.model = clamsa_only_model()
                self.model.load_weights(self.model_path+"/variables/variables", 
                                    custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.batch_size),
                                        'loss_': custom_cce_f1_loss(2, self.batch_size)})
                if self.hmm:
                    self.lstm_model = Model(
                                    inputs=self.model.input, 
                                    outputs=self.model.get_layer('lstm_out').output
                                    )
                    self.gene_pred_hmm_layer = self.model.get_layer('gene_pred_hmm_layer')

                    if self.parallel_factor is not None:
                        self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
                    print(f"Running gene pred hmm layer with parallel factor {self.gene_pred_hmm_layer.parallel_factor}")

                    self.gene_pred_hmm_layer.cell.recurrent_init()
            if True:
                self.model = keras.models.load_model(self.model_path, 
                                        custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, self.batch_size),
                                            'loss_': custom_cce_f1_loss(2, self.batch_size)})
                
                if self.hmm:
                    try:
                        lstm_output=self.model.get_layer('out').output
                    except ValueError as e:
                        lstm_output=self.model.get_layer('lstm_out').output
                    self.lstm_model = Model(
                                    inputs=self.model.input, 
                                    outputs=lstm_output
                                    )
                    # self.lstm_model.compile(run_eagerly=True)
                    # self.gene_pred_hmm_layer = self.model.get_layer('gene_pred_hmm_layer')
                    self.gene_pred_hmm_layer = self.model.layers[-1]

                    if self.parallel_factor is not None:
                        self.gene_pred_hmm_layer.parallel_factor = self.parallel_factor
                    print(f"Running gene pred hmm layer with parallel factor {self.gene_pred_hmm_layer.parallel_factor}")

                    self.gene_pred_hmm_layer.cell.recurrent_init()
            if summary:
                self.model.summary()
        else: 
            self.make_default_hmm()
    
    # def transformer_model(self, seq_len, lstm_load):
    #     input_ids = Input(shape=(None,), dtype='int32', name='input_ids')
    #     attention_mask = Input(shape=(None,), dtype='int32', name='attention_mask')

    #     transformer_model = TFEsmForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    #     trans_out = transformer_model(input_ids, 
    #                               attention_mask=attention_mask, 
    #                               output_hidden_states=True)
    #     trans_out = trans_out['hidden_states'][-1][:,1:]   
    #     trans_out = lstm_load.get_layer('dense_transformer')(trans_out) 
    #     # ouput size: batch_size, 918, 600
    #     model = Model(inputs=[input_ids, attention_mask], outputs=trans_out)
    #     return model

    # def trans_lstm_model(self, lstm_load):
    #     nuc_input = Input(shape=(None, 6), name='main_input')
    #     trans_input = Input(shape=(None, 134), name='trans_emb')
        
    #     x = lstm_load.get_layer('initial_conv')(nuc_input) 
    #     x = lstm_load.get_layer('batch_normalization1')(x) 
    #     x = lstm_load.get_layer('conv_1')(x) 
    #     x = lstm_load.get_layer('batch_normalization2')(x) 
    #     x = lstm_load.get_layer('conv_2')(x) 
    #     x = tf.concat([nuc_input, x], axis=-1)
    #     x = tf.keras.layers.Add()([x, trans_input])
    #     x = lstm_load.get_layer('R1')(x) 
    #     x = lstm_load.get_layer('biLSTM_1')(x) 
    #     x = lstm_load.get_layer('biLSTM_2')(x) 
    #     x = lstm_load.get_layer('dense')(x) 
    #     x = lstm_load.get_layer('Reshape2')(x) 
    #     x = lstm_load.get_layer('out')(x) 
    #     x = tf.concat([
    #         x[:,:,0:1], 
    #         tf.reduce_sum(x[:,:,1:4], axis=-1, keepdims=True, name='reduce_inp_introns'), 
    #         x[:,:,4:]
    #         ], 
    #         axis=-1, name='concat_outp')
    #     return Model(inputs=[nuc_input, trans_input], outputs=x)
    
    def init_fasta(self,  genome_path=None, chunk_len=None, strand=None):
        if genome_path is None:
            genome_path = self.genome_path
        if chunk_len is None:
            chunk_len = self.seq_len
        
            
        fasta = GenomeSequences(fasta_file=genome_path,
            chunksize=chunk_len, 
            overlap=0)
        return fasta
    
    def load_genome_data(self, fasta_object, seq_names, strand='', softmask=True):
        if strand is None:
            strand = self.strand
            
        fasta_object.encode_sequences(seq=seq_names) 
        f_chunk, coords = fasta_object.get_flat_chunks(strand=strand, coords=True, 
                                                       sequence_name=seq_names)
        if not softmask:
            f_chunk = f_chunk[:,:,:5]
        return f_chunk, coords
     
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
        
        fasta = GenomeSequences(fasta_file=genome_path,
            chunksize=chunk_len, 
            overlap=0)
        
        fasta.encode_sequences() 
        f_chunk, coords = fasta.get_flat_chunks(strand=strand, coords=chunk_coords, pad=pad)
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
    
    # def tokenize_inp(self, inp_chunks):
    #     """Tokenizes input sequences for nucleotide transformer.

    #     Arguments:
    #         - inp_chunks (np.array): A numpy array of one-hot encoded nucleotide sequences.
            

    #         Returns:
    #         - tokens (BatchEncoding): Dictionary with input_ids and attention masks of tokens
    #     """
    #     token_len = inp_chunks.shape[1]
    #     # cutoff = ((inp_chunks.shape[0] * token_len) // seq_len) * seq_len // token_len
    #     # inp_chunks = inp_chunks[:cutoff]
    #     def decode_one_hot(encoded_seq):
    #         # Define the mapping from index to nucleotide
    #         index_to_nucleotide = np.array(['A', 'C', 'G', 'T', 'A'])
    #         # Use np.argmax to find the index of the maximum value in each row
    #         nucleotide_indices = np.argmax(encoded_seq, axis=-1)
    #         # Map indices to nucleotides
    #         decoded_seq = index_to_nucleotide[nucleotide_indices]
    #         # Convert from array of characters to string for each sequence
    #         decoded_seq_str = [''.join(seq) for seq in decoded_seq]
    #         return decoded_seq_str

    #     tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    #     tokens = decode_one_hot(inp_chunks[:,:,:5])
    #     tokens = tokenizer.batch_encode_plus(tokens, return_tensors="tf", 
    #                                           padding="max_length",
    #                                           max_length=token_len//6+1)
    #     return tokens
    
    # def transformer_prediction(self, inp_ids, attention_mask, save=True):
    #     """Generates predictions using the transformer only model based on input IDs and attention masks.

    #     Arguments:
    #         inp_ids (np.array): The input IDs for the transformer model, expected to be in a numpy array format.
    #         attention_mask (np.array): The attention mask for the transformer model, aligned with `inp_ids`.
    #         save (bool): A flag to indicate whether the predictions should be saved/loaded to/from a file.

    #     Returns:
    #         trans_predictions (np.array or list of np.array): The predictions generated by the transformer model.
    #     """
    #     trans_predictions = []
    #     batch_size = 30
    #     num_batches = inp_ids.shape[0] // batch_size
    #     if save and self.temp_dir and os.path.exists(f'{self.temp_dir}/../../../inp/trans_predictions.npz'):
    #         trans_predictions = np.load(f'{self.temp_dir}/../../../inp/trans_predictions.npz')            
    #         if self.emb:
    #             trans_predictions = [trans_predictions['array1'],  trans_predictions['array2']]
    #         else:
    #             trans_predictions = trans_predictions['array1']
    #         return trans_predictions

    #     print('### Transformer prediction')
    #     if inp_ids.shape[0] % batch_size > 0:
    #         num_batches += 1
    #     for i in range(num_batches):
    #         start_pos = i * batch_size
    #         end_pos = (i+1) * batch_size
    #         y = self.trans_model.predict_on_batch((inp_ids[start_pos:end_pos],
    # attention_mask[start_pos:end_pos]))                
    #         if len(y.shape) == 1:
    #             y = np.expand_dims(y,0)            
    #         trans_predictions.append(y)
    #         print(len(trans_predictions)+1, '/', num_batches, file=sys.stderr)
    #     trans_predictions = np.concatenate(trans_predictions, axis=0)  
    #     if save and self.temp_dir:
    #         if self.emb:
    #             np.savez(f'{self.temp_dir}/trans_predictions.npz', array1=trans_predictions[0], array2=trans_predictions[1])
    #         else:
    #             np.savez(f'{self.temp_dir}/trans_predictions.npz', array1=trans_predictions)  
    #     return trans_predictions
   
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
            batch_size = self.batch_size
        num_batches = inp_chunks.shape[0] // batch_size
        lstm_predictions = []
    
        print('### LSTM prediction')
        if save and self.temp_dir and os.path.exists(f'{self.temp_dir}/lstm_predictions.npz'):
            lstm_predictions = np.load(f'{self.temp_dir}/lstm_predictions.npz')
            if self.emb:
                lstm_predictions = [lstm_predictions['array1'],  lstm_predictions['array2']]
            else:
                lstm_predictions = lstm_predictions['array1']
            return lstm_predictions
        
        if inp_chunks.shape[0] % batch_size > 0:
            num_batches += 1
        for i in range(num_batches):
            start_pos = i * batch_size
            end_pos = (i+1) * batch_size
            # if self.trans_lstm:
            #     y = self.lstm_model(
            #         (inp_chunks[start_pos:end_pos],
            #         trans_emb[start_pos:end_pos]))
            if clamsa_inp is not None:
                y = self.lstm_model.predict_on_batch([
                    inp_chunks[start_pos:end_pos],
                    clamsa_inp[start_pos:end_pos]
                ])           
            else:
                # print(start_pos,end_pos, len(inp_chunks), inp_chunks[start_pos].shape)
                # y = self.lstm_model.predict_on_batch(inp_chunks[start_pos:end_pos])
                y = self.lstm_model(inp_chunks[start_pos:end_pos])
            if not self.emb and len(y.shape) == 1:
                y = np.expand_dims(y,0)
            elif self.emb and len(y[0].shape) == 1:
                y[0] = np.expand_dims(y[0],0)
                y[1] = np.expand_dims(y[1],0)
            lstm_predictions.append(y)
            #print(len(lstm_predictions), '/', num_batches, file=sys.stderr) 
        if self.emb:
            lstm_predictions = [np.concatenate([l[0] for l in lstm_predictions], axis=0),
                               np.concatenate([l[1] for l in lstm_predictions], axis=0)]
        else:
            lstm_predictions = np.concatenate(lstm_predictions, axis=0)
        if save and self.temp_dir:
            if self.emb:
                np.savez(f'{self.temp_dir}/lstm_predictions.npz', array1=lstm_predictions[0], array2=lstm_predictions[1])
            else:
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
            batch_size = self.batch_size
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
            if self.emb:
                y_hmm = self.predict_vit(nuc_seq[start_pos:end_pos], 
                    [lstm_predictions[0][start_pos:end_pos], 
                     lstm_predictions[1][start_pos:end_pos]]
                           ).numpy().squeeze()
            else:
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
            batch_size = self.batch_size
            
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
            if self.emb:
                slide_mean = sliding_window_avg(lstm_predictions[0][i], 200)[:,0].min()
            if slide_mean < 0.8:
                batch_i += [i]
            else:
                hmm_predictions[i] = lstm_predictions[0][i].argmax(-1)

            if len(batch_i) == batch_size*self.hmm_factor or i == inp_chunks.shape[0]-1:
                #print(i, '/', inp_chunks.shape[0], file=sys.stderr)  
                if self.emb:
                    y_hmm = self.predict_vit(inp_chunks[batch_i], 
                            [lstm_predictions[0][batch_i], 
                             lstm_predictions[1][batch_i]], border_hints=False
                                   ).numpy().squeeze()
                else:
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
    
    def get_predictions(self, inp_chunks, clamsa_inp=None, hmm_filter=False, save=True, encoding_layer_oracle=None, batch_size=None):
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
            batch_size = self.batch_size
            
        start_time = time.time()
        if encoding_layer_oracle is not None:
            encoding_layer_pred = encoding_layer_oracle
        # elif self.transformer:
        #     # NT prediction
        #     tokens = self.tokenize_inp(
        #         inp_chunks.reshape(-1, 5994), seq_len=inp_chunks.shape[1])
        #     encoding_layer_pred = self.transformer_prediction(tokens['input_ids'], 
        #                                                       tokens['attention_mask'],save=save)
        # elif self.trans_lstm:
        #     tokens = self.tokenize_inp(
        #         inp_chunks.reshape(-1, 5502), seq_len=inp_chunks.shape[1])
        #     trans_emb = self.transformer_prediction(tokens['input_ids'], tokens['attention_mask'], 
        #                                             save=save)
        #     trans_emb = trans_emb.reshape((-1, trans_emb.shape[1]*6, 134))
        #     trans_emb = trans_emb.reshape((-1, inp_chunks.shape[1], 134))
        #     encoding_layer_pred = self.lstm_prediction(inp_chunks, trans_emb=trans_emb, save=save,
        #                                               batch_size=batch_size)
        else:
            # LSTM prediction
            encoding_layer_pred = self.lstm_prediction(inp_chunks, clamsa_inp=clamsa_inp, save=save,
                                                      batch_size=batch_size)   
        
        self.lstm_pred = encoding_layer_pred
        lstm_end = time.time()
        duration = lstm_end - start_time
        print(f"LSTM took {duration/60} minutes to execute.")
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
        print(f"HMM took {duration/60} minutes to execute.")
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
    def predict_vit(self, x, y_lstm, border_hints=False):
        """Perform prediction using the Viterbi algorithm on the output of an LSTM model.
        
        This method applies the Viterbi algorithm to the sequence probabilities output by
        the LSTM model to find the most likely sequence of hidden states.
        
        Args:
            x (tf.Tensor): Input sequence tensor for which the predictions are to be made.
            y_lstm (np.array): LSTM predictions used as input for viterbi.
            
        Returns:
            tf.Tensor: The predicted state sequence tensor after applying Viterbi decoding.
        """
        # print(x.shape, y_lstm.shape)
        if self.emb and border_hints:
            if y_lstm[0].shape[-1] > 5:
                class_emb = tf.concat([y_lstm[0][:,:,0:1], 
                        tf.reduce_sum(y_lstm[0][:,:,1:4], axis=-1, keepdims=True), 
                        y_lstm[0][:,:,4:]], axis=-1)
            else:
                class_emb = y_lstm[0]
            nuc = tf.cast(x[:,:,:5], tf.float32)
            A = make_aggregation_matrix(k=self.num_hmm)
            hints = tf.stack([class_emb[:,0,:],class_emb[:,-1,:]], axis=1)
            hints = tf.matmul(hints, A, transpose_b=True)
            y_vit = self.gene_pred_hmm_layer.viterbi(class_emb, nucleotides=nuc,
                                    embeddings=y_lstm[1],
                                    end_hints=hints)
        elif self.emb:
            if y_lstm[0].shape[-1] > 5 and self.gene_pred_hmm_layer.share_intron_parameters:
                class_emb = tf.concat([y_lstm[0][:,:,0:1],
                        tf.reduce_sum(y_lstm[0][:,:,1:4], axis=-1, keepdims=True),
                        y_lstm[0][:,:,4:]], axis=-1)
            else:
                class_emb = y_lstm[0]
            nuc = tf.cast(x[:,:,:5], tf.float32)
            y_vit = self.gene_pred_hmm_layer.viterbi(class_emb, nucleotides=nuc,
                                                     embeddings=y_lstm[1])
        elif self.lstm_model and self.hmm:
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
            # print(new_y_lstm.shape)
            nuc = tf.cast(x[:,:,:5], tf.float32)
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
                   strand='+', border_hints=False, correct_y_label=None, anno=None, tx_id=0,
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
        batch_size = self.batch_size//2

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
#                         re_pred_inp.append(
#                             np.concatenate([f_chunks[i, self.seq_len//2:],
#                                             f_chunks[i+1, :self.seq_len//2]], axis=1))
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
            
#             print(c, y_label[i-1, -1], y_label[i, 0])
            
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
            # print(len(inp_nuc[i]))
            y_hmm = self.predict_vit(
                    np.expand_dims(np.concatenate(inp_nuc[i], axis=0), 0),
                    np.expand_dims(np.concatenate(inp_lstm[i], axis=0), 0),
                    ).numpy().squeeze()
            y_ranges = self.get_ranges(y_hmm, 1)
            start_fragment, txs, end_fragment = self.get_tx_from_range(y_ranges)
            #print(start_fragment, end_fragment)
            
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
                #print(tx.transcript_lines)
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
        if inp_size == 5:
            em_kernel = make_5_class_emission_kernel(smoothing=1e-6)
        elif inp_size == 15:
            em_kernel = make_15_class_emission_kernel(smoothing=1e-6)
        self.gene_pred_hmm_layer = GenePredHMMLayer(
                emitter_init=ConstantInitializer(em_kernel),
                initial_exon_len=200, 
                initial_intron_len=4500,
                initial_ir_len=10000,
                emit_embeddings=False,
                start_codons=[("ATG", 1.)],
                stop_codons=[("TAG", .34), ("TAA", 0.33), ("TGA", 0.33)],
                intron_begin_pattern=[("NGT", 0.99), ("NGC", 0.01)],
                intron_end_pattern=[("AGN", 1.)],
                starting_distribution_init="zeros",
                simple=False,
                trainable_nucleotides_at_exons=False,
                parallel_factor=self.parallel_factor,
                use_border_hints=False
        )
        self.gene_pred_hmm_layer.build([self.batch_size, self.seq_len, inp_size])