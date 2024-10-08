import sys, os, json, csv, argparse
sys.path.append("../../programs/learnMSA")
sys.path.append("/home/gabriell/gene_pred_deepl/bin")
sys.path.append("/home/gabriell/programs/learnMSA")
sys.path.append("/home/jovyan/brain//programs/learnMSA")
import subprocess as sp
import numpy as np
from gene_pred_hmm import  GenePredHMMLayer
import tensorflow as tf
# from transformers import AutoTokenizer, TFAutoModelForMaskedLM, TFEsmForMaskedLM
import tensorflow.keras as keras
from learnMSA.msa_hmm.Viterbi import viterbi

def decode_one_hot(encoded_seq):
        # Define the mapping from index to nucleotide
        index_to_nucleotide = np.array(['A', 'C', 'G', 'T', 'A'])
        # Use np.argmax to find the index of the maximum value in each row
        nucleotide_indices = np.argmax(encoded_seq, axis=-1)
        # Map indices to nucleotides
        decoded_seq = index_to_nucleotide[nucleotide_indices]
        # Convert from array of characters to string for each sequence
        decoded_seq_str = [''.join(seq) for seq in decoded_seq]
        return decoded_seq_str
    
def load_val_data(file, hmm_factor=False, reduce_output=True, ):
    data = np.load(file)
    x_val = data["array1"]
    y_val = data["array2"]
    
    if reduce_output:
        # reduce y_label size to 5
        y_new = np.zeros((y_val.shape[0], y_val.shape[1], 5), np.float32)
        y_new[:,:,0] = y_val[:,:,0]
        y_new[:,:,1] = np.sum(y_val[:,:,1:4], axis=-1)         
        y_new[:,:,2:] = y_val[:,:,4:]
        y_val = y_new        
    data.close()
    
    if hmm_factor:
        step_width = y_val.shape[1] // hmm_factor
        start = y_val[:,::step_width,:] # shape (batch_size, hmm_factor, 5)
        end = y_val[:,step_width-1::step_width,:] # shape (batch_size, hmm_factor, 5)
        hints = np.concatenate([start[:,:,tf.newaxis,:], end[:,:,tf.newaxis,:]],-2)
        return ([np.array(x_val), hints], [y_val, y_val])
    
    # if trans:
    #     x_val = x_val[:,:99036]
    #     y_val = y_val[:,:99036]
    #     tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    #     max_token_len = 5502
    #     x_token = np.reshape(x_val[:,:,:5], (-1, max_token_len, 5))
    #     x_token = decode_one_hot(x_token)
    #     x_token = tokenizer.batch_encode_plus(x_token, return_tensors="tf", 
    #               padding="max_length", max_length=max_token_len//6+1)  
    #     x_token['input_ids'] = x_token['input_ids'].numpy()
    #     x_token['input_ids'] = x_token['input_ids'].reshape(
    #             x_val.shape[0],-1,
    #             x_token['input_ids'].shape[1])
    #     x_token['attention_mask'] = x_token['attention_mask'].numpy()
    #     x_token['attention_mask'] = x_token['attention_mask'].reshape(
    #             x_val.shape[0],-1,
    #             x_token['attention_mask'].shape[1])
        
    #     x = [[np.expand_dims(x_val[i],0), x_token['input_ids'][i], x_token['attention_mask'][i]] for i in range(x_val.shape[0])]
    #     return x, y_val
    # return (x_val, y_val)

def main():
    args = parseCmd()    
    val_data_path = f'/home/gabriell/deepl_data/tfrecords/data/99999_hmm/val/validation_lstm.npz'
    val_data = load_val_data(val_data_path)
    model = tf.keras.models.load_model(args.model, 
                                          custom_objects={'TFEsmForMaskedLM': TFEsmForMaskedLM})
    result = model.evaluate(x=val_data[0], y=val_data[1], batch_size=args.batch_size, verbose=1)
    print(';'.join(model.metrics_names), '\n', ';'.join(list(map(str, result))))

def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--species', type=str,
#         help='')
    parser.add_argument('--model', type=str,
        help='')
    parser.add_argument('--batch_size', type=int,
        help='')
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
