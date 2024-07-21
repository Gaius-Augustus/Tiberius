import sys, json, os, re, sys, csv, argparse
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{script_dir}/../bin/")
import subprocess as sp
import numpy as np
import os
import warnings
import time

def print_class_freq(array, name='', out=f''):
    out_str = f'### Class frequencies {name}\n# 0 - IR, 1 - Intron, 2 - CDS\n'
    for i in range(3):
        out_str += f'{i} - {(array == i).sum():>10,}\n'
    print(out_str)
    if out:
        with open(out, 'a+') as f:
            f.write(out_str)
            
def main():
    start_time = time.time()
    args = parseCmd()
    sys.path.append(args.learnMSA)
    sys.path.append(args.LRU)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import tensorflow as tf
    from eval_model_class import PredictionGTF
    from transformers import TFEsmForMaskedLM
    from models import make_weighted_cce_loss
    if args.bigwig:
        import pyBigWig
    
    model_path = os.path.abspath(args.model) if args.model else None
    model_path_lstm = os.path.abspath(args.model_lstm) if args.model_lstm else None
    model_path_hmm = os.path.abspath(args.model_hmm) if args.model_hmm else None
    out_dir = args.out_dir
    
    batch_size = 8 if not args.batch_size else args.batch_size
    seq_len = 500004 if not args.seq_len else args.seq_len
    strand = '+' if not args.strand else args.strand
    # if args.trans_lstm:
    #     seq_len = 99036*5
    #     batch_size = 6

    print(batch_size, seq_len, strand)
    hmm_factor = 1
    num_hmm = 1
    inp_data_dir = f'{args.species_dir}/inp/' if args.species_dir else None
    
    genome_path = f'{inp_data_dir}/genome.fa' if not args.genome else args.genome
    annot_path= f'{inp_data_dir}/annot_{strand}.gtf' if not args.annot else args.annot
    temp_dir = '' if args.no_temp else f'{out_dir}/temp'
    
    eval_dir = f'{out_dir}/eval'
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=True)
    if temp_dir and  not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    
    gtf_out = f'{out_dir}/deepfinder.gtf'
    
    pred_gtf = PredictionGTF( 
        model_path=model_path,
         model_path_lstm=model_path_lstm,
        model_path_hmm=model_path_hmm,
         seq_len=seq_len, batch_size=batch_size, hmm=True, 
        temp_dir=temp_dir,
        emb=args.emb, 
        num_hmm=num_hmm,
        hmm_factor=hmm_factor,
        trans_lstm=args.trans_lstm,    
        genome_path=genome_path,
        annot_path=annot_path, 
        softmask=True, strand=strand,
        parallel_factor=args.parallel_factor,
        lstm_cfg=args.lstm_cfg,
    )

    tf.keras.utils.get_custom_objects()["weighted_cce_loss"] = make_weighted_cce_loss()
    pred_gtf.load_model()
    
    if args.clamsa:
        f_chunks, r_chunks, coords, clamsa_track= pred_gtf.load_inp_data(    
            strand=strand, 
            softmask=True, 
            clamsa_path=args.clamsa,
        )
        
    else:
        f_chunks, r_chunks, coords = pred_gtf.load_inp_data(    
            strand=strand, 
            softmask=True
        )
        clamsa_track = None
    
    if args.no_softmasking:
        f_chunks = f_chunks[:,:,:5]
        
    print(r_chunks.shape)
    print('IR', np.sum(r_chunks[:,:,0]))
    print('Intron', np.sum(r_chunks[:,:,1]))
    print('CDS', np.sum(r_chunks[:,:,2:]))
    
    if args.oracle:
        warnings.warn('Oracle mode is on. Using correct labels as HMM input. Beware, this is only for debugging.')
        from gene_pred_hmm import make_aggregation_matrix
        encoding_layer_oracle = np.matmul(r_chunks.astype(np.float32), make_aggregation_matrix())
        hmm_pred = pred_gtf.get_predictions(f_chunks, hmm_filter=True, clamsa_inp=clamsa_track, encoding_layer_oracle=encoding_layer_oracle)
    else:
        hmm_pred = pred_gtf.get_predictions(f_chunks, hmm_filter=True, clamsa_inp=clamsa_track)

    pred_gtf.create_gtf(y_label=hmm_pred, coords=coords,
        out_file=gtf_out, f_chunks=f_chunks[:hmm_pred.shape[0]], 
        clamsa_inp=clamsa_track, strand=strand, border_hints=True,
        correct_y_label=encoding_layer_oracle if args.oracle else None)
    
    end_time = time.time()    
    duration = end_time - start_time
    print(f"DeepFinder took {duration/60} minutes to execute.")
    
    cmd = f'{script_dir}/../bin/compare_intervals_exact.pl --f1 {annot_path} --f2 {gtf_out} > {eval_dir}/gtf_eval.txt'
    sp.call(cmd, shell=True)
    
    cmd = f'{script_dir}/../bin/compare_intervals_exact.pl --f1 {annot_path} --f2 {gtf_out} --gene >> {eval_dir}/gtf_eval.txt'
    sp.call(cmd, shell=True)
    if not args.no_temp:
        def get_class_label(array):
            if array.shape[-1] > 5:
                new_arr = np.concatenate([array[:,:,0:1], 
                            np.sum(array[:,:,1:4], axis=-1, keepdims=True), 
                            np.sum(array[:,:,4:], axis=-1, keepdims=True)], axis=-1)

            else:
        #         new_arr = np.concatenate(array, 0)        
                new_arr = np.concatenate([array[:,:,0:1], array[:,:,1:2],
                            np.sum(array[:,:,2:], axis=-1, keepdims=True)], axis=-1)    
            new_arr = new_arr.argmax(-1)
            return new_arr

        def print_metrics(repre, tpfnfp=None, name='', out=f'{eval_dir}/metrics.txt'):
            out_str = f'### Class accuracies {name}\n'
            classes = ['0 - IR', '1 - Intron', '2 - CDS']
            for i in range(3): 
                out_str += f'# {classes[i]}\n'
                if tpfnfp:
                    for k in ["TP", "FP", "FN"]:
                        out_str += f'{k} - {tpfnfp[i][k]:>11,}\n'
                    print('')
                for k in ["Precision", 'Recall', 'F1']:
                    out_str += f'{k:<9} - {repre[i][k]:>3.20f}\n'

            if tpfnfp:
                for k in ["TP", "FP"]:
                    out_str += f'{k} - {tpfnfp["all"][k]:>10}\n'
            out_str += f'All classes accuracy - {repre["all"]["F1"]:>3.20f}\n'

            print(out_str)
            if out:
                with open(out, 'a+') as f:
                    f.write(out_str)

        pred_gtf = PredictionGTF( 
            model_path=model_path,
             model_path_lstm=model_path_lstm, 
             seq_len=seq_len, batch_size=batch_size, hmm=True, 
            temp_dir=temp_dir,
            emb=args.emb, 
            num_hmm=num_hmm,
            hmm_factor=hmm_factor,
            trans_lstm=args.trans_lstm,    
            genome_path=genome_path,
            annot_path=gtf_out, 
            softmask=True, strand=strand,
            parallel_factor=args.parallel_factor,
            lstm_cfg=args.lstm_cfg
        )            
        _, pred_chunks, _ = pred_gtf.load_inp_data(    
            strand=strand, 
            softmask=True
        )

        lstm_predictions = encoding_layer_oracle if args.oracle else np.load(f'{temp_dir}/lstm_predictions.npz')['array1']
        hmm_predictions = np.load(f'{temp_dir}/hmm_predictions.npy')

        annot_lstm = get_class_label(r_chunks)
        pred_label = get_class_label(pred_chunks)
        lstm_label = get_class_label(lstm_predictions)
        hmm_label = pred_gtf.reduce_label(hmm_predictions, 1)

        print_class_freq(annot_lstm, name='reference annotation', out=f'{eval_dir}/clas_freq.txt')
        print('\n')
        print_class_freq(lstm_label, name='LSTM prediction', out=f'{eval_dir}/clas_freq.txt')
        print('\n')
        print_class_freq(hmm_label, name='HMM prediction', out=f'{eval_dir}/clas_freq.txt')
        print('\n')
        print_class_freq(pred_label, name='GTF prediction', out=f'{eval_dir}/clas_freq.txt')

        tpfnfp_lstm = pred_gtf.get_tp_fn_fp(lstm_label, annot_lstm)
        metric_lstm = pred_gtf.calculate_metrics(tpfnfp_lstm)
        print_metrics(metric_lstm, name='LSTM')

        print('')
        tpfnfp_hmm = pred_gtf.get_tp_fn_fp(hmm_label, annot_lstm)
        metric_hmm = pred_gtf.calculate_metrics(tpfnfp_hmm)
        print_metrics(metric_hmm, name='HMM')

        print('')
        tpfnfp_gtf = pred_gtf.get_tp_fn_fp(pred_label, annot_lstm)
        metric_gtf = pred_gtf.calculate_metrics(tpfnfp_gtf)
        print_metrics(metric_gtf, name='GTF')

        if args.bigwig:
            seq_len = {}
            lstm_seq = {}
            for i, c in enumerate(coords):
                if c[0] not in seq_len.keys():
                    seq_len[c[0]] = 0
                    lstm_seq[c[0]] = []
                seq_len[c[0]] += lstm_predictions.shape[1]
                lstm_seq[c[0]].append(lstm_predictions[i])

            for seq_name in lstm_seq:
                lstm_seq[seq_name] = np.array(lstm_seq[seq_name])
                print(lstm_seq[seq_name].shape)
                if lstm_seq[seq_name].shape[-1] > 5:
                    y_new = np.zeros((lstm_seq[seq_name].shape[0], lstm_seq[seq_name].shape[1], 5), np.float32)
                    y_new[:,:,0] = lstm_seq[seq_name][:,:,0]
                    y_new[:,:,1] = np.sum(lstm_seq[seq_name][:,:,1:4], axis=-1)            
                    y_new[:,:,2:] = lstm_seq[seq_name][:,:,4:]
                    lstm_seq[seq_name] = y_new
                lstm_seq[seq_name] = lstm_seq[seq_name].reshape((-1,5))

            with pyBigWig.open(f"{eval_dir}/ir.bw", "w") as bw:
                bw.addHeader(list(seq_len.items()))
                for seq_name in lstm_seq:
                    starts = np.arange(0,lstm_seq[seq_name].shape[0],1)
                    bw.addEntries([seq_name]*lstm_seq[seq_name].shape[0], starts, ends=starts+1, values=lstm_seq[seq_name][:,0])

            with pyBigWig.open(f"{eval_dir}/intron.bw", "w") as bw:
                bw.addHeader(list(seq_len.items()))
                for seq_name in lstm_seq:
                    starts = np.arange(0,lstm_seq[seq_name].shape[0],1)
                    bw.addEntries([seq_name]*lstm_seq[seq_name].shape[0], starts, ends=starts+1, values=lstm_seq[seq_name][:,1])

            with pyBigWig.open(f"{eval_dir}/cds.bw", "w") as bw:
                bw.addHeader(list(seq_len.items()))
                for seq_name in lstm_seq:
                    starts = np.arange(0,lstm_seq[seq_name].shape[0],1)
                    values = lstm_seq[seq_name][:,2:].sum(-1)
                    bw.addEntries([seq_name]*lstm_seq[seq_name].shape[0], starts, ends=starts+1, values=values)

        if args.rm_temp:
            cmd = f"rm -r {temp_dir}"
            sp.call(cmd, shell=True)
        
def parseCmd():
    """Parse command line arguments

    Returns:
        dictionary: Dictionary with arguments
    """
    parser = argparse.ArgumentParser(
        description="""Uses a LSTM+HMM hybrid model to infer gene structures from a nucleotide sequences and evaluate them with a reference annotation.

    There are flexible configuration to load the model, including options to:
    - Load a complete LSTM+HMM model
    - Load only the LSTM model and use a default HMM layer
    - Operate in 'oracle mode' for debugging with HMM predictions based on the true labels

    Returns:
        argparse.Namespace: Namespace containing the command line arguments

    Example usage:
        Load LSTM+HMM model:
        test_vit.py --annot reference_annot.gtf --genome genome.fa --model lstm_hmm_keras_save

        Load only LSTM model:
        test_vit.py --annot reference_annot.gtf --genome genome.fa --model_lstm lstm_keras_save

        Load LSTM model and custom HMM Layer:
        test_vit.py --annot reference_annot.gtf --genome genome.fa --model_lstm lstm_keras_save --model_hmm hmm_layer_keras_save

        Oracle mode without LSTM model:
        test_vit.py --annot reference_annot.gtf --genome genome.fa --oracle
        test_vit.py --annot reference_annot.gtf --genome genome.fa --model_hmm hmm_layer_keras_save --oracle
    """)
    parser.add_argument('--model_lstm', type=str, default='',
        help='Path to the LSTM model file. Use with --model_hmm to add a custom HMM layer, otherwise a default HMM layer is added.')
    parser.add_argument('--model_hmm', type=str, default='',
        help='Path to the HMM layer file. Can be used with --model_lstm or in oracle mode.')
    parser.add_argument('--model', type=str,
        help='Path to LSTM model file with HMM Layer.', default='')
    parser.add_argument('--species_dir', type=str,
        help='', default='Panthera_pardus')
    parser.add_argument('--out_dir', type=str,
        help='Output directory for the script’s files.', default='./')
    parser.add_argument('--seq_len', type=int,
        help='Length of sub-sequences used for parallelizing the prediction.', default=0)
    parser.add_argument('--batch_size', type=int,
        help='Number of sub-sequences per batch.', default=0)
    parser.add_argument('--strand', type=str,
        help='Either "+" or "-".', default='+')
    parser.add_argument('--trans_lstm', action='store_true',
        help='Indicates if the LSTM model includes nucleotide transformer layers.')
    parser.add_argument('--emb', action='store_true',
        help='Indicates if the HMM layer uses embedding input.')
    parser.add_argument('--no_softmasking', action='store_true',
        help='Disables softmasking.')
    parser.add_argument('--clamsa', type=str, default='',
        help='')
    parser.add_argument('--learnMSA',  type=str, default='../learnMSA',
        help='Path to the learnMSA repository')
    parser.add_argument('--LRU',  type=str, default='../LRU',
        help='Path to the LRU repository')
    parser.add_argument('--parallel_factor',  type=int, default=817,
        help='Parallel factor used in Viterbi. Use the factor of w_size that is closest to sqrt(w_size) (817 works well for 500004)')
    parser.add_argument('--gpu', type=str, default='',
        help='Number of GPUS to be used.')
    parser.add_argument('--bigwig',  action='store_true',
        help='Generates BigWig files for visualizing LSTM output probabilities.')
    parser.add_argument('--rm_temp',  action='store_true',
        help='Enables automatic removal of temporary files, which can be large.')
    parser.add_argument('--no_temp',  action='store_true',
        help='Temporary files are not stored.')
    parser.add_argument('--annot', type=str, default='',
        help='Path to the GTF file of reference annotations.')
    parser.add_argument('--genome',  type=str, default='',
        help='Path to the genome sequence file in FASTA format.')
    parser.add_argument('--lstm_cfg',  type=str, default='',
        help='')
    parser.add_argument('--oracle', action='store_true',
        help='Uses the correct labels as inputs for debugging.')
    return parser.parse_args()

if __name__ == '__main__':
    main()
