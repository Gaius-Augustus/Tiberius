import sys
sys.path.append("../")   
sys.path.append("../../learnMSA") 
import tensorflow as tf
from models import custom_cce_f1_loss
from packaging import version
import json 
from models import custom_cce_f1_loss, lstm_model, add_hmm_layer



assert((version.parse(tf.__version__) >= version.parse("2.10.0") and version.parse(tf.__version__) < version.parse("2.11.0") or
        version.parse(tf.__version__) >= version.parse("2.17.0")), 
        "Run this script with tensorflow 2.10.x or 2.17.x. Your TF: " + tf.__version__)


# loads the old Tiberius models and converts them to H5 format, which can be loaded by keras 3 (tf 2.17+)

model_path = "../model_weights/"

# if run under tf 2.10.x; convert to H5
if version.parse(tf.__version__) >= version.parse("2.10.0") and version.parse(tf.__version__) < version.parse("2.11.0"):
    for model_name in ["tiberius_weights", "tiberius_nosm_weights"]:
        model = tf.keras.models.load_model(model_path + model_name, 
                                            custom_objects={'custom_cce_f1_loss': custom_cce_f1_loss(2, 32),
                                                'loss_': custom_cce_f1_loss(2, 32)})
        model.save_weights(model_path + model_name + ".h5")
#else if run under tf 2.17.x; load from H5 and save again
elif version.parse(tf.__version__) >= version.parse("2.17.0"):
    for model_name in ["tiberius_weights"]:
        cfg_file  = "../config.json"
        with open(cfg_file, 'r') as f: 
            config = json.load(f)
            
        relevant_keys = ['units', 'filter_size', 'kernel_size', 
                                'numb_conv', 'numb_lstm', 'dropout_rate', 
                                'pool_size', 'stride', 'lstm_mask', 'clamsa',
                                'output_size', 'residual_conv', 'softmasking',
                                'clamsa_kernel', 'lru_layer']
        relevant_args = {key: config[key] for key in relevant_keys if key in config}
        model1 = lstm_model(**relevant_args)
        model1.summary()
        model = add_hmm_layer(model1, None,
                                            dense_size=config['hmm_dense'], 
                                            pool_size=config['pool_size'],
                                            output_size=config['output_size'], 
                                            num_hmm=config['num_hmm_layers'],
                                            l2_lambda=config['l2_lambda'],
                                            hmm_factor=config['hmm_factor'], 
                                            batch_size=config['batch_size'],
                                            seq_len=config['w_size'],
                                            initial_variance=config['initial_variance'],
                                            temperature=config['temperature'],
                                            emit_embeddings=config['hmm_emit_embeddings'], 
                                            share_intron_parameters=config['hmm_share_intron_parameters'],
                                            trainable_nucleotides_at_exons=config['hmm_nucleotides_at_exons'],
                                            trainable_emissions=config['hmm_trainable_emissions'],
                                            trainable_transitions=config['hmm_trainable_transitions'],                                    trainable_starting_distribution=config['hmm_trainable_starting_distribution'],
                                            use_border_hints=False,
                                            include_lstm_in_output=config['multi_loss'],
                                            neutral_hmm=config['neutral_hmm'])
        model.load_weights(model_path + model_name + ".h5")
        model.save(model_path + model_name + ".keras")