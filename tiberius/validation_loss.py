#!/usr/bin/env python3


import argparse
from pathlib import Path 
import sys
import json
from tensorflow import keras
from tiberius import DataGenerator
from tiberius.models import lstm_model, custom_cce_f1_loss, Cast, add_hmm_layer

def parse_args(argv = None) -> argparse.Namespace:
    """Parse CLI arguments, convert to absolute paths, and perform basic path sanity checks."""
    parser = argparse.ArgumentParser(
        description="Run training/validation pipeline and record validation loss."
    )

    parser.add_argument(
        "--species_list",
        type=Path,
        required=True,
        help="Path to the species list file.",
    )
    parser.add_argument(
        "--epochs_dir",
        type=Path,
        required=True,
        help="Directory containing training-epoch checkpoints and config.json file.",
    )
    parser.add_argument(
        "--val_loss_out",
        type=Path,
        required=True,
        help="Output path for writing validation-loss values.",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory with input tfRecords with validation data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False, default=200,
        help="Chunk length used for training.",
    )
    parser.add_argument(
        "--tfrec_per_species",
        type=int,
        required=False, default=100,
        help="Number of tfRecords used per species.",
    )

    args = parser.parse_args(argv)

    # Convert all provided paths to absolute paths
    args.species_list = args.species_list.resolve()
    args.epochs_dir = args.epochs_dir.resolve()
    args.val_loss_out = args.val_loss_out.resolve()
    args.data_dir = args.data_dir.resolve()

    # --- basic validation ---------------------------------------------------
    if not args.species_list.is_file():
        parser.error(f"Species list not found: {args.species_list}")
    if not args.epochs_dir.is_dir():
        parser.error(f"Epochs directory not found: {args.epochs_dir}")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory not found: {args.data_dir}")

    # Ensure the output directory exists (create parents if needed)
    args.val_loss_out.parent.mkdir(parents=True, exist_ok=True)

    return args

def load_model(model_path: str, batch_size: int = 12):
    model_weights = None
    if Path(model_path + "/model.weights.h5").is_file():
        model_weights = Path(model_path + "/model.weights.h5")
    elif Path(model_path + "/weights.h5").is_file():
        model_weights = Path(model_path + "/weights.h5")
    else: 
        try:
            model = keras.models.load_model(
                    str(model_weights), 
                    custom_objects={
                    'custom_cce_f1_loss': custom_cce_f1_loss(2, batch_size),
                    'loss_': custom_cce_f1_loss(2, batch_size),
                    "Cast": Cast}, 
                    compile=False,
                    )
            return model
        except Exception as e:
            print(f"Error loading the model from {model_path}: {e}")
            sys.exit(1)
    
    try:        
        with open(f"{model_path}/model_config.json", 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error could not find config of the model. It should be located at {model_path}/model_config.json: {e}")
        sys.exit(1)
    relevant_keys = ['units', 'filter_size', 'kernel_size', 
                'numb_conv', 'numb_lstm', 'dropout_rate', 
                'pool_size', 'stride', 'lstm_mask', 'clamsa',
                'output_size', 'residual_conv', 'softmasking',
                'clamsa_kernel', 'lru_layer']
    relevant_args = {key: config[key] for key in relevant_keys if key in config}
    model = lstm_model(**relevant_args)
    if config["use_hmm"]:
        relevant_keys = ['output_size', 'num_hmm', 'num_copy', 'hmm_factor', 
                    'share_intron_parameters', 'trainable_nucleotides_at_exons',
                    'trainable_emissions', 'trainable_transitions',
                    'trainable_starting_distribution', 'include_lstm_in_output',
                    'emission_noise_strength']        
        relevant_args = {key: config[key] for key in relevant_keys if key in config}
        model = add_hmm_layer(model,
                           **relevant_args)

    return model


def main(argv = None) -> None:
    """Main entry point.

    Expand this function with your model-specific training / evaluation code.
    """
    args = parse_args(argv)

    # read config file from epochs_dir
    config_path = args.epochs_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    print(f"Using config file: {config_path}")
    # Load the config file 
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # load species list
    with open(args.species_list, 'r') as f:
        species = [line.strip() for line in f if line.strip()]
    print(f"Loaded species list: {species}")

    file_paths = [str(args.data_dir / f'{s}_{i}.tfrecords') for s in species for i in range(args.tfrec_per_species)\
            if (args.data_dir / f'{s}_{i}.tfrecords').is_file()]
    # print(file_paths)
    if not file_paths:
        raise FileNotFoundError(f"No tfRecords found in {args.data_dir} for species {species}")

    # create data generator
    generator = DataGenerator(file_path=file_paths, 
          batch_size=args.batch_size, 
          shuffle=False,
          repeat=False,
          filter=config["filter"],
          output_size=config["output_size"],
          hmm_factor=0,
          seq_weights=config["seq_weights"], 
          softmasking=config["softmasking"],
          clamsa=False if not "clamsa" in config else config["clamsa"],
          oracle=False if 'oracle' not in config else config['oracle'],
          threads=config["threads"] if "threads" in config else 48,
          tx_filter=[]
      ).get_dataset()


    # predict and save loss and accuracy for each epoch
    # find all dirs in data_dir that match the pattern epoch_\d+ and sort them

    epochs_dirs = sorted(args.epochs_dir.glob("epoch_*"), key=lambda x: int(x.name.split('_')[1]))
    # print(epochs_dirs)
    result = []
    for epoch in epochs_dirs:
        print(str(epoch))
        model = load_model(str(epoch), args.batch_size)
        # model.summary()
        use_hmm = any("gene_pred_hmm_layer" in layer.name for layer in model.layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config["lr"]),
            loss=custom_cce_f1_loss(2, args.batch_size, from_logits=use_hmm),
            metrics=['accuracy']
        )
        print(f"Evaluating model from epoch: {epoch.name}")
        # Evaluate the model on the validation data
        val_loss, val_accuracy = model.evaluate(generator, verbose=1)
        print(f"Validation loss: {val_loss}, Validation accuracy: {val_accuracy}")
        result.append({
            "epoch": epoch.name,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    # Write results to the output file
    with args.val_loss_out.open('w') as f:
        f.write("Epoch,Validation Loss,Validation Accuracy\n")
        for res in result:
            f.write(f"{res['epoch']},{res['val_loss']},{res['val_accuracy']}\n")


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
