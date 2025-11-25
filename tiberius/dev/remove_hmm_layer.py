#!/usr/bin/env python3

import argparse
import sys
from tiberius.gene_pred_hmm import GenePredHMMLayer
from tiberius.models import custom_cce_f1_loss, Cast
from pathlib import Path
import json, tensorflow as tf
from tensorflow.keras.models import Model

def remove_last_layer(model):
    """
    Create a new model by removing the last layer from the input model.
    This function assumes that the model has at least two layers.
    """
    if len(model.layers) < 2:
        raise ValueError("The model must have at least two layers to remove the last layer.")

    # Rebuild the model by taking the output from the second-to-last layer.
    # This works for both functional and Sequential models provided that the 
    # connectivity can be traced back to model.input.
    new_output = model.layers[-3].output
    new_model = Model(inputs=model.input, outputs=new_output)
    return new_model

def main():
    parser = argparse.ArgumentParser(
        description="Load a TensorFlow model, remove the last layer, and save the modified model."
    )
    parser.add_argument(
        "--input_model", type=str, required=True,
        help="Path to the input TensorFlow model. This can be a saved model directory or an HDF5 file."
    )
    parser.add_argument(
        "--output_model", type=str, required=True,
        help="Path to save the modified TensorFlow model."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Training config path"
    )
    args = parser.parse_args()

    # Load the model from the input path

    print(args.input_model)
    try:
        model = tf.keras.models.load_model(args.input_model, 
                        custom_objects={
                        'custom_cce_f1_loss': custom_cce_f1_loss(2, 12),
                        'loss_': custom_cce_f1_loss(2, 12),
                        "Cast": Cast}
                        )
    except Exception as e:
        print(f"Error loading the model from {args.input_model}: {e}")
        sys.exit(1)

    print("Original model summary:")
    model.summary()

    # Remove the last layer
    try:
        new_model = remove_last_layer(model)
    except Exception as e:
        print(f"Error removing the last layer: {e}")
        sys.exit(1)

    print("\nModified model summary:")
    new_model.summary()

    model_config = {
            "units": 200,
            "filter_size": 64,
            "kernel_size": 9,
            "numb_conv": 2,
            "numb_lstm": 3,
            "dropout_rate": 0.0,
            "pool_size": 10,
            "stride": 0,
            "lstm_mask": False,
            "output_size": 7,
            "multi_loss": False,
            "residual_conv": False,
            "clamsa": False,
            "clamsa_kernel": 6,
            "softmasking": True,
            "lru_layer": False,
            "hmm": False
        }

    with Path(args.config).open("r", encoding="utf-8") as f:
        train_config = json.load(f)

    for k in model_config:
        if k in train_config:
            model_config[k] = train_config[k]

    # Save the modified model to the output path

    new_model.save_weights(args.output_model + "/weights.h5")
    print(f"\nModified model successfully saved to {args.output_model}")
    Path(args.output_model + "/model_layers.json").write_text(new_model.to_json())
    with Path(args.output_model + "/model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)


if __name__ == "__main__":
    main()
