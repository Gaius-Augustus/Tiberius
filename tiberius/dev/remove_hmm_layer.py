#!/usr/bin/env python3

import argparse
import sys
sys.path.append("/home/gabriell/Tiberius/bin/")   
from gene_pred_hmm import GenePredHMMLayer
from models import custom_cce_f1_loss, Cast
import tensorflow as tf
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
    # Set up command-line argument parsing
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

    # Save the modified model to the output path
    try:
        new_model.save(args.output_model)
        print(f"\nModified model successfully saved to {args.output_model}")
    except Exception as e:
        print(f"Error saving the modified model to {args.output_model}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
