#!/usr/bin/env python3
import os
import argparse
import yaml  # make sure PyYAML is installed (pip install pyyaml)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a config.yaml file for a Snakemake workflow automatically based on the data directory and specified splits."
    )
    parser.add_argument("data_dir", help="Path to the data directory (e.g., data/)")
    parser.add_argument(
        "--val",
        nargs="*",
        default=[],
        help="List of species names to use as validation data",
    )
    parser.add_argument(
        "--test",
        nargs="*",
        default=[],
        help="List of species names to use as test data",
    )
    parser.add_argument(
        "--output",
        default="config.yaml",
        help="Path for the output config file (default: config.yaml)",
    )
    args = parser.parse_args()

    # List all subdirectories in the data directory (each assumed to be a species)
    try:
        species_all = [
            d for d in os.listdir(args.data_dir + '/species')
            if os.path.isdir(os.path.join(args.data_dir + '/species', d))
        ]
    except FileNotFoundError:
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    species_all = sorted(species_all)
    
    # Warn if specified validation/test species are not present in the data directory
    for sp in args.val:
        if sp not in species_all:
            print(f"Warning: Validation species '{sp}' not found in '{args.data_dir}'.")
    for sp in args.test:
        if sp not in species_all:
            print(f"Warning: Test species '{sp}' not found in '{args.data_dir}'.")

    # Determine training species as those not in validation or test lists
    train_species = [sp for sp in species_all if sp not in args.val and sp not in args.test]

    # Create the configuration dictionary
    config = {
        "species_split": {
            "train": train_species,
            "val": args.val,
            "test": args.test,
        },
        "work_dir": args.data_dir,  # Default work directory
    }

    # Write the configuration to a YAML file
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Configuration successfully written to '{args.output}'.")

if __name__ == "__main__":
    main()
