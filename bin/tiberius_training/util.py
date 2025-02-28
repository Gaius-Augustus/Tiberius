import os
import glob
import re
import scipy

import matplotlib.pyplot as plt
def read_training_stats(training_dir):
    """
    Reads stats for all epochs for a given training number.
    Assumes folder structure: train<number>/epoch_<numb>.stats
    File format example:
    
    #-----------------| Sensitivity | Precision  |
            Base level:    88.0     |    87.3    |
            Exon level:    68.8     |    64.9    |
          Intron level:    74.2     |    70.3    |
    Intron chain level:    33.0     |    32.8    |
      Transcript level:    38.6     |    36.4    |
           Locus level:    41.7     |    36.4    |
    
    Returns:
        dict: Mapping epoch numbers (int) to a dict mapping each level name to its metrics.
              For example:
              {
                  0: {
                      "Base": {"sensitivity": 88.0, "precision": 87.3},
                      "Exon": {"sensitivity": 68.8, "precision": 64.9},
                      ...
                  },
                  1: { ... },
                  ...
              }
    """
    directory = training_dir
    stats_files = sorted(glob.glob(os.path.join(directory, "epoch_*.stats")))
    
    epoch_stats = {}
    
    # Regular expression to capture the level name, sensitivity, and precision.
    # It looks for lines like:
    # "        Base level:    88.0     |    87.3    |"
    pattern = re.compile(r'^\s*(.*?)\s+level:\s+([\d.]+)\s+\|\s+([\d.]+)', re.IGNORECASE)
    
    for file_path in stats_files:
        # Extract epoch number from the filename: "epoch_<number>.stats"
        match_epoch = re.search(r'epoch_(\d+)\.stats', file_path)
        if not match_epoch:
            continue
        epoch_num = int(match_epoch.group(1))
        
        level_metrics = {}
        with open(file_path, 'r') as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    level_name = m.group(1).strip()
                    sensitivity = float(m.group(2))
                    precision = float(m.group(3))
                    level_metrics[level_name] = {"sensitivity": sensitivity, "precision": precision}
                    
        epoch_stats[epoch_num] = level_metrics
        
    return epoch_stats


def plot_training_metrics(stats):
    """
    Plots sensitivity and precision for each level over epochs.

    Parameters:
        stats (dict): A dictionary where each key is an epoch (int) and each value is a dictionary
                      that maps level names (str) to their metrics (a dict with keys 'sensitivity' and 'precision').

    The function creates a figure with two subplots:
      - The top subplot plots sensitivity values over epochs.
      - The bottom subplot plots precision values over epochs.
    Each level is plotted with its own line for easy comparison.
    """
    # Get sorted list of epochs
    epochs = sorted(stats.keys())
    
    # Determine all unique levels from the stats (union across all epochs)
    levels = set()
    for ep in epochs:
        levels.update(stats[ep].keys())
    levels = sorted(levels)  # sorted alphabetically
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot sensitivity for each level
    for level in ["Base", "Exon", "Intron", "Transcript"]:
        sens_values = []
        for ep in epochs:
            # If the level isn't found for a given epoch, use None (or you could use np.nan)
            metrics = stats[ep].get(level, {})
            sens_values.append(metrics.get("sensitivity", None))
        axs[0].plot(epochs, sens_values,  label=level)
    
    axs[0].set_ylabel("Sensitivity (%)")
    axs[0].set_title("Sensitivity over Epochs")
    axs[0].grid(True)
    axs[0].legend(loc="best")
    
    # Plot precision for each level
    for level in levels:
        prec_values = []
        for ep in epochs:
            metrics = stats[ep].get(level, {})
            prec_values.append(metrics.get("precision", None))
        axs[1].plot(epochs, prec_values,  label=level)
    
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Precision (%)")
    axs[1].set_title("Precision over Epochs")
    axs[1].grid(True)
    axs[1].legend(loc="best")
    
    plt.tight_layout()
    plt.show()


def plot_accuracies(stats_list, run_labels=None):
    """
    Plots the accuracies (computed as (sensitivity + precision)/2) on Transcript and Exon levels,
    for a list of stat dictionaries from different training runs.

    Each stats dictionary should be structured as:
      {
         epoch_number: {
             "Transcript": {"sensitivity": value, "precision": value},
             "Exon": {"sensitivity": value, "precision": value},
             ...
         },
         ...
      }
    
    Parameters:
      stats_list (list): A list of stat dictionaries, each corresponding to a training run.
      run_labels (list, optional): A list of labels for the training runs. If not provided,
                                   labels "Run 1", "Run 2", ... will be used.

    The function creates a figure with two subplots:
      - The top subplot shows Transcript level accuracies over epochs.
      - The bottom subplot shows Exon level accuracies over epochs.
    """
    # Create default labels if none are provided
    if run_labels is None:
        run_labels = [f"Run {i+1}" for i in range(len(stats_list))]
    
    # Create two subplots sharing the same x-axis
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Iterate over each stats dictionary (i.e. each training run)
    for stats, label in zip(stats_list, run_labels):
        # Get sorted list of epochs
        epochs = sorted(stats.keys())
        
        transcript_accuracies = []
        exon_accuracies = []
        
        for ep in epochs:
            metrics = stats[ep]
            # Compute Transcript accuracy if available, else use None
            if "Transcript" in metrics:
                trans = metrics["Transcript"]
                transcript_accuracies.append(scipy.stats.hmean([trans["sensitivity"], trans["precision"]]))
            else:
                transcript_accuracies.append(None)
            
            # Compute Exon accuracy if available, else use None
            if "Exon" in metrics:
                exon = metrics["Exon"]
                exon_accuracies.append(scipy.stats.hmean([exon["sensitivity"] + exon["precision"]]))
            else:
                exon_accuracies.append(None)
        
        axs[0].plot(epochs, transcript_accuracies,  label=label)
        axs[1].plot(epochs, exon_accuracies,  label=label)
    
    # Configure the Transcript subplot
    axs[0].set_ylabel("Transcript Accuracy (%)")
    axs[0].set_title("Transcript Level Accuracy over Epochs")
    axs[0].grid(True)
    axs[0].legend(loc="best")
    
    # Configure the Exon subplot
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Exon Accuracy (%)")
    axs[1].set_title("Exon Level Accuracy over Epochs")
    axs[1].grid(True)
    axs[1].legend(loc="best")
    
    plt.tight_layout()
    plt.show()
