# DeepFinder (preliminary name): End-to-End Deep Learning with HMMs for Gene Prediction

### Prerequisites

Clone the repository, including learnMSA as submodule:
```
git clone --recursive https://github.com/LarsGab/gene_pred_deepl
```
In case you cloned the repository without the submodule, you can load the submodule with:
```
git submodule update --init --recursive
```
Ensure that learnMSA was loaded and that is on the branch `parallel`.

The following Python libraries are required:
- tensorflow==2.10.1
- tensorflow_probability==0.18.0
- transformers (optional)
- pyBigWig
- biopython 
- bcbio-gff

They can be installed with:
```
pip install tensorflow==2.10.1 tensorflow_probability==0.18.0 transformers pyBigWig bio scikit-learn biopython bcbio-gff
```

On brain cluster, if above does not give you access to GPUs try:
```
mamba create -n genepred python==3.10
conda activate genepred
mamba install -c conda-forge cudatoolkit ipykernel cudnn tensorflow==2.10.*
```

### Model Architecture
> relevant code: `bin/models.py`

The DeepFinder model comprises three primary types of layers applied sequentially: 
Convolutional Neural Networks (CNNs), bidirectional Long Short-Term Memory (bLSTM), and a Hidden Markov Model (HMM).

![model](figures/model_fig.png)

The DeepFinder model or its CNN-bLSTM components can be constructed and customized using different hyperparameters (see `bin/models.py`):

- `lstm_model()`:  Constructs a model with only the CNN-bLSTM part (excludes HMM).
- `add_hmm_layer()`: Integrates an HMM Layer into the CNN-bLSTM model.

Work in progress:
- `add_clamsa2lstm()`: Appends CNN and dense layers to a CNN-bLSTM model toinclude a ClaMSA track.
- `add_transformer2lstm()`: Incorporates a nucleotide transformer layer after the CNN-bLSTM model.

### HMM Layer
> relevant code: `bin/gene_pred_hmm.py`, `bin/gene_pred_hmm_emitter.py`, `gene_pred_hmm_transitioner.py`

The HMM Layer consists of 15 states and can be integrated and trained like other TensorFlow Keras layers. It enforces biological constraints on gene structures, such as splice site patterns. Inputs to the HMM layer are one-hot encoded nucleotides, class label predictions (typically 5 labels), and optionally, an additional embedding vector.It outputs a prediction for the 15 HMM states, reducible to desired labels size (usually 5). Beyond training, this layer can infer the most likely sequence of hidden states using the Viterbi algorithm. 

![hmm_chart](figures/hmm_chart.png)

### Input:
> relevant code: `bin/genome_fasta.py`, `bin/wig_class.py`, `bin/eval_model_class.py`

The primary inputs to the DeepFinder model are:
- One-hot encoded nucleotide sequences (A, C, G, T, N) accompanied by a track for softmasking. The input structure is (sequence_length, 6): the first five tracks correspond to the nucleotides in the specified order, and the sixth track indicates softmasking. It is important to note that the sequence length must be divisible by `pool_size=9`, as this factor reduces sequence lengths in the bLSTM using a `Reshape` layer. This reshaping can be bypassed if `pool_size` is set to 1.

Work in progress:
- **ClaMSA Input Track**: This secondary input vector (sequence_length, 4) comprises:
    1. Average ClaMSA score for the start of a protein-coding codon at the position.
    2. Average ClaMSA score for the start of a protein-coding codon at the position on the reverse strand.
    3. Count of values averaged for point 1.
    4. Count of values averaged for point 2.
- **Tokens for Nucleotide Transformer**: Additional input features to enhance model predictions.


The `GenomeSequences` class from `bin/genome_fasta.py` processes FASTA genome files into one-hot encoded input sub-sequences suitable for the DeepFinder model. See `def load_inp_data()` in `bin/eval_model_class.py` for an example how this class can be used to load a single genome sequence. A detailed example of input data loading is provided in `test_data/Panthera_pardus/example_prediction.ipynb`.

With the `Wig_util` class from `bin/wig_class.py`, the input data for the ClaMSA tracks are prepared

### Output:
> relevant code: `bin/annotation_gtf.py`, `bin/eval_model_class.py`

Label of gene structure feature for each position of the sequence (shape (sequence_length, number labels)), the number of labels can be set to:
* 5 (default): intergenic region, intron, exon-0, exon-1, exon-2
* 3: intergenic region, intron, exon
* 15 (representing the states of the HMM): intergenic region, intron-0, intron-1, intron-2, exon-0, exon-1, exon-2, START, exon-intron-0, exon-intron-1, exon-intron-2, intron-exon-0, intron-exon-0, intron-exon-0, STOP


Where k in Exon-k denotes the position inside the current codon (reading frame).


The `GeneStructure` class from `bin/genome_fasta.py` is used to read a GTF file of a reference annotation. It is crucial that the reference annotation includes only one transcript isoform per gene as alternative splicing is not supported. For example, using the transcript with longest conding sequence for each gene is a reasonable choice (`TSEBRA/bin/get_longest_isoform.py`).
See `def load_inp_data()` in `bin/eval_model_class.py` for an example how this class can be used to load a single reference annotation. A detailed example of the class labels loading is provided in `test_data/Panthera_pardus/example_prediction.ipynb`.

### Training
> relevant code: `bin/train.py`, `bin/write_tfrecord_species.py`, `bin/data_generator.py`, `test_data/Panthera_pardus/example_training.ipynb`

The DeepFinder models are trained similarly to other neural networks using Keras. The training process is adaptable to datasets of various sizes, from individual species to large multi-species datasets.

#### Training for a Small Number of Species
For smaller datasets, such as training models specific to individual species:
- An example of setting up the model, loading input data for a single species, and conducting training can be found in `test_data/Panthera_pardus/example_training.ipynb`.
- This example includes a recommended set of hyperparameters as a starting point.
- Example data for this training setup is available at `test_data/Panthera_pardus/inp/`.

#### Training with tfRecords
For larger datasets, where the input data exceeds memory capacity:
- tfRecords can be utilized to efficiently read the data during training
- Use `bin/write_tfrecord_species.py` to prepare tfRecords for individual species.
- Pre-prepared tfRecords for a set of mammalian species are accessible at `/home/nas-hs/projs/lars/deepfinder/tfrecords` on the brain cluster.

In order to load trainings examples (input, output data) you can use `DataGenerator` from `bin/data_generator.py`. You can find an example how to set the Generator up in `bin/train.py`.

Automated training with tfRecords:
- The `bin/train.py` script automates the entire training process, from data loading to model training and validation.
- For detailed script configurations and command-line arguments, refer to `bin/parse_args.py`.

The following example shows how to start a training on the brain cluster using 1 vison node with 4 GPUs:
```
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=slurm/train_%j.out
#SBATCH --error=slurm/train_%j.err
#SBATCH --nodes=1
#SBATCH --partition=vision
#SBATCH --gpus=4
#SBATCH --time=72:00:00
#SBATCH --mem=150gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude vision-[01,02,08]

python3 bin/train.py --hmm --out ~/deepfinder/exclude_primates/weights/train \
  --data ~/deepfinder/tfrecords/train \
  --val_data ~/deepfinder/tfrecords/validation_lstm.npz \
  --learnMSA ~/learnMSA
```


### Gene structure prediction
> relevant code: `bin/eval_model_class.py`, `test_data/test_vit.py`, 

Full gene structures from a genome can be predicted using the `PredictionGTF` class from `bin/eval_model_class.py`. Fully automated prediction and evaluation of a model can be done using the `test_data/test_vit.py` script (see its documetation for usage information). A detailed example for gene structure prediction with a model can be found at `test_data/Panthera_pardus/example_prediction.ipynb`.

