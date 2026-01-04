![Docker Pulls](https://img.shields.io/docker/pulls/larsgabriel23/tiberius) 

### ⚠️ Important Update — October 2025 
#### Tiberius models for Mammals, Eudicots, Monocots, Fungi, Diatoms and Vertebrates are now available.  

# Tiberius: End-to-End Deep Learning with an HMM for Gene Prediction

For more information, see the [Tiberius paper](https://academic.oup.com/bioinformatics/article/40/12/btae685/7903281).


Tiberius is a deep learning-based *ab initio* gene structure prediction tool that end-to-end integrates convolutional
and long short-term memory layers with a differentiable HMM layer. It can be used to predict gene structures from **genomic sequences only** (*ab initio*), while matching the accuracy of tool that use extrinsic evidence. 

Additionally, Tiberius provides an evidence mode that generates highly precise gene structures from extrinsic evidence, which are then combined with Tiberius *ab inito* predictions. Tiberius can also be parallized on HPC systems using Nextflow.


![Accuracy comparison for Tiberius *ab initio* predicitons for the human genome](figures/acc_hsap.png)



We are providing different models for following species (see model_cfg/README.md for more information):
- Diatoms 
- Eudicotyledons
- Lepidoptera
- Insects
- Monocotyledonae
- Mucoromycota
- Mammalia 
- Saccharomycota
- Sordariomycota
- Vertebrates


## Installation

This repository must always be cloned locally, as Tiberius relies on a local launcher script that manages execution and that can pull the Singularity image.
```
git clone https://github.com/Gaius-Augustus/Tiberius
cd Tiberius
pip install .
```

The command above installs the Tiberius Python package itself and is required in all cases, including when running Tiberius via Singularity.

Tiberius can be executed either using Singularity or with a local installation with all dependencies.


### Installation from Source

If Singularity is **not** used, Tiberius must be installed together with its dependencies as defined in the `from_source` dependencies in pyproject.toml:
```
pip install .[from_source]
```

**Note:**  
If TensorFlow is *not* installed beforehand, `pip install .` installs the latest available TensorFlow version. This version may restrict the usable `--seq_len` to **≤ 259 992**, which can slightly reduce prediction accuracy.

If you want to use a longer sequence length, install **TensorFlow 2.10** manually by following the [instructions](docs/install_tensorflow.md) or use the Singularity container.

#### Python Libraries

The following Python libraries are required and installed with Tiberius:
- tensorflow
- pyBigWig
- biopython 
- bcbio-gff
- requests
- rich

Make sure TensorFlow is installed with GPU support. Tiberius was built on TensorFlow 2.10 and runs best with that version. If you are using conda, you can install Tensorflow 2.10 with these [instructions](docs/install_tensorflow.md).

Tiberius does also work with TensorFlow >2.10, however, **it will produce an error if you use a `--seq_len` parameter > 259.992 during inference!**  


## Running Tiberius for *Ab Initio* Gene Prediction

To run Tiberius with `tiberius.py`, you need to provide a FASTA file containing the genomic sequences and a model config file. The sequence can either include repeat softmasking (recommended) or be run without softmasking. See [softmasking_workflow](docs/softmasking_workflow.md) for recommandations on how to mask repeats for Tiberius.


### Choosing the Model Weights

Tiberius requires a model config file (`--model_cfg`). The model config is a YAML file that contains the model weights URL and other information about the model. See [model_cfg](model_cfg/README.md) for available configurations and how to create custom ones. You can pass either a full path or a short name (e.g., `diatoms` or `diatoms.yaml`).

### Running Tiberius
Tiberius can be run using **Singularity** for example for a mammalian species with softmasking with following command:
```shell
# Run Tiberius for a softmasked mammalian genome
python tiberius.py --singularity --genome input.fasta \
    --out output.gtf --model_cfg mammalia_softmasking_v2.yaml
```

In order to run Tiberius with a local installation omit `--singularity`.


Tiberius produces a GTF file containing the predicted gene structures. It can also generate FASTA-formatted files of coding sequences and protein sequences when locations are specified using the `--codingseq` and `--protseq` options, respectively.

If you want to write custom code for Tiberius gene prediction, see [example_prediction.ipynb](test_data/Panthera_pardus/example_prediction.ipynb) for an example on how to write a custom prediction script.


## Running Tiberius with Nextflow

Tiberius can launch also be parallized across multiple GPU nodes on an HPC with Nextflow. For this, you have to set up a nextflow configuration for your specific cluster and extend [conf/base.config](conf/base.config). As an example, see [conf/slurm_generic.config](conf/slurm_generic.config) and see section *Adapting Tiberius to an HPC* in [conf/README.md](conf/README.md).

You can start Tiberius with Nextflow by providing it with your Nextflow config file:
```shell
# Nextflow for a Diatom genome
python tiberius.py --nf_config conf/slurm_generic.config --genome input.fasta --model_cfg diatoms
```



## Running Tiberius with Extrinsic Evidence (Nextflow-only)
You can also run the Tiberius Evidence Pipeline. A set of high confidence genes is generated and added to the Tiberius prediction that improves its accuracy, it adds some alternative splicing forms and it includes UTR regions for the evidence-only predictions. 

To provide Tiberius with the files and required parameters, it is recommended to generate a parameter file `params.yaml`. See [conf/README.md](conf/README.md) for details about the parameter file and [conf/params.yaml](conf/params.yaml) for a template. 


```shell
# Evidence pipeline with an existing params file
python tiberius.py --params_yaml params.yaml --nf_config conf/slurm_generic.config
```

Parameters in the params.yaml file can be overwritten with command line arguments

```shell
# Nextflow with params file and commandline overwrites
python tiberius.py --nf_config conf/base.config --genome input.fasta --model_cfg diatoms --outdir results
```
![Workflow of the Tiberius Evidence Pipeline](figures/evi_wflow.png)



### Running Tiberius with evolutionary information
To run Tiberius in *de novo* mode, evolutionary information data has to be generated with ClaMSA. See [docs/clamsa_data.md](docs/clamsa_data.md) for instructions on how to generate the data. Afterwards, you should have a directory with files named `$clamsa/{prefix}{seq_name}.npz` for each sequence of your FASTA file. You can then run Tiberius with the `--clamsa` argument. Note that your genome has to be softmasked for this mode and that you have to use different training weights than in *ab initio* mode. You can downloaded the model weights from [https://bioinf.uni-greifswald.de/bioinf/tiberius/models/tiberius_denovo_weights.tgz](https://bioinf.uni-greifswald.de/bioinf/tiberius/models//tiberius_denovo_weights_v2.tar.gz). Or you can provide Tiberius with the model configuration file `model_cfg/mammalia_clamsa_v2.yaml`
```shell
# Run Tiberius with softmasking
python tiberius.py --genome input.fasta --clamsa $clamsa/{prefix} --out output.gtf --model_cfg mammalia_clamsa_v2.yaml
```

### Running Tiberius on Differnet GPUs

Tiberius can run on any GPU with at least 8GB of memory. However, you will need to adjust the batch size to match the memory capacity of your GPU using the `--batch_size` argument. Below is a list of recommended batch sizes for different GPUs:
Here is a list of GPUs to batch siezes:
- **A100 (80GB):** batch size of 16
- **RTX 3090 (25GB):** batch size of 8
- **RTX 2070 (8GB):** batch size of 2



## Training Tiberius
We recommend using one of the provided trained models. However, if you want to train Tiberius on your own data, you need at least a genomic seqeunce file (FASTA) and reference annotations (GTF) for each species. **Note that you can only train on genes with one transcript isoform per gene.** Please remove alternative splicing variants before training. There two ways to train Tiberius:
1. Training Tiberius with a large dataset that does not fit into memory. See [training_large_data.md](docs/training_large_data.md) for documentation on how to prepare a dataset and train Tiberius with it.
2. Training Tiberius with a small dataset that fits into memory. See [example_train_full.ipynb](test_data/Panthera_pardus/example_train_full.ipynb) for an example on how to load data and train Tiberius on a single genome. This can easily be adapted to train Tiberius on several genomes by first loading the data for all genome and then training the model. See [training_large_data.md](docs/training_large_data.md) (Step 1) and [softmasking_workflow.md](docs/softmasking_workflow.md) for the preparation of the genome and annotation files.

## Tiberius Model

Tiberius' model consists of a model that consist of CNN, biLSTM, and a differentiable HMM layer. 
  
![Tiberius Architecture](figures/tiberius_architecture.png)


## Annotations from Tiberius

 [Tiberius predictions for 1314 mamalian assemblies](https://bioinf.uni-greifswald.de/bioinf/tiberius/genes/tib-tbl.html)
 
We also provide example annotations for *Homo sapiens* (genome assembly GCF_000001405.40), *Bos taurus* (genome assembly GCF_000003205.7) and *Delphinapterus leucas* (genome assembly GCF_002288925.1) that were generated at the time of writing the paper with Tiberius using the default weights:
```shell
wget https://bioinf.uni-greifswald.de/bioinf/tiberius/anno/Homo_sapiens.gtf.gz
wget https://bioinf.uni-greifswald.de/bioinf/tiberius/anno/Bos_taurus.gtf.gz
wget https://bioinf.uni-greifswald.de/bioinf/tiberius/anno/Delphinapterus_leucas.gtf.gz
```
[Human genome annotation](https://genome.ucsc.edu/s/MaSta/Tiberius_hg38) track on UCSC Genome Browser.


## References
  - Gabriel, Lars, et al. "Tiberius: End-to-End Deep Learning with an HMM for Gene Prediction." 2024, [*Bioinformatics*](https://academic.oup.com/bioinformatics/article/40/12/btae685/7903281), [bioRxiv](https://www.biorxiv.org/content/early/2024/07/23/2024.07.21.604459)
  - [Popular science podcast on this paper generated with NotebookLM](https://bioinf.uni-greifswald.de/bioinf/tiberius/pod/Tiberius.wav)
  - Processed RefSeq annotations used for training, validation and evaluation as described in the paper.  
`wget https://bioinf.uni-greifswald.de/bioinf/tiberius/anno/ref_annot.tar.gz`
