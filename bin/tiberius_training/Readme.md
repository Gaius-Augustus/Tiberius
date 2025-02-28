This repository contains a Snakemake workflow and a helper Python script to generate training data for a TensorFlow model. The pipeline processes species data by running BUSCO, filtering annotations to keep only the longest isoforms, and generating TFRecords for each species. Species are split into training, validation, and test sets as specified in a configuration file.

## Directory Structure
Your input data is expected to be organized as follows:

````
data/
  ├── <species_name>/
       ├── genome/
             └── genome.fa
       ├── annot/
             ├── annot.fa
             └── pseudo.gff3
       └── prot/
             └── protein.faa
````

## Generate Config
This Python script automatically generates a YAML configuration file (config.yaml) by scanning the provided data directory and assigning species to training, validation, or test splits:
```
./create_config.py data/ --val species3 species5 --test species7
```
- data/: Path to the main data directory.
- --val: A list of species names to assign to the validation set.
- --test: A list of species names to assign to the test set.
- Any species in the data directory not specified under validation or test will automatically be assigned to the training set.

Example Config:
```yaml
species_split:
  train:
    - species1
    - species2
  val:
    - species3
  test:
    - species4
```

## Snakemake Workflow
The Snakemake workflow defines the following steps for each species:
1. BUSCO Analysis:
Runs BUSCO on the protein file.

```
busco -m protein -i data/{species}/prot/protein.faa -o data/{species}/BUSCO -l viridiplantae_odb11
```

2. Get Longest Isoform for each Species:
```
get_longest_isoform.py data/{species}/annot/annot.fa data/{species}/annot/annot_longest.fa
```

3. TFRecord Generation:
Generates TFRecords for each species and stores them in the proper subdirectory (tfrecords/train, tfrecords/val, or tfrecords/test) based on the config.
```
write_tfrecord_species.py --gtf data/{species}/annot/annot_longest.fa --fasta data/{species}/genome/genome.fa --out tfrecords/<split>/{species}
```

### Running the Workflow
Once you have generated the config.yaml and verified your data directory structure, run the workflow using Slurm:

```
snakemake -s Snakefile_dataprep --executor slurm --default-resources slurm_account=none slurm_partition=batch --jobs=10 --use-apptainer
```
