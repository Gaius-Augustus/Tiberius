### Recommended Workflow for Training Tiberius with a Large Dataset

For trainign Tiberius with a large dataset we recommend generating tfRecords files. This will allow you to train on a large dataset without having to load the entire dataset into memory. The following steps will guide you through the process of generating tfrecords files from the data of several genomes. For each genome, you need a FASTA file with the genomic sequences (ideally softmasked) and a GTF file with the gene annotations.

For following instructions, we will assume that the files are named after the species as `${SPECIES}.fa` and `${SPECIES}.gtf`. And that [learnMSA](https://github.com/Gaius-Augustus/learnMSA) is installed at `$leanMSA`:

1. Remove alternative transcripts from each GTF file. This is necessary because Tiberius does not support alternative splicing. A common way to do this would be to choose the alternative with the longest codeing sequence, you can use for example [`get_longest_isoform.py`](https://github.com/Gaius-Augustus/TSEBRA/blob/main/bin/get_longest_isoform.py) for this:

    ```shell
    python bin/get_longest_isoform.py --gtf ${SPECIES}.gtf --out ${SPECIES}.longest.gtf
    ```

2. Create tfrecords files for each species. This will create 100 tfrecords files for each genome in the directory `$tfrecords`. The tfrecords files will contain the genomic sequences and the gene annotations, split into trainings examples with a sequence length of `${seq_size}`, in the format that is required for training. For training with the mammalian genomes, we used a `${seq_size}` of `9999`, which has proven to be a reasonable choice.

    ```shell
    python bin/write_tfrecord_species.py --fasta ${SPECIES}.fa --gtf ${SPECIES}.longest.gtf --out $tfrecords/${SPECIES} --wsize ${seq_size}
    ```

    If you want to train in *de novo* mode, you have to generate ClaMSA data for each species. See [docs/clamsa_data.md](docs/clamsa_data.md) for instructions on how to generate the data. Afterwards, you should have a directory with files named `$clamsa/{prefix}{seq_name}.npz` for each sequence of your FASTA file and a file with the list of sequence names (`$seq_names`). 
    You can create tfrecords files with the ClaMSA data with the following command:

    ```shell
    python bin/write_tfrecord_species.py --fasta ${SPECIES}.fa --gtf ${SPECIES}.longest.gtf --out $tfrecords/${SPECIES} --clamsa $clamsa/{prefix} --seq_names $seq_names  --wsize ${seq_size}
    
    ```


3. Create a list of all species that you want to train on. This list (separated by `\n` ) will be used to search for all tfrecords file that start with the species name, for example:    
    ```shell
    Desmodus_rotundus
    Dipodomys_ordii
    Enhydra_lutris
    ```

4. Create a config file that contains the parameters for training, a config file with default parameter is located at `docs/config.json`. You can find descriptions of key parametes in `bin/train.py`. Start training:
    
    ```shell
    python bin/train.py --data $tfrecords/ --learnMSA $leanMSA  --cfg config.json
    ```

    If you want to train without the HMM layer, you can set 'use_hmm' to false in the config file. This will speed up training and reduce the memory requirements. 

    You can also start a training from an existing model by providing the path to the model with the `--load` argument. This will continue training from the existing model.
