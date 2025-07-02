### Recommended Workflow for Training Tiberius with a Large Dataset

For training Tiberius with a large dataset we recommend generating tfRecords files. This will allow you to train on a large dataset without having to load the entire dataset into memory. The following steps will guide you through the process of generating tfrecords files from the data of several genomes. For each genome, you need a FASTA file with the genomic sequences (ideally softmasked) and a GTF file with the gene annotations.

For following instructions, we will assume that the files are named after the species as `${SPECIES}.fa` and `${SPECIES}.gtf`. And that [learnMSA](https://github.com/Gaius-Augustus/learnMSA) is installed at `$leanMSA`:

1. Remove alternative transcripts from each GTF file. This is necessary because Tiberius does not support alternative splicing. A common way to do this would be to choose the alternative with the longest codeing sequence, you can use for example [`select_single_isoform.py`] for this:

    ```shell
    python tiberius/select_single_isoform.py  ${SPECIES}.gtf > ${SPECIES}.longest.gtf
    ```

2. Create tfrecords files for each species. This will create 100 tfrecords files for each genome in the directory `$tfrecords`. The tfrecords files will contain the genomic sequences and the gene annotations, split into trainings examples with a sequence length of `${seq_size}`, in the format that is required for training. For training with the mammalian genomes, we used a `${seq_size}` of `9999`, which has proven to be a reasonable choice.

    ```shell
    python tiberius/write_tfrecord_species.py --fasta ${SPECIES}.fa --gtf ${SPECIES}.longest.gtf --out $tfrecords/${SPECIES} --wsize ${seq_size}
    ```

    If you want to train in *de novo* mode, you have to generate ClaMSA data for each species. See [docs/clamsa_data.md](docs/clamsa_data.md) for instructions on how to generate the data. Afterwards, you should have a directory with files named `$clamsa/{prefix}{seq_name}.npz` for each sequence of your FASTA file and a file with the list of sequence names (`$seq_names`). 
    You can create tfrecords files with the ClaMSA data with the following command:

    ```shell
    python tiberius/write_tfrecord_species.py --fasta ${SPECIES}.fa --gtf ${SPECIES}.longest.gtf --out $tfrecords/${SPECIES} --clamsa $clamsa/{prefix} --seq_names $seq_names  --wsize ${seq_size}
    
    ```


3. Create a list of all species that you want to train on. This list (separated by `\n` ) will be used to search for all tfrecords file that start with the species name as `species.txt`, for example:    
    ```shell
    Desmodus_rotundus
    Dipodomys_ordii
    Enhydra_lutris
    ```
4. (Optional) Prepare validation data. To monitor validation loss and accuracy during training, you can prepare validation examples by repeating steps 1–3 for a validation dataset. Assume you have:

-  A directory containing validation TFRecord files: `val_tfrecords_dir/`
- A list of validation species: `val_species.txt`

You can then generate a `.npz` file containing a fixed number of validation examples (e.g., `num_val` = 500) using the following command:
```bash
python3 tiberius/validation_from_tfrecords \
        --tfrec_dir val_tfrecords_dir/ \
       --species val_species.txt \
       --tfrec_per_species 100 \
       --out val.npz --val_size num_val
```

5. Create a config file that contains the parameters for training, a config file with default parameter is located at `docs/config.json`. You can find descriptions of key parametes in `tiberius/train.py`. Start training:
    
    ```shell
    python tiberius/train.py --data $tfrecords/ --learnMSA $leanMSA  --cfg config.json --train_species_file species.txt --val_data val.npz
    ```

    If you want to train with the HMM layer, you can use the '--hmm' argument. This will however require more memory and slow training down.

    You can also start a training from an existing model by providing the path to the model with the `--load` argument. This will continue training from the existing model.




You can load training save points using tiberius.py by providing the path with the appropriate argument:
- Use `--model_lstm_old` if the model was trained without the HMM layer.
- If you trained with the HMM layer (`--hmm`), use `--model_old` instead.
