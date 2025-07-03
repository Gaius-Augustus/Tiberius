## Recommended Workflow for Training Tiberius with a Large Dataset

For training Tiberius with a large dataset we recommend generating tfRecords files. This will allow you to train on a large dataset without having to load the entire dataset into memory. The following steps will guide you through the process of generating tfrecords files from the data of several genomes. For each genome, you need a FASTA file with the genomic sequences (ideally softmasked) and a GTF file with the gene annotations.

For following instructions, we will assume that the files are named after the species as `${SPECIES}.fa` and `${SPECIES}.gtf`. And that [learnMSA](https://github.com/Gaius-Augustus/learnMSA) is installed at `$leanMSA`:

### 1. Prepare GTF files
Remove alternative transcripts from each GTF file. This is necessary because Tiberius does not support alternative splicing. A common way to do this would be to choose the alternative with the longest codeing sequence, you can use for example [`select_single_isoform.py`] for this:

    ```shell
    python tiberius/select_single_isoform.py  ${SPECIES}.gtf > ${SPECIES}.longest.gtf
    ```

### 2. Create tfrecords files for each species

Following  will create 100 tfrecords files for each genome in the directory `$tfrecords`. The tfrecords files will contain the genomic sequences and the gene annotations, split into trainings examples with a sequence length of `${seq_size}`, in the format that is required for training. For training with the mammalian genomes, we used a `${seq_size}` of `9999`, which has proven to be a reasonable choice.

```shell
python tiberius/write_tfrecord_species.py --fasta ${SPECIES}.fa --gtf ${SPECIES}.longest.gtf --out $tfrecords/${SPECIES} --wsize ${seq_size}
```

If you want to train in *de novo* mode, you have to generate ClaMSA data for each species. See [docs/clamsa_data.md](docs/clamsa_data.md) for instructions on how to generate the data. Afterwards, you should have a directory with files named `$clamsa/{prefix}{seq_name}.npz` for each sequence of your FASTA file and a file with the list of sequence names (`$seq_names`). 
You can create tfrecords files with the ClaMSA data with the following command:

```shell
python tiberius/write_tfrecord_species.py --fasta ${SPECIES}.fa --gtf ${SPECIES}.longest.gtf --out $tfrecords/${SPECIES} --clamsa $clamsa/{prefix} --seq_names $seq_names  --wsize ${seq_size}

```


### 3. Species List

Create a list of all species that you want to train on. This list (separated by `\n` ) will be used to search for all tfrecords file that start with the species name as `species.txt`, for example:    
```shell
Desmodus_rotundus
Dipodomys_ordii
Enhydra_lutris
```

### 4. (Optional) Prepare validation data. 

To monitor validation loss and accuracy during training, you can prepare validation examples by repeating steps 1–3 for a validation dataset. Assume you have:

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

### 5. Training 

#### 5.1 Standart Training without HMM Layer

Create a config file that contains the parameters for training, a config file with default parameter is located at `docs/config.json`. You can find descriptions of key parametes in `tiberius/train.py`. Start training:

```shell
python tiberius/train.py --data $tfrecords/ --learnMSA $leanMSA  --cfg config.json --train_species_file species.txt --val_data val.npz
```

If you want to train with the HMM layer, you can use the '--hmm' argument. This will however require more memory and slow training down.

You can also start a training from an existing model by providing the path to the model with the `--load` argument. This will continue training from the existing model.

#### 5.2 Training with HMM Layer

You can start a training with the HMM layer by using the `--hmm` argument. This will require more memory and will slow down training. You will need to reduce batch size compared to the training without the HMM layer. Usually, the training with HMM layer is done after first training without the HMM layer and the HMM layer is added to the model afterwards for fine-tuning. 
You can start a fine-tuning with the HMM layer by providing the path to the model without the HMM layer with the `--load_lstm` argument and using the `--hmm` argument:
```shell
python tiberius/train.py --data $tfrecords/ --learnMSA $leanMSA  --cfg config.json --train_species_file species.txt --val_data val.npz --load_lstm path/to/model_without_hmm --hmm
```

#### 5.3 Masking out transcript regions

If you want to mask out transcript regions during training. For this, you have to generate a list of transcript IDs from the preprocessed GTF files (one per line). And provide this list to the `--mask_tx_list` argument of `train.py`. During training, Tiberius will mask out the regions of the transcripts with a flanking region around these transcripts. You can set the number of bases of the flanking regions with `--mask_flank`. This means these regions will not be evaluated by the loss and completly ignored during training. This is useful for transcripts, where you have low confidence that these are correct. However, this feature is not systematically tested and is not yet clear how much the training benefits from this. 

Some details on the logic of the masking for the case that a unfiltered transcript overlaps with the flanking region of a filtered transcript:
- For the actual region of each transcript that is included in `mask_tx_list`, it is guaranteed that the region is masked out
- For the actual region of each transcript that is not included in `mask_tx_list`, it is guaranteed that the region is not masked out
- In case a transcript overlaps with the flanking region of a filtered transcript, it is guaranteed that around the unfiltered transcript a region of half `mask_flank` is *not* masked out. In these cases the flanking region around a filtered transcript is reduced.  

#### 5.4 Loading a trained model with Tiberius
You can load training save points using tiberius.py by providing the path with the appropriate argument:
- Use `--model_lstm_old` if the model was trained without the HMM layer.
- If you trained with the HMM layer (`--hmm`), use `--model_old` instead.

### 6. (Optional) Validation Loss and Accuracy after Training:
Once training completes, you can assess each save-point (e.g. epoch_*/ folders) on your validation set to compute validation loss and accuracy.

Prepare validation data by repeating steps 1–3 for a validation dataset. Assume you have:
- A directory containing validation TFRecord files: `val_tfrecords_dir/`
- A list of validation species: `val_species.txt`
- directory where savepoints (`epoch_*/`) are stored `train/`

Generate loss and accuracy for each save-point using the first K TFRecords per species (e.g. K=10):
```bash
python3 tiberius/validation_loss.py \
    --epochs_dir train \
    --species_list val_species.txt \
    --data_dir val_tfrecords_dir/ \
    --val_loss_out val.txt   --batch_size 200 --tfrec_per_species $K
```
You may want to adjust the batch size depending on your GPU's memory.