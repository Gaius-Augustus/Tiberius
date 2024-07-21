This document provides instructions on how to generate the data required for Tiberius in *de novo* mode. The data is generated with [ClaMSA](https://github.com/Gaius-Augustus/clamsa), a tool that uses multiple sequence alignments to generate evolutionary information for a given species. See instrutions on how to install [ClaMSA](https://github.com/Gaius-Augustus/clamsa) and how to train ClaMSA ([here](https://github.com/Gaius-Augustus/clamsa/blob/master/docs/usage-train.md)) and to create splice site predictions ([here](https://github.com/Gaius-Augustus/clamsa/blob/master/docs/usage-predict-sitewise.md)). Afterwards, you should have 6 files in UCSCs WIG format for the 6 possible reading frames. 

Tiberius de novo training and predicition uses ClaMSA data as numpy array as input. Following describes the steps to generate the data for one species for Tiberius in de novo mode:
1. Get sequence names from the genomic sequence of the FASTA file:
    ```shell
    grep ">" $genome | sed 's/>//g' > seq_names.txt
    ```
2. Convert ClaMSA wig files into numpy arrays. Note that the ClaMSA files have to be located in a directory `$inp_dir` and the output will be saved in `$out_dir`. You can provide the script with a prefix for the input and output files with `--prefix`. For example, if the files are named `Desmodus_rotundus_plus_0.wig`, `Desmodus_rotundus_minus_0.wig`, etc. you can use `--prefix Desmodus_rotundus_` and it will produce numpy arrays named `Desmodus_rotundus_chr1.npy`, `Desmodus_rotundus_chr2.npy`, etc.:
    ```shell
    python bin/wig2npy.py --inp_dir $inp_dir --out_dir $out_dir --prefix $prefix --seq_names seq_names.txt
    ```

