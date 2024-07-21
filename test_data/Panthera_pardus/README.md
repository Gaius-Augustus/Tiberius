Example data is included in `inp.tar.gz.`
```bash
tar -xvzf inp.tar.gz
```
This produces a directory `inp` with the following contents:
```bash
inp/annot.gtf # Reference annotation for evaluation and training
inp/genome.fa # Genomic sequence as input for Tiberius
```

Jupyther notebooks as examples for training or prediction are included in:
* `example_prediction.ipynb` for loading data of a single genome and using Tiberius code to predict genes.
* `example_training_lstm.ipynb` for training preHMM of Tiberius on a single genome.
* `example_training_hmm.ipynb` for training Tiberius (including HMM layer) on a single genome.