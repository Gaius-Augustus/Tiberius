# Parameters YAML Guide for Tiberius
Tiberius allows supplying input data and settings via a single YAML file.
You pass the YAML file to the pipeline using the wrapper:
```bash
./tiberius.py --params-yaml parameters.yaml -c ~/nf_configs/mycluster.config
```
## Input Data
### Genomic Sequences (Required Input)
Path to the genome FASTA file.

Example:
```bash
genome: /path/to/genome.fa
```

### Protein Sequences (Required Input)
Path to one or more FASTA files with protein sequences aligned to the target genome.
Multiple files are concatenated in input order.

Example:
```bash
proteins: /path/to/proteins.faa
```

Or a list:
```bash
proteins: [
  /path/to/proteins1.faa,
  /path/to/proteins2.faa,
]
```

### ODB12 Partitions (Optional)
Provide one or more named partitions to download and merge into the protein evidence.
Available partitions: `Metazoa`, `Vertebrata`, `Viridiplantae`, `Arthropoda`, `Fungi`,
`Alveolata`, `Stramenopiles`, `Amoebozoa`, `Euglenozoa`, `Eukaryota`.

Example:
```bash
odb12Partitions: [
  Metazoa,
  Fungi,
]
```

### RNA-Seq/Iso-Seq (Local)
Tiberius accepts local long and short read data, add the absolut paths to your files to :
`rnaseq_single`, `rnaseq_paired`, `isoseq`. 

You can also use a pattern instead of listing all files, for example for paired reads:
`rnaseq_paired: "/path/to/rnaseq/*_{1,2}.fastq.gz"`


### RNA-Seq/Iso-Seq (SRA-download)
RNA-Seq ad Iso-Seq libraries can also be automatically downloaded from the Sequence-Read-Archive by specifying their SRA IDs at
`rnaseq_sra_paired`, `rnaseq_sra_single`, `isoseq_sra`.  


## Parameters
You can also set parameters of the pipeline within the file, default parameters that you can overwrite are:

| Parameter                  | Default Value                          | Description                                                                                                  |
| -------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `threads`                  | `48`                                   | Global default number of CPU threads used by many processes.                                                 |
| `outdir`                   | `"results"`                            | Directory where all final results are written.                                                               |
| `scoring_matrix`           | `"conf/blosum62.csv"` | Amino acid substitution scoring matrix used by homology-based tools.                                         |

The location of the required executables is set by default so that they are available in your path, unless you are using the Singularity container:
| Tool Parameter                       | Default Value                           | Description                                                         |
| ------------------------------------ | --------------------------------------- | ------------------------------------------------------------------- |
| `tools.hisat2`                       | `"hisat2"`                              | Path or module name for HISAT2 aligner.                             |
| `tools.hisat2_build`                 | `"hisat2-build"`                        | Path to HISAT2 genome index builder.                                |
| `tools.minimap2`                     | `"minimap2"`                            | Path to Minimap2 aligner (used for Iso-Seq).                        |
| `tools.stringtie`                    | `"stringtie"`                           | Path to StringTie3 for transcript assembly.                          |
| `tools.samtools`                     | `"samtools"`                            | Path to Samtools for BAM/FASTQ operations.                          |
| `tools.transdecoder_longorfs`        | `"TransDecoder.LongOrfs"`               | Tool to detect long ORFs for protein prediction.                    |
| `tools.transdecoder_predict`         | `"TransDecoder.Predict"`                | Main TransDecoder protein prediction step.                          |
| `tools.transdecoder_util_gtf2fa`     | `"gtf_genome_to_cdna_fasta.pl"`         | Utility for converting GTF + genome → cDNA FASTA.                   |
| `tools.transdecoder_util_orf2genome` | `"cdna_alignment_orf_to_genome_orf.pl"` | Maps ORF predictions from cDNA-level to genome coordinates.         |
| `tools.transdecoder_gtf2gff`         | `"gtf_to_alignment_gff3.pl"`            | Converts GTF to alignment-style GFF3 for TransDecoder.              |
| `tools.diamond`                      | `"diamond"`                             | Diamond alignment tool for protein-to-genome searches.              |
| `tools.bedtools`                     | `"bedtools"`                            | Bedtools for genomic interval operations.                           |
| `tools.miniprot`                     | `"miniprot"`                            | MiniProt — protein-to-genome aligner.                               |
| `tools.miniprot_boundary_scorer`     | `"miniprot_boundary_scorer"`            | Additional MiniProt scoring utility for boundary refinement.        |
| `tools.miniprothint`                 | `"miniprothint.py"`                     | MiniProtHint wrapper script for generating protein hints.           |
| `tools.bam2hints`                    | `"bam2hints"`                           | Converts RNA-Seq BAM alignments into genome annotation hint format. |


## Tiberius

To enable Tiberius set `tiberius.run: true` and choose the correct model parameter for your model, see [Tiberius/model_cfg](https://github.com/Gaius-Augustus/Tiberius/tree/main/model_cfg) for a complete list of availble Tiberius parameter.

For the Tiberius prediction, the genome is split into smaller FASTA files so that it can run on multiple GPUs. You can set the upper limit of parallel Tiberius tasks with `tiberius.max_files` and the minimal size of a file with `tiberius.min_split_size`.
You can also provide the result of Tiberius to the pipeline so that it is used but not rerun, user here `tiberius.result`.

# Adapting Tiberius to an HPC

This section explains how to configure the nextflow config file for a specific HPC environment (queues, scratch paths, GPU settings, etc.).

---

## 1. Create your personal HPC config

Copy the template shipped with the repository:

```bash
mkdir -p ~/nf_configs
cp conf/ user_hpc_template.config mycluster.config
```

Otherwise, use one of the example configs `local.config` for non HPC usage and `slurm_generic` a generic example for a slurm HPC.

## 2. Edit the config for your cluster

Open `mycluster.config` and adjust:

### Queues / partitions

```bash
process {
  executor = 'slurm'
  queue    = 'batch'  // change to your default queue
  time     = '12h'
  cpus     = 8
  memory   = '32 GB'

  withLabel: gpu {
    queue          = 'gpu'          // your GPU queue
    cpus           = 16
    memory         = '64 GB'
    time           = '24h'
    clusterOptions = '--gpus=1'     // or '--gres=gpu:1' etc.
    containerOptions = '--nv'
  }
}
```

Set the correct queue/partition names and any required `clusterOptions`
(e.g. constraints, accounts, QoS, etc.) for non GPU tasks like Miniprot and for GPU tasks if you want to run Tiberius. 
Make sure to include `containerOptions = '--nv'` for GPU processes if you want to use the Singularity container.


###  Singularity

If you want to use the Singularity container for the dependencies:
```
singularity {
  enabled    = true
  autoMounts = true
  // envWhitelist = 'CUDA_VISIBLE_DEVICES,NVIDIA_VISIBLE_DEVICES'
}
```
For Slurm tou need to set `envWhitelist = 'CUDA_VISIBLE_DEVICES'`
Make sure to include `containerOptions = '--nv'` to include in the GPU process section. 
