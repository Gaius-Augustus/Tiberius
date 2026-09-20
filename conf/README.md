# Nextflow configs

The evidence pipeline, its configs and its documentation live in the
[Paludamentum](https://github.com/Gaius-Augustus/Paludamentum) submodule at
`paludamentum/`. The `.config` files in this directory only include their
counterparts there, so that `--nf_config conf/<name>.config` keeps working.

- Parameters of the params YAML file: [paludamentum/docs/parameters.md](../paludamentum/docs/parameters.md)
- Template for the params file: [paludamentum/conf/parameters.yaml](../paludamentum/conf/parameters.yaml)
- Adapting the pipeline to an HPC: [paludamentum/docs/hpc.md](../paludamentum/docs/hpc.md)
- Template for your own cluster config: [paludamentum/conf/user_hpc_template.config](../paludamentum/conf/user_hpc_template.config)

If `paludamentum/` is empty, run `git submodule update --init --recursive`.
