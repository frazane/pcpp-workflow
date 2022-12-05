# pcpp-workflow
Workflow for the physics-constrained deep learning postprocessing of temperature and humidity. This work investigates the effect of enforcing dependencies between variables by constraining the optimization of neural networks with thermodynamic state equations. The pre-print is available at: []
A colab-notebook containing the relevant code and a minimal reproducible example is availabile here [].

## Reproducing the complete workflow
Due to restrictions by our data provider we cannot publish the full dataset used to c
## Data availability

## Installation

Requirements:

* [Mamba package manager](https://github.com/conda-forge/miniforge#mambaforge)
* [Snakemake workflow management system](https://snakemake.github.io)

Download and install Mambaforge via

    curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
    bash Mambaforge-Linux-x86_64.sh

Clone the `pcpp-workflow` repository and cd into the project root directory

    git clone git@github.com:frazane/pcdlpp-workflow.git
    cd pcdlpp-workflow

Create and activate the conda environment for snakemake via

    mamba env create -f environment.yml
    conda activate snakemake

The `config/config.yaml` file contains the default configurations, in `config/experiments.yaml` we defined experiment-specific configurations.


## Environment variables

It is possible to specify few optional environment variables to
control where datasets are stored (defaults to `data/`) with `SNAKEMAKE_DATA_DIR`
and where the conda environments are created (defaults to `.snakemake`) with `SNAKEMAKE_CONDA_PREFIX`.

To permanently set those variables within this project only, use
conda env config vars set `SNAKEMAKE_DATA_DIR=<path>` and `SNAKEMAKE_CONDA_PREFIX=<path>`.
Alternatively, these can be set in the `environment.yaml` file.

To reproduce the entire workflow we recommend working on a cluster (see [here](https://snakemake.readthedocs.io/en/stable/executing/cluster.html#)). If you are using SLURM, you can use the profile provided in the `profile` directory. You can find more profiles for other job scheduling systems [here](https://github.com/snakemake-profiles/doc).

    snakemake all --profile profile 

to reproduce a single experiment, use

    snakemake <experiment name> --profile profile