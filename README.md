# Physics-constrained deep learning postprocessing of temperature and humidity

Workflow for the physics-constrained deep learning postprocessing of temperature and humidity (paper under review). This work investigates the effect of enforcing dependencies between variables by constraining the optimization of neural networks with thermodynamic state equations. 

Pre-print: [https://arxiv.org/abs/2212.04487](https://arxiv.org/abs/2212.04487)

A self-contained, minimal example of this work was presented by Tom Beucler in the "Physics-Guided Machine Learning" e-learning module of ECMWF's MOOC on Machine Learning in Weather and Climate. <a href="https://colab.research.google.com/github/tbeucler/2023_MOOC_ECMWF/blob/main/Physically_constrained_postprocessing.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


__Installation:__
If using conda, simply replace mamba with conda. We reccommend you set the two environment variables that indicate where the workflow environments and data will be located. By default, these will be located in a `.snakemake/conda` and `data/` respectively.
```
mamba env create -f environment.yaml
mamba env config vars set SNAKEMAKE_CONDA_PREFIX=<path> SNAKEMAKE_DATA_DIR=<path>
```

Visualize the workflow:
```
snakemake all_results --dag | dot -Tpdf > dag.pdf
snakemake all_results --rulegraph | dot -Tpdf > rulegraph.pdf
```
