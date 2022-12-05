import pandas as pd
import os
from functools import partial 

configfile: "config/config.yaml"
configfile: "config/experiments.yaml"
configfile: "config/extras.yaml"
configfile: "config/tuning.yaml"

APPROACHES = [
    "unconstrained",
    "loss_constrained",
    "architecture_constrained",
    "offline_constrained"
]

TASKS = [
    "air_temperature",
    "dew_point_temperature",
    "surface_air_pressure",
    "relative_humidity",
    "water_vapor_mixing_ratio",
]

N_SPLITS = config["data_partitioning"]["forecast_reference_time"]["n_splits"]
DATA_DIR = Path(os.getenv("SNAKEMAKE_DATA_DIR", "data/"))


include: "rules/common.smk"
include: "rules/eda.smk"
include: "rules/main.smk"
include: "rules/extras.smk"

# rule all_eda:
#     "results/eda/stations/", "results/eda/model"

rule all_results:
    input:
        expand(
            "results/experiments/{experiment}/{partition}/{focus}",
            partition=["test"], 
            experiment=["default","time_generalization"],
            focus=["performance","physical_consistency"]
        ),
        expand(
            "results/experiments/{experiment}/{partition}/analysis",
            partition=["test"],
            experiment=["loss_alpha", "data_efficiency", "time_generalization"]
        ),
        expand(
            "results/experiments/{experiment}/{partition}/physical_consistency",
            partition=["test"], 
            experiment=["data_reduction_consistency"]
        )


rule all_performance:
    input:
        expand(
            "results/experiments/{experiment}/{partition}/performance",
            partition=["test", "train"], 
            experiment=["default","time_generalization"]
        )


rule all_physical_consistency:
    input:
        expand(
            "results/experiments/{experiment}/{partition}/physical_consistency",
            partition=["test"], 
            experiment=["default","time_generalization","data_reduction_consistency"]
        )


rule all_analysis:
    input:
        expand(
            "results/experiments/{experiment}/{partition}/analysis",
            partition=["test"],
            experiment=["loss_alpha", "data_efficiency", "time_generalization"]
        )



