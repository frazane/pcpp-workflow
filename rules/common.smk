from dataclasses import dataclass, field
from itertools import product
import pandas as pd

VALUE_SEP = "~"
PARAM_SEP = "-"


def parametrization_patterns(params_list: list[dict]) -> str:
    """Returns a formatted string like `param1~value1-param2~value2`"""
    if len(params_list) == 0:
        params = {}
    else:
        params = list(params_list[0].keys())
    pattern = PARAM_SEP.join([r"{}"] * (len(params)))

    instance_patterns = []
    for params in params_list:
        param_values = []
        for param, value in params.items():
            param_values.append(VALUE_SEP.join(("{}", "{}")).format(param, value))
        instance_patterns.append(pattern.format(*param_values))
    
    return instance_patterns


def params_from_wildcards(wildcards: snakemake.io.Wildcards) -> dict:
    wildcard_string = wildcards.params
    free = wildcard_string.split("-")
    free = dict([p.split("~") for p in free])

    for param in free:
        if param == "":
            free = {}
            break
        try:
            free[param] = eval(free[param])
        except:
            free[param] = free[param]

    fixed = config["experiments"][wildcards.experiment].get("param_override", {})

    return {"free": free, "fixed": fixed}


def experiment_params(experiment: str) -> list[dict]:
    """Return a list of parametrization dictionaries, one for each experiment run."""
    approaches = config["experiments"][experiment]["approaches"]

    grid = config["experiments"][experiment].get("param_grid", None)
    runs = config["experiments"][experiment].get("param_runs", None)

    if grid is None and runs is None:
        params_list = []
    elif grid is not None and runs is not None:
        raise ValueError("grid_params and grid_runs are mutually exclusive!")
    elif grid is not None:
        params_list = [dict(zip(list(grid.keys()), x)) for x in product(*grid.values())]
    elif runs is not None:
        params_list = runs

    return params_list

def experiment_inputs(wildcards: str) -> list[str]:
    """Return the input files needed for a given experiment."""
    experiment = wildcards.experiment
    partition = wildcards.partition
    cfg = config["experiments"][experiment]
    approaches = cfg["approaches"]

    splits = cfg["splits"] if cfg.get("splits", None) is not None else range(N_SPLITS)
    seeds = cfg["seeds"] if cfg.get("splits", None) is not None else config["random_seeds"]

    params_list = experiment_params(experiment)
    train_params = parametrization_patterns(params_list)
    if len(train_params) == 0:
        train_params = "~" 

    RUN_DIR_PATTERN = "03_model_output/split~{split}/seed~{seed}/{approach}/{experiment}/{params}/{partition}_predictions.nc"


    filenames = expand(
        RUN_DIR_PATTERN,
        split=splits,
        seed=seeds,
        approach=approaches,
        experiment=experiment,
        params=train_params,
        partition=partition,
        allow_missing=True
    )
    filenames = [fn.replace(" ", "") for fn in filenames] # if parameter is a list, we must avoid spaces
    filenames = [DATA_DIR / fn for fn in filenames]
    return filenames