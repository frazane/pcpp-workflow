from ast import parse
from pathlib import Path
import json
from itertools import chain, product
import logging
from collections import defaultdict

import xarray as xr
from torch.utils.data import DataLoader
import torch
from torch import optim

import ray  # type: ignore
from ray import tune  # type: ignore
from ray.tune.schedulers import create_scheduler  # type: ignore
from ray.tune.search import create_searcher, ConcurrencyLimiter  # type: ignore

# from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune import Stopper

from common import (
    Data,
    DataLoader,
    Net,
    MultiTaskLoss,
    split_data,
    normalize_data,
    training_step,
    validation_step,
)

LOGGER = logging.getLogger(__name__)


class EarlyStopper(Stopper):
    def __init__(self, metric="nmae", patience=5):
        self.metric = metric
        self.patience = patience
        self.iterations = defaultdict(lambda: 0)
        self.best_metric = defaultdict(lambda: torch.inf)

    def __call__(self, trial_id, result):
        metric_result = result.get(self.metric)
        if metric_result < self.best_metric[trial_id]:
            self.best_metric[trial_id] = metric_result
            self.iterations[trial_id] = 0
        else:
            self.iterations[trial_id] += 1
        return self.iterations[trial_id] >= self.patience

    def stop_all(self):
        return False


def parse_search_space(search_space: dict) -> dict:
    for hp, v in search_space.items():
        target = v.pop("_target_")
        search_space[hp] = getattr(tune, target)(*v.values())
    return search_space

def train(
    config,
    train_datasets=None,
    val_datasets=None,
    run_config=None,
):

    torch.set_num_threads(run_config["cpus_per_trial"])

    train_dataloaders = [
        DataLoader(ds, batch_size=config["fit.batch_size"], shuffle=True)
        for ds in train_datasets
    ]
    val_dataloaders = [
        DataLoader(ds, batch_size=config["fit.batch_size"] * 8) for ds in val_datasets
    ]
    for epoch in range(300):
        cv_val_loss = 0.0
        cv_train_loss = 0.0
        cv_val_mae = 0.0
        cv_val_mae = 0.0
        cv_val_nmae = 0.0
        for i, (train_dl, val_dl) in enumerate(zip(train_dataloaders, val_dataloaders)):
            val_y_std = val_dl.dataset.y.std(dim=0)
            net_kwargs = dict(
                in_size=train_dl.dataset.x.shape[-1],
                n_stations=run_config["n_stations"],
                embedding_size=config["net.embedding_size"],
                l1=config["net.l1"],
                l2=config["net.l2"],
            )
            net = Net(**net_kwargs)

            loss_kwargs = dict(
                alpha=run_config["loss.alpha"], mask=run_config["loss.mask"]
            )
            loss = MultiTaskLoss(**loss_kwargs)

            optimizer = optim.Adam(
                chain(net.parameters(), loss.parameters()), lr=config["fit.lr"]
            )
            lr_scheduler = None

            if epoch > 0:
                with tune.checkpoint_dir(step=epoch - 1) as checkpoint_dir_:
                    checkpoint_dir_ = Path(checkpoint_dir_)
                    checkpoint = checkpoint_dir_ / f"checkpoint_{i}.pt"
                    net_state, loss_state, optimizer_state = torch.load(checkpoint)
                    net.load_state_dict(net_state)
                    loss.load_state_dict(loss_state)
                    optimizer.load_state_dict(optimizer_state)

            train_loss = training_step(
                net, loss, optimizer, train_dl, lr_scheduler=lr_scheduler
            )
            val_loss, val_p, val_mae = validation_step(net, loss, val_dl)
            val_nmae = torch.mean(val_mae / val_y_std)

            cv_train_loss += train_loss
            cv_val_loss += val_loss
            cv_val_mae += val_mae
            cv_val_nmae += val_nmae

            with tune.checkpoint_dir(step=epoch) as checkpoint_dir_:
                checkpoint_dir_ = Path(checkpoint_dir_)
                path = checkpoint_dir_ / f"checkpoint_{i}.pt"
                torch.save(
                    (net.state_dict(), loss.state_dict(), optimizer.state_dict()), path
                )

        cv_train_loss /= len(train_dataloaders)
        cv_val_loss /= len(val_dataloaders)
        cv_val_mae /= len(val_dataloaders)
        cv_val_nmae /= len(val_dataloaders)
        tune.report(
            train_loss=float(cv_train_loss),
            loss=float(cv_val_loss),
            nmae=float(cv_val_nmae),
            p=float(val_p),
            t_mae=float(cv_val_mae[0]),
            td_mae=float(cv_val_mae[1]),
            p_mae=float(cv_val_mae[2]),
            rh_mae=float(cv_val_mae[3]),
            r_mae=float(cv_val_mae[4]),
        )


def main(inputs, outputs, config, params):

    x = xr.open_dataset(inputs["x"]).to_array("var").transpose("s","var")
    y = xr.open_dataset(inputs["y"]).to_array("var").transpose("s","var")
    with open(inputs["stations_list"], "r") as f:
        stations_list = json.load(f)
    config["n_stations"] = len(stations_list["id"])
    config["cpus_per_trial"] = params["cpus_per_trial"]

    train_datasets = []
    val_datasets = []
    data_kwargs = {"reduction": config["data.reduction"], "season": config["data.train_season"]}
    for split in inputs["splits"]:
        with open(split, "r") as f:
            split = json.load(f)
        train_x, train_y, val_x, val_y = split_data(x, y, split)
        train_x, val_x = normalize_data(train_x, val_x)
        train_datasets.append(Data(train_x, train_y, **data_kwargs))
        val_datasets.append(Data(val_x, val_y, season=data_kwargs["season"]))

    scheduler_kws = dict(
        metric="nmae",
        time_attr=params["time_attr"],
        mode="min",
        max_t=params["max_t"],
        grace_period=params["grace_period"],
    )
    scheduler = create_scheduler("async_hyperband", **scheduler_kws)

    run_kws = dict(
        name=params["name"],
        local_dir=params["ray_results_dir"],
        resources_per_trial={"cpu": params["cpus_per_trial"]},
        num_samples=params["num_samples"],
        resume=params["resume"],
    )
    search_space = parse_search_space(params["search_space"])

    searcher_kws = dict(metric="nmae", mode="min")
    searcher = create_searcher("hyperopt", **searcher_kws)
    searcher = ConcurrencyLimiter(
        searcher, max_concurrent=params["max_concurrent_trials"], batch=False
    )

    stopper = EarlyStopper(metric="nmae", patience=5)
    result = tune.run(
        tune.with_parameters(
            train,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            run_config=config,
        ),
        config=search_space,
        scheduler=scheduler,
        search_alg=searcher,
        verbose=1,
        **run_kws,
        stop=stopper,
        reuse_actors=True,
    )

    best_config = result.get_best_config(metric="nmae", mode="min", scope="all")
    drop_cols = [
        "timesteps_total",
        "done",
        "episodes_total",
        "experiment_id",
        "date",
        "timestamp",
        "pid",
        "hostname",
        "node_ip",
        "time_since_restore",
        "timesteps_since_restore",
        "iterations_since_restore",
        "warmup_time",
        "experiment_tag",
    ]
    best_n = result.results_df.sort_values(by="nmae").head(5)
    best_n = best_n.drop(columns=drop_cols).to_dict("index")

    out_dir = Path(outputs[0])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    with open(out_dir / "best_n_config.json", "w") as f:
        json.dump(best_n, f, indent=4)


if __name__ == "__main__":

    workdir = Path(__file__).parents[1]
    ray.init(runtime_env={"working_dir": workdir / "scripts/"})

    SNAKEMAKE = snakemake  # type: ignore
    wildcards = SNAKEMAKE.wildcards
    inputs = SNAKEMAKE.input
    outputs = SNAKEMAKE.output
    config = SNAKEMAKE.config
    logfile = SNAKEMAKE.log[0]
    params = SNAKEMAKE.params[0]
    params["name"] = f"{wildcards.experiment}/{wildcards.approach}/"
    config = config | params.get("override", {})

    logging.basicConfig(
        **config["logging"],
        handlers=[logging.FileHandler(logfile)],
    )

    try:
        main(inputs, outputs, config, params)
    except Exception as e:
        LOGGER.exception("Exception occurred")
        raise e
