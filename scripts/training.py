from pathlib import Path
import json
from itertools import chain
import logging
import multiprocessing

import numpy as np
import xarray as xr
import torch  # type: ignore
import torch.optim as optim  # type: ignore


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

TASKS = [
    "air_temperature",
    "dew_point_temperature",
    "surface_air_pressure",
    "relative_humidity",
    "water_vapor_mixing_ratio",
]

LOGGER = logging.getLogger(__name__)


def main(inputs, outputs, config):

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    LOGGER.info(f"System info: cpu_count={multiprocessing.cpu_count()}")
    LOGGER.info(f"Inputs: {inputs}")
    LOGGER.info(f"Outputs: {outputs}")
    LOGGER.info(f"Approach: {config['approach']}")

    x = xr.open_dataset(inputs["x"]).to_array("var").transpose("s","var")
    y = xr.open_dataset(inputs["y"]).to_array("var").transpose("s","var")
    with open(inputs["split"], "r") as f:
        split = json.load(f)
    with open(inputs["stations_list"], "r") as f:
        stations_list = json.load(f)

    train_x, train_y, val_x, val_y = split_data(x, y, split)
    train_x, val_x = normalize_data(train_x, val_x, save_json=outputs["scaling_values"])

    # data_kwargs = {k.split(".")[1].split("_")[-1]: v for k, v in config.items() if "data." in k}
    data_kwargs = {"reduction": config["data.reduction"], "season": config["data.train_season"]}
    train = Data(train_x, train_y, **data_kwargs)
    val = Data(val_x, val_y, season=data_kwargs["season"])

    LOGGER.info(f"Training dataset:\n{train}")
    LOGGER.info(f"Validation dataset:\n{val}")

    train_dl = DataLoader(train, batch_size=config["fit.batch_size"], shuffle=True)
    val_dl = DataLoader(val, batch_size=config["fit.batch_size"] * 8)

    LOGGER.info("Dataloaders ready.")

    # model
    net_kwargs = dict(
        in_size=len(x["var"]),
        n_stations=len(stations_list["id"].keys()),
        embedding_size=config["net.embedding_size"],
        l1=config["net.l1"],
        l2=config["net.l2"],
        constraint=config["net.constraint"],
    )
    net = Net(**net_kwargs)
    LOGGER.info(f"Model configuration: {net_kwargs}")
    LOGGER.info(f"Model:\n{net}")

    loss_kwargs = dict(alpha=config["loss.alpha"], mask=config["loss.mask"], trainable=config["loss.trainable"])
    loss = MultiTaskLoss(**loss_kwargs)
    LOGGER.info(f"Loss configuration: {loss_kwargs}")

    fit_kwargs = {k: v for k, v in config.items() if "fit." in k}
    LOGGER.info(f"Fit configuration: {fit_kwargs}")

    optimizer = optim.Adam(
        chain(net.parameters(), loss.parameters()), lr=config["fit.lr"]
    )

    LOGGER.info(f"Start training: num_threads = {torch.get_num_threads()}")
    best_val_nmae = torch.inf
    val_y_var = val.y.var(dim=0)
    val_y_std = val.y.std(dim=0)
    for epoch in range(config["fit.max_epochs"]):

        train_loss = training_step(net, loss, optimizer, train_dl)
        val_loss, val_p, val_mae = validation_step(net, loss, val_dl)
        val_nmae = torch.mean((val_mae / val_y_std)[config["loss.mask"]])

        if val_nmae < best_val_nmae:
            best_val_nmae = val_nmae
            patience = 0
            torch.save(net.state_dict(), outputs["state"])
        else:
            patience += 1
            if patience == config["fit.patience"]:
                break

        LOGGER.info(
            f"{epoch+1:<2}{'':>4}loss: {train_loss:<10.4}val_loss: {val_loss:<10.4}"
            f"val_t_mae: {val_mae[0]:<10.4} val_rh_mae: {val_mae[3]:<10.4}"
            f"val_nmae: {val_nmae:<10.4} val_p: {val_p:<10.4}"
        )

    loss_kwargs["log_var_init"] = loss.log_var.detach().tolist()
    run_config = {"net": net_kwargs, "loss": loss_kwargs, "fit": fit_kwargs}
    with open(outputs["config"], "w") as f:
        json.dump(run_config, f, indent=4)

    LOGGER.info("Done! \N{SNAKE}")


if __name__ == "__main__":

    SNAKEMAKE = snakemake  # type: ignore
    config = SNAKEMAKE.config
    config["approach"] = SNAKEMAKE.wildcards.approach
    config["experiment"] = SNAKEMAKE.wildcards.experiment
    config["seed"] = int(SNAKEMAKE.wildcards.seed)
    config["split"] = int(SNAKEMAKE.wildcards.split)
    config = config | config[config["approach"]]

    config = config | dict(SNAKEMAKE.params[0]["free"])
    config = config | dict(SNAKEMAKE.params[0]["fixed"])
    inputs = SNAKEMAKE.input
    outputs = SNAKEMAKE.output
    logfile = SNAKEMAKE.log[0]

    logging.basicConfig(
        **config["logging"],
        handlers=[logging.FileHandler(logfile)],
    )

    try:
        main(inputs, outputs, config)
    except Exception as e:
        LOGGER.exception("Exception occurred")
        raise e
