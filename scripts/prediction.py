from pathlib import Path
import json
import logging

import numpy as np
import xarray as xr
import pandas as pd
import torch

from common import Data, Net

LOGGER = logging.getLogger(__name__)


def remove_source_prefix(ds):
    """Remove"""
    for var in ds.data_vars:
        _, name = var.split(":")
        ds = ds.rename({var: name})
    return ds


def unstack(ds):
    """Unstack dataset that does not have a multiindex"""
    dims = ["forecast_reference_time", "t", "station"]
    samples = pd.MultiIndex.from_arrays([ds[dim].values for dim in dims], names=dims)
    ds = ds.reset_coords(drop=True).assign_coords(s=samples).unstack("s")
    return ds


def to_xarray(tensor, coords):
    """Convert a 2D torch tensor to a xarray dataset"""
    ds = unstack(xr.DataArray(tensor.detach(), coords=coords)).to_dataset("var")
    return remove_source_prefix(ds)


def main(inputs, outputs, config):

    x = xr.open_dataset(inputs["x"]).to_array("var").transpose("s","var")
    y = xr.open_dataset(inputs["y"]).to_array("var").transpose("s","var")
    with open(inputs["split"], "r") as f:
        split = json.load(f)
    with open(inputs["config"], "r") as f:
        run_config = json.load(f)
    with open(inputs["stations_list"], "r") as f:
        stations = pd.DataFrame.from_dict(json.load(f))
        stations.index.name = "station"
        stations = stations.to_xarray()

    k = "forecast_reference_time"
    data_sel = x[k].isin(np.array(split[config["partition"]][k], dtype=np.datetime64))
    scaling_values = pd.read_json(inputs["scaling_values"])
    x, y = [x.where(data_sel, drop=True), y.where(data_sel, drop=True)]
    x = (x - scaling_values["mean"].values) / scaling_values["std"].values

    data_kwargs = {"reduction": config["data.reduction"], "season": config["data.test_season"]}
    dataset = Data(x.astype("float32"), y.astype("float32"), season=data_kwargs["season"])

    net = Net(**run_config["net"])
    net.load_state_dict(torch.load(inputs["state"]))

    preds = to_xarray(net(dataset.x, dataset.station_id), dataset.y_coords)
    preds = preds.reindex_like(stations).assign_coords(stations)
    preds.to_netcdf(outputs["predictions"])


if __name__ == "__main__":

    SNAKEMAKE = snakemake  # type: ignore
    config = SNAKEMAKE.config
    logfile = SNAKEMAKE.log[0]
    config["partition"] = SNAKEMAKE.wildcards.partition
    config = config | dict(SNAKEMAKE.params[0]["free"])
    config = config | dict(SNAKEMAKE.params[0]["fixed"])

    logging.basicConfig(
        **config["logging"],
        handlers=[logging.FileHandler(logfile)],
    )

    try:
        main(SNAKEMAKE.input, SNAKEMAKE.output, config)
    except Exception as e:
        LOGGER.exception("Exception occurred")
        raise e
