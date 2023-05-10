import json
import yaml
from pathlib import Path
import logging

import xarray as xr
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


def filter_stations(targets, threshold=0.9):
    """Only keep stations with completeness above a certain threshold."""
    dims = ["forecast_reference_time", "t"]
    n_stations = len(targets.station.values)
    ds = targets.stack(s=dims).to_array("var")
    missing = np.isnan(ds).sum("var")
    completeness = (missing == 0).sum("s") / len(ds.s)
    targets = targets.where(completeness > threshold, drop=True)
    n_bad_stations = n_stations - len(targets.station.values)
    LOGGER.info(f"Filtered out {n_bad_stations} out of {n_stations}")
    return targets


def reshape(features: xr.Dataset, targets: xr.Dataset) -> tuple[xr.DataArray]:
    """Reshape data to 2-d (sample, variable) tensors."""
    dims = ["forecast_reference_time", "t", "station"]
    x = (
        features.to_array("var")
        .stack(s=dims, create_index=False)
        .transpose("s", ..., "var")
    )
    y = (
        targets.to_array("var")
        .stack(s=dims, create_index=False)
        .transpose("s", ..., "var")
    )
    LOGGER.info(f"Reshaped: x -> {dict(x.sizes)} and y -> {dict(y.sizes)}")
    return x, y


def drop_missing(x: xr.DataArray, y: xr.DataArray) -> tuple[xr.DataArray]:
    """Only keep complete (all features and targets available) samples."""
    n_samples = len(x.s.values)
    mask_x_dims = [dim for dim in x.dims if dim != "s"]
    mask_y_dims = [dim for dim in y.dims if dim != "s"]
    x = x[np.isfinite(y).all(dim=mask_y_dims)]
    y = y[np.isfinite(y).all(dim=mask_y_dims)]
    y = y[np.isfinite(x).all(dim=mask_x_dims)]
    x = x[np.isfinite(x).all(dim=mask_x_dims)]
    n_incomplete_samples = n_samples - len(x.s.values)
    LOGGER.info(f"Dropped {n_incomplete_samples} incomplete samples out of {n_samples}")
    return x, y


# def unstack(ds):
#     import pandas as pd
#     dims = ["forecast_reference_time", "t", "station"]
#     samples = pd.MultiIndex.from_arrays([ds[dim].values for dim in dims], names=dims)
#     ds = ds.assign_coords(s=samples).unstack("s")


def main(inputs, outputs, config):

    LOGGER.info(f"Inputs: {inputs}")
    LOGGER.info(f"Outputs: {outputs}")

    features = xr.open_zarr(inputs["features"])
    model_height_difference = features["coe:model_height_difference"].isel(forecast_reference_time=0, drop=True)
    features = features[config["features"]]
    targets = xr.open_zarr(inputs["targets"])[config["targets"]]
    targets = targets.assign_coords(model_height_difference=model_height_difference)
    targets = targets.where(targets.owner_id == 1, drop=True).load()

    # filter stations and leadtimes
    targets = filter_stations(targets)
    features = features.reindex_like(targets).load()
    features = features.isel(t=slice(3, 24, 1))
    targets = targets.isel(t=slice(3, 24, 1))
    features = features.where(features.forecast_reference_time.dt.hour == 0, drop=True)
    targets = targets.where(targets.forecast_reference_time.dt.hour == 0, drop=True)

    LOGGER.info("Loading data...")
    LOGGER.info(targets.dims)
    features.load()
    targets.load()

    # reshape, clean
    x, y = reshape(features, targets)
    x, y = drop_missing(x, y)

    # add station id mapping
    stations_list = (
        targets.reset_coords()[
            ["elevation", "longitude", "latitude", "model_height_difference"]
        ]
        .to_dataframe()
        .to_dict()
    )

    station_id_map = {s: i for i, s in enumerate(targets.station.values)}
    stations_list["id"] = station_id_map
    station_id_coord = x.station.to_pandas().map(station_id_map).values
    x = x.reset_coords(["owner_id", "elevation", "longitude", "latitude", "model_height_difference"], drop=True)
    y = y.reset_coords(["owner_id", "elevation", "longitude", "latitude", "model_height_difference"], drop=True)
    x = x.assign_coords(station_id=("s", station_id_coord))

    

    # write
    LOGGER.debug(x)
    LOGGER.debug(y)

    Path(outputs["x"]).parent.mkdir(parents=True, exist_ok=True)
    x.to_dataset("var").to_netcdf(outputs["x"])
    y.to_dataset("var").to_netcdf(outputs["y"])
    with open(outputs["stations_list"], "w") as f:
        json.dump(stations_list, f, indent=4)
    LOGGER.info(f"Saved: {outputs['x']}, {outputs['y']} and {outputs['stations_list']}")


if __name__ == "__main__":

    try:
        SNAKEMAKE = snakemake  # type: ignore
        config = SNAKEMAKE.config
        inputs = SNAKEMAKE.input
        outputs = SNAKEMAKE.output
        logfile = SNAKEMAKE.log[0]
    except NameError:
        workdir = Path(__file__).parents[1]
        with open(workdir / "config/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        inputs = {
            "features": "/scratch/fzanetta/pcpp-workflow/data/01_raw/features.zarr",
            "targets": "/scratch/fzanetta/pcpp-workflow/data/01_raw/targets.zarr",
        }
        outputs = dict(
            x="results/preprocess/x.nc",
            y="results/preprocess/y.nc",
            stations_list="results/preprocess/stations_list.json",
        )
        logfile = "logs/preprocess/preprocess.log"

    logging.basicConfig(
        **config["logging"],
        handlers=[logging.FileHandler(logfile)],
    )

    try:
        main(inputs, outputs, config)
    except Exception as e:
        LOGGER.exception("Exception occurred")
        raise e
