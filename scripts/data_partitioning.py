from itertools import product, combinations
from pathlib import Path
import json

import xarray as xr
import numpy as np
from sklearn import model_selection as ms
import pandas as pd
import matplotlib.pyplot as plt


def cluster_single_class(classes: list) -> list:
    """Cluster together single class occurrences."""
    classes = classes.copy()
    classes = [el if classes.count(el) > 1 else 0 for el in classes]
    assert classes.count(0) > 1
    return classes


def split_time(
    time_array: xr.DataArray,
    p: list,
    n_splits: int,
    shuffle: bool = False,
    seed: int = None,
    gap="5D",
):

    gap_idx = int(pd.Timedelta(gap) / np.mean(np.diff(time_array)))
    train_val, test = ms.train_test_split(
        time_array.values,
        test_size=p[-1],
        shuffle=shuffle,
        random_state=seed,
    )

    train_val = train_val[:-gap_idx]
    cv_kwargs = {"gap": gap, "random_state": seed}
    cv = UniformTimeSeriesSplit(n_splits, test_size=p[1] / (1 - p[-1]), **cv_kwargs)
    split_kwargs = {"X": train_val}

    splits = []
    for train, val in cv.split(**split_kwargs):
        split = {
            "train": np.sort(train).astype(str).tolist(),
            "val": np.sort(val).astype(str).tolist(),
            "test": np.sort(test).astype(str).tolist(),
        }
        splits.append(split)
    return splits


def plot_time_splits(time_array, splits, fn=None):
    time_array = time_array.values.astype("str").tolist()
    fig, axs = plt.subplots(len(splits), sharex=True, figsize=(10, 4))
    for i, split in enumerate(splits):
        non_gaps = set(split["train"]) | set(split["val"]) | set(split["test"])
        gaps = list(set(time_array) - non_gaps)
        for set_ in ["train", "val", "test"]:
            axs[i].scatter(
                pd.DatetimeIndex(split[set_]),
                [0.5] * len(split[set_]),
                marker="_",
                linewidth=10,
                label=set_,
            )
        axs[i].scatter(
            pd.DatetimeIndex(gaps),
            [0.5] * len(gaps),
            marker="_",
            linewidth=10,
            label="gap",
        )

        axs[i].set(yticks=[], yticklabels=[], ylim=(0.2, 0.8), ylabel=f"{i}")
    axs[0].legend(
        bbox_to_anchor=(0.0, 1.15, 1.0, 0.8),
        loc="lower left",
        ncol=4,
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )

    if fn is not None:
        plt.savefig(fn)

    return axs


class UniformTimeSeriesSplit:
    """Create train/test splits that are uniform throughout the year."""

    def __init__(
        self,
        n_splits: int = 4,
        test_size: float = None,
        year_fraction_interval: float = 1.0,
        gap: str = "5D",
        random_state: int = 1234,
    ):

        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = pd.Timedelta(gap)
        self.year_fraction_interval = year_fraction_interval
        self.random_state = random_state

    def split(
        self,
        X,
    ):
        X = pd.DatetimeIndex(X)
        interval = pd.Timedelta("365D") * self.year_fraction_interval
        gap_idx = self.gap / np.mean(np.diff(X))
        n_intervals_by_year = pd.Timedelta("365D") // interval
        n_intervals = np.ceil((X[-1] - X[0]) / interval).astype("int")
        n_test_intervals = np.floor(n_intervals * self.test_size).astype("int")

        day_intervals = (
            np.array(
                [pd.Timedelta(f"{day}D") // interval for day in range(1, 367)],
                dtype="int64",
            )
            + 1
        )
        map_day_to_interval = dict(zip(np.arange(1, 367, dtype="int64"), day_intervals))
        element_interval_idx = (
            X.day_of_year.map(map_day_to_interval).values
            + (X.year - X.year.min()).values * n_intervals_by_year
        )
        interval_array = np.hstack(list(range(1, n_intervals_by_year + 1)) * 20)[
            :n_intervals
        ]
        test_idx = np.array(
            list(combinations(range(n_intervals), n_test_intervals)), dtype="int"
        )
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(test_idx)

        good_splits = 0
        for i, split in enumerate(test_idx):
            if good_splits == self.n_splits:
                break
            is_good = set(interval_array[split]) == set(
                range(1, n_intervals_by_year + 1)
            )
            # is_good &= len(interval_array) - 1 not in split
            if not is_good:
                continue
            train = ~np.isin(element_interval_idx, split + 1)
            gaps = np.argwhere(np.diff(train) != 0) + np.arange(
                -gap_idx / 2, gap_idx / 2
            )
            gaps[gaps > len(X)] = len(X) - 1
            train[gaps.astype(int).ravel()] = False
            test = np.isin(element_interval_idx, split + 1)
            gaps = np.argwhere(np.diff(test) != 0) + np.arange(
                -gap_idx / 2, gap_idx / 2
            )
            gaps[gaps > len(X)] = len(X) - 1
            test[gaps.astype(int).ravel()] = False
            good_splits += 1
            yield X[train], X[test]


def main(inputs, outputs, config):
    """"""

    seed = 1
    dp_config = config["data_partitioning"]

    features_ds = xr.open_zarr(inputs["features"])
    time_array = features_ds.sel(forecast_reference_time=slice("2017-01-01", "2022-01-01")).forecast_reference_time
    station_array = features_ds.station

    time_splits = split_time(
        time_array, **dp_config["forecast_reference_time"], seed=seed
    )


    station_splits = [None]
    # rearrange dictionary and write json
    output_dir = Path(outputs[0]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (t, s) in enumerate(product(time_splits, station_splits)):
        split = {}
        for set in ["train", "val", "test"]:
            if s is None:
                split[set] = {"forecast_reference_time": t[set]}
            else:
                split[set] = {"forecast_reference_time": t[set], "station": s[set]}
        fn = Path(outputs[i])
        with open(fn, "w") as f:
            json.dump(split, f, indent=4)

    plot_time_splits(time_array, time_splits, output_dir / "time_split.png")


if __name__ == "__main__":

    # snakemake run
    try:
        SNAKEMAKE = snakemake  # type: ignore
        inputs = SNAKEMAKE.input
        outputs = SNAKEMAKE.output
        config = SNAKEMAKE.config

    # debugging (no snakemake)
    except NameError:
        print("Running this script in DEBUG mode.")
        import yaml

        workdir = Path(__file__).parents[2]
        with open(workdir / "config/config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        inputs = {"features": "data/features.zarr"}
        outputs = {}

    main(inputs, outputs, config)
