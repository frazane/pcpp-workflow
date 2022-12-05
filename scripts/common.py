import json

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch.utils.data import Dataset  # type: ignore

import numpy as np
import pandas as pd


class Data(Dataset):
    def __init__(self, x, y, season=None, reduction=None, station_ids=True):
        self.season = season
        self.reduction = reduction
        if season is not None:
            x = x.where(x.forecast_reference_time.dt.season.isin(list(season)), drop=True)
            y = y.where(y.forecast_reference_time.dt.season.isin(list(season)), drop=True)

        if reduction is not None:
            ids = x.station_id.values

            x = (x.to_dataset("var")
                .to_dataframe().groupby(ids).sample(frac=reduction, random_state=0)
                .to_xarray().set_coords(c for c in list(x.coords) if c != "var")
                .to_array("var").transpose("s","var")
            )
            y = (y.to_dataset("var")
                .to_dataframe().groupby(ids).sample(frac=reduction, random_state=0)
                .to_xarray().set_coords(c for c in list(y.coords) if c != "var")
                .to_array("var").transpose("s","var")
            )

        self.x = torch.tensor(x.values)
        self.y = torch.tensor(y.values)

        if station_ids:
            self.station_id = torch.tensor(x.station_id.values, dtype=torch.int32)
        self.y_coords = y.coords

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.station_id[idx]), self.y[idx]

    def __repr__(self):
        return f"x of shape {self.x.shape}, and y of shape {self.y.shape} \n" \
                f"Options: season = {self.season} and reduction = {self.reduction}"


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.dataset_len = self.dataset.x.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.dataset.x = self.dataset.x[r]
            self.dataset.station_id = self.dataset.station_id[r]
            self.dataset.y = self.dataset.y[r]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = self.dataset[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class PhysicsLayer(nn.Module):
    def __init__(self):
        super(PhysicsLayer, self).__init__()

    def forward(self, direct):
        t, t_def, p = direct[:, 0], direct[:, 1], direct[:, 2]
        t_d = t - torch.relu(t_def)
        e_s = torch.where(
            t >= 0.0,
            6.107 * torch.exp((17.368 * t) / (t + 238.83)),
            6.108 * torch.exp((17.856 * t) / (t + 245.52)),
        )
        e = torch.where(
            t >= 0.0,
            6.107 * torch.exp((17.368 * t_d) / (t_d + 238.83)),
            6.108 * torch.exp((17.856 * t_d) / (t_d + 245.52)),
        )
        rh = e / (e_s + 1e-5) * 100.0
        r = 622.0 * (e / (p - e))
        pred = torch.stack([t, t_d, p, rh, r], dim=1)
        return pred


class Net(nn.Module):

    out_bias = [15.0, 10.0, 900.0, 70.0, 5.0]  # t, t_d, p, rh, r
    out_bias_constrained = [15.0, 5.0, 900]  # t, t_def, p

    def __init__(self, in_size, n_stations, embedding_size, l1, l2, constraint=False):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(n_stations, embedding_size)
        self.l1 = nn.Linear(in_size + embedding_size, l1)
        self.l2 = nn.Linear(l1, l2)
        if constraint:
            self.out = nn.Sequential(nn.Linear(l2, 3), PhysicsLayer())
            self.out[0].bias = nn.Parameter(torch.Tensor(self.out_bias_constrained))
        else:
            self.out = nn.Linear(l2, 5)
            self.out.bias = nn.Parameter(torch.Tensor(self.out_bias))

        self.hp = {"embedding_size": embedding_size, "l1": l1, "l2": l2}

    def forward(self, x, station_id):
        station_embedding = self.embedding(station_id)
        x = torch.concat([x, station_embedding], dim=-1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        out = self.out(x)
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.0, mask=None, log_var_init=None, trainable=True):
        super(MultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.mask = [True] * 5 if mask is None else mask

        log_var = torch.zeros(5) if log_var_init is None else log_var_init

        if trainable:
            self.log_var = nn.Parameter(log_var)
        else:
            self.log_var = log_var

        self.hp = {"alpha": alpha}

    def forward(self, pred, y):
        loss = torch.mean((pred - y) ** 2, axis=0)
        loss = torch.exp(-self.log_var) * loss + self.log_var
        loss = torch.sum(loss[self.mask])
        rh_res, r_res = physical_penalty(pred)
        physics_loss = rh_res / torch.var(y[:, 3]) + r_res / torch.var(y[:, 4])
        if self.alpha > 0.0:
            loss = (1 - self.alpha) * loss + self.alpha * physics_loss
        return loss, physics_loss


def physical_penalty(pred):
    t, t_d, p, rh, r = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    e = torch.where(
        t >= 0.0,
        6.107 * torch.exp((17.368 * t_d) / (t_d + 238.83)),
        6.108 * torch.exp((17.856 * t_d) / (t_d + 245.52)),
    )
    e_s = torch.where(
        t >= 0.0,
        6.107 * torch.exp((17.368 * t) / (t + 238.83)),
        6.108 * torch.exp((17.856 * t) / (t + 245.52)),
    )
    rh_derived = e / (e_s + 1e-5) * 100.0
    r_derived = 622.0 * (e / (p - e))

    return (
        torch.mean((rh_derived - rh) ** 2),
        torch.mean((r_derived - r) ** 2),
    )


def split_data(x, y, split):
    k = "forecast_reference_time"
    train = x[k].isin(np.array(split["train"][k], dtype=np.datetime64))
    train_x, train_y = [x.where(train, drop=True), y.where(train, drop=True)]
    val = x[k].isin(np.array(split["val"][k], dtype=np.datetime64))
    val_x, val_y = [x.where(val, drop=True), y.where(val, drop=True)]
    return train_x, train_y, val_x, val_y


def normalize_data(train_x, val_x, save_json=None):
    train_x_mean = train_x.mean("s")
    train_x_std = train_x.std("s")
    train_x = (train_x - train_x_mean) / train_x_std
    val_x = (val_x - train_x_mean) / train_x_std
    if save_json is not None:
        pd.concat(
            [train_x_mean.to_dataframe("mean"), train_x_std.to_dataframe("std")], axis=1
        ).to_json(save_json, indent=4)
    return train_x, val_x


def training_step(model, loss_fn, optimizer, train_dataloader, lr_scheduler=None):
    model.train(True)
    loss_fn.train(True)
    running_loss = 0.0
    num_batches = len(train_dataloader)
    iterator = enumerate(train_dataloader)
    for i, (X, y) in iterator:
        pred = model(*X)
        loss, p = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if lr_scheduler is not None:
            lr_scheduler.step()
    return running_loss / num_batches


def validation_step(model, loss_fn, val_dataloader):
    model.train(False), loss_fn.train(False)
    val_loss = 0.0
    val_p = 0.0
    val_mae = 0.0
    with torch.no_grad():
        iterator = val_dataloader
        for X, y in iterator:
            pred = model(*X)
            loss, p = loss_fn(pred, y)
            val_loss += loss.item()
            val_p += p.item()
            val_mae += torch.mean(torch.abs(pred - y), dim=0)
    val_loss /= len(val_dataloader)
    val_p /= len(val_dataloader)
    val_mae /= len(val_dataloader)
    return val_loss, val_p, val_mae
