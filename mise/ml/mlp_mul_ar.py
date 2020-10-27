from argparse import Namespace
import copy
import datetime as dt
from math import sqrt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
import tqdm

from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png, export_svgs

import data
from constants import SEOUL_STATIONS, SEOULTZ

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ml_mlp_mul_ar(station_name="종로구"):
    print("Start Multivariate MLP AR model")
    targets = ["PM10", "PM25"]
    sample_size = 48
    output_size = 24

    # Hyper parameter
    epoch_size = 500
    batch_size = 256
    learning_rate = 1e-3

    train_fdate = dt.datetime(2012, 1, 1, 0).astimezone(SEOULTZ)
    train_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    #test_tdate = dt.datetime(2019, 12, 31, 23).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2019, 12, 31, 23).astimezone(SEOULTZ)

    # check date range assumption
    assert test_tdate > train_fdate
    assert test_fdate > train_tdate

    train_features = ["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp", "u", "v", "pres", "humid", "prep", "snow"]

    for target in targets:
        print("Training " + target + "...")
        target_sea_h_path = Path("/input/python/input_jongro_imputed_hourly_pandas.csv")

        df_sea_h = pd.read_csv(target_sea_h_path,
                               index_col=[0],
                               parse_dates=[0])

        output_dir = Path("/mnt/data/MLPARMultivariate/" +
                          station_name + "/" + target + "/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)

        if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            # load imputed result
            _df_h = data.load_imputed(HOURLY_DATA_PATH)
            df_h = _df_h.query('stationCode == "' +
                            str(SEOUL_STATIONS[station_name]) + '"')
            df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

        if target == 'PM10':
            unit_size = 32
            hparams = Namespace(
                input_size=sample_size*len(train_features),
                layer1_size=32,
                layer2_size=32,
                output_size=output_size,
                learning_rate=learning_rate,
                sample_size=sample_size,
                batch_size=batch_size)
            model = BaseMLPModel(hparams=hparams,
                station_name=station_name,
                target=target,
                features=["PM10"],
                train_fdate=train_fdate, train_tdate=train_tdate,
                test_fdate=test_fdate, test_tdate=test_tdate,
                output_dir=output_dir)
        elif target == 'PM25':
            unit_size = 16
            hparams = Namespace(
                input_size=sample_size*len(train_features),
                layer1_size=16,
                layer2_size=16,
                output_size=output_size,
                learning_rate=learning_rate,
                sample_size=sample_size,
                batch_size=batch_size)
            model = BaseMLPModel(hparams=hparams,
                station_name=station_name,
                target=target,
                features=["PM25"],
                train_fdate=train_fdate, train_tdate=train_tdate,
                test_fdate=test_fdate, test_tdate=test_tdate,
                output_dir=output_dir)
        # first, plot periodicity
        # second, univariate or multivariate

        # early stopping
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30,
            verbose=True,
            mode='auto'
        )
        # most basic trainer, uses good defaults
        trainer = Trainer(gpus=1,
                          precision=32,
                          min_epochs=1, max_epochs=epoch_size,
                          early_stop_callback=early_stop_callback,
                          default_save_path=output_dir,
                          fast_dev_run=True,
                          logger=model.logger,
                          row_log_interval=10)


        trainer.fit(model)

        # run test set
        trainer.test()

class BaseMLPModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams = kwargs.get('hparams', Namespace(
                input_size=48*24,
                layer1_size=32,
                layer2_size=32,
                output_size=24,
                learning_rate=1e-3,
                sample_size=48,
                batch_size=32
                ))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = [self.target]
        self.metrics = kwargs.get('metrics', ['MAE', 'MSE', 'R2'])
        self.train_fdate = kwargs.get('train_fdate', dt.datetime(
            2012, 1, 1, 0).astimezone(SEOULTZ))
        self.train_tdate = kwargs.get('train_tdate', dt.datetime(
            2017, 12, 31, 23).astimezone(SEOULTZ))
        self.test_fdate = kwargs.get('test_fdate', dt.datetime(
            2018, 1, 1, 0).astimezone(SEOULTZ))
        self.test_tdate = kwargs.get('test_tdate', dt.datetime(
            2018, 12, 31, 23).astimezone(SEOULTZ))
        self.num_workers = kwargs.get('num_workers', 1)
        self.output_dir = kwargs.get(
            'output_dir', Path('/mnt/data/MLPMSARUnivariate/'))
        self.log_dir = kwargs.get('log_dir', self.output_dir / Path('log'))
        Path.mkdir(self.log_dir, parents=True, exist_ok=True)
        self.png_dir = kwargs.get(
            'plot_dir', self.output_dir / Path('png/'))
        Path.mkdir(self.png_dir, parents=True, exist_ok=True)
        self.svg_dir = kwargs.get(
            'plot_dir', self.output_dir / Path('svg/'))
        Path.mkdir(self.svg_dir, parents=True, exist_ok=True)
        self.data_dir = kwargs.get(
            'data_dir', self.output_dir / Path('csv/'))
        Path.mkdir(self.data_dir, parents=True, exist_ok=True)

        self.fc1 = nn.Linear(self.hparams.input_size, self.hparams.layer1_size)
        self.fc2 = nn.Linear(self.hparams.layer1_size, self.hparams.layer2_size)
        self.fc3 = nn.Linear(self.hparams.layer2_size, self.hparams.output_size)
        self.loss = nn.MSELoss(reduction='mean')

        self._train_set = None
        self._valid_set = None
        self._test_set = None

        log_name = self.target + "_" + dt.date.today().strftime("%y%m%d-%H-%M")
        self.logger = TensorBoardLogger(self.log_dir, name=log_name)

        self.train_logs = {}
        self.valid_logs = {}

    def forward(self, x):
        # vectorize
        x = x.view(-1, self.hparams.input_size).to(device)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr|self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        yas, xms, ys, dates = batch

        y_hat = self.forward(xms)
        _loss = self.loss(yas + y_hat, ys)

        _ys = ys.detach().cpu().clone().numpy()
        _y_hat = y_hat.detach().cpu().clone().numpy()
        _mae = mean_absolute_error(_ys, _y_hat)
        _mse = mean_squared_error(_ys, _y_hat)
        _r2 = r2_score(_ys, _y_hat)

        return {
            'loss': _loss,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'R2': _r2
            }
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        tensorboard_logs = {'train/loss': avg_loss}
        _log = {}
        for name in self.metrics:
            tensorboard_logs['train/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
            _log[name] = float(torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean())
        tensorboard_logs['step'] = self.current_epoch
        _log['loss'] = float(avg_loss.detach().cpu())

        self.train_logs[self.current_epoch] = _log

        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        yas, xms, ys, dates = batch

        y_hat = self.forward(xms)
        _loss = self.loss(yas + y_hat, ys)

        _ys = ys.detach().cpu().clone().numpy()
        _y_hat = y_hat.detach().cpu().clone().numpy()
        _mae = mean_absolute_error(_ys, _y_hat)
        _mse = mean_squared_error(_ys, _y_hat)
        _r2 = r2_score(_ys, _y_hat)

        return {
            'loss': _loss,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'R2': _r2
            }
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        tensorboard_logs = {'valid/loss': avg_loss}
        _log = {}
        for name in self.metrics:
            tensorboard_logs['valid/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
            _log[name] = float(torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean())
        tensorboard_logs['step'] = self.current_epoch
        _log['loss'] = float(avg_loss.detach().cpu())

        self.valid_logs[self.current_epoch] = _log

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        yas, xms, ys, dates = batch

        y_hat = self.forward(xms)
        _loss = self.loss(yas + y_hat, ys)

        _ys = ys.detach().cpu().clone().numpy()
        _y_hat = y_hat.detach().cpu().clone().numpy()
        _mae = mean_absolute_error(_ys, _y_hat)
        _mse = mean_squared_error(_ys, _y_hat)
        _r2 = r2_score(_ys, _y_hat)

        return {
            'loss': _loss,
            'obs': ys,
            'sim': y_hat,
            'dates': dates,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'R2': _r2
            }
        }

    def test_epoch_end(self, outputs):
        # column to indicate offset to key_date
        cols = [str(t) for t in range(self.hparams.output_size)]

        df_obs = pd.DataFrame(columns=cols)
        df_sim = pd.DataFrame(columns=cols)

        for out in outputs:
            ys = out['obs']
            y_hats = out['sim']
            dates = out['dates']

            _df_obs, _df_sim = self.single_batch_to_df(ys, y_hats, dates, cols)

            df_obs = pd.concat([df_obs, _df_obs])
            df_sim = pd.concat([df_sim, _df_sim])
        df_obs.index.name = 'date'
        df_sim.index.name = 'date'

        df_obs.sort_index(inplace=True)
        df_sim.sort_index(inplace=True)

        df_obs.to_csv(self.data_dir / "df_test_obs.csv")
        df_sim.to_csv(self.data_dir / "df_test_sim.csv")

        plot_line(self.hparams, df_obs, df_sim, self.target,
                  self.data_dir, self.png_dir, self.svg_dir)
        plot_scatter(self.hparams, df_obs, df_sim,
                     self.data_dir, self.png_dir, self.svg_dir)
        plot_corr(self.hparams, df_obs, df_sim,
                  self.data_dir, self.png_dir, self.svg_dir)
        plot_rmse(self.hparams, df_obs, df_sim,
                  self.data_dir, self.png_dir, self.svg_dir)
        plot_logs(self.train_logs, self.valid_logs, self.target,
                  self.data_dir, self.png_dir, self.svg_dir)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        tensorboard_logs = {'test/loss': avg_loss}
        for name in self.metrics:
            tensorboard_logs['test/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
        tensorboard_logs['step'] = self.current_epoch

        return {
            'test_loss': avg_loss,
            'log': tensorboard_logs,
            'obs': df_obs,
            'sim': df_sim,
        }

    def single_batch_to_df(self, ys, y_hats, dates, cols):
        # single batch to dataframe
        # dataframe that index is starting date
        values, indicies = [], []
        for _d, _y in zip(dates, ys):
            values.append(_y.cpu().detach().numpy())
            # just append single key date
            indicies.append(_d[0])
        _df_obs = pd.DataFrame(data=values, index=indicies, columns=cols)

        values, indicies = [], []
        for _d, _y_hat in zip(dates, y_hats):
            values.append(_y_hat.cpu().detach().numpy())
            # just append single key date
            indicies.append(_d[0])
        _df_sim = pd.DataFrame(data=values, index=indicies, columns=cols)

        return _df_obs, _df_sim

    def prepare_data(self):
        # create custom dataset
        train_valid_set = data.UnivariateAutoRegressiveSubDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=[self.target],
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            train_valid_ratio=0.8,
            num_workers=4)
        test_set = data.UnivariateAutoRegressiveSubDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=[self.target],
            fdate=self.test_fdate,
            tdate=self.test_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            num_workers=4)

        # save train/valid set
        train_valid_set.to_csv(
            self.data_dir / ("df_train_valid_set_" + self.target + ".csv"))
        test_set.to_csv(self.data_dir / ("df_testset_" + self.target + ".csv"))

        # split train/valid/test set
        train_len = int(len(train_valid_set) * train_valid_set.train_valid_ratio)
        valid_len = len(train_valid_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(train_valid_set, [train_len, valid_len])

        # assign to use in dataloaders
        self.train_dataset = train_set
        self.val_dataset = valid_set
        self.test_dataset = test_set

        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)

        # arima table preprocess in constructor
        train_valid_set.plot_arima(self.data_dir, self.plot_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """Creates mini-batch tensors from from list of tuples (x, y, dates)

        dates will not be trained but need to construct output, so don't put dates into Tensors
        Args:
        batch: list of tuple  (xas, yas, xms, xmi, ys, dates).
            - xas: pandas DataFrame or numpy of shape (input_size, num_features);
            - yas: pandas DataFrame or numpy of shape (input_size, num_features);
            - xms: pandas DataFrame or numpy of shape (input_size, num_features);
            - ys: pandas DataFrame or numpy of shape (output_size);
            - dates: pandas DateTimeIndex of shape (output_size):

        Returns:
            - xas: torch Tensor of shape (batch_size, input_size, num_features);
            - mask: torch Tensor of shape (batch_size, input_size, num_features);
            - yas: torch Tensor of shape (batch_size, input_size, num_features);
            - xms: torch Tensor of shape (batch_size, input_size, num_features);
            - ys: pandas DataFrame or numpy of shape (output_size);
            - dates: pandas DateTimeIndex of shape (output_size):
        """

        # seperate source and target sequences
        # data goes to tuple (thanks to *) and zipped
        yas, xms, ys, dates = zip(*batch)

        return torch.as_tensor(yas), torch.as_tensor(xms), torch.as_tensor(ys), dates


def plot_line(hparams, df_obs, df_sim, target, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    for t in range(hparams.output_size):
        dates = df_obs.index + dt.timedelta(hours=t)

        png_dir_h = png_dir / str(t).zfill(2)
        svg_dir_h = svg_dir / str(t).zfill(2)
        Path.mkdir(png_dir_h, parents=True, exist_ok=True)
        Path.mkdir(svg_dir_h, parents=True, exist_ok=True)
        png_path = png_dir_h / ("line_" + str(t).zfill(2) + "h.png")
        svg_path = svg_dir_h / ("line_" + str(t).zfill(2) + "h.svg")

        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        p = figure(title="OBS & Model")
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label = "dates"
        p.xaxis.formatter = DatetimeTickFormatter()
        p.line(dates, obs, line_color="dodgerblue", legend_label="obs")
        p.line(dates, sim, line_color="lightcoral", legend_label="sim")
        export_png(p, filename=png_path)
        p.output_backend = "svg"
        export_svgs(p, filename=str(svg_path))

        data_dir_h = data_dir / str(t).zfill(2)
        Path.mkdir(data_dir_h, parents=True, exist_ok=True)
        csv_path = data_dir_h / ("line_" + str(t).zfill(2) + "h.csv")
        df_line = pd.DataFrame.from_dict(
            {'date': dates, 'obs': obs, 'sim': sim})
        df_line.set_index('date', inplace=True)
        df_line.to_csv(csv_path)


def plot_logs(train_logs, valid_logs, target,
              data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    df_train_logs = pd.DataFrame.from_dict(train_logs, orient='index',
                                           columns=['MAE', 'MSE', 'R2', 'LOSS'])
    df_train_logs.index.rename('epoch', inplace=True)

    df_valid_logs = pd.DataFrame.from_dict(valid_logs, orient='index',
                                           columns=['MAE', 'MSE', 'R2', 'LOSS'])
    df_valid_logs.index.rename('epoch', inplace=True)

    csv_path = data_dir / ("log_train.csv")
    df_train_logs.to_csv(csv_path)

    epochs = df_train_logs.index.to_numpy()
    for col in df_train_logs.columns:
        png_path = png_dir / ("log_train_" + col + ".png")
        svg_path = svg_dir / ("log_train_" + col + ".svg")
        p = figure(title=col)
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label = "epoch"
        p.line(epochs, df_train_logs[col].to_numpy(), line_color="dodgerblue")
        export_png(p, filename=png_path)
        p.output_backend = "svg"
        export_svgs(p, filename=str(svg_path))

    csv_path = data_dir / ("log_valid.csv")
    df_valid_logs.to_csv(csv_path)

    epochs = df_valid_logs.index.to_numpy()
    for col in df_valid_logs.columns:
        png_path = png_dir / ("log_valid_" + col + ".png")
        svg_path = svg_dir / ("log_valid_" + col + ".svg")
        p = figure(title=col)
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label = "epoch"
        p.line(epochs, df_valid_logs[col].to_numpy(), line_color="dodgerblue")
        export_png(p, filename=png_path)
        p.output_backend = "svg"
        export_svgs(p, filename=str(svg_path))


def plot_scatter(hparams, df_obs, df_sim, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    for t in range(hparams.output_size):
        png_dir_h = png_dir / str(t).zfill(2)
        svg_dir_h = svg_dir / str(t).zfill(2)
        Path.mkdir(png_dir_h, parents=True, exist_ok=True)
        Path.mkdir(svg_dir_h, parents=True, exist_ok=True)
        png_path = png_dir_h / ("scatter_" + str(t).zfill(2) + "h.png")
        svg_path = svg_dir_h / ("scatter_" + str(t).zfill(2) + "h.svg")

        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        p = figure(title="Model/OBS")
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label = "OBS"
        p.yaxis.axis_label = "Model"
        maxval = np.nanmax([np.nanmax(obs), np.nanmax(sim)])
        p.xaxis.bounds = (0.0, maxval)
        p.yaxis.bounds = (0.0, maxval)
        p.x_range = Range1d(0.0, maxval)
        p.y_range = Range1d(0.0, maxval)
        p.scatter(obs, sim)
        export_png(p, filename=png_path)
        p.output_backend = "svg"
        export_svgs(p, filename=str(svg_path))

        data_dir_h = data_dir / str(t).zfill(2)
        Path.mkdir(data_dir_h, parents=True, exist_ok=True)
        csv_path = data_dir_h / ("scatter_" + str(t).zfill(2) + "h.csv")
        df_scatter = pd.DataFrame({'obs': obs, 'sim': sim})
        df_scatter.to_csv(csv_path)


def plot_corr(hparams, df_obs, df_sim, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    png_path = png_dir / ("corr_time.png")
    svg_path = svg_dir / ("corr_time.svg")
    csv_path = data_dir / ("corr_time.csv")

    times = list(range(hparams.output_size + 1))
    corrs = [1.0]
    for t in range(hparams.output_size):
        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        corrs.append(np.corrcoef(obs, sim)[0, 1])

    p = figure(title="Correlation of OBS & Model")
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.axis_label = "lags"
    p.yaxis.axis_label = "corr"
    p.yaxis.bounds = (0.0, 1.0)
    p.y_range = Range1d(0.0, 1.0)
    p.line(times, corrs)
    export_png(p, filename=png_path)
    p.output_backend = "svg"
    export_svgs(p, filename=str(svg_path))

    df_corrs = pd.DataFrame({'time': times, 'corr': corrs})
    df_corrs.set_index('time', inplace=True)
    df_corrs.to_csv(csv_path)


def plot_rmse(hparams, df_obs, df_sim, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    png_path = png_dir / ("rmse_time.png")
    svg_path = svg_dir / ("rmse_time.svg")
    csv_path = data_dir / ("rmse_time.csv")

    times = list(range(1, hparams.output_size + 1))
    rmses = []
    for t in range(hparams.output_size):
        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        rmses.append(sqrt(mean_squared_error(obs, sim)))

    p = figure(title="RMSE of OBS & Model")
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.axis_label = "lags"
    p.yaxis.axis_label = "RMSE"
    p.line(times, rmses)
    export_png(p, filename=png_path)
    p.output_backend = "svg"
    export_svgs(p, filename=str(svg_path))

    df_rmses = pd.DataFrame({'time': times, 'rmse': rmses})
    df_rmses.set_index('time', inplace=True)
    df_rmses.to_csv(csv_path)

def swish(_input, beta=1.0):
    """
        Swish function in [this paper](https://arxiv.org/pdf/1710.05941.pdf)

    Args:
        input: Tensor

    Returns:
        output: Activated tensor
    """
    return _input * beta * nn.Sigmoid(_input)

