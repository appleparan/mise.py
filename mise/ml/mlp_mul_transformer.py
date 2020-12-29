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
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import optuna.visualization as optv

from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png, export_svgs

import matplotlib.pyplot as plt

from data import load_imputed, MultivariateRNNDataset, MultivariateRNNMeanSeasonalityDataset
from constants import SEOUL_STATIONS, SEOULTZ
import utils

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)

def ml_mlp_mul_transformer(station_name="종로구"):
    print("Start Multivariate Transformer + Time2Vec Model")
    targets = ["PM10", "PM25"]
    # 24*14 = 336
    sample_size = 24*7
    output_size = 24
    # If you want to debug, fast_dev_run = True and n_trials should be small number
    fast_dev_run = False
    n_trials = 216
    # fast_dev_run = True
    # n_trials = 1

    # Hyper parameter
    epoch_size = 500
    batch_size = 256
    learning_rate = 1e-3

    train_fdate = dt.datetime(2012, 1, 1, 0).astimezone(SEOULTZ)
    train_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    #test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2019, 12, 31, 23).astimezone(SEOULTZ)

    # check date range assumption
    assert test_tdate > train_fdate
    assert test_fdate > train_tdate

    train_features = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                      "temp", "u", "v", "pres", "humid", "prep", "snow"]
    train_features_aerosol = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",]
    train_features_weather = ["temp", "u", "v", "pres", "humid", "prep", "snow"]

    for target in targets:
        print("Training " + target + "...")
        target_sea_h_path = Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv")

        df_sea_h = pd.read_csv(target_sea_h_path,
                               index_col=[0],
                               parse_dates=[0])

        output_dir = Path("/mnt/data/MLPTransformerT2VMultivariate_" + str(sample_size) + "/" +
                          station_name + "/" + target + "/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)
        model_dir = output_dir / "models"
        Path.mkdir(model_dir, parents=True, exist_ok=True)

        if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            # load imputed result
            _df_h = load_imputed(HOURLY_DATA_PATH)
            df_h = _df_h.query('stationCode == "' +
                               str(SEOUL_STATIONS[station_name]) + '"')
            df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=30,
            verbose=True,
            mode='auto')

        hparams = Namespace(
            nhead=16,
            head_dim=128,
            d_feedforward=256,
            num_layers=3,
            learning_rate=learning_rate,
            batch_size=batch_size)

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        metrics_callback = MetricsCallback()

        def objective(trial):
            # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
            # filenames match. Therefore, the filenames for each trial must be made unique.
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                os.path.join(model_dir, "trial_{}".format(trial.number)), monitor="val_loss",
                period=10
            )

            model = BaseTransformerModel(trial=trial,
                                     hparams=hparams,
                                     sample_size=sample_size,
                                     output_size=output_size,
                                     target=target,
                                     features=train_features,
                                     features_aerosol=train_features_aerosol,
                                     features_weather=train_features_weather,
                                     train_fdate=train_fdate, train_tdate=train_tdate,
                                     test_fdate=test_fdate, test_tdate=test_tdate,
                                     output_dir=output_dir)

            # most basic trainer, uses good defaults
            trainer = Trainer(gpus=1 if torch.cuda.is_available() else None,
                              precision=32,
                              min_epochs=1, max_epochs=20,
                              early_stop_callback=PyTorchLightningPruningCallback(
                                  trial, monitor="val_loss"),
                              default_root_dir=output_dir,
                              fast_dev_run=fast_dev_run,
                              logger=model.logger,
                              row_log_interval=10,
                              checkpoint_callback=checkpoint_callback,
                              callbacks=[metrics_callback, PyTorchLightningPruningCallback(
                                  trial, monitor="val_loss")])

            trainer.fit(model)

            return metrics_callback.metrics[-1]["val_loss"].item()

        if n_trials > 1:
            study = optuna.create_study(direction="minimize")
            # timeout = 3600*48 = 48h
            study.optimize(lambda trial: objective(
                trial), n_trials=n_trials, timeout=3600*48)

            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            print("sample_size : ", sample_size)
            print("output_size : ", output_size)

            # plot optmization results
            fig_cont1 = optv.plot_contour(
                study, params=['nhead', 'head_dim'])
            fig_cont1.write_image(
                str(output_dir / "contour_n_head_head_dim.png"))
            fig_cont1.write_image(
                str(output_dir / "contour_n_head_head_dim.svg"))

            fig_cont2 = optv.plot_contour(
                study, params=['head_dim', 'd_feedforward'])
            fig_cont2.write_image(
                str(output_dir / "contour_head_dim_d_feedforward.png"))
            fig_cont2.write_image(
                str(output_dir / "contour_head_dim_d_feedforward.svg"))

            fig_cont3 = optv.plot_contour(
                study, params=['d_feedforward', 'num_layers'])
            fig_cont3.write_image(
                str(output_dir / "contour_d_feedforward_num_layers.png"))
            fig_cont3.write_image(
                str(output_dir / "contour_d_feedforward_num_layers.svg"))

            fig_edf = optv.plot_edf(study)
            fig_edf.write_image(str(output_dir / "edf.png"))
            fig_edf.write_image(str(output_dir / "edf.svg"))

            fig_iv = optv.plot_intermediate_values(study)
            fig_iv.write_image(str(output_dir / "intermediate_values.png"))
            fig_iv.write_image(str(output_dir / "intermediate_values.svg"))

            fig_his = optv.plot_optimization_history(study)
            fig_his.write_image(str(output_dir / "opt_history.png"))
            fig_his.write_image(str(output_dir / "opt_history.svg"))

            fig_pcoord = optv.plot_parallel_coordinate(
                study, params=['nhead', 'head_dim', 'd_feedforward'])
            fig_pcoord.write_image(str(output_dir / "parallel_coord.png"))
            fig_pcoord.write_image(str(output_dir / "parallel_coord.svg"))

            fig_slice = optv.plot_slice(
                study, params=['nhead', 'head_dim', 'd_feedforward'])
            fig_slice.write_image(str(output_dir / "slice.png"))
            fig_slice.write_image(str(output_dir / "slice.svg"))

            # set hparams with optmized value
            hparams.nhead = trial.params['nhead']
            hparams.head_dim = trial.params['head_dim']
            hparams.d_feedforward = trial.params['d_feedforward']
            hparams.num_layers = trial.params['num_layers']

            dict_hparams = copy.copy(vars(hparams))
            dict_hparams["sample_size"] = sample_size
            dict_hparams["output_size"] = output_size
            with open(output_dir / 'hparams.json', 'w') as f:
                print(dict_hparams, file=f)

        model = BaseTransformerModel(hparams=hparams,
                                 sample_size=sample_size,
                                 output_size=output_size,
                                 target=target,
                                 features=train_features,
                                 features_aerosol=train_features_aerosol,
                                 features_weather=train_features_weather,
                                 train_fdate=train_fdate, train_tdate=train_tdate,
                                 test_fdate=test_fdate, test_tdate=test_tdate,
                                 output_dir=output_dir)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "train"), monitor="val_loss",
            period=10
        )

        # most basic trainer, uses good defaults
        trainer = Trainer(gpus=1 if torch.cuda.is_available() else None,
                          precision=32,
                          min_epochs=1, max_epochs=epoch_size,
                          early_stop_callback=early_stop_callback,
                          default_root_dir=output_dir,
                          fast_dev_run=fast_dev_run,
                          logger=model.logger,
                          row_log_interval=10,
                          checkpoint_callback=checkpoint_callback)

        trainer.fit(model)

        # run test set
        trainer.test()


class EmbeddingLayer(nn.Module):
    """Embedding Time Series by Learnable Linear Projection
    """
    def __init__(self, input_size, embed_size):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size

        # Nonlinear term : on-periodic patterns depends on time
        # frequency of sine function
        self.W = nn.Linear(self.input_size, self.embed_size)
        # phase-shift of sine function
        self.b = nn.Linear(self.input_size, self.embed_size)

    def forward(self, x):
        # 1 <= i <= k, parallel computation by matrix
        return torch.matmul(x, self.W) + self.b


class Time2Vec(nn.Module):
    """Encode time information

    phi and omega has k + 1 elements per each time step
    so, from input (batch_size, sample_size) will be
    ouptut (batch_size, sample_size, embed_size)

    Reference
    * https://arxiv.org/abs/1907.05321
    * https://github.com/ojus1/Time2Vec-PyTorch
    """
    def __init__(self, input_size, embed_size):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size

        self.lin = nn.Linear(self.input_size, 1)
        self.nonlin = nn.Linear(self.input_size, self.embed_size - 1)

        # activation
        self.F = lambda x: torch.sin(x)

    def forward(self, x):
        """Compute following equation

        t2v(t)[i] = omega[i] * x[t] + phi[i] if i == 0
        t2v(t)[i] = f(omega[i] * x[t] + phi[i]) if 1 <= i <= k

        so, just applying Linear layer twice

        x: (batch_size, feature_size, sample_size)
        v1: (batch_size, feature_size, 1)
        v2: (batch_size, feature_size, embed_size-1)
        """
        batch_size = x.size(0)

        v1 = self.lin(x)
        v2 = self.F(self.nonlin(x))

        return torch.cat([v1, v2], dim=2)


class TransformerEncoderBatchNormLayer(nn.TransformerEncoderLayer):
    r"""
    Use BatchNorm instead of LayerNorm
    """
    def __init__(self, d_model, nhead, dim_features=13, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__(d_model, nhead, \
            dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)

        # apply BatchNorm over C, computing statistics of (N, L) when input is (N, C, L)
        # Our input: (batch_size, feature_size, d_model) = (N, C, L)
        # this will normalize batch by column-wise
        self.norm1 = nn.BatchNorm1d(dim_features)
        self.norm2 = nn.BatchNorm1d(dim_features)


class BaseTransformerModel(LightningModule):
    """
    Temporal Pattern Attention Seasonality Embedding model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.hparams = kwargs.get('hparams', Namespace(
            nhead=16,
            head_dim=128,
            d_feedforward=256,
            num_layers=3,
            learning_rate=1e-3,
            batch_size=32
        ))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features', ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                                                "temp", "u", "v", "pres", "humid", "prep", "snow"])
        self.features_aerosol = kwargs.get('features_aerosol',
                                            ["SO2", "CO", "O3", "NO2", "PM10", "PM25"])
        self.features_weather = kwargs.get('features_weather',
                                            ["temp", "u", "v", "pres", "humid", "prep", "snow"])
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
            'output_dir', Path('/mnt/data/MLPTransformerT2VMultivariate/'))
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

        self.trial = kwargs.get('trial', None)
        self.sample_size = kwargs.get('sample_size', 48)
        self.output_size = kwargs.get('output_size', 24)

        if self.trial:
            self.hparams.head_dim = self.trial.suggest_int(
                "head_dim", 8, 128)
            self.hparams.nhead = self.trial.suggest_int(
                "nhead", 1, 16)
            self.hparams.d_feedforward = self.trial.suggest_int(
                "d_feedforward", 128, 4096)
            self.hparams.num_layers = self.trial.suggest_int(
                "num_layers", 3, 12)

        self.d_model = self.hparams.nhead * self.hparams.head_dim
        self.loss = nn.MSELoss(reduction='mean')

        self._train_set = None
        self._valid_set = None
        self._test_set = None

        log_name = self.target + "_" + dt.date.today().strftime("%y%m%d-%H-%M")
        self.logger = TensorBoardLogger(self.log_dir, name=log_name)

        # convert input vector to d_model
        self.proj = nn.Linear(self.sample_size, self.d_model)

        # use Time2Vec instead of positional encoding
        # embed sample_size -> d_model by column-wise
        self.t2v = Time2Vec(self.sample_size, self.d_model)

        # method in A Transformer-based Framework for Multivariate Time Series Representation Learning
        #self.embedding = EmbeddingLayer(self.sample_size, self.d_model)
        # also needs positional Encoding

        self.encoder_layer = TransformerEncoderBatchNormLayer(d_model=self.d_model,
                                                              nhead=self.hparams.nhead,
                                                              dim_features=len(self.features),
                                                              dim_feedforward=self.hparams.d_feedforward,
                                                              activation="gelu")
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.hparams.num_layers)

        self.outW = nn.Linear(len(self.features) * self.d_model, self.output_size)
        self.ar = nn.Linear(self.sample_size, self.output_size)
        self.out = nn.Linear(self.output_size, self.output_size)

        self.train_logs = {}
        self.valid_logs = {}

    def forward(self, x, x1d):
        """
        Args:
            x  : 2D input
            x1d : 1D input for target column

        Returns:
            outputs: output tensor

        Reference:
            * https://arxiv.org/abs/2010.02803 :
                A Transformer-based Framework for Multivariate Time Series Representation Learning
            * https://arxiv.org/abs/1907.05321 :
                Time2Vec: Learning a Vector Representation of Time
            * https://arxiv.org/abs/2001.08317 :
                Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case
        """
        batch_size = x.shape[0]
        sample_size = x.shape[1]
        feature_size = x.shape[2]

        # section 3.1
        # to apply transformer by column-wise
        # x: (batch_size, sample_size, feature_size)
        # x.permute(0,2,1): (batch_size, feature_size, sample_size) d x m
        # x_t2v: (batch_size, feature_size, d_model)
        x_t2v = self.t2v(x.permute(0, 2, 1))
        # u in paper
        x_prj = self.proj(x.permute(0, 2, 1))

        # then apply x_t2v (same as Positional Encoding) to TransformerEncoder
        # u: (batch_size, feature_size, d_model)
        u = self.encoder(x_t2v + x_prj)

        # z: (batch_size, feature_size * d_model)
        # section 3.3
        z = u.reshape(batch_size, feature_size * self.d_model)

        # yhat: (batch_size, output_size)
        yhat = self.out(self.outW(z) + self.ar(x1d))

        return yhat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        # without seasonality
        x, _x1d, _y0, _y, _y_raw, y_dates = batch
        # with seasonality
        #x, _y0, _y, _y_raw, x_dates, y_dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y, _y_hat)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y, y_hat)
        _mse = mean_squared_error(y, y_hat)
        _r2 = r2_score(y, y_hat)

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

    def validation_step(self, batch, batch_idx):
        # without seasonality
        x, _x1d, _y0, _y, _y_raw, y_dates = batch
        # with seasonality
        #x, _y0, _y, _y_raw, x_dates, y_dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y, _y_hat)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y, y_hat)
        _mse = mean_squared_error(y, y_hat)
        _r2 = r2_score(y, y_hat)

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
        # without seasonality
        x, _x1d, _y0, _y, _y_raw, y_dates = batch
        # with seasonality
        #x, _y0, _y, _y_raw, x_dates, y_dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y, _y_hat)

        y = _y.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_hat_inv = np.array(self.test_dataset.inverse_transform(y_hat, y_dates))

        _mae = mean_absolute_error(y_raw, y_hat_inv)
        _mse = mean_squared_error(y_raw, y_hat_inv)
        _r2 = r2_score(y_raw, y_hat_inv)

        return {
            'loss': _loss,
            'obs': y_raw,
            'sim': y_hat_inv,
            'dates': y_dates,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'R2': _r2
            }
        }

    def test_epoch_end(self, outputs):
        # column to indicate offset to key_date
        cols = [str(t) for t in range(self.output_size)]

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

        plot_line(self.output_size, df_obs, df_sim, self.target,
                  self.data_dir, self.png_dir, self.svg_dir)
        plot_scatter(self.output_size, df_obs, df_sim,
                     self.data_dir, self.png_dir, self.svg_dir)
        plot_corr(self.output_size, df_obs, df_sim,
                  self.data_dir, self.png_dir, self.svg_dir)
        plot_rmse(self.output_size, df_obs, df_sim,
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
            if isinstance(_y, torch.Tensor):
                values.append(_y.cpu().detach().numpy())
            elif isinstance(_y, np.ndarray):
                values.append(_y)

            # just append single key date
            indicies.append(_d[0])
        _df_obs = pd.DataFrame(data=values, index=indicies, columns=cols)

        values, indicies = [], []
        for _d, _y_hat in zip(dates, y_hats):
            if isinstance(_y_hat, torch.Tensor):
                values.append(_y_hat.cpu().detach().numpy())
            elif isinstance(_y_hat, np.ndarray):
                values.append(_y_hat)

            # just append single key date
            indicies.append(_d[0])
        _df_sim = pd.DataFrame(data=values, index=indicies, columns=cols)

        return _df_obs, _df_sim

    def setup(self, stage=None):
        """Data operations on every GPU
        Wrong usage of LightningModule. Need to Refactored

        * TODO : Refactoring https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
        """
        # create custom dataset
        train_valid_set = MultivariateRNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.sample_size,
            output_size=self.output_size,
            train_valid_ratio=0.8)

        # first mkdir of seasonality
        Path.mkdir(self.png_dir / "seasonality", parents=True, exist_ok=True)
        Path.mkdir(self.svg_dir / "seasonality", parents=True, exist_ok=True)
        Path.mkdir(self.data_dir / "seasonality", parents=True, exist_ok=True)

        # fit & transform (seasonality)
        # without seasonality
        train_valid_set.preprocess()
        # with seasonality
        # train_valid_set.preprocess(
        #     self.data_dir / "seasonality", self.png_dir / "seasonality", self.svg_dir / "seasonality")


        # save train/valid set
        train_valid_set.to_csv(
            self.data_dir / ("df_train_valid_set_" + self.target + ".csv"))

        # split train/valid/test set
        train_len = int(len(train_valid_set) *
                        train_valid_set.train_valid_ratio)
        valid_len = len(train_valid_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(
            train_valid_set, [train_len, valid_len])

        test_set = MultivariateRNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.test_fdate,
            tdate=self.test_tdate,
            sample_size=self.sample_size,
            output_size=self.output_size,
            scaler_X=train_valid_set.scaler_X,
            scaler_Y=train_valid_set.scaler_Y)
        test_set.to_csv(self.data_dir / ("df_testset_" + self.target + ".csv"))

        test_set.transform()

        # assign to use in dataloaders
        self.train_dataset = train_set
        self.val_dataset = valid_set
        self.test_dataset = test_set

        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)

        print("Data Setup Completed")

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
        data: list of tuple  (x, x1d, y0, y, dates).
            - x: numpy of shape (sample_size, num_features);
            - y0: scalar
            - y: numpy of shape (output_size);
            - y_date: pandas DateTimeIndex of shape (output_size):

        Returns:
            - xs: torch Tensor of shape (batch_size, sample_size, num_features);
            - xs_1d: torch Tensor of shape (batch_size, 1, num_features);
            - ys: torch Tensor of shape (batch_size, output_size);
            - y0: torch scalar Tensor
            - dates: pandas DateTimeIndex of shape (batch_size, output_size):
        """
        # without seasonality
        # seperate source and target sequences
        # data goes to tuple (thanks to *) and zipped
        # MutlivariateRNNDataset
        xs, xs_1d, ys0, ys, ys_raw, y_dates = zip(*batch)

        return torch.as_tensor(xs), \
            torch.as_tensor(xs_1d), \
            torch.as_tensor(ys0), \
            torch.as_tensor(ys), \
            torch.as_tensor(ys_raw), \
            y_dates

        # with seasonality
        # MutlivariateRNNMeanSeasonalityDataset
        # xs, ys0, ys, ys_raw, x_dates, y_dates = zip(*batch)

        # return torch.as_tensor(xs), \
        #     torch.as_tensor(ys0), \
        #     torch.as_tensor(ys), \
        #     torch.as_tensor(ys_raw), \
        #     x_dates, y_dates


def plot_line(output_size, df_obs, df_sim, target, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    for t in range(output_size):
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
                                           columns=['MAE', 'MSE', 'R2', 'loss'])
    df_train_logs.index.rename('epoch', inplace=True)

    df_valid_logs = pd.DataFrame.from_dict(valid_logs, orient='index',
                                           columns=['MAE', 'MSE', 'R2', 'loss'])
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

def plot_scatter(output_size, df_obs, df_sim, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    for t in range(output_size):
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

def plot_corr(output_size, df_obs, df_sim, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    png_path = png_dir / ("corr_time.png")
    svg_path = svg_dir / ("corr_time.svg")
    csv_path = data_dir / ("corr_time.csv")

    times = list(range(output_size + 1))
    corrs = [1.0]
    for t in range(output_size):
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

def plot_rmse(output_size, df_obs, df_sim, data_dir, png_dir, svg_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    png_path = png_dir / ("rmse_time.png")
    svg_path = svg_dir / ("rmse_time.svg")
    csv_path = data_dir / ("rmse_time.csv")

    times = list(range(1, output_size + 1))
    rmses = []
    for t in range(output_size):
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
    return _input * beta * torch.sigmoid(_input)