from argparse import Namespace
import copy
import datetime as dt
from math import sqrt
import os
from pathlib import Path
import shutil

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

import data
from constants import SEOUL_STATIONS, SEOULTZ

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


def ml_rnn_mul_lstnet_skip(station_name="종로구"):
    print("Start Multivariate LSTNet (Skip Layer) Model")
    targets = ["PM10", "PM25"]
    # 24*14 = 336
    sample_size = 48
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

    train_fdate = dt.datetime(2008, 1, 5, 0).astimezone(SEOULTZ)
    train_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    #test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2019, 12, 31, 23).astimezone(SEOULTZ)

    # check date range assumption
    assert test_tdate > train_fdate
    assert test_fdate > train_tdate

    train_features = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                      "temp", "u", "v", "pres", "humid", "prep", "snow"]

    for target in targets:
        print("Training " + target + "...")
        target_sea_h_path = Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv")

        df_sea_h = pd.read_csv(target_sea_h_path,
                               index_col=[0],
                               parse_dates=[0])

        output_dir = Path("/mnt/data/RNNLSTNetSkipMultivariate_" + str(sample_size) + "/" +
                          station_name + "/" + target + "/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)
        model_dir = output_dir / "models"
        Path.mkdir(model_dir, parents=True, exist_ok=True)

        if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            # load imputed result
            _df_h = data.load_imputed(HOURLY_DATA_PATH)
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
            filter_size=3,
            hidCNN=16,
            hidSkip=16,
            hidden_size=16,
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

            model = BaseLSTNetModel(trial=trial,
                                    hparams=hparams,
                                    sample_size=sample_size,
                                    output_size=output_size,
                                    target=target,
                                    features=train_features,
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
            # timeout = 3600*36 = 36h
            study.optimize(lambda trial: objective(
                trial), n_trials=n_trials, timeout=3600*36)

            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            print("sample_size : ", sample_size)
            print("output_size : ", output_size)

            # plot optmization results
            fig_cont1 = optv.plot_contour(
                study, params=['filter_size', 'hidCNN'])
            fig_cont1.write_image(
                str(output_dir / "contour_filter_size_hidCNN.png"))
            fig_cont1.write_image(
                str(output_dir / "contour_filter_size_hidCNN.svg"))

            fig_cont2 = optv.plot_contour(
                study, params=['hidCNN', 'hidden_size'])
            fig_cont2.write_image(
                str(output_dir / "contour_hidCNN_hidden_size.png"))
            fig_cont2.write_image(
                str(output_dir / "contour_hidCNN_hidden_size.svg"))

            fig_cont3 = optv.plot_contour(
                study, params=['filter_size', 'hidden_size'])
            fig_cont3.write_image(
                str(output_dir / "contour_filter_size_hidden_size.png"))
            fig_cont3.write_image(
                str(output_dir / "contour_filter_size_hidden_size.svg"))

            fig_cont3 = optv.plot_contour(
                study, params=['hidSkip', 'hidden_size'])
            fig_cont3.write_image(
                str(output_dir / "contour_hidSkip_hidden_size.png"))
            fig_cont3.write_image(
                str(output_dir / "contour_hidSkip_hidden_size.svg"))

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
                study, params=['filter_size', 'hidCNN', 'hidSkip', 'hidden_size'])
            fig_pcoord.write_image(str(output_dir / "parallel_coord.png"))
            fig_pcoord.write_image(str(output_dir / "parallel_coord.svg"))

            fig_slice = optv.plot_slice(
                study, params=['filter_size', 'hidCNN', 'hidSkip',  'hidden_size'])
            fig_slice.write_image(str(output_dir / "slice.png"))
            fig_slice.write_image(str(output_dir / "slice.svg"))

            # set hparams with optmized value
            hparams.filter_size = trial.params['filter_size']
            hparams.hidCNN = trial.params['hidCNN']
            hparams.hidSkip = trial.params['hidSkip']
            hparams.hidden_size = trial.params['hidden_size']

            dict_hparams = copy.copy(vars(hparams))
            dict_hparams["sample_size"] = sample_size
            dict_hparams["output_size"] = output_size
            with open(output_dir / 'hparams.json', 'w') as f:
                print(dict_hparams, file=f)

        model = BaseLSTNetModel(hparams=hparams,
                                sample_size=sample_size,
                                output_size=output_size,
                                target=target,
                                features=train_features,
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

        shutil.rmtree(model_dir)


class BaseLSTNetModel(LightningModule):
    """
    LSTNet + Skip layer
    """
    def __init__(self, **kwargs):
        super().__init__()
        # h_out = (h_in + 2 * padding[0] - dilation[0]*(kernel_size[0] - 1) - 1) / stride[0] + 1
        # to make h_out == h_in, dilation[0] == 1, stride[0] == 1,
        # 2*padding[0] + 1 = kernel_size[0]

        # w_out = (w_in + 2 * padding[1] - dilation[1]*(kernel_size[1] - 1) - 1) / stride[1] + 1
        # to make w_out == w_in, dilation[1] == 1, stride[1] == 1,
        # 2*padding[1] + 1 = kernel_size[1]

        self.hparams = kwargs.get('hparams', Namespace(
            hidCNN=4,
            hidSkip=16,
            hidden_size=16,
            filter_size=5,
            learning_rate=1e-3,
            batch_size=32
        ))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features', ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                                                "temp", "u", "v", "pres", "humid", "prep", "snow"])
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
            'output_dir', Path('/mnt/data/RNNLSTNetSkipMultivariate/'))
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
            self.hparams.filter_size = self.trial.suggest_int(
                "filter_size", 1, 7, step=2)
            self.hparams.hidden_size = self.trial.suggest_int(
                "hidden_size", 8, 1024)
            self.hparams.hidCNN = self.trial.suggest_int(
                "hidCNN", 8, 1024)
            self.hparams.hidSkip = self.trial.suggest_int(
                "hidSkip", 8, 1024)

        self.kernel_shape = (self.hparams.filter_size, len(self.features))

        #self.loss = nn.MSELoss()
        self.loss = nn.L1Loss()

        self._train_set = None
        self._valid_set = None
        self._test_set = None

        log_name = self.target + "_" + dt.date.today().strftime("%y%m%d-%H-%M")
        self.logger = TensorBoardLogger(self.log_dir, name=log_name)

        self.train_logs = {}
        self.valid_logs = {}

        padding_size = int(self.hparams.filter_size - 1)
        self.pad = nn.ZeroPad2d((0, 0, padding_size, 0))

        self.conv = nn.Conv2d(1, self.hparams.hidCNN, self.kernel_shape)
        self.dropout = nn.Dropout(p = 0.1)

        # normal GRU
        self.gru_no_skip = nn.GRU(self.hparams.hidCNN, self.hparams.hidden_size, batch_first=True)

        # skip interval
        self.p = 24
        # total length for skip, remove remainer step
        # i.e. if sample_size is 25, kernel_shape == (1, 3), and self.p == 24,
        # self.pt should be (25 - 1) / 24 = 1
        self.pt = int(self.sample_size / self.p)

        assert self.sample_size > self.p

        # skip layer
        self.gru_skip = nn.GRU(self.hparams.hidCNN,
                               self.hparams.hidSkip, batch_first=True)

        self.ar = nn.Linear(self.sample_size, self.output_size)

        self.proj1 = nn.Linear(self.hparams.hidden_size + self.p * self.hparams.hidSkip,
                               self.output_size)
        self.proj2 = nn.Linear(self.output_size, self.output_size)
        self.act = nn.ReLU()

    def forward(self, _x, _x1d):
        """
        Args:
            _x  : 2D Input (with input features), shape is (batch_size, sample_size, feature_size)
            _x1d : 1D Input (only target column), shape is (batch_size, sample_size, feature_size)
            y0 : First step output feed to Decoder, shape is (batch_size, 1)
            y  : Output, shape is (batch_size, output_size)
        """
        # _xx : [batch_size, sample_size, feature_size]
        # use this batch_size,
        # because batch_size could be different with hparams.batch_size on last batch
        batch_size = _x.shape[0]
        sample_size = _x.shape[1]
        feature_size = _x.shape[2]
        # _x.unsqueeze(1) : (batch_size, 1, sample_size, feature_size), NxC_inxH_inxW_in
        # x: (batch_size, 1, sample_size + pad_size, feature_size), NxC_inxH_inxW_in
        x = self.pad(_x.unsqueeze(1))
        # c: (batch_size, hidCNN, sample_size, 1), NxC_outxH_outxW_out
        c = self.conv(x)
        c = self.dropout(F.relu(c))
        # c: (batch_size, hidCNN, sample_size)
        c = c.squeeze(3)

        # Recurrent Layer
        # x_rnn = (num_layers * num_directions, batch_size, hidden_size)
        _, x_rnn = self.gru_no_skip(c.permute(0, 2, 1))

        # x_rnn = (batch_size, hidden_size)
        x_rnn = self.dropout(x_rnn.squeeze(0))

        # Recurrent Skip Layer
        # Best implementation is laiguokun/LSTNet
        # [1, 2, 3, 4] -> [[1, 2], [3, 4]]
        # -> [[1, 3], [2, 4]] using permute
        s = c[:, :, (-self.pt * self.p):].contiguous()
        s = s.reshape(batch_size, self.hparams.hidCNN, self.pt, self.p)
        # s.permute(0, 3, 2, 1): (batch_size, self.p, self.pt, hidCNN)
        s = s.permute(0, 3, 2, 1)
        # s.reshape: (batch_size * self.p, self.pt, hidCNN) == (batch, seq, input)
        s = s.reshape(batch_size * self.p, self.pt, self.hparams.hidCNN)

        # x_rnn_skip: (num_layers * num_directions, batch_size * self.p, hidSkip)
        _, x_rnn_skip = self.gru_skip(s)
        # x_rnn_skip: (batch_size, self.p * hidSkip)
        x_rnn_skip = x_rnn_skip.squeeze(0)
        x_rnn_skip = self.dropout(
            x_rnn_skip.reshape(batch_size, self.p * self.hparams.hidSkip))

        # (batch_size, hidden_size + self.p * hidSkip
        output1 = torch.cat((x_rnn, x_rnn_skip), dim=1)
        # (batch_first, output_size)
        output2 = self.ar(_x1d)

        # Sum Autoregressive and Recurrent Layer output
        output = self.act(self.proj1(output1) + self.proj2(output2))

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, _x1d, _y0, _y, _y_raw, dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y_hat, _y_raw)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y_raw, y_hat)
        _mse = mean_squared_error(y_raw, y_hat)
        _r2 = r2_score(y_raw, y_hat)

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
        x, _x1d, _y0, _y, _y_raw, dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y_hat, _y_raw)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y_raw, y_hat)
        _mse = mean_squared_error(y_raw, y_hat)
        _r2 = r2_score(y_raw, y_hat)

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
        x, _x1d, _y0, _y, _y_raw, dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y_hat, _y_raw)

        y = _y.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y_raw, y_hat)
        _mse = mean_squared_error(y_raw, y_hat)
        _r2 = r2_score(y_raw, y_hat)

        return {
            'loss': _loss,
            'obs': y_raw,
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
            #values.append(_y.cpu().detach().numpy())
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

    def prepare_data(self):
        # create custom dataset
        train_valid_set = data.MultivariateRNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.sample_size,
            output_size=self.output_size,
            train_valid_ratio=0.8)

        # save train/valid set
        train_valid_set.to_csv(
            self.data_dir / ("df_train_valid_set_" + self.target + ".csv"))

        # fit & transform (standardization)
        train_valid_set.preprocess()

        # split train/valid/test set
        train_len = int(len(train_valid_set) *
                        train_valid_set.train_valid_ratio)
        valid_len = len(train_valid_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(
            train_valid_set, [train_len, valid_len])

        test_set = data.MultivariateRNNDataset(
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
            - x: pandas DataFrame or numpy of shape (sample_size, num_features);
            - x1d: pandas DataFrma or numpy of shape (sample_size)
            - y0: scalar
            - y: pandas DataFrame or numpy of shape (output_size);
            - date: pandas DateTimeIndex of shape (output_size):

        Returns:
            - xs: torch Tensor of shape (batch_size, sample_size, num_features);
            - xs_1d: torch Tensor of shape (batch_size, sample_size);
            - ys: torch Tensor of shape (batch_size, output_size);
            - y0: torch scalar Tensor
            - dates: pandas DateTimeIndex of shape (batch_size, output_size):
        """

        # seperate source and target sequences
        # data goes to tuple (thanks to *) and zipped
        xs, xs_1d, ys0, ys, ys_raw, dates = zip(*batch)

        return torch.as_tensor(xs), \
            torch.as_tensor(xs_1d), \
            torch.as_tensor(ys0), \
            torch.as_tensor(ys), \
            torch.as_tensor(ys_raw), \
            dates


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


class LogCoshLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Implement numerically stable log-cosh which is used in Keras

        log(cosh(x)) = logaddexp(x, -x) - log(2)
                = abs(x) + log1p(exp(-2 * abs(x))) - log(2)

        Reference:
            * https://stackoverflow.com/a/57786270
        """
        # not to compute log(0), add 1e-24 (small value)
        def _log_cosh(x):
            return torch.abs(x) + \
                torch.log1p(torch.exp(-2 * torch.abs(x))) + \
                torch.log(torch.full_like(x, 2, dtype=x.dtype))

        return torch.mean(_log_cosh(input - target))
