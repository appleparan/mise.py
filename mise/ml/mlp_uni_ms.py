from argparse import Namespace
import copy
import datetime as dt
from math import sqrt
import os
from pathlib import Path
import shutil
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
import tqdm

from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, ConcatDataset
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

def construct_dataset(fdate, tdate,
    scaler_X=None, scaler_Y=None,
    filepath=HOURLY_DATA_PATH, station_name='종로구', target='PM10',
    sample_size=48, output_size=24,
    transform=True):
    if scaler_X == None or scaler_Y == None:
        data_set = data.UnivariateMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=filepath,
            features=[target],
            fdate=fdate,
            tdate=tdate,
            sample_size=sample_size,
            output_size=output_size)
    else:
        data_set = data.UnivariateMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=filepath,
            features=[target],
            fdate=fdate,
            tdate=tdate,
            sample_size=sample_size,
            output_size=output_size,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y)

    if transform:
        data_set.transform()

    return data_set

def ml_mlp_uni_ms(station_name="종로구"):
    print("Start Univariate MLP Mean Seasonality Decomposition Model")
    targets = ["PM10", "PM25"]
    # 24*14 = 336
    #sample_size = 336
    sample_size = 48
    output_size = 24
    # If you want to debug, fast_dev_run = True and n_trials should be small number
    fast_dev_run = False
    n_trials = 200
    # fast_dev_run = True
    # n_trials = 2

    # Hyper parameter
    epoch_size = 500
    batch_size = 128
    learning_rate = 1e-4

    # Blocked Cross Validation
    # neglect small overlap between train_dates and valid_dates
    # 11y = ((2y, 0.5y), (2y, 0.5y), (2y, 0.5y), (2.5y, 1y))
    train_dates = [
        (dt.datetime(2008, 1, 3, 1).astimezone(SEOULTZ), dt.datetime(2009, 12, 31, 23).astimezone(SEOULTZ)),
        (dt.datetime(2010, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2012, 6, 30, 23).astimezone(SEOULTZ)),
        (dt.datetime(2013, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2014, 12, 31, 23).astimezone(SEOULTZ)),
        (dt.datetime(2015, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2017, 12, 31, 23).astimezone(SEOULTZ))]
    valid_dates = [
        (dt.datetime(2010, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2010, 6, 30, 23).astimezone(SEOULTZ)),
        (dt.datetime(2012, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2012, 12, 31, 23).astimezone(SEOULTZ)),
        (dt.datetime(2015, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2015, 6, 30, 23).astimezone(SEOULTZ)),
        (dt.datetime(2018, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ))]

    # Debug
    # train_dates = [
    #     (dt.datetime(2013, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2014, 12, 31, 23).astimezone(SEOULTZ)),
    #     (dt.datetime(2015, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2017, 12, 31, 23).astimezone(SEOULTZ))]
    # valid_dates = [
    #     (dt.datetime(2015, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2015, 6, 30, 23).astimezone(SEOULTZ)),
    #     (dt.datetime(2018, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ))]

    train_valid_fdate = dt.datetime(2008, 1, 3, 1).astimezone(SEOULTZ)
    train_valid_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2020, 10, 31, 23).astimezone(SEOULTZ)

    # check date range assumption
    assert len(train_dates) == len(valid_dates)
    for i, (td, vd) in enumerate(zip(train_dates, valid_dates)):
        assert vd[0] > td[1]
    assert test_fdate > train_dates[-1][1]
    assert test_fdate > valid_dates[-1][1]

    train_features = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                      "temp", "u", "v", "pres", "humid", "prep", "snow"]

    for target in targets:
        print("Training " + target + "...", flush=True)
        output_dir = Path(f"/mnt/data/MLPMSUnivariate/{station_name}/{target}/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)
        model_dir = output_dir / "models"
        Path.mkdir(model_dir, parents=True, exist_ok=True)

        _df_h = data.load_imputed(HOURLY_DATA_PATH)
        df_h = _df_h.query('stationCode == "' +
                            str(SEOUL_STATIONS[station_name]) + '"')

        if station_name == '종로구' and \
            not Path("/input/python/input_jongno_imputed_hourly_pandas.csv").is_file():
            # load imputed result

            df_h.to_csv("/input/python/input_jongno_imputed_hourly_pandas.csv")

        # construct dataset for seasonality
        train_valid_dataset = construct_dataset(train_valid_fdate, train_valid_tdate,
            filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
            sample_size=sample_size, output_size=output_size, transform=False)
        # compute seasonality
        train_valid_dataset.preprocess()

        # For Block Cross Validation..
        # load dataset in given range dates and transform using scaler from train_valid_set
        # all dataset are saved in tuple
        train_datasets = tuple(construct_dataset(td[0], td[1],
                                                scaler_X=train_valid_dataset.scaler_X,
                                                scaler_Y=train_valid_dataset.scaler_Y,
                                                filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                                sample_size=sample_size, output_size=output_size, transform=True) for td in train_dates)

        valid_datasets = tuple(construct_dataset(vd[0], vd[1],
                                                scaler_X=train_valid_dataset.scaler_X,
                                                scaler_Y=train_valid_dataset.scaler_Y,
                                                filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                                sample_size=sample_size, output_size=output_size, transform=True) for vd in valid_dates)

        # just single test set
        test_dataset = construct_dataset(test_fdate, test_tdate,
                                        scaler_X=train_valid_dataset.scaler_X,
                                        scaler_Y=train_valid_dataset.scaler_Y,
                                        filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                        sample_size=sample_size, output_size=output_size, transform=True)

        # convert tuple of datasets to ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(valid_datasets)

        # num_layer == number of hidden layer
        hparams = Namespace(
            num_layers=1,
            layer_size=128,
            learning_rate=learning_rate,
            batch_size=batch_size)

        # The default logger in PyTorch Lightning writes to event files to be consumed by
        # TensorBoard. We don't use any logger here as it requires us to implement several abstract
        # methods. Instead we setup a simple callback, that saves metrics from each validation step.
        # metrics_callback = [MetricsCallback() for _ in range(len(train_dates_opt))]
        metrics_callback = MetricsCallback()

        def objective(trial):
            # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
            # filenames match. Therefore, the filenames for each trial must be made unique.
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                os.path.join(model_dir, "trial_{}".format(trial.number)), monitor="val_loss",
                period=10
            )

            hparams = Namespace(
                num_layers=1,
                layer_size=128,
                learning_rate=learning_rate,
                batch_size=batch_size)

            # models are independent from dataset and dates
            model = BaseMLPModel(trial=trial,
                                 hparams=hparams,
                                 input_size=sample_size,
                                 sample_size=sample_size,
                                 output_size=output_size,
                                 station_name=station_name,
                                 target=target,
                                 features=[target],
                                 train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 test_dataset=test_dataset,
                                 scaler_X=train_valid_dataset.scaler_X,
                                 scaler_Y=train_valid_dataset.scaler_Y,
                                 output_dir=output_dir)

            # most basic trainer, uses good defaults
            # TODO: PytorchLightningPruningCallback wheree to put?
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
                study, params=['num_layers', 'layer_size'])
            fig_cont1.write_image(
                str(output_dir / "contour_num_layers_layer_size.png"))
            fig_cont1.write_image(
                str(output_dir / "contour_num_layers_layer_size.svg"))

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
                study, params=['num_layers', 'layer_size'])
            fig_pcoord.write_image(str(output_dir / "parallel_coord.png"))
            fig_pcoord.write_image(str(output_dir / "parallel_coord.svg"))

            fig_slice = optv.plot_slice(
                study, params=['num_layers', 'layer_size'])
            fig_slice.write_image(str(output_dir / "slice.png"))
            fig_slice.write_image(str(output_dir / "slice.svg"))

            # set hparams with optmized value
            hparams.num_layers = trial.params['num_layers']
            hparams.layer_size = trial.params['layer_size']

            dict_hparams = copy.copy(vars(hparams))
            dict_hparams["sample_size"] = sample_size
            dict_hparams["output_size"] = output_size
            with open(output_dir / 'hparams.json', 'w') as f:
                print(dict_hparams, file=f)
            with open(output_dir / 'hparams.csv', 'w') as f:
                print(pd.DataFrame.from_dict(dict_hparams, orient='index'), file=f)

        model = BaseMLPModel(hparams=hparams,
                             input_size=sample_size,
                             sample_size=sample_size,
                             output_size=output_size,
                             station_name=station_name,
                             target=target,
                             features=[target],
                             train_dataset=train_dataset,
                             val_dataset=val_dataset,
                             test_dataset=test_dataset,
                             scaler_X=train_valid_dataset.scaler_X,
                             scaler_Y=train_valid_dataset.scaler_Y,
                             output_dir=output_dir)

        # record input
        for i, _train_set in enumerate(train_datasets):
            _train_set.to_csv(model.data_dir / ("df_trainset_{0}_".format(str(i).zfill(2)) + target + ".csv"))
        for i, _valid_set in enumerate(valid_datasets):
            _valid_set.to_csv(model.data_dir / ("df_validset_{0}_".format(str(i).zfill(2)) + target + ".csv"))
        train_valid_dataset.to_csv(model.data_dir / ("df_trainvalidset_" + target + ".csv"))
        test_dataset.to_csv(model.data_dir / ("df_testset_" + target + ".csv"))

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join(model_dir, "train_{epoch}_{val_loss:.2f}"), monitor="val_loss",
            period=10
        )

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00001,
            patience=30,
            verbose=True,
            mode='min')

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


class BaseMLPModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams = kwargs.get('hparams', Namespace(
            num_layers=1,
            learning_rate=1e-3,
            batch_size=32))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features', [self.target])
        self.metrics = kwargs.get('metrics', ['MAE', 'MSE', 'R2'])
        self.num_workers = kwargs.get('num_workers', 1)
        self.output_dir = kwargs.get(
            'output_dir', Path('/mnt/data/MLPMS2Univariate/'))
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

        self.train_dataset = kwargs.get('train_dataset', None)
        self.val_dataset = kwargs.get('val_dataset', None)
        self.test_dataset = kwargs.get('test_dataset', None)

        # Set ColumnTransformer if provided
        self._scaler_X = kwargs.get('scaler_X', None)
        self._scaler_Y = kwargs.get('scaler_Y', None)

        self.trial = kwargs.get('trial', None)
        self.sample_size = kwargs.get('sample_size', 48)
        self.output_size = kwargs.get('output_size', 24)
        self.input_size = kwargs.get(
            'input_size', self.sample_size)

        # select layer sizes
        # num_layer == number of hidden layer
        self.layer_sizes = [self.input_size, self.output_size]
        if self.trial:
            # if trial, there is no element of layer name such as "layer0_size"
            self.hparams.num_layers = self.trial.suggest_int(
                "num_layers", 2, 32)
            self.hparams.layer_size = self.trial.suggest_int(
                "layer_size", 24, 96)

        for l in range(self.hparams.num_layers):
            # insert another layer_size to end of list of layer_size
            # initial self.layer_sizes = [input_size, output_size]
            self.layer_sizes.insert(
                len(self.layer_sizes)-1, self.hparams.layer_size)

        # because of input_size and output_size,
        # total length of layer_sizes is num_layers + 2
        # num_layer == number of hidden layer
        assert len(self.layer_sizes) == self.hparams.num_layers + 2

        # construct Layers
        # if n_layers == 0 -> (in, out)
        # if n_layers > 1 -> (in, tmp0), (tmp0, tmp2), ..., (tmpN, out)
        # layer size are pair from slef.layer_sizes
        self.linears = nn.ModuleList()
        for i in range(self.hparams.num_layers + 1):
            self.linears.append(
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
        print("Linear Layers :")
        print(self.linears)

        self.dropout = nn.Dropout(p=0.2)

        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()

        log_name = self.target + "_" + dt.date.today().strftime("%y%m%d-%H-%M")
        self.logger = TensorBoardLogger(self.log_dir, name=log_name)

        self.train_logs = {}
        self.valid_logs = {}

    def forward(self, x):
        # vectorize
        x = x.view(-1, self.input_size).to(device)

        for (i, layer) in enumerate(self.linears):
            if i != len(self.linears):
                x = self.dropout(F.leaky_relu(layer(x)))
            else:
                x = layer(x)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr | self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, _y, _y_raw, dates = batch
        _y_hat = self(x)
        _loss = self.loss(_y_hat, _y)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        x, _y, _y_raw, dates = batch
        _y_hat = self(x)
        _loss = self.loss(_y_hat, _y)

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
        x, _y, _y_raw, dates = batch
        _y_hat = self(x)
        _loss = self.loss(_y_hat, _y)

        # transformed y might be smoothed
        y = _y.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_hat_inv = np.array(self.test_dataset.inverse_transform(y_hat, dates))

        _mae = mean_absolute_error(y_raw, y_hat_inv)
        _mse = mean_squared_error(y_raw, y_hat_inv)
        _r2 = r2_score(y_raw, y_hat_inv)

        return {
            'loss': _loss,
            'obs': y_raw,
            'sim': y_hat_inv,
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
            if isinstance(_y, torch.Tensor):
                values.append(_y.cpu().detach().numpy())
            elif isinstance(_y, np.ndarray):
                values.append(_y)
            else:
                raise TypeError("Wrong type: _y")
            # just append single key date
            indicies.append(_d[0])
        _df_obs = pd.DataFrame(data=values, index=indicies, columns=cols)

        values, indicies = [], []
        for _d, _y_hat in zip(dates, y_hats):
            if isinstance(_y_hat, torch.Tensor):
                values.append(_y_hat.cpu().detach().numpy())
            elif isinstance(_y_hat, np.ndarray):
                values.append(_y_hat)
            else:
                raise TypeError("Wrong type: _y_hat")
            # just append single key date
            indicies.append(_d[0])
        _df_sim = pd.DataFrame(data=values, index=indicies, columns=cols)

        return _df_obs, _df_sim

    def setup(self, stage=None):
        """Data operations on every GPU
        Wrong usage of LightningModule. Need to Refactored
        * TODO: Refactoring https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html
        """
        # first mkdir of seasonality
        Path.mkdir(self.png_dir / "seasonality", parents=True, exist_ok=True)
        Path.mkdir(self.svg_dir / "seasonality", parents=True, exist_ok=True)
        Path.mkdir(self.data_dir / "seasonality", parents=True, exist_ok=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """Creates mini-batch tensors from from list of tuples (x, y, dates)

        dates will not be trained but need to construct output, so don't put dates into Tensors
        Args:
            batch: list of tuple  (x, y, y_raw, dates).
                - x: pandas DataFrame or numpy of shape (input_size, num_features);
                - y: pandas DataFrame or numpy of shape (output_size);
                - y_raw: pandas DataFrame or numpy of shape (output_size);
                - date: pandas DateTimeIndex of shape (output_size):

        Returns:
            - xs: torch Tensor of shape (batch_size, input_size, num_features);
            - ys: torch Tensor of shape (batch_size, output_size);
            - ys_raw: torch Tensor of shape (batch_size, output_size);
            - dates: pandas DateTimeIndex of shape (batch_size, output_size):
        """

        # seperate source and target sequences
        # data goes to tuple (thanks to *) and zipped
        xs, ys, ys_raw, dates = zip(*batch)

        return torch.as_tensor(xs), \
            torch.as_tensor(ys), \
            torch.as_tensor(ys_raw), \
            dates

    @property
    def scaler_X(self):
        return self._scaler_X

    @property
    def scaler_Y(self):
        return self._scaler_Y

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
