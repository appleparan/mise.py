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
import scipy as sp
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
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics

import optuna
from optuna.integration import PyTorchLightningPruningCallback, TensorBoardCallback
import optuna.visualization as optv

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

import data
from constants import SEOUL_STATIONS, SEOULTZ

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def construct_dataset(fdate, tdate,
    scaler_X=None, scaler_Y=None,
    filepath=HOURLY_DATA_PATH, station_name='종로구', target='PM10',
    sample_size=48, output_size=24,
    features=["SO2", "CO", "NO2", "PM10", "PM25",
                      "temp", "wind_spd", "wind_cdir", "wind_sdir",
                      "pres", "humid", "prep"],
    features_periodic=["SO2", "CO", "NO2", "PM10", "PM25", "temp",
                                "wind_spd", "wind_cdir", "wind_sdir", "pres", "humid"],
    features_nonperiodic=["prep"],
    transform=True):
    """Crate dataset and transform
    """
    if scaler_X == None or scaler_Y == None:
        data_set = data.MultivariateMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=filepath,
            features=features,
            features_1=features_nonperiodic,
            features_2=features_periodic,
            fdate=fdate,
            tdate=tdate,
            sample_size=sample_size,
            output_size=output_size)
    else:
        data_set = data.MultivariateMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=filepath,
            features=features,
            features_1=features_nonperiodic,
            features_2=features_periodic,
            fdate=fdate,
            tdate=tdate,
            sample_size=sample_size,
            output_size=output_size,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y)

    if transform:
        data_set.transform()

    return data_set

def ml_mlp_mul_ms_mccr(station_name="종로구"):
    print("Start Multivariate MLP Mean Seasonality Decomposition (MCCR) Model")
    targets = ["PM10", "PM25"]
    # targets = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
    #                   "temp", "u", "v", "pres", "humid", "prep", "snow"]
    # 24*14 = 336
    #sample_size = 336
    sample_size = 72
    output_size = 24
    # If you want to debug, fast_dev_run = True and n_trials should be small number
    fast_dev_run = False
    n_trials = 128
    # fast_dev_run = True
    # n_trials = 1

    # Hyper parameter
    epoch_size = 500
    batch_size = 64
    learning_rate = 1e-3

    # Blocked Cross Validation
    # neglect small overlap between train_dates and valid_dates
    # 11y = ((2y, 0.5y), (2y, 0.5y), (2y, 0.5y), (2.5y, 1y))
    train_dates = [
        (dt.datetime(2008, 1, 4, 1).astimezone(SEOULTZ), dt.datetime(2009, 12, 31, 23).astimezone(SEOULTZ)),
        (dt.datetime(2010, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2012, 6, 30, 23).astimezone(SEOULTZ)),
        (dt.datetime(2013, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2014, 12, 31, 23).astimezone(SEOULTZ)),
        (dt.datetime(2015, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2017, 12, 31, 23).astimezone(SEOULTZ))]
    valid_dates = [
        (dt.datetime(2010, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2010, 6, 30, 23).astimezone(SEOULTZ)),
        (dt.datetime(2012, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2012, 12, 31, 23).astimezone(SEOULTZ)),
        (dt.datetime(2015, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2015, 6, 30, 23).astimezone(SEOULTZ)),
        (dt.datetime(2018, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ))]
    train_valid_fdate = dt.datetime(2008, 1, 3, 1).astimezone(SEOULTZ)
    train_valid_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)

    # Debug
    if fast_dev_run:
        train_dates = [
            (dt.datetime(2015, 7, 1, 0).astimezone(SEOULTZ), dt.datetime(2017, 12, 31, 23).astimezone(SEOULTZ))]
        valid_dates = [
            (dt.datetime(2018, 1, 1, 0).astimezone(SEOULTZ), dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ))]
        train_valid_fdate = dt.datetime(2015, 7, 1, 0).astimezone(SEOULTZ)
        train_valid_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)

    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2020, 10, 31, 23).astimezone(SEOULTZ)

    # check date range assumption
    assert len(train_dates) == len(valid_dates)
    for i, (td, vd) in enumerate(zip(train_dates, valid_dates)):
        assert vd[0] > td[1]
    assert test_fdate > train_dates[-1][1]
    assert test_fdate > valid_dates[-1][1]

    train_features = ["SO2", "CO", "NO2", "PM10", "PM25",
                      "temp", "wind_spd", "wind_cdir", "wind_sdir",
                      "pres", "humid", "prep"]
    train_features_periodic = ["SO2", "CO", "NO2", "PM10", "PM25", "temp",
                                "wind_spd", "wind_cdir", "wind_sdir", "pres", "humid"]
    train_features_nonperiodic = ["prep"]

    for target in targets:
        print("Training " + target + "...")
        output_dir = Path(f"/mnt/data/MLPMSMCCRMultivariate/{station_name}/{target}/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)
        model_dir = output_dir / "models"
        Path.mkdir(model_dir, parents=True, exist_ok=True)
        log_dir = output_dir / "log"
        Path.mkdir(log_dir, parents=True, exist_ok=True)

        _df_h = data.load_imputed(HOURLY_DATA_PATH)
        df_h = _df_h.query('stationCode == "' +
                            str(SEOUL_STATIONS[station_name]) + '"')

        if station_name == '종로구' and \
            not Path("/input/python/input_jongno_imputed_hourly_pandas.csv").is_file():
            # load imputed result

            df_h.to_csv("/input/python/input_jongno_imputed_hourly_pandas.csv")

        # construct dataset for seasonality
        print("Construct Train/Validation Sets...", flush=True)
        train_valid_dataset = construct_dataset(train_valid_fdate, train_valid_tdate,
            filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
            sample_size=sample_size, output_size=output_size, transform=False)
        # compute seasonality
        train_valid_dataset.preprocess()

         # For Block Cross Validation..
        # load dataset in given range dates and transform using scaler from train_valid_set
        # all dataset are saved in tuple
        print("Construct Training Sets...", flush=True)
        train_datasets = tuple(construct_dataset(td[0], td[1],
                                                scaler_X=train_valid_dataset.scaler_X,
                                                scaler_Y=train_valid_dataset.scaler_Y,
                                                filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                                sample_size=sample_size, output_size=output_size,
                                                features=train_features,
                                                features_periodic=train_features_periodic,
                                                features_nonperiodic=train_features_nonperiodic,
                                                transform=True) for td in train_dates)

        print("Construct Validation Sets...", flush=True)
        valid_datasets = tuple(construct_dataset(vd[0], vd[1],
                                                scaler_X=train_valid_dataset.scaler_X,
                                                scaler_Y=train_valid_dataset.scaler_Y,
                                                filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                                sample_size=sample_size, output_size=output_size,
                                                features=train_features,
                                                features_periodic=train_features_periodic,
                                                features_nonperiodic=train_features_nonperiodic,
                                                transform=True) for vd in valid_dates)

        # just single test set
        print("Construct Test Sets...", flush=True)
        test_dataset = construct_dataset(test_fdate, test_tdate,
                                        scaler_X=train_valid_dataset.scaler_X,
                                        scaler_Y=train_valid_dataset.scaler_Y,
                                        filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                        sample_size=sample_size, output_size=output_size,
                                        features=train_features,
                                        features_periodic=train_features_periodic,
                                        features_nonperiodic=train_features_nonperiodic,
                                        transform=True)

        # convert tuple of datasets to ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(valid_datasets)

        # num_layer == number of hidden layer
        hparams = Namespace(
            sigma=1.0,
            num_layers=4,
            layer_size=1024,
            learning_rate=learning_rate,
            batch_size=batch_size)

        def objective(trial):
            model = BaseMLPModel(trial=trial,
                                 hparams=hparams,
                                 input_size=sample_size * len(train_features),
                                 sample_size=sample_size,
                                 output_size=output_size,
                                 station_name=station_name,
                                 target=target,
                                 features=train_features,
                                 features_periodic=train_features_periodic,
                                 features_nonperiodic=train_features_nonperiodic,
                                 train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 test_dataset=test_dataset,
                                 scaler_X=train_valid_dataset.scaler_X,
                                 scaler_Y=train_valid_dataset.scaler_Y,
                                 output_dir=output_dir)

            # most basic trainer, uses good defaults
            trainer = Trainer(gpus=1 if torch.cuda.is_available() else None,
                              precision=32,
                              min_epochs=1, max_epochs=20,
                              default_root_dir=output_dir,
                              fast_dev_run=fast_dev_run,
                              logger=True,
                              checkpoint_callback=False,
                              callbacks=[PyTorchLightningPruningCallback(
                                    trial, monitor="valid/MSE")])

            trainer.fit(model)

            # Don't Log
            # hyperparameters = model.hparams
            # trainer.logger.log_hyperparams(hyperparameters)

            return trainer.callback_metrics.get("valid/MSE")

        if n_trials > 1:
            study = optuna.create_study(direction="minimize")
            study.enqueue_trial({
                'sigma': 1.3,
                'num_layers': 4,
                'layer_size': 64,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 1.3,
                'num_layers': 4,
                'layer_size': 256,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 1.3,
                'num_layers': 4,
                'layer_size': 1024,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 1.3,
                'num_layers': 4,
                'layer_size': 64,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 1.3,
                'num_layers': 8,
                'layer_size': 64,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 1.3,
                'num_layers': 12,
                'layer_size': 64,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 0.7,
                'num_layers': 4,
                'layer_size': 64,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            study.enqueue_trial({
                'sigma': 2.0,
                'num_layers': 4,
                'layer_size': 64,
                'learning_rate': learning_rate,
                'batch_size': batch_size})
            # timeout = 3600*36 = 36h
            study.optimize(objective,
                n_trials=n_trials, timeout=3600*36)

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
            hparams.sigma = trial.params['sigma']
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
                             input_size=sample_size * len(train_features),
                             sample_size=sample_size,
                             output_size=output_size,
                             station_name=station_name,
                             target=target,
                             features=train_features,
                             features_periodic=train_features_periodic,
                             features_nonperiodic=train_features_nonperiodic,
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
            os.path.join(model_dir, "train_{epoch}_{valid/MSE:.2f}"), monitor="valid/MSE",
            period=10
        )

        early_stop_callback = EarlyStopping(
            monitor='valid/MSE',
            min_delta=0.001,
            patience=30,
            verbose=True,
            mode='min')

        log_version = dt.date.today().strftime("%y%m%d-%H-%M")
        loggers = [ \
            TensorBoardLogger(log_dir, version=log_version),
            CSVLogger(log_dir, version=log_version)]

        # most basic trainer, uses good defaults
        trainer = Trainer(gpus=1 if torch.cuda.is_available() else None,
                          precision=32,
                          min_epochs=1, max_epochs=epoch_size,
                          default_root_dir=output_dir,
                          fast_dev_run=fast_dev_run,
                          logger=loggers,
                          log_every_n_steps=5,
                          flush_logs_every_n_steps=10,
                          callbacks=[early_stop_callback],
                          checkpoint_callback=checkpoint_callback)

        trainer.fit(model)

        # run test set
        trainer.test(ckpt_path=None)

        shutil.rmtree(model_dir)


class BaseMLPModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams = kwargs.get('hparams', Namespace(
            sigma=1.0,
            num_layers=4,
            layer_size=128,
            learning_rate=1e-3,
            batch_size=32))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features', ["SO2", "CO", "NO2", "PM10", "PM25",
                                        "temp", "wind_spd", "wind_cdir", "wind_sdir",
                                        "pres", "humid", "prep"])
        self.features_periodic = kwargs.get('features_periodic',
                                            ["SO2", "CO", "NO2", "PM10", "PM25"])
        self.features_nonperiodic = kwargs.get('features_nonperiodic',
                                            ["temp", "wind_spd", "wind_cdir", "wind_sdir",
                                            "pres", "humid", "prep"])
        self.metrics = kwargs.get('metrics', ['MAE', 'MSE', 'R2', 'MAD'])
        self.num_workers = kwargs.get('num_workers', 1)
        self.output_dir = kwargs.get(
            'output_dir', Path('/mnt/data/MLPMS2Multivariate/'))
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
        self.input_size = kwargs.get('input_size', self.sample_size * len(self.features))

        # select layer sizes
        # num_layer == number of hidden layer
        self.layer_sizes = [self.input_size, self.output_size]
        if self.trial:
            self.hparams.sigma = self.trial.suggest_float(
                "sigma", 0.5, 5.0, step=0.1)
            self.hparams.num_layers = self.trial.suggest_int(
                "num_layers", 2, 12)
            self.hparams.layer_size = self.trial.suggest_int(
                "layer_size", 8, 1024)

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

        self.ar = nn.Linear(self.sample_size, self.output_size)

        self.act = nn.ReLU()

        self.dropout = nn.Dropout(p=0.2)
        # self.loss = nn.MSELoss()
        self.loss = MCCRLoss(sigma=self.hparams.sigma)
        # self.loss = nn.L1Loss()

        self.train_logs = {}
        self.valid_logs = {}

        self.df_obs = pd.DataFrame()
        self.df_sim = pd.DataFrame()

    def forward(self, x, x1d):
        # vectorize
        x = x.view(-1, self.input_size).to(device)

        for (i, layer) in enumerate(self.linears):
            if i != len(self.linears) - 1:
                x = F.leaky_relu(layer(x))
            else:
                x = layer(x)

        y = x + self.ar(x1d)

        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=0.001)

    def training_step(self, batch, batch_idx):
        x, x1d, _y, _y_raw, dates = batch
        _y_hat = self(x, x1d)
        _loss = self.loss(_y_hat, _y)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y, y_hat)
        _mse = mean_squared_error(y, y_hat)
        _r2 = r2_score(y, y_hat)
        _mad = median_abs_deviation(y - y_hat)

        return {
            'loss': _loss,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'MAD': _mad,
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
        _log['loss'] = avg_loss.detach().cpu().item()

        self.train_logs[self.current_epoch] = _log

        # self.log('train/loss', tensorboard_logs['train/loss'].item(), prog_bar=True)
        self.log('train/MSE', tensorboard_logs['train/MSE'].item(), on_epoch=True, logger=self.logger)
        self.log('train/MAE', tensorboard_logs['train/MAE'].item(), on_epoch=True, logger=self.logger)
        self.log('train/MAD', tensorboard_logs['train/MAD'].item(), on_epoch=True, logger=self.logger)
        self.log('train/avg_loss', _log['loss'], on_epoch=True, logger=self.logger)

    def validation_step(self, batch, batch_idx):
        x, x1d, _y, _y_raw, dates = batch
        _y_hat = self(x, x1d)
        _loss = self.loss(_y_hat, _y)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y, y_hat)
        _mse = mean_squared_error(y, y_hat)
        _r2 = r2_score(y, y_hat)
        _mad = median_abs_deviation(y - y_hat)

        return {
            'loss': _loss,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'MAD': _mad,
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
            _log[name] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean().item()
        tensorboard_logs['step'] = self.current_epoch
        _log['loss'] = avg_loss.detach().cpu().item()

        self.valid_logs[self.current_epoch] = _log

        self.log('valid/MSE', tensorboard_logs['valid/MSE'].item(), on_epoch=True, logger=self.logger)
        self.log('valid/MAE', tensorboard_logs['valid/MAE'].item(), on_epoch=True, logger=self.logger)
        self.log('valid/MAD', tensorboard_logs['valid/MAD'].item(), on_epoch=True, logger=self.logger)
        self.log('valid/loss', _log['loss'], on_epoch=True, logger=self.logger)

    def test_step(self, batch, batch_idx):
        x, x1d, _y, _y_raw, dates = batch

        _y_hat = self(x, x1d)

        y = _y.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_hat2 = relu_mul(
            np.array(self.test_dataset.inverse_transform(y_hat, dates)))
        _loss = self.loss(_y_raw, torch.as_tensor(y_hat2).to(device))

        _mae = mean_absolute_error(y_raw, y_hat2)
        _mse = mean_squared_error(y_raw, y_hat2)
        _r2 = r2_score(y_raw, y_hat2)
        _mad = median_abs_deviation(y_raw - y_hat2)

        return {
            'loss': _loss,
            'obs': y_raw,
            'sim': y_hat2,
            'dates': dates,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'MAD': _mad,
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
        plot_scatter(self.output_size, df_obs, df_sim, self.target,
                     self.data_dir, self.png_dir, self.svg_dir)
        plot_logs(self.train_logs, self.valid_logs, self.target,
                  self.data_dir, self.png_dir, self.svg_dir)
        for metric in ['MAPE', 'PCORR', 'SCORR', 'R2', 'FB', 'NMSE', 'MG', 'VG', 'FAC2']:
            plot_metrics(metric, self.output_size, df_obs, df_sim,
                         self.data_dir, self.png_dir, self.svg_dir)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().item()
        tensorboard_logs = {'test/loss': avg_loss}
        for name in self.metrics:
            tensorboard_logs['test/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
        tensorboard_logs['step'] = self.current_epoch

        self.log('test/MSE', tensorboard_logs['test/MSE'].item(), on_epoch=True, logger=self.logger)
        self.log('test/MAE', tensorboard_logs['test/MAE'].item(), on_epoch=True, logger=self.logger)
        self.log('test/MAD', tensorboard_logs['test/MAD'].item(), on_epoch=True, logger=self.logger)
        self.log('test/loss', avg_loss, on_epoch=True, logger=self.logger)

        self.df_obs = df_obs
        self.df_sim = df_sim

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
        # round decimal
        _df_sim = pd.DataFrame(data=np.around(
            values), index=indicies, columns=cols)

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
        data: list of tuple  (x, y, dates).
            - x: pandas DataFrame or numpy of shape (input_size, num_features);
            - y: pandas DataFrame or numpy of shape (output_size);
            - date: pandas DateTimeIndex of shape (output_size):

        Returns:
            - xs: torch Tensor of shape (batch_size, input_size, num_features);
            - ys: torch Tensor of shape (batch_size, output_size);
            - dates: pandas DateTimeIndex of shape (batch_size, output_size):
        """

        # seperate source and target sequences
        # data goes to tuple (thanks to *) and zipped
        xs, x1ds, ys, ys_raw, dates = zip(*batch)

        #return torch.as_tensor(xs.reshape(1, -1)), \
        return torch.as_tensor(xs), \
            torch.as_tensor(x1ds), \
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

        # save data first
        data_dir_h = data_dir / str(t).zfill(2)
        Path.mkdir(data_dir_h, parents=True, exist_ok=True)
        csv_path = data_dir_h / ("line_" + str(t).zfill(2) + "h.csv")
        df_line = pd.DataFrame.from_dict(
            {'date': dates, 'obs': obs, 'sim': sim})
        df_line.set_index('date', inplace=True)
        df_line.to_csv(csv_path)

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(dates, obs, color="tab:blue", alpha=0.7, label="obs")
        ax.plot(dates, sim, color="tab:orange", alpha=0.7, label="sim")
        ax.legend()

        # Major ticks every 3 months.
        fmt_half_year = mdates.MonthLocator(interval=3)
        fmt_month = mdates.MonthLocator()
        ax.xaxis.set_major_locator(fmt_half_year)
        ax.xaxis.set_minor_locator(fmt_month)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        ax.set_xlabel('dates')
        ax.set_ylabel(target)
        ax.set_title('OBS & Model')
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

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
    csv_path = data_dir / ("log_valid.csv")
    df_valid_logs.to_csv(csv_path)

    epochs = df_train_logs.index.to_numpy()
    for col in df_train_logs.columns:
        png_path = png_dir / ("log_train_" + col + ".png")
        svg_path = svg_dir / ("log_train_" + col + ".svg")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, df_train_logs[col].to_numpy(), color="tab:blue")

        # leg = plt.legend()
        # ax.get_legend().remove()

        ax.set_xlabel('epoch')
        ax.set_ylabel(col)
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

    csv_path = data_dir / ("log_valid.csv")
    df_valid_logs.to_csv(csv_path)

    epochs = df_valid_logs.index.to_numpy()
    for col in df_valid_logs.columns:
        png_path = png_dir / ("log_valid_" + col + ".png")
        svg_path = svg_dir / ("log_valid_" + col + ".svg")

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, df_valid_logs[col].to_numpy(), color="tab:blue")

        # leg = plt.legend()
        # ax.get_legend().remove()

        ax.set_xlabel('epoch')
        ax.set_ylabel(col)
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

    for col1, col2 in zip(df_train_logs.columns, df_train_logs.columns):
        if col1 != col2:
            continue

        png_path = png_dir / ("log_train_valid_" + col + ".png")
        svg_path = svg_dir / ("log_train_valid_" + col + ".svg")

        # plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(epochs, df_train_logs[col].to_numpy(), color="tab:blue", label="train")
        ax.plot(epochs, df_valid_logs[col].to_numpy(), color="tab:orange", label="valid")

        leg = plt.legend()
        ax.get_legend().remove()

        ax.set_xlabel('epoch')
        ax.set_ylabel(col1)
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

def plot_scatter(output_size, df_obs, df_sim, target, data_dir, png_dir, svg_dir):
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

        # save data first
        data_dir_h = data_dir / str(t).zfill(2)
        Path.mkdir(data_dir_h, parents=True, exist_ok=True)
        csv_path = data_dir_h / ("scatter_" + str(t).zfill(2) + "h.csv")

        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()
        maxval = np.nanmax([np.nanmax(obs), np.nanmax(sim)])

        df_scatter = pd.DataFrame({'obs': obs, 'sim': sim})
        df_scatter.to_csv(csv_path)

        # plot
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(obs, sim, color="tab:blue", alpha=0.8, s=(10.0,))
        ax.set_aspect(1.0)

        ax.set_xlabel('target')
        ax.set_ylabel('predicted')
        ax.set_title(target)
        plt.xlim([0.0, maxval])
        plt.ylim([0.0, maxval])
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

def plot_metrics(metric, output_size, df_obs, df_sim, data_dir, png_dir, svg_dir):
    """
    Reference:
        * Chang, Joseph C., and Steven R. Hanna.
            "Air quality model performance evaluation." Meteorology and Atmospheric Physics 87.1-3 (2004): 167-196.
    """
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(png_dir, parents=True, exist_ok=True)
    Path.mkdir(svg_dir, parents=True, exist_ok=True)

    png_path = png_dir / (metric.lower() + "_time.png")
    svg_path = svg_dir / (metric.lower() + "_time.svg")
    csv_path = data_dir / (metric.lower() + "_time.csv")

    times = list(range(1, output_size + 1))
    metric_vals = []

    for t in range(output_size):
        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        # Best case
        # MG, VG, R, and FAC2=1.0;
        # FB and NMSE = 0.0.
        if metric == 'MAPE':
            metric_vals.append(
                sklearn.metrics.mean_absolute_percentage_error(obs, sim))
        elif metric == 'PCORR':
            pcorr, p_val = sp.stats.pearsonr(obs, sim)
            metric_vals.append(pcorr)
        elif metric == 'SCORR':
            scorr, p_val = sp.stats.spearmanr(obs, sim)
            metric_vals.append(scorr)
        elif metric == 'R2':
            metric_vals.append(
                sklearn.metrics.r2_score(obs, sim))
        elif metric == 'FB':
            # fractional bias
            avg_o = np.mean(obs)
            avg_s = np.mean(sim)
            metric_vals.append(
                2.0 * ((avg_o - avg_s) / (avg_o + avg_s + np.finfo(float).eps)))
        elif metric == 'NMSE':
            # normalized mean square error
            metric_vals.append(
                np.square(np.mean(obs - sim)) / (np.mean(obs) * np.mean(sim) + np.finfo(float).eps))
        elif metric == 'MG':
            # geometric mean bias
            metric_vals.append(
                np.exp(np.mean(np.log(obs + 1.0)) - np.mean(np.log(sim + 1.0))))
        elif metric == 'VG':
            # geometric variance
            metric_vals.append(
                np.exp(np.mean(np.square(np.log(obs + 1.0) - np.log(sim + 1.0)))))
        elif metric == 'FAC2':
            # the fraction of predictions within a factor of two of observations
            frac = sim / obs
            metric_vals.append(
                ((0.5 <= frac) & (frac <= 2.0)).sum())

    title = ''
    if metric == 'MAPE':
        # Best MAPE => 1.0
        title = 'MAPE'
        ylabel = 'MAPE'
    elif metric == 'R2':
        # Best R2 => 1.0
        metric_vals.insert(0, 1.0)
        times = list(range(len(metric_vals)))
        title = 'R2'
        ylabel = 'R2'
    elif metric == 'PCORR':
        # Best R2 => 1.0
        metric_vals.insert(0, 1.0)
        times = list(range(len(metric_vals)))
        title = 'Pearson correlation coefficient (p=' + str(p_val) + ')'
        ylabel = 'corr'
    elif metric == 'SCORR':
        # Best R2 => 1.0
        metric_vals.insert(0, 1.0)
        times = list(range(len(metric_vals)))
        title = 'Spearman\'s rank-order correlation coefficient (p=' + str(
            p_val) + ')'
        ylabel = 'corr'
    elif metric == 'FB':
        # Best FB => 0.0
        title = 'Fractional Bias'
        ylabel = 'FB'
    elif metric == 'NMSE':
        # Best NMSE => 0.0
        title = 'Normalized Mean Square Error'
        ylabel = 'NMSE'
    elif metric == 'MG':
        # Best MG => 1.0
        title = 'Geometric Mean Bias'
        ylabel = 'MG'
    elif metric == 'VG':
        # Best VG => 1.0
        title = 'Geometric Mean Variance'
        ylabel = 'VG'
    elif metric == 'FAC2':
        # Best FAC2 => 1.0
        title = 'The Fraction of predictions within a factor of two of observations'
        ylabel = 'FAC2'

    df_metric = pd.DataFrame({'time': times, metric.lower(): metric_vals})
    df_metric.set_index('time', inplace=True)
    df_metric.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, metric_vals, color="tab:blue")

    if title:
        ax.set_title(title)
    ax.set_xlabel('time')
    if ylabel:
        ax.set_ylabel(ylabel)
    if metric == 'MAPE':
        plt.ylim([0.0, 1.0])
    elif metric == 'R2' or metric == 'PCORR' or metric == 'SCORR':
        ymin = min(0.0, min(metric_vals))
        plt.ylim([ymin, 1.0])

    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()


def swish(_input, beta=1.0):
    """
        Swish function in [this paper](https://arxiv.org/pdf/1710.05941.pdf)

    Args:
        input: Tensor

    Returns:
        output: Activated tensor
    """
    return _input * beta * torch.sigmoid(_input)


class MCCRLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        # save sigma
        assert sigma > 0
        self.sigma2 = sigma**2

    def forward(self, _input: torch.Tensor, _target: torch.Tensor) -> torch.Tensor:
        """
        Implement maximum correntropy criterion for regression

        loss(y, t) = sigma^2 * (1.0 - exp(-(y-t)^2/sigma^2))

        where sigma > 0 (parameter)

        Reference:
            * Feng, Yunlong, et al. "Learning with the maximum correntropy criterion induced losses for regression." J. Mach. Learn. Res. 16.1 (2015): 993-1034.
        """
        return torch.mean(
            self.sigma2 * (1-torch.exp(-(_input - _target)**2 / self.sigma2)))


def relu_mul(x):
    """[fastest method](https://stackoverflow.com/a/32109519/743078)
    """
    return x * (x > 0)
