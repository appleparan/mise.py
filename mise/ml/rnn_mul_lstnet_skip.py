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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn.metrics

import madgrad

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
        data_set = data.MultivariateRNNMeanSeasonalityDataset(
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
        data_set = data.MultivariateRNNMeanSeasonalityDataset(
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
        # you can braodcast seasonality only if scaler was fit
        data_set.broadcast_seasonality()

    return data_set


def ml_rnn_mul_lstnet_skip(station_name="종로구"):
    print("Start Multivariate LSTNet (Skip Layer) Model")
    targets = ["PM10", "PM25"]
    # 24*14 = 336
    sample_size = 48
    output_size = 24
    # If you want to debug, fast_dev_run = True and n_trials should be small number
    fast_dev_run = False
    n_trials = 96
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
        (dt.datetime(2008, 1, 3, 1).astimezone(SEOULTZ), dt.datetime(2009, 12, 31, 23).astimezone(SEOULTZ)),
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
        output_dir = Path(f"/mnt/data/RNNLSTNetSkipMultivariate/{station_name}/{target}/")
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
            sample_size=sample_size, output_size=output_size, features=train_features,
            features_periodic=train_features_periodic, features_nonperiodic=train_features_nonperiodic,
            transform=False)
        # scaler in trainn_valid_set is not fitted, so fit!
        train_valid_dataset.preprocess()
        # then it can broadcast its seasonalities!
        train_valid_dataset.broadcast_seasonality()

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

        # Dummy hyperparameters
        hparams = Namespace(
            filter_size=3,
            hidCNN=16,
            hidSkip=16,
            hidRNN=16,
            learning_rate=learning_rate,
            batch_size=batch_size)

        def objective(trial):
            model = BaseLSTNetModel(trial=trial,
                                    hparams=hparams,
                                    sample_size=sample_size,
                                    output_size=output_size,
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
                study, params=['filter_size', 'hidCNN'])
            fig_cont1.write_image(
                str(output_dir / "contour_filter_size_hidCNN.png"))
            fig_cont1.write_image(
                str(output_dir / "contour_filter_size_hidCNN.svg"))

            fig_cont2 = optv.plot_contour(
                study, params=['hidCNN', 'hidRNN'])
            fig_cont2.write_image(
                str(output_dir / "contour_hidCNN_hidRNN.png"))
            fig_cont2.write_image(
                str(output_dir / "contour_hidCNN_hidRNN.svg"))

            fig_cont3 = optv.plot_contour(
                study, params=['filter_size', 'hidRNN'])
            fig_cont3.write_image(
                str(output_dir / "contour_filter_size_hidRNN.png"))
            fig_cont3.write_image(
                str(output_dir / "contour_filter_size_hidRNN.svg"))

            fig_cont3 = optv.plot_contour(
                study, params=['hidSkip', 'hidRNN'])
            fig_cont3.write_image(
                str(output_dir / "contour_hidSkip_hidRNN.png"))
            fig_cont3.write_image(
                str(output_dir / "contour_hidSkip_hidRNN.svg"))

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
                study, params=['filter_size', 'hidCNN', 'hidSkip', 'hidRNN'])
            fig_pcoord.write_image(str(output_dir / "parallel_coord.png"))
            fig_pcoord.write_image(str(output_dir / "parallel_coord.svg"))

            fig_slice = optv.plot_slice(
                study, params=['filter_size', 'hidCNN', 'hidSkip',  'hidRNN'])
            fig_slice.write_image(str(output_dir / "slice.png"))
            fig_slice.write_image(str(output_dir / "slice.svg"))

            # set hparams with optmized value
            hparams.filter_size = trial.params['filter_size']
            hparams.hidCNN = trial.params['hidCNN']
            hparams.hidSkip = trial.params['hidSkip']
            hparams.hidRNN = trial.params['hidRNN']

            dict_hparams = copy.copy(vars(hparams))
            dict_hparams["sample_size"] = sample_size
            dict_hparams["output_size"] = output_size
            with open(output_dir / 'hparams.json', 'w') as f:
                print(dict_hparams, file=f)
            with open(output_dir / 'hparams.csv', 'w') as f:
                print(pd.DataFrame.from_dict(dict_hparams, orient='index'), file=f)

        model = BaseLSTNetModel(hparams=hparams,
                                sample_size=sample_size,
                                output_size=output_size,
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
            hidRNN=16,
            filter_size=5,
            learning_rate=1e-3,
            batch_size=32
        ))

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
        self.metrics = kwargs.get('metrics', ['MAE', 'MSE', 'R2'])
        self.num_workers = kwargs.get('num_workers', 1)
        self.output_dir = kwargs.get(
            'output_dir', Path('/mnt/data/RNNLSTNetSkipMultivariate/'))
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

        self.trial = kwargs.get('trial', None)
        self.sample_size = kwargs.get('sample_size', 48)
        self.output_size = kwargs.get('output_size', 24)

        if self.trial:
            self.hparams.filter_size = self.trial.suggest_int(
                "filter_size", 1, 5, step=2)
            self.hparams.hidRNN = self.trial.suggest_int(
                "hidRNN", 8, 128)
            self.hparams.hidCNN = self.trial.suggest_int(
                "hidCNN", 8, 64)
            self.hparams.hidSkip = self.trial.suggest_int(
                "hidSkip", 8, 128)

        self.kernel_shape = (self.hparams.filter_size, len(self.features))

        padding_size = int(self.hparams.filter_size - 1)
        self.pad = nn.ZeroPad2d((0, 0, padding_size, 0))

        self.conv = nn.Conv2d(1, self.hparams.hidCNN, self.kernel_shape)
        self.dropout = nn.Dropout(p = 0.1)

        # normal GRU
        self.gru_no_skip = nn.GRU(self.hparams.hidCNN, self.hparams.hidRNN, batch_first=True)

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

        self.proj1 = nn.Linear(self.hparams.hidRNN + self.p * self.hparams.hidSkip,
                               self.output_size)
        self.proj2 = nn.Linear(self.output_size, self.output_size)
        # self.act = nn.ReLU()

        self.loss = nn.MSELoss()
        #self.loss = nn.L1Loss()

        self.train_logs = {}
        self.valid_logs = {}

        self.df_obs = pd.DataFrame()
        self.df_sim = pd.DataFrame()

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
        # x_rnn = (num_layers * num_directions, batch_size, hidRNN)
        _, x_rnn = self.gru_no_skip(c.permute(0, 2, 1))

        # x_rnn = (batch_size, hidRNN)
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

        # (batch_size, hidRNN + self.p * hidSkip
        output1 = torch.cat((x_rnn, x_rnn_skip), dim=1)
        # (batch_first, output_size)
        output2 = self.ar(_x1d)

        # Sum Autoregressive and Recurrent Layer output
        output = self.proj1(output1) + self.proj2(output2)

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=0.001)

    def training_step(self, batch, batch_idx):
        x, _x1d, _y, _y_raw, dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y_hat, _y)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y_hat, y)
        _mse = mean_squared_error(y_hat, y)
        _r2 = r2_score(y_hat, y)

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
        _log['loss'] = avg_loss.detach().cpu().item()

        self.train_logs[self.current_epoch] = _log

        # self.log('train/loss', tensorboard_logs['train/loss'].item(), prog_bar=True)
        self.log('train/MSE', tensorboard_logs['train/MSE'].item(), on_epoch=True, logger=self.logger)
        self.log('train/MAE', tensorboard_logs['train/MAE'].item(), on_epoch=True, logger=self.logger)
        self.log('train/avg_loss', _log['loss'], on_epoch=True, logger=self.logger)

    def validation_step(self, batch, batch_idx):
        x, _x1d, _y, _y_raw, dates = batch

        _y_hat = self(x, _x1d)
        _loss = self.loss(_y_hat, _y)

        y = _y.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()

        _mae = mean_absolute_error(y_hat, y)
        _mse = mean_squared_error(y_hat, y)
        _r2 = r2_score(y_hat, y)

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
            _log[name] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean().item()
        tensorboard_logs['step'] = self.current_epoch
        _log['loss'] = avg_loss.detach().cpu().item()

        self.valid_logs[self.current_epoch] = _log

        self.log('valid/MSE', tensorboard_logs['valid/MSE'].item(), on_epoch=True, logger=self.logger)
        self.log('valid/MAE', tensorboard_logs['valid/MAE'].item(), on_epoch=True, logger=self.logger)
        self.log('valid/loss', _log['loss'], on_epoch=True, logger=self.logger)

    def test_step(self, batch, batch_idx):
        x, _x1d, _y, _y_raw, dates = batch

        _y_hat = self(x, _x1d)

        y = _y.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_hat2 = relu_mul(
            np.array(self.test_dataset.inverse_transform(y_hat, dates)))
        _loss = self.loss(torch.as_tensor(y_hat2).to(device), _y_raw)

        _mae = mean_absolute_error(y_hat2, y_raw)
        _mse = mean_squared_error(y_hat2, y_raw)
        _r2 = r2_score(y_hat2, y_raw)

        return {
            'loss': _loss,
            'obs': y_raw,
            'sim': y_hat2,
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
        for metric in ['MAPE', 'PCORR', 'SCORR', 'R2', 'FB', 'NMSE', 'MG', 'VG', 'FAC2']:
            plot_metrics(metric, self.output_size, df_obs, df_sim,
                         self.data_dir, self.png_dir, self.svg_dir)
        plot_rmse(self.output_size, df_obs, df_sim,
                  self.data_dir, self.png_dir, self.svg_dir)
        plot_logs(self.train_logs, self.valid_logs, self.target,
                  self.data_dir, self.png_dir, self.svg_dir)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().item()
        tensorboard_logs = {'test/loss': avg_loss}
        for name in self.metrics:
            tensorboard_logs['test/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
        tensorboard_logs['step'] = self.current_epoch

        self.log('test/MSE', tensorboard_logs['test/MSE'].item(), on_epoch=True, logger=self.logger)
        self.log('test/MAE', tensorboard_logs['test/MAE'].item(), on_epoch=True, logger=self.logger)
        self.log('test/loss', avg_loss, on_epoch=True, logger=self.logger)

        self.df_obs = df_obs
        self.df_sim = df_sim

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
        xs, xs_1d, _, _, _, \
            ys, ys_raw, _, _, _, dates = zip(*batch)

        return torch.as_tensor(xs), \
            torch.as_tensor(xs_1d), \
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

    p = figure(title=title)
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.axis_label = "time"
    if ylabel:
        p.yaxis.axis_label = ylabel
    if metric == 'MAPE':
        p.yaxis.bounds = (0.0, 1.0)
        p.y_range = Range1d(0.0, 1.0)
    elif metric == 'R2' or metric == 'PCORR' or metric == 'SCORR':
        ymin = min(0.0, min(metric_vals))
        p.yaxis.bounds = (ymin, 1.0)
        p.y_range = Range1d(ymin, 1.0)
    p.line(times, metric_vals)
    export_png(p, filename=png_path)
    p.output_backend = "svg"
    export_svgs(p, filename=str(svg_path))

    df_corrs = pd.DataFrame({'time': times, metric.lower(): metric_vals})
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


class MCCRLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        # save sigma
        assert sigma > 0
        self.sigma2 = sigma**2

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Implement maximum correntropy criterion for regression

        loss(y, t) = sigma^2 * (1.0 - exp(-(y-t)^2/sigma^2))

        where sigma > 0 (parameter)

        Reference:
            * Feng, Yunlong, et al. "Learning with the maximum correntropy criterion induced losses for regression." J. Mach. Learn. Res. 16.1 (2015): 993-1034.
        """
        # not to compute log(0), add 1e-24 (small value)
        def _mccr(x):
            return self.sigma2 * (1-torch.exp(-x**2 / self.sigma2))

        return torch.mean(_mccr(input - target))


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


def relu_mul(x):
    """[fastest method](https://stackoverflow.com/a/32109519/743078)
    """
    return x * (x > 0)
