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
    transform=True):
    if scaler_X == None or scaler_Y == None:
        data_set = data.UnivariateRNNMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=filepath,
            features=[target],
            fdate=fdate,
            tdate=tdate,
            sample_size=sample_size,
            output_size=output_size)
    else:
        data_set = data.UnivariateRNNMeanSeasonalityDataset(
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

def ml_rnn_uni_attn(station_name="종로구"):
    print("Start Univariate Attention Model")
    targets = ["PM10", "PM25"]
    # 24*14 = 336
    #sample_size = 336
    sample_size = 48
    output_size = 24
    # If you want to debug, fast_dev_run = True and n_trials should be small number
    fast_dev_run = False
    n_trials = 16
    # fast_dev_run = True
    # n_trials = 3

    # Hyper parameter
    epoch_size = 500
    batch_size = 64
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
    # train_valid_fdate = dt.datetime(2013, 1, 1, 1).astimezone(SEOULTZ)
    # train_valid_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)

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

    for target in targets:
        print("Training " + target + "...", flush=True)
        output_dir = Path(f"/mnt/data/RNNAttentionUnivariate/{station_name}/{target}/")
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
                                                sample_size=sample_size, output_size=output_size, transform=True) for td in train_dates)

        print("Construct Validation Sets...", flush=True)
        valid_datasets = tuple(construct_dataset(vd[0], vd[1],
                                                scaler_X=train_valid_dataset.scaler_X,
                                                scaler_Y=train_valid_dataset.scaler_Y,
                                                filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                                sample_size=sample_size, output_size=output_size, transform=True) for vd in valid_dates)

        # just single test set
        print("Construct Test Sets...", flush=True)
        test_dataset = construct_dataset(test_fdate, test_tdate,
                                        scaler_X=train_valid_dataset.scaler_X,
                                        scaler_Y=train_valid_dataset.scaler_Y,
                                        filepath=HOURLY_DATA_PATH, station_name=station_name, target=target,
                                        sample_size=sample_size, output_size=output_size, transform=True)

        # convert tuple of datasets to ConcatDataset
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(valid_datasets)

        # Dummy hyperparameters
        hparams = Namespace(
            sigma=1.0,
            hidden_size=32,
            learning_rate=learning_rate,
            batch_size=batch_size)

        def objective(trial):
            model = BaseAttentionModel(trial=trial,
                                       hparams=hparams,
                                       input_size=1,
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
                              default_root_dir=output_dir,
                              fast_dev_run=fast_dev_run,
                              logger=False,
                              checkpoint_callback=False,
                              callbacks=[PyTorchLightningPruningCallback(
                                    trial, monitor="val_loss")])

            trainer.fit(model)

            # Don't Log
            # hyperparameters = model.hparams
            # trainer.logger.log_hyperparams(hyperparameters)

            return trainer.callback_metrics["valid/MSE"].item()

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
                study, params=['hidden_size'])
            fig_pcoord.write_image(str(output_dir / "parallel_coord.png"))
            fig_pcoord.write_image(str(output_dir / "parallel_coord.svg"))

            fig_slice = optv.plot_slice(
                study, params=['hidden_size'])
            fig_slice.write_image(str(output_dir / "slice.png"))
            fig_slice.write_image(str(output_dir / "slice.svg"))

            # set hparams with optmized value
            hparams.sigma = trial.params['sigma']
            hparams.hidden_size = trial.params['hidden_size']

            dict_hparams = copy.copy(vars(hparams))
            dict_hparams["sample_size"] = sample_size
            dict_hparams["output_size"] = output_size
            with open(output_dir / 'hparams.json', 'w') as f:
                print(dict_hparams, file=f)
            with open(output_dir / 'hparams.csv', 'w') as f:
                print(pd.DataFrame.from_dict(dict_hparams, orient='index'), file=f)

        model = BaseAttentionModel(hparams=hparams,
                                   input_size=1,
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

        log_version = dt.date.today().strftime("%y%m%d-%H-%M")
        loggers = [ \
            TensorBoardLogger(log_dir, version=log_version),
            CSVLogger(log_dir, version=log_version)]

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


class EncoderRNN(nn.Module):
    """Encoder for Attention model

    Args:
        input_size : The number of expected features in the input x
        hidden_size : The number of features in the hidden state h

    Reference:
        * Neural Machine Translation by Jointly Learning to Align and Translate, Bahdanau et al.
        * https://github.com/bentrevett/pytorch-seq2seq
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # usually input_size == 1
        # hidden_size is given at hparams

        # GRU input 1 : (L, N, H_in)
        #       where L : seq_len, N : batch_size, H_in : input_size (usually 1), the number of input feature
        #       however, batch_first=True, (N, L, H_in)
        # GRU output 1 : (L, N, H_all) where H_all = num_directions (== 1 now) * hidden_size
        # GRU output 2 : (S, N, H_out) wehre S : num_layers * num_directions, H_out : hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, _input):
        """Encode _input

        Args:
            _input: [batch_size, sample_size, input_size]

        Returns:
            outputs: [batch_size, sample_size, hidden]
            hidden: [batch_size, num_layers * num_direction, hidden_size]
        """
        # no embedding
        # _input: [batch_size, sample_size, input_size]
        outputs, _hidden = self.gru(_input)

        # create context vector for attention by simple linear layer
        # _hidden: [batch_size, num_layers * num_direction, hidden_size]
        # hidden: [batch_size, num_layers * num_direction, hidden_size]
        hidden = torch.tanh(self.fc(_hidden))

        # outputs: [batch_size, sample_size, num_directions*hidden_size]
        # hidden: [num_layers * num_direction, batch_size, hidden_size]
        return outputs, hidden


class Attention(nn.Module):
    """ Attention Layer merging Encoder and Decoder


    Args:
        hidden_size
        hidden_size

    Reference:
        * https://github.com/bentrevett/pytorch-seq2seq
    """
    def __init__(self, hidden_size):
        super().__init__()
        # input: previous hidden state of decoder +
        #   hidden state of encoder
        # output: attention vector; length of source sentence
        self.attn = nn.Linear(hidden_size +
                              hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: previous hidden state of the decoder [batch_size, hidden_size]
            encoder_outputs: outputs of encoder  [batch_size, sample_size, num_directions*hidden_size]
        """
        # hidden: [num_layers * num_direction, batch_size, hidden_size]
        # encoder_outputs = [batch_size, sample_size, num_directions*hidden_size]
        # encoder_outputs: tensor containing the output features h_t from the 'last layer' of the GRU
        batch_size = encoder_outputs.shape[0]
        sample_size = encoder_outputs.shape[1]

        # repeat decoder hidden state to sample_size times to compute energy with encoder_outputs
        # hidden = [num_layers * num_direction, batch_size, hidden_size]
        # ignore num_layers and num_direction (already 1)
        hidden = hidden.squeeze(0)

        # hidden = [batch size, sample_size, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, sample_size, 1)

        # encoder_outputs = [batch size, sample_size, num_directions*hidden_size]
        # energy: [batch size, sample_size, hidden_size]
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, sample_size, hidden_size]
        attention = self.v(energy).squeeze(2)

        # attention = [batch size, sample_size]
        return F.softmax(attention, dim=1)


class DecoderRNN(nn.Module):
    """
        Reference: https://github.com/bentrevett/pytorch-seq2seq
    """

    def __init__(self, output_size, hidden_size, attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = attention

        # no embedding -> GRU input size is 1
        self.gru = nn.GRU(hidden_size + 1, hidden_size)
        self.out = nn.Linear(hidden_size + 1 + hidden_size, 1)

    def forward(self, _input, hidden, encoder_outputs):
        """
        Args:
            _input: [L, N, H_in] : (seq_len, batch_size, hidden_size)
            hidden: previous hidden state of the decoder [batch_size, hidden_size]
            encoder_outputs: outputs of encoder  [batch_size, sample_size, num_directions*hidden_size]

        https://github.com/bentrevett/pytorch-seq2seq
        Decoder : hidden state + context vector as input
        """
        # first input is output of key_date (where output starts)
        # so actual output size is output_size + 1,
        # we just truncated to output_size

        # no embeding
        # [batch_size, hidden_size] -> [1, batch_size, hidden_size]
        _input = _input.unsqueeze(0)

        # [batch size, sample_size]
        a = self.attention(hidden, encoder_outputs)

        # [batch size, 1, sample_size]
        a = a.unsqueeze(1)

        # encoder_outputs: [batch size, sample_size, hidden_size]
        #encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # batch matrix multiplication for score function
        # a = [batch size, 1, sample_size]
        # encoder_outputs = [batch size, sample_size, hidden_size]
        # matrix multiplication -> 1xsample_size * sample_sizexhidden_size
        # weighted: [batch_size, 1, hidden_size]
        weighted = torch.bmm(a, encoder_outputs)

        # weighted: [1, batch size, hidden_size]
        weighted = weighted.permute(1, 0, 2)

        # _input: [1, batch_size, hidden_size]
        # weighted: [1, batch size, hidden_size]
        # rnn_input: [1, batch_size, hidden_size + hidden_size]
        rnn_input = torch.cat((_input, weighted), dim = 2)

        # do rnn with hidden state (initial hidden state = hidden state from encoder)
        # rnn_input: [1, batch_size, hidden_size + hidden_size]
        # output: [1, batch_size, hidden_size]
        # hidden: [1, batch_size, hidden_size]
        output, hidden = self.gru(rnn_input, hidden)

        assert (output == hidden).all()

        # prediction: [batch size, 1]
        # swish
        #cat_vec = torch.cat((output.squeeze(0), weighted.squeeze(0), _input.squeeze(0)), dim=1)
        #prediction = self.out(swish(cat_vec))

        # sigmoid
        # cat_vec = torch.cat((output.squeeze(0), weighted.squeeze(0), _input.squeeze(0)), dim=1)
        # prediction = self.out(torch.sigmoid(cat_vec))

        # no activation
        cat_vec = torch.cat((output.squeeze(0), weighted.squeeze(0), _input.squeeze(0)), dim=1)
        prediction = self.out(cat_vec)

        # current hidden state is a input of next hidden state
        return prediction, hidden


class BaseAttentionModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams = kwargs.get('hparams', Namespace(
            sigma=1.0,
            hidden_size=16,
            learning_rate=1e-3,
            batch_size=32
        ))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features', [self.target])
        self.metrics = kwargs.get('metrics', ['MAE', 'MSE', 'R2'])
        self.train_fdate = kwargs.get('train_fdate', dt.datetime(
            2012, 1, 1, 0).astimezone(SEOULTZ))
        self.train_tdate = kwargs.get('train_tdate', dt.datetime(
            2016, 12, 31, 23).astimezone(SEOULTZ))
        self.valid_fdate = kwargs.get('valid_fdate', dt.datetime(
            2017, 1, 1, 0).astimezone(SEOULTZ))
        self.valid_tdate = kwargs.get('valid_tdate', dt.datetime(
            2018, 12, 31, 23).astimezone(SEOULTZ))
        self.test_fdate = kwargs.get('test_fdate', dt.datetime(
            2019, 1, 1, 0).astimezone(SEOULTZ))
        self.test_tdate = kwargs.get('test_tdate', dt.datetime(
            2020, 10, 31, 23).astimezone(SEOULTZ))
        self.num_workers = kwargs.get('num_workers', 1)
        self.output_dir = kwargs.get(
            'output_dir', Path('/mnt/data/RNNAttnUnivariate/'))
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

        self.trial = kwargs.get('trial', None)
        self.sample_size = kwargs.get('sample_size', 48)
        self.output_size = kwargs.get('output_size', 24)
        self.input_size = kwargs.get(
            'input_size', self.sample_size)

        if self.trial:
            self.hparams.sigma = self.trial.suggest_float(
                "sigma", 0.8, 1.5, step=0.05)
            self.hparams.hidden_size = self.trial.suggest_int(
                "hidden_size", 4, 64)

        self.encoder = EncoderRNN(
            self.input_size, self.hparams.hidden_size)
        self.attention = Attention(
            self.hparams.hidden_size)
        self.decoder = DecoderRNN(
            self.output_size, self.hparams.hidden_size, self.attention)

        # self.loss = nn.MSELoss()
        self.loss = MCCRLoss(sigma=self.hparams.sigma)
        # self.loss = nn.L1Loss()

        self.train_logs = {}
        self.valid_logs = {}

        self.df_obs = pd.DataFrame()
        self.df_sim = pd.DataFrame()

    def forward(self, x, y0, y):
        # x  : [batch_size, sample_size]
        # y0 : [batch_size, 1]
        # y  : [batch_size, output_size]

        # _x : [batch_size, sample_size, 1]
        batch_size = x.shape[0]
        _x = x.unsqueeze(2)
        # last hidden state of the encoder is the context
        # hidden : [num_layers * num_direction, batch_size, hidden_size]
        encoder_outputs, hidden = self.encoder(_x)

        # first input to the decoder is the first output of y
        _input = y0

        # x  : [batch_size, sample_size, 1]
        # y0 : [batch_size, 1]
        # y  : [batch_size, output_size]
        outputs = torch.zeros(batch_size, self.output_size).to(device)

        # iterate sequence
        # x[ei] : single value of PM10 or PM25
        for t in range(self.output_size + 1):
            output, hidden = self.decoder(_input, hidden, encoder_outputs)

            if t > 0:
                # skip initial result
                outputs[:, t - 1] = output[:, 0]

            _input = output

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01)

    def training_step(self, batch, batch_idx):
        # x, _y0, _y, dates = batch
        x, _y, _y0, _y_raw, y_dates = batch
        _y_hat = self(x, _y0, _y)
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

        self.log('train/loss', tensorboard_logs['train/loss'], prog_bar=True)
        self.log('train/MSE', tensorboard_logs['train/MSE'], on_epoch=True, logger=self.logger)
        self.log('train/MAE', tensorboard_logs['train/MAE'], on_epoch=True, logger=self.logger)
        self.log('train/avg_loss', _log['loss'], on_epoch=True, logger=self.logger)

    def validation_step(self, batch, batch_idx):
        # x, _y0, _y, dates = batch
        x, _y, _y0, _y_raw, y_dates = batch
        _y_hat = self(x, _y0, _y)
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

        self.log('valid/MSE', tensorboard_logs['valid/MSE'], on_epoch=True, logger=self.logger)
        self.log('valid/MAE', tensorboard_logs['valid/MAE'], on_epoch=True, logger=self.logger)
        self.log('valid/loss', _log['loss'], on_epoch=True, logger=self.logger)

    def test_step(self, batch, batch_idx):
        # x, _y0, _y, dates = batch
        x, _y, _y0, _y_raw, y_dates = batch
        _y_hat = self(x, _y0, _y)
        _loss = self.loss(_y_hat, _y)

        y = _y.detach().cpu().clone().numpy()
        y_raw = _y_raw.detach().cpu().clone().numpy()
        y_hat = _y_hat.detach().cpu().clone().numpy()
        y_hat_inv = np.array(self.test_dataset.inverse_transform(y_hat, y_dates))

        _mae = mean_absolute_error(y, y_hat)
        _mse = mean_squared_error(y, y_hat)
        _r2 = r2_score(y, y_hat)

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
        for metric in ['MAPE', 'PCORR', 'SCORR', 'R2', 'FB', 'NMSE', 'MG', 'VG', 'FAC2']:
            plot_metrics(metric, self.output_size, df_obs, df_sim,
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
        avg_loss = float(avg_loss.detach().cpu())

        self.log('test/MSE', tensorboard_logs['test/MSE'], on_epoch=True, logger=self.logger)
        self.log('test/MAE', tensorboard_logs['test/MAE'], on_epoch=True, logger=self.logger)
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
        """ dataloader for train_dataset (ConcatDataset)
        * TODO: Type Checking for ConcatDataset
        """
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        """ dataloader for val_dataset (ConcatDataset)
        * TODO: Type Checking for ConcatDataset
        """
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        """ dataloader for test_dataset (Dataset)
        """
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        """Creates mini-batch tensors from from list of tuples (x, y, dates)

        dates will not be trained but need to construct output, so don't put dates into Tensors
        Args:
            batch: list of tuple  (x, y, y_raw, date).
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
        xs, ys, ys0, ys_raw, y_dates = zip(*batch)

        return torch.as_tensor(xs), \
               torch.as_tensor(ys), \
               torch.as_tensor(ys0), \
               torch.as_tensor(ys_raw), \
               y_dates

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
