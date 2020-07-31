from argparse import Namespace
import copy
import datetime as dt
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

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png

import data
from constants import SEOUL_STATIONS, SEOULTZ

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ml_rnn_uni_attention(station_name="종로구"):
    print("Start Univariate Seq2Seq Model")
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

        output_dir = Path("/mnt/data/RNNAttentionUnivariate/" +
                          station_name + "/" + target + "/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)

        if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            # load imputed result
            _df_h = data.load_imputed(HOURLY_DATA_PATH)
            df_h = _df_h.query('stationCode == "' +
                               str(SEOUL_STATIONS[station_name]) + '"')
            df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

        if target == 'PM10':
            hparams = Namespace(
                seq_size=sample_size,
                input_size=1,
                hidden_size_enc=32,
                hidden_size_dec=32,
                output_size=output_size,
                learning_rate=learning_rate,
                sample_size=sample_size,
                batch_size=batch_size)
            model = BaseAttentionModel(hparams=hparams,
                                     station_name=station_name,
                                     target=target,
                                     features=["PM10"],
                                     train_fdate=train_fdate, train_tdate=train_tdate,
                                     test_fdate=test_fdate, test_tdate=test_tdate,
                                     output_dir=output_dir)
        elif target == 'PM25':
            hparams = Namespace(
                seq_size=sample_size,
                input_size=1,
                hidden_size_enc=32,
                hidden_size_dec=32,
                output_size=output_size,
                learning_rate=learning_rate,
                sample_size=sample_size,
                batch_size=batch_size)
            model = BaseAttentionModel(hparams=hparams,
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
                          #fast_dev_run=True,
                          logger=model.logger,
                          row_log_interval=10)

        trainer.fit(model)

        # run test set
        trainer.test()


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
        hidden_size_enc
        hidden_size_dec

    Reference:
        * https://github.com/bentrevett/pytorch-seq2seq
    """
    def __init__(self, hidden_size_enc, hidden_size_dec):
        super().__init__()
        # input: previous hidden state of decoder +
        #   hidden state of encoder
        # output: attention vector; length of source sentence
        self.attn = nn.Linear(hidden_size_enc +
                              hidden_size_dec, hidden_size_dec)
        self.v = nn.Linear(hidden_size_dec, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            hidden: previous hidden state of the decoder [batch_size, hidden_size_dec]
            encoder_outputs: outputs of encoder  [batch_size, sample_size, num_directions*hidden_size]
        """
        # hidden: [num_layers * num_direction, batch_size, hidden_size]
        # encoder_outputs = [batch_size, sample_size, num_directions*hidden_size_enc]
        # encoder_outputs: tensor containing the output features h_t from the 'last layer' of the GRU
        batch_size = encoder_outputs.shape[0]
        sample_size = encoder_outputs.shape[1]

        # repeat decoder hidden state to sample_size times to compute energy with encoder_outputs
        # hidden = [num_layers * num_direction, batch_size, hidden_size]
        # ignore num_layers and num_direction (already 1)
        hidden = hidden.squeeze(0)

        # hidden = [batch size, sample_size, hidden_size_dec]
        hidden = hidden.unsqueeze(1).repeat(1, sample_size, 1)

        # encoder_outputs = [batch size, sample_size, num_directions*hidden_size_enc]
        # energy: [batch size, sample_size, hidden_size_dec]
        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, sample_size, hidden_size_dec]
        attention = self.v(energy).squeeze(2)

        # attention = [batch size, sample_size]
        return F.softmax(attention, dim=1)


class DecoderRNN(nn.Module):
    """
        Reference: https://github.com/bentrevett/pytorch-seq2seq
    """

    def __init__(self, output_size, hidden_size_enc, hidden_size_dec, attention):
        super().__init__()
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        # decoder's initial hidden state is encoder's last hidden state
        assert hidden_size_enc == hidden_size_dec
        self.output_size = output_size
        self.attention = attention

        # no embedding -> GRU input size is 1
        self.gru = nn.GRU(hidden_size_enc + 1, hidden_size_dec)
        self.out = nn.Linear(hidden_size_enc + 1 + hidden_size_dec, 1)

    def forward(self, _input, hidden, encoder_outputs):
        """
        Args:
            _input: [L, N, H_in] : (seq_len, batch_size, hidden_size)
            hidden: previous hidden state of the decoder [batch_size, hidden_size_dec]
            encoder_outputs: outputs of encoder  [batch_size, sample_size, num_directions*hidden_size_enc]
        
        https://github.com/bentrevett/pytorch-seq2seq
        Decoder : hidden state + context vector as input
        """
        # first input is output of key_date (where output starts)
        # so actual output size is output_size + 1,
        # we just truncated to output_size

        # no embeding
        # [batch_size, hidden_size_enc] -> [1, batch_size, hidden_size_enc]
        _input = _input.unsqueeze(0)

        # [batch size, sample_size]
        a = self.attention(hidden, encoder_outputs)
        
        # [batch size, 1, sample_size]
        a = a.unsqueeze(1)

        # encoder_outputs: [batch size, sample_size, hidden_size_enc]
        #encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # batch matrix multiplication for score function
        # a = [batch size, 1, sample_size]
        # encoder_outputs = [batch size, sample_size, hidden_size_enc]
        # matrix multiplication -> 1xsample_size * sample_sizexhidden_size_enc
        # weighted: [batch_size, 1, hidden_size_enc]
        weighted = torch.bmm(a, encoder_outputs)

        # weighted: [1, batch size, hidden_size_enc]
        weighted = weighted.permute(1, 0, 2)

        # _input: [1, batch_size, hidden_size_enc]
        # weighted: [1, batch size, hidden_size_enc]
        # rnn_input: [1, batch_size, hidden_size_enc + hidden_size_dec]
        rnn_input = torch.cat((_input, weighted), dim = 2)

        # do rnn with hidden state (initial hidden state = hidden state from encoder)
        # rnn_input: [1, batch_size, hidden_size_enc + hidden_size_dec]
        # output: [1, batch_size, hidden_size_dec]
        # hidden: [1, batch_size, hidden_size_dec]
        output, hidden = self.gru(rnn_input, hidden)
        
        assert (output == hidden).all()

        # prediction: [batch size, 1]
        prediction = self.out(
            torch.cat((output.squeeze(0), weighted.squeeze(0), _input.squeeze(0)), dim=1))

        # current hidden state is a input of next hidden state
        return prediction, hidden


class BaseAttentionModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.hparams = kwargs.get('hparams', Namespace(
            seq_size=48,
            input_size=1,
            hidden_size_enc=16,
            hidden_size_dec=32,
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
            'output_dir', Path('/mnt/data/DNNUnivariate/'))
        self.log_dir = kwargs.get('log_dir', self.output_dir / Path('log'))
        Path.mkdir(self.log_dir, parents=True, exist_ok=True)
        self.plot_dir = kwargs.get(
            'plot_dir', self.output_dir / Path('png/'))
        Path.mkdir(self.plot_dir, parents=True, exist_ok=True)
        self.data_dir = kwargs.get(
            'data_dir', self.output_dir / Path('csv/'))
        Path.mkdir(self.data_dir, parents=True, exist_ok=True)

        self.loss = nn.MSELoss(reduction='mean')

        self._train_set = None
        self._valid_set = None
        self._test_set = None

        log_name = self.target + "_" + dt.date.today().strftime("%y%m%d-%H-%M")
        self.logger = TensorBoardLogger(self.log_dir, name=log_name)

        self.encoder = EncoderRNN(
            self.hparams.input_size, self.hparams.hidden_size_enc)
        self.attention = Attention(
            self.hparams.hidden_size_enc, self.hparams.hidden_size_dec)
        self.decoder = DecoderRNN(
            self.hparams.output_size, self.hparams.hidden_size_enc, self.hparams.hidden_size_dec, self.attention)

    def forward(self, x, y0, y):
        # x  : [batch_size, sample_size]
        # y0 : [batch_size, 1]
        # y  : [batch_size, output_size]

        # _x : [batch_size, sample_size, 1]
        batch_size = x.shape[0]
        _x = x.unsqueeze(2)
        # last hidden state of the encoder is the context
        # hidden : [num_layers * num_direction, batch_size, hidden_size_enc]
        encoder_outputs, hidden = self.encoder(_x)

        # first input to the decoder is the first output of y
        _input = y0

        # x  : [batch_size, sample_size, 1]
        # y0 : [batch_size, 1]
        # y  : [batch_size, output_size]
        outputs = torch.zeros(batch_size, self.hparams.output_size).to(device)

        # iterate sequence
        # x[ei] : single value of PM10 or PM25
        for t in range(self.hparams.output_size + 1):
            output, hidden = self.decoder(_input, hidden, encoder_outputs)

            if t > 0:
                # skip initial result
                outputs[:, t - 1] = output[:, 0]

            _input = output

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y0, y, dates = batch
        y_hat = self(x, y0, y)
        _loss = self.loss(y_hat, y)

        _y = y.detach().cpu().clone().numpy()
        _y_hat = y_hat.detach().cpu().clone().numpy()
        _mae = mean_absolute_error(_y, _y_hat)
        _mse = mean_squared_error(_y, _y_hat)
        _r2 = r2_score(_y, _y_hat)

        return {
            'loss': _loss,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'R2': _r2
            }
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'train/loss': avg_loss}
        for name in self.metrics:
            tensorboard_logs['train/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
        tensorboard_logs['step'] = self.current_epoch

        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y0, y, dates = batch
        y_hat = self(x, y0, y)

        _loss = self.loss(y, y_hat)
        _y = y.detach().cpu().clone().numpy()
        _y_hat = y_hat.detach().cpu().clone().numpy()
        _mae = mean_absolute_error(_y, _y_hat)
        _mse = mean_squared_error(_y, _y_hat)
        _r2 = r2_score(_y, _y_hat)

        return {
            'loss': _loss,
            'metric': {
                'MSE': _mse,
                'MAE': _mae,
                'R2': _r2
            }
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'valid/loss': avg_loss}
        for name in self.metrics:
            tensorboard_logs['valid/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
        tensorboard_logs['step'] = self.current_epoch

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y0, y, dates = batch
        y_hat = self(x, y0, y)

        _loss = self.loss(y, y_hat)
        _y = y.detach().cpu().clone().numpy()
        _y_hat = y_hat.detach().cpu().clone().numpy()
        _mae = mean_absolute_error(_y, _y_hat)
        _mse = mean_squared_error(_y, _y_hat)
        _r2 = r2_score(_y, _y_hat)

        return {
            'loss': _loss,
            'obs': y,
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
                  self.data_dir, self.plot_dir)
        plot_scatter(self.hparams, df_obs, df_sim,
                     self.data_dir, self.plot_dir)
        plot_corr(self.hparams, df_obs, df_sim,
                  self.data_dir, self.plot_dir)

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
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
        train_valid_set = data.UnivariateRNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            train_valid_ratio=0.8)
        test_set = data.UnivariateRNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.test_fdate,
            tdate=self.test_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size)

        # save train/valid set
        train_valid_set.to_csv(
            self.data_dir / ("df_train_valid_set_" + self.target + ".csv"))
        test_set.to_csv(self.data_dir / ("df_testset_" + self.target + ".csv"))

        # split train/valid/test set
        train_len = int(len(train_valid_set) *
                        train_valid_set.train_valid_ratio)
        valid_len = len(train_valid_set) - train_len
        train_set, valid_set = torch.utils.data.random_split(
            train_valid_set, [train_len, valid_len])

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
        xs, ys0, ys, dates = zip(*batch)

        # x : [batch_size x sample_size] -> [sample_size x batch_size]
        # y : [batch_size x output_size] -> [output_size x batch_size]
        return torch.as_tensor(xs), torch.as_tensor(ys0), torch.as_tensor(ys), dates


def plot_line(hparams, df_obs, df_sim, target, data_dir, plot_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(plot_dir, parents=True, exist_ok=True)

    for t in range(hparams.output_size):
        dates = df_obs.index + dt.timedelta(hours=t)

        plot_dir_h = plot_dir / str(t).zfill(2)
        Path.mkdir(plot_dir_h, parents=True, exist_ok=True)
        plt_path = plot_dir_h / ("line_" + str(t).zfill(2) + "h.png")

        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        p = figure(title="OBS & Model")
        p.toolbar.logo = None
        p.toolbar_location = None
        p.xaxis.axis_label = "dates"
        p.xaxis.formatter = DatetimeTickFormatter()
        p.line(dates, obs, line_color="dodgerblue", legend_label="obs")
        p.line(dates, sim, line_color="lightcoral", legend_label="sim")
        export_png(p, filename=plt_path)

        data_dir_h = data_dir / str(t).zfill(2)
        Path.mkdir(data_dir_h, parents=True, exist_ok=True)
        csv_path = data_dir_h / ("line_" + str(t).zfill(2) + "h.csv")
        df_line = pd.DataFrame.from_dict(
            {'date': dates, 'obs': obs, 'sim': sim})
        df_line.set_index('date', inplace=True)
        df_line.to_csv(csv_path)


def plot_scatter(hparams, df_obs, df_sim, data_dir, plot_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(plot_dir, parents=True, exist_ok=True)

    for t in range(hparams.output_size):
        plot_dir_h = plot_dir / str(t).zfill(2)
        Path.mkdir(plot_dir_h, parents=True, exist_ok=True)
        plt_path = plot_dir_h / ("scatter_" + str(t).zfill(2) + "h.png")

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
        export_png(p, filename=plt_path)

        data_dir_h = data_dir / str(t).zfill(2)
        Path.mkdir(data_dir_h, parents=True, exist_ok=True)
        csv_path = data_dir_h / ("scatter_" + str(t).zfill(2) + "h.csv")
        df_scatter = pd.DataFrame({'obs': obs, 'sim': sim})
        df_scatter.to_csv(csv_path)


def plot_corr(hparams, df_obs, df_sim, data_dir, plot_dir):
    Path.mkdir(data_dir, parents=True, exist_ok=True)
    Path.mkdir(plot_dir, parents=True, exist_ok=True)

    plt_path = plot_dir / ("corr_time.png")
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
    export_png(p, filename=plt_path)

    df_corrs = pd.DataFrame({'time': times, 'corr': corrs})
    df_corrs.set_index('time', inplace=True)
    df_corrs.to_csv(csv_path)


def swish(_input, beta=1.0):
    """
        Swish function in [this paper](https://arxiv.org/pdf/1710.05941.pdf)

    Args:
        input: Tensor

    Returns:
        output: Activated tensor
    """
    return _input * beta * nn.Sigmoid(_input)