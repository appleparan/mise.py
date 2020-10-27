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

def ml_rnn_mul_tpa_attn(station_name="종로구"):
    print("Start Multivariate Temporal Pattern Attention(TPA) Model")
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

        output_dir = Path("/mnt/data/RNNTPAAttnMultivariate/" +
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
                num_filters=32,
                hidden_size=16,
                # should be (smaple_size, filter_size)
                kernel_size=(sample_size, 5),
                output_size=output_size,
                learning_rate=learning_rate,
                sample_size=sample_size,
                batch_size=batch_size)
            model = BaseTPAAttnModel(hparams=hparams,
                                     station_name=station_name,
                                     target=target,
                                     features=train_features,
                                     train_fdate=train_fdate, train_tdate=train_tdate,
                                     test_fdate=test_fdate, test_tdate=test_tdate,
                                     output_dir=output_dir)
        elif target == 'PM25':
            hparams = Namespace(
                seq_size=sample_size,
                input_size=1,
                num_filters=32,
                hidden_size=16,
                # should be (smaple_size, filter_size)
                kernel_size=(sample_size, 5),
                output_size=output_size,
                learning_rate=learning_rate,
                sample_size=sample_size,
                batch_size=batch_size)
            model = BaseTPAAttnModel(hparams=hparams,
                                     station_name=station_name,
                                     target=target,
                                     features=train_features,
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
                          default_root_dir=output_dir,
                          #fast_dev_run=True,
                          logger=model.logger,
                          row_log_interval=10)

        trainer.fit(model)

        # run test set
        trainer.test()


class EncoderRNN(nn.Module):
    """
    Encoder, but not same as Seq2Seq's
    Map data to hidden represeentation.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, _input):
        """

        Args:
            _input: tensor containing the features of the input sequence.
                input_size == feature_size
                [batch_size, sample_size, input_size]

        Return:
            outputs: tensor containing the output features h_t from the last layer of the GRU
                [sample_size, batch_size, num_directions*hidden_size]
            hidden: hidden state of last sequence (when t = batch_size)
                [num_layers * num_direction, batch_size, hidden_size]
        """
        # no embedding
        # _input: [batch_size, sample_size, input_size]
        outputs, hidden = self.gru(_input)

        # outputs: [sample_size, batch_size, num_directions*hidden_size]
        # hidden: [num_layers * num_direction, batch_size, hidden_size]
        return outputs, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Attention(nn.Module):
    """ Attention Layer (Luong Attention)
    Use Global Attention with general score function

    Args:
        hidden_size

    Reference:
        * http://www.phontron.com/class/nn4nlp2017/assets/slides/nn4nlp-09-attention.pdf
    """

    def __init__(self, hidden_size, num_filters, filter_size):
        super().__init__()
        # input: previous hidden state of decoder +
        #   hidden state of encoder
        # output: attention vector; length of source sentence
        # RNN hidden size, hidRNN
        self.hidden_size = hidden_size
        # number of CNN filter, num_filters
        self.num_filters = num_filters
        # filter_size, same as kernel_size[1]
        self.filter_size = filter_size
        # attn_feature_size for attention layer, because there is no padding
        self.attn_feature_size = hidden_size - filter_size + 1
        # must be positive
        assert self.attn_feature_size > 0

        self.attn = nn.Linear(hidden_size, num_filters)

    def forward(self, query, values):
        """
        Args:
            query: single hidden state for querying its attention weights,
                usually last hidden state
                [num_layer*num_direction, batch_size, hidden_size
            values:
                convolution result of hidden states
                [batch_size, num_filters, num_directions*attn_feature_size]

        Returns
            context: context vector for each sequence [batch_size, hidden_size]
            attention_weights: attention weight for save [batch size, sample_size]
        """
        # query : [num_layer*num_direction, batch_size, hidden_size]
        # in our model, num_directions always 1
        # values : [batch_size, num_filters, attn_feature_size]
        batch_size = values.shape[0]
        num_filters = values.shape[1]

        # query: (1, batch_size, num_filters)
        # attention_weights: (batch_size, attn_feature_size)
        attention_weights = self.attn(query.squeeze(0))

        # a: (batch_size, num_filters, self.attn_feature_size)
        # H^c(values) * W * h_t(query)
        a = attention_weights.unsqueeze(2).repeat(1, 1, self.attn_feature_size)
        # attention score (alpha)
        # score: (batch_size, attn_feature_size)
        score = torch.sigmoid(torch.sum(values * a, 1))

        # apply attention core to convonlution results
        # \sum{\alpha * H_c}
        # context: (batch_size, num_filters)
        context = torch.sum(score.unsqueeze(1).repeat(1, num_filters, 1) * values, dim=2)

        # context : [batch_size, num_filters]
        # attention_weights : [batch size, num_filters]
        return context, attention_weights


class DecoderRNN(nn.Module):
    """
    Decoder, but not same as Seq2Seq's

    input is last hidden state of Encoder

    # Reference
    * https://arxiv.org/abs/1409.3215
    * https://arxiv.org/abs/1406.1078
    * https://github.com/bentrevett/pytorch-seq2seq
    * https://arxiv.org/abs/1502.04681
    """

    def __init__(self, hidden_size, num_filters, output_size, attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.output_size = output_size
        self.attention = attention

        # no embedding -> GRU input size is 1
        self.gru = nn.GRU(hidden_size + num_filters, hidden_size, batch_first=True)
        self.out1 = nn.Linear(num_filters, 1)
        self.out2 = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        """
        Args:
            x: x is a decoder input for single step, so shape must be (batch_size, 1)
            hidden: previous hidden state of the decoder, and doesn't affect by batch_first
                (num_layers * num_directions, batch, hidden_size)
            encoder_outputs: outputs of encoder for context (batch_size, sample_size, num_directions*hidden_size)

        Returns:
            prediction: output for this step and input for next step,
            hidden: hidden state for next step

        # Reference
        * https://github.com/bentrevett/pytorch-seq2seq
        * https://www.tensorflow.org/tutorials/text/nmt_with_attention
        * https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

        """
        # context : [batch_size, num_filters]
        # attention_weights : [batch size, num_filters]
        # context is always composed of new hidden state and same encoder_outputs over sequences
        context, a = self.attention(hidden, encoder_outputs)

        # every output needs ATTENTION!, now I just use feedforward network to combine
        # x: (batch_size, 1)
        # context: (batch_size, num_filters)
        # attention_vector: (batch_size, num_filters)
        # prediction: (batch size, 1)
        prediction = self.out1(context) + self.out2(hidden.squeeze(0))

        # apply rnn(GRU) to attention_vector
        # GRU input must be [batch_size, seq_len, hidden_size], because batch_first=True
        # seq_len for Decoder is just 1
        # attention_vector : (batch_size, hidden_size + num_filters)
        # attention_vector.unsqueeze(1) : [batch_size, 1, hidden_size + num_filters]
        attention_vector = torch.cat((context, hidden.squeeze(0)), dim=1)
        # hidden : [num_layers * num_directions, batch_size, hidden_size]
        output, hidden = self.gru(attention_vector.unsqueeze(1), hidden)

        # prediction: [batch size, 1]
        # current hidden state is a input of next step's hidden state
        # hidden : [num_layers * num_directions, batch_size, hidden_size]
        return prediction, hidden


class BaseTPAAttnModel(LightningModule):
    """
    Temporal Pattern Attention model
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # h_out = (h_in + 2 * padding[0] - dilation[0]*(kernel_size[0] - 1) - 1) / stride[0] + 1
        # to make h_out == h_in, dilation[0] == 1, stride[0] == 1,
        # 2*padding[0] + 1 = kernel_size[0]

        # w_out = (w_in + 2 * padding[1] - dilation[1]*(kernel_size[1] - 1) - 1) / stride[1] + 1
        # to make w_out == w_in, dilation[1] == 1, stride[1] == 1,
        # 2*padding[1] + 1 = kernel_size[1]

        self.hparams = kwargs.get('hparams', Namespace(
            seq_size=72,
            num_filters=16,
            hidden_size=16,
            kernel_size=(1, 1),
            output_size=24,
            learning_rate=1e-3,
            sample_size=48,
            batch_size=32
        ))

        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                         "temp", "u", "v", "pres", "humid", "prep", "snow"]
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
            'output_dir', Path('/mnt/data/RNNTPAAttnMultivariate/'))
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

        self.loss = nn.MSELoss(reduction='mean')

        self._train_set = None
        self._valid_set = None
        self._test_set = None

        log_name = self.target + "_" + dt.date.today().strftime("%y%m%d-%H-%M")
        self.logger = TensorBoardLogger(self.log_dir, name=log_name)

        self.train_logs = {}
        self.valid_logs = {}

        #if self.hparams.num_filters > \
        #    pow(2, self.hparams.kernel_size[0] + self.hparams.kernel_size[1]):
        #    raise ValueError("CNN channel size limit exceed: ", self.hparams.num_filters)

        # padding_size is determined by kernel_size to keep same height and width between input and output
        #if self.hparams.kernel_size[0] % 2 == 0 or self.hparams.kernel_size[1] % 2 == 0:
        #    raise ValueError("kernel_size should be odd number: ", self.hparams.kernel_size)

        # kernel_size = 2*padding_size+1
        # convolution filter to time array
        #padding_size = (int((self.hparams.kernel_size[0] - 1)/2), 0)
        # Time is horizontal axis
        #padding_size = (0, int((self.hparams.kernel_size[1] - 1)/2))

        # no padding
        self.conv = nn.Conv2d(1, self.hparams.num_filters, self.hparams.kernel_size)

        self.encoder = EncoderRNN(
            len(self.features), self.hparams.hidden_size)
        #self.encoder = nn.GRU(len(self.features),
        #                      self.hparams.hidden_size, batch_first=True)
        #self.gru = nn.GRU(self.hparams.hidden_size + 1, self.hparams.hidden_size, batch_first=True)
        self.attention = Attention(
            self.hparams.hidden_size, self.hparams.num_filters, self.hparams.kernel_size[1])
        self.decoder = DecoderRNN(
            self.hparams.hidden_size, self.hparams.num_filters, self.hparams.output_size, self.attention)

    def forward(self, _x, y0, y):
        """
        Args:
            _x  : Input feed to Encoder, shape is (batch_size, sample_size, feature_size)
            y0 : First step output feed to Decoder, shape is (batch_size, 1)
            y  : Output, shape is (batch_size, output_size)
        """
        batch_size = _x.shape[0]
        sample_size = _x.shape[1]
        feature_size = _x.shape[2]

        # first, encode RNN hidden state
        encoder_outputs, encoder_hidden = self.encoder(_x)

        # encoder_outputs : (batch_size, sample_size, feature_size)
        # x_rnn : (batch_size, sample_size, feature_size)
        x_rnn = F.leaky_relu(encoder_outputs)

        # CNN needs shape as NxC_inxHxW
        # H : sample_size
        # W : feature_size
        # x_rnn.unsqueeze(1): (batch_size, 1, sample_size, feature_size)
        # self.conv(x_rnn.unsqueeze(1)): (batch_size, num_filters, 1, feature_size)
        # x_cnn: (batch_size, num_filters, (hidRNN - kernel_size[0] + 1))
        x_cnn = self.conv(x_rnn.unsqueeze(1)).squeeze(2)

        # set input of decoder as last hidden state of encoder
        # batch_first doesn't affect to hidden's size
        hidden = encoder_hidden

        # placeholder for multi-horization
        # x  : [batch_size, sample_size, 1]
        # y0 : [batch_size, 1]
        # y  : [batch_size, output_size]
        outputs = torch.zeros(batch_size, self.hparams.output_size).to(device)

        for t in range(self.hparams.output_size):
            # x_cnn is "Attention"
            output, hidden = self.decoder(hidden, x_cnn)

            outputs[:, t] = output.squeeze(1)

        return outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, x1d, y0, y, dates = batch
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
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        tensorboard_logs = {'train/loss': avg_loss}
        _log = {}
        for name in self.metrics:
            tensorboard_logs['train/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
            _log[name] = float(torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean())
        tensorboard_logs['step'] = self.current_epoch
        _log['loss'] = float(avg_loss)

        self.train_logs[self.current_epoch] = _log

        return {'train_loss': avg_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, x1d, y0, y, dates = batch
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
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu()
        tensorboard_logs = {'valid/loss': avg_loss}
        _log = {}
        for name in self.metrics:
            tensorboard_logs['valid/{}'.format(name)] = torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean()
            _log[name] = float(torch.stack(
                [torch.tensor(x['metric'][name]) for x in outputs]).mean())
        tensorboard_logs['step'] = self.current_epoch
        _log['loss'] = float(avg_loss)

        self.valid_logs[self.current_epoch] = _log

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, x1d, y0, y, dates = batch
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
        train_valid_set = data.MultivariateRNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            train_valid_ratio=0.8)

        # save train/valid set
        train_valid_set.to_csv(
            self.data_dir / ("df_train_valid_set_" + self.target + ".csv"))

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
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size)
        test_set.to_csv(self.data_dir / ("df_testset_" + self.target + ".csv"))

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
        xs, xs_1d, ys0, ys, dates = zip(*batch)

        return torch.as_tensor(xs), torch.as_tensor(xs_1d), \
            torch.as_tensor(ys0), torch.as_tensor(ys), dates


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
