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
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import data
from constants import SEOUL_STATIONS, SEOULTZ

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ml_dnn_msea(station_name="종로구"):
    print("Start DNN model")
    targets = ["PM10", "PM25"]
    sample_size = 48
    output_size = 24

    # Hyper parameter
    epoch_size = 500
    batch_size = 256
    learning_rate = 1e-3

    train_fdate = dt.datetime(2012, 1, 1, 0).astimezone(SEOULTZ)
    train_tdate = dt.datetime(2017, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2018, 1, 1, 0).astimezone(SEOULTZ)
    #test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)

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

        output_dir = Path("/mnt/data/dnn_mean_sea/" + station_name + "/"+ target + "/")
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
            model = BaseDNNModel(hparams=hparams,
                station_name=station_name,
                target=target,
                features=train_features,
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
            model = BaseDNNModel(hparams=hparams,
                station_name=station_name,
                target=target,
                features=train_features,
                train_fdate=train_fdate, train_tdate=train_tdate,
                test_fdate=test_fdate, test_tdate=test_tdate,
                output_dir=output_dir)

        log_name = target + dt.date.today().strftime("%y%m%d-%H-%M")
        log_dir = output_dir / "logs"
        Path.mkdir(log_dir, parents=True, exist_ok=True)
        logger = TensorBoardLogger(log_dir, name=log_name + " DNN")
        # most basic trainer, uses good defaults
        trainer = Trainer(gpus=1,
            precision=32,
            min_epochs=1, max_epochs=epoch_size,
            early_stop_checkpoint=True,
            default_save_path=output_dir,
            logger=logger)

        trainer.fit(model)
                
        # run test set
        trainer.test()

class BaseDNNModel(LightningModule):
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
        self.features = kwargs.get('features',
            ["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp", "u", "v", "pres", "humid", "prep", "snow"])
        self.train_fdate = kwargs.get('train_fdate', dt.datetime(
            2012, 1, 1, 0).astimezone(SEOULTZ))
        self.train_tdate = kwargs.get('train_tdate', dt.datetime(
            2017, 12, 31, 23).astimezone(SEOULTZ))
        self.test_fdate = kwargs.get('test_fdate', dt.datetime(
            2018, 1, 1, 0).astimezone(SEOULTZ))
        self.test_tdate = kwargs.get('test_tdate', dt.datetime(
            2018, 12, 31, 23).astimezone(SEOULTZ))
        self.num_workers = kwargs.get('num_workers', 1)
        self.output_dir = kwargs.get('output_dir', Path('.'))

        self.fc1 = nn.Linear(self.hparams.input_size, self.hparams.layer1_size)
        self.fc2 = nn.Linear(self.hparams.layer1_size, self.hparams.layer2_size)
        self.fc3 = nn.Linear(self.hparams.layer2_size, self.hparams.output_size)
        self.loss = nn.MSELoss(reduction='mean')

        self._train_set = None
        self._valid_set = None
        self._test_set = None

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
        x, y, dates = batch
        y_hat = self(x)
        _loss = self.loss(y_hat, y)
        tensorboard_logs = {'Loss/Train': _loss}
        return {'loss': _loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        x, y, dates = batch
        y_hat = self(x)
        return {'val_loss': self.loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'Loss/Valid': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y, dates = batch
        y_hat = self(x)

        return {
            'test_loss': self.loss(y_hat, y),
            'obs': y,
            'sim': y_hat,
            'dates': dates
        }

    def test_epoch_end(self, out_dicts):
        avg_loss = torch.stack([x['test_loss'] for x in out_dicts]).mean()
        tensorboard_logs = {'Loss/Test': avg_loss}

        # column to indicate offset to key_date
        cols = [str(t) for t in range(self.hparams.output_size)]

        df_obs = pd.DataFrame(columns=cols)
        df_sim = pd.DataFrame(columns=cols)

        for out_dict in out_dicts:
            ys = out_dict['obs']
            y_hats = out_dict['sim']
            dates = out_dict['dates']

            _df_obs, _df_sim = self.single_batch_to_df(ys, y_hats, dates, cols)

            df_obs = pd.concat([df_obs, _df_obs])
            df_sim = pd.concat([df_sim, _df_sim])
        df_obs.index.name = 'date'
        df_sim.index.name = 'date'

        df_obs.sort_index(inplace=True)
        df_sim.sort_index(inplace=True)
        
        df_obs.to_csv(self.output_dir / "df_test_obs.csv")
        df_sim.to_csv(self.output_dir / "df_test_sim.csv")

        plot_line(self.hparams, df_obs, df_sim, self.target,
            self.output_dir, "line_DNN_" + self.target)
        plot_scatter(self.hparams, df_obs, df_sim,
            self.output_dir, "scatter_DNN_" + self.target)
        plot_corr(self.hparams, df_obs, df_sim,
            self.output_dir, "corrs_DNN_" + self.target)

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
        train_valid_set = data.DNNMeanSeasonalityDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            train_valid_ratio=0.8)
        test_set = data.DNNMeanSeasonalityDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.test_fdate,
            tdate=self.test_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            avg_hourly=train_valid_set.dict_avg_hourly,
            avg_annual=train_valid_set.dict_avg_annual)

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
        xs, ys, dates = zip(*batch)

        elem = batch[0]
        elem_type = type(elem)

        return torch.as_tensor(xs), torch.as_tensor(ys), dates


def plot_line(hparams, df_obs, df_sim, target, output_dir, fname_prefix):
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    for t in range(hparams.output_size):
        dates = df_obs.index + dt.timedelta(hours=t)
        plt_fname = output_dir / (fname_prefix + "_h" + str(t) + ".png")

        obs = df_obs[[str(t)]]
        sim = df_sim[[str(t)]]

        plt.figure()
        plt.title("OBS & Model")
        plt.xlabel("dates")
        plt.ylabel(target)
        plt.plot(dates, obs, "b", dates, sim, "r")
        plt.savefig(plt_fname)
        plt.close()


def plot_scatter(hparams, df_obs, df_sim, output_dir, fname_prefix):
    Path.mkdir(output_dir, parents=True, exist_ok=True)

    for t in range(hparams.output_size):
        plt_fname = output_dir / (fname_prefix + "_h" + str(t).zfill(2) + ".png")

        obs = df_obs[[str(t)]]
        sim = df_sim[[str(t)]]
        
        plt.figure()
        plt.title("Model/OBS")
        plt.xlabel("OBS")
        plt.ylabel("Model")
        maxval = np.nanmax([np.nanmax(obs), np.nanmax(sim)])
        plt.xlim((0, maxval))
        plt.ylim((0, maxval))
        plt.scatter(obs, sim)
        plt.savefig(plt_fname)
        plt.close()

def plot_corr(hparams, df_obs, df_sim, output_dir, fname_prefix):
    Path.mkdir(output_dir, parents=True, exist_ok=True)
    plt_fname = output_dir / (fname_prefix + ".png")
    csv_fname = output_dir / (fname_prefix + ".csv")
    
    times = list(range(hparams.output_size))
    corrs = []
    for t in range(hparams.output_size):
        obs = df_obs[str(t)].to_numpy()
        sim = df_sim[str(t)].to_numpy()

        corrs.append(np.corrcoef(obs, sim)[0, 1])

    plt.figure()
    plt.title("Correlation of OBS & Model")
    plt.xlabel("lags")
    plt.ylabel("corr")
    plt.plot(times, corrs)
    plt.savefig(plt_fname)
    plt.close()

    df_corrs = pd.DataFrame({'time': times, 'corr': corrs})
    df_corrs.to_csv(output_dir / csv_fname)

def swish(_input, beta=1.0):
    """
        Swish function in [this paper](https://arxiv.org/pdf/1710.05941.pdf)

    Args:
        input: Tensor

    Returns:
        output: Activated tensor
    """
    return _input * beta * nn.Sigmoid(_input)

