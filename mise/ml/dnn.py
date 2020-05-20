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
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, RandomSampler, SequentialSampler

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer

import data
from constants import SEOUL_STATIONS, SEOULTZ

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ml_dnn(station_name="종로구"):
    print("Start DNN model")
    targets = ["PM10", "PM25"]
    sample_size = 48
    output_size = 24

    # Hyper parameter
    epoch_size = 3
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

        output_dir = Path("/mnt/data/dnn/" + target + "/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)

        if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            # load imputed result
            _df_h = data.load_imputed(HOURLY_DATA_PATH)
            df_h = _df_h.query('stationCode == "' +
                            str(SEOUL_STATIONS[station_name]) + '"')
            df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

        # construct data loader
        """
        train_loader = DataLoader(train_set,
            batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(valid_set,
            batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set,
            batch_size=batch_size, shuffle=True, pin_memory=True)
        """

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
                num_workers=4,
                train_fdate=train_fdate, train_tdate=train_tdate,
                test_fdate=test_fdate, test_tdate=test_tdate)
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
                num_workers=4,
                train_fdate=train_fdate, train_tdate=train_tdate,
                test_fdate=test_fdate, test_tdate=test_tdate)

        # Loss and optimizer
        #criterion = nn.MSELoss(reduction='sum')
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        #total_step = len(train_loader)
        #curr_lr = learning_rate

        # most basic trainer, uses good defaults
        trainer = Trainer(gpus=1,
            precision=32,
            min_epochs=1, max_epochs=1000,
            early_stop_checkpoint=True,
            auto_lr_find=True,
            default_save_path=output_dir)

        # Invoke method
        #scaled_batch_size = trainer.scale_batch_size(model)

        # Override old batch size
        #model.hparams.batch_size = scaled_batch_size

        #lr_finder = trainer.lr_find(model)

        # Results can be found in
        #print(lr_finder.results)

        # Pick point based on plot, or get suggestion
        #new_lr = lr_finder.suggestion()

        # update hparams of the model
        #model.hparams.lr = new_lr

        trainer.fit(model)

        """
        # training
        for epoch in range(epoch_size):
            for batch_idx, (xs, ys) in enumerate(train_loader):
                #xs = xs.view(-1, sample_size*len(train_features)).to(device)
                #ys = ys.view(-1, output_size).to(device)
                #y = y.view(-1, output_size).to(device)

                # Forward pass
                ys_pred = model(xs)
                loss = criterion(ys_pred, ys)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_idx+1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                        .format(epoch+1, epoch_size, batch_idx+1, total_step, loss.item()))

            # Decay learning rate
            if (epoch+1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

        model.eval()
        # test
        with torch.no_grad():
            evals = 0
            for xs, ys in test_loader:
                xs = xs.view(-1, sample_size*len(train_features)).to(device)
                ys = ys.view(-1, output_size).to(device)

                ys_pred = model(xs)
                evals = criterion(ys_pred, ys)

            print('Accuracy of the model {}'.format(evals))
        """
        #torch.save(model.state_dict(),  output_dir / (target + '.ckpt'))

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
        x, y = batch
        y_hat = self(x)
        _loss = self.loss(y_hat, y)
        tensorboard_logs = {'train_loss': _loss}
        return {'loss': _loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': self.loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': self.loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        # create custom dataset
        train_valid_set = data.DNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.train_fdate,
            tdate=self.train_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size,
            train_valid_ratio=0.8)
        test_set = data.DNNDataset(
            station_name=self.station_name,
            target=self.target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=self.features,
            fdate=self.test_fdate,
            tdate=self.test_tdate,
            sample_size=self.hparams.sample_size,
            output_size=self.hparams.output_size)

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
            shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers)

def swish(_input, beta=1.0):
    """
        Swish function in [this paper](https://arxiv.org/pdf/1710.05941.pdf)

    Args:
        input: Tensor

    Returns:
        output: Activated tensor
    """
    return _input * beta * nn.Sigmoid(_input)

