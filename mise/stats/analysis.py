from argparse import Namespace
import copy
import datetime as dt
from math import sqrt
import os
from pathlib import Path
import random
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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
import sklearn.metrics

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from data import load_imputed, MultivariateRNNDataset, MultivariateRNNMeanSeasonalityDataset
from constants import SEOUL_STATIONS, SEOULTZ
import utils

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def stats_analysis(station_name="종로구"):
    """
    References
    * Ghil, M., et al. "Extreme events: dynamics, statistics and prediction." Nonlinear Processes in Geophysics 18.3 (2011): 295-350.
    """
    print("Start Analysis of input")
    targets = ["PM10", "PM25"]
    sea_targets = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                   "temp", "u", "v", "pres", "humid", "prep", "snow"]
    # sea_targets = ["prep", "snow"]
    # 24*14 = 336
    sample_size = 24*2
    output_size = 24

    # Hyper parameter
    epoch_size = 500
    batch_size = 256
    learning_rate = 1e-3

    train_fdate = dt.datetime(2015, 1, 5, 0).astimezone(SEOULTZ)
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
    train_features_periodic = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                               "temp", "u", "v", "pres", "humid"]
    train_features_nonperiodic = ["prep", "snow"]

    def plot_sea():
        for target in sea_targets:
            print("Analyze " + target + "...")
            target_sea_h_path = Path(
                "/input/python/input_jongro_imputed_hourly_pandas.csv")

            df_sea_h = pd.read_csv(target_sea_h_path,
                                index_col=[0],
                                parse_dates=[0])
            print(df_sea_h.head(5))

            output_dir = Path("/mnt/data/STATS_ANALYSIS_" + str(sample_size) + "/" +
                            station_name + "/" + target + "/")
            Path.mkdir(output_dir, parents=True, exist_ok=True)

            data_dir = output_dir / Path('csv/')
            png_dir = output_dir / Path('png/')
            svg_dir = output_dir / Path('svg/')
            Path.mkdir(data_dir, parents=True, exist_ok=True)
            Path.mkdir(png_dir, parents=True, exist_ok=True)
            Path.mkdir(svg_dir, parents=True, exist_ok=True)

            if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
                # load imputed result
                _df_h = load_imputed(HOURLY_DATA_PATH)
                df_h = _df_h.query('stationCode == "' +
                                str(SEOUL_STATIONS[station_name]) + '"')
                df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

            hparams = Namespace(
                nhead=8,
                head_dim=128,
                d_feedforward=256,
                num_layers=3,
                learning_rate=learning_rate,
                batch_size=batch_size)

            # prepare dataset
            train_valid_set = MultivariateRNNMeanSeasonalityDataset(
                station_name=station_name,
                target=target,
                filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
                features=train_features,
                features_1=["SO2", "CO", "O3", "NO2", "PM10", "PM25",
                            "temp", "v", "pres", "humid", "prep", "snow"],
                features_2=['u'],
                fdate=train_fdate,
                tdate=train_tdate,
                sample_size=sample_size,
                output_size=output_size,
                train_valid_ratio=0.8)

            # first mkdir of seasonality
            Path.mkdir(png_dir / "seasonality", parents=True, exist_ok=True)
            Path.mkdir(svg_dir / "seasonality", parents=True, exist_ok=True)
            Path.mkdir(data_dir / "seasonality", parents=True, exist_ok=True)

            # fit & transform (seasonality)
            # without seasonality
            # train_valid_set.preprocess()
            # with seasonality
            train_valid_set.preprocess(
                data_dir / "seasonality_fused",
                png_dir / "seasonality_fused",
                svg_dir / "seasonality_fused")
            # save seasonality index-wise
            # train_valid_set.broadcast_seasonality()

            train_valid_set.plot_fused_seasonality(
                data_dir / "seasonality_fused",
                png_dir / "seasonality_fused",
                svg_dir / "seasonality_fused")

    plot_sea()

    for target in targets:
        print("Analyze " + target + "...")
        target_sea_h_path = Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv")

        df_sea_h = pd.read_csv(target_sea_h_path,
                               index_col=[0],
                               parse_dates=[0])

        output_dir = Path("/mnt/data/STATS_ANALYSIS_" + str(sample_size) + "/" +
                          station_name + "/" + target + "/")
        Path.mkdir(output_dir, parents=True, exist_ok=True)

        data_dir = output_dir / Path('csv/')
        png_dir = output_dir / Path('png/')
        svg_dir = output_dir / Path('svg/')
        Path.mkdir(data_dir, parents=True, exist_ok=True)
        Path.mkdir(png_dir, parents=True, exist_ok=True)
        Path.mkdir(svg_dir, parents=True, exist_ok=True)

        if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            # load imputed result
            _df_h = load_imputed(HOURLY_DATA_PATH)
            df_h = _df_h.query('stationCode == "' +
                               str(SEOUL_STATIONS[station_name]) + '"')
            df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

        hparams = Namespace(
            nhead=8,
            head_dim=128,
            d_feedforward=256,
            num_layers=3,
            learning_rate=learning_rate,
            batch_size=batch_size)

        # prepare dataset
        train_valid_set = MultivariateRNNMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=train_features,
            features_1=train_features_nonperiodic,
            features_2=train_features_periodic,
            fdate=train_fdate,
            tdate=train_tdate,
            sample_size=sample_size,
            output_size=output_size,
            train_valid_ratio=0.8)

        # first mkdir of seasonality
        Path.mkdir(png_dir / "seasonality", parents=True, exist_ok=True)
        Path.mkdir(svg_dir / "seasonality", parents=True, exist_ok=True)
        Path.mkdir(data_dir / "seasonality", parents=True, exist_ok=True)

        # fit & transform (seasonality)
        # without seasonality
        # train_valid_set.preprocess()
        # with seasonality
        train_valid_set.preprocess(
            data_dir / "seasonality_fused",
            png_dir / "seasonality_fused",
            svg_dir / "seasonality_fused")
        # save seasonality index-wise
        train_valid_set.broadcast_seasonality()

        test_set = MultivariateRNNMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath="/input/python/input_jongro_imputed_hourly_pandas.csv",
            features=train_features,
            features_1=train_features_nonperiodic,
            features_2=train_features_periodic,
            fdate=test_fdate,
            tdate=test_tdate,
            sample_size=sample_size,
            output_size=output_size,
            scaler_X=train_valid_set.scaler_X,
            scaler_Y=train_valid_set.scaler_Y)

        test_set.transform()
        # save seasonality index-wise
        test_set.broadcast_seasonality()

        def run_01_CLT():
            """
            1. Is data sufficient?
                * Central Limit Theorem =>
                * Distribution of sample mean & sample std =>
                * Is it normal or log-normal?
            """
            _data_dir = data_dir / "01-CLT"
            _png_dir = png_dir / "01-CLT"
            _svg_dir = svg_dir / "01-CLT"
            Path.mkdir(_data_dir, parents=True, exist_ok=True)
            Path.mkdir(_png_dir, parents=True, exist_ok=True)
            Path.mkdir(_svg_dir, parents=True, exist_ok=True)

            # statistics of decomposed samples
            means_d = np.zeros(len(train_valid_set))
            means_r = np.zeros(len(train_valid_set))
            sample_means_d = np.zeros(len(train_valid_set))

            # statistics of raw samples
            sample_means_r = np.zeros(len(train_valid_set))

            # save sample statistics
            # len(train_valid_set) == 34895
            for i, s in enumerate(train_valid_set):
                x, x_1d, x_sa, x_sw, x_sh, \
                    y, y_raw, y_sa, y_sw, y_sh, y_date = s

                if len(y) != output_size:
                    break

                # it's not random sampling
                means_d[i] = np.mean(y)
                means_r[i] = np.mean(y_raw)

            nchoice = 64
            for i in range(100):
                dr = np.random.choice(means_d, size = nchoice)
                sample_means_d[i] = np.mean(dr)
                rr = np.random.choice(means_r, size = nchoice)
                sample_means_r[i] = np.mean(rr)

            print("Sample & Pop. Mean : ", np.mean(
                sample_means_d), np.mean(means_d))
            print("Sample & Pop. STD : ", np.std(
                sample_means_d) / sqrt(nchoice), np.std(means_d))

            print("Sample & Pop. Mean : ", np.mean(
                sample_means_r), np.mean(means_r))
            print("Sample & Pop. STD : ", np.std(
                sample_means_r) / sqrt(nchoice), np.std(means_r))

        def run_02_LRD():
            _data_dir = data_dir / "02-LRD"
            _png_dir = png_dir / "02-LRD"
            _svg_dir = svg_dir / "02-LRD"
            Path.mkdir(_data_dir, parents=True, exist_ok=True)
            Path.mkdir(_png_dir, parents=True, exist_ok=True)
            Path.mkdir(_svg_dir, parents=True, exist_ok=True)

            # enumerate window
            # for i, s in enumerate(train_valid_set):
            #     x, x_1d, x_sa, x_sw, x_sh, \
            #         y, y_raw, y_sa, y_sw, y_sh, y_date = s

            #     if len(y) != output_size:
            #         break

            # Define unbounded process
            Xs = train_valid_set.ys - train_valid_set.ys.mean()[target]

            # iterate
            for td in train_valid_set.ys.iterrows():
                x, x_1d, x_sa, x_sw, x_sh, \
                    y, y_raw, y_sa, y_sw, y_sh, y_date = s

                if len(y) != output_size:
                    break
                # 2. Do our data have Long-Range Dependence (LRD)?
                # Hurst phenomenon : slow decay of ACF (AutoCorrelation Function)
                # Hurst exponennt for original data and seasonlity decomposed data

        #run_01_CLT()
        #run_02_LRD()
    #plot_sea()


def dfa(data: np.ndarray, ss=[10, 20, 30], order=1,
        debug_plot=False, debug_data=False, plot_file=None):
    """
    Performs a detrended fluctuation analysis (DFA) on the given data to detect long-range dependence (LRD)


    Recommendations for parameter settings by Hardstone et al.:
        * nvals should be equally spaced on a logarithmic scale so that each window
        scale hase the same weight
        * min(nvals) < 4 does not make much sense as fitting a polynomial (even if
        it is only of order 1) to 3 or less data points is very prone.
        * max(nvals) > len(data) / 10 does not make much sense as we will then have
        less than 10 windows to calculate the average fluctuation
        * use overlap=True to obtain more windows and therefore better statistics
        (at an increased computational cost)

    # Reference
    * Kantelhardt, Jan W., et al. "Detecting long-range correlations with detrended fluctuation analysis." Physica A: Statistical Mechanics and its Applications 295.3-4 (2001): 441-454.
    * Inspiration from [this code](https://github.com/CSchoel/nolds/blob/b52530a783d2d2aa39351ea285ab0d8e3a502ab3/nolds/measures.py)
    """
    if data.ndims != 1:
        raise ValueError("ndims of data should be 1")

    # Step 1.
    # Compute fluctuation
    Y = data - np.mean(data)

    total_N = len(data)

    if len(total_N) < 70:
        raise ValueError("Data length is too small!")

    def seg_polyfit(s):
        return s - np.polynomial.Polynomial.fit(range(len(s)), s, deg=order)

    for s in ss:
        # s : segment size
        # Step 2
        # Split data by non-overlapping segments
        # i.e. s == 3 -> N_s = [N/s] == 3 and 2N_s segments are computed
        # not to discard remainders
        # [1, 2, 3], [4, 5, 6], [7, 8, 9]
        # [2, 3, 4], [5, 6, 7], [8, 9, 10]
        Ns, Ns_rem = divmod(total_N, s)
        if Ns_rem == 0:
            seg1 = np.array_split(Y, s)
            seg2 = seg1.copy()
        else:
            seg1 = np.array_split(Y, s)[:Ns]
            seg2 = np.array_split(Y[Ns_rem:], s)

        segs = np.concatenate(seg1, seg2)

        # Step 3
        # Compute basis of variance (Y - fitted)
        # eq. 4 in Kantelhardt, et. al
        seg1_detrend = map(seg_polyfit, seg1)
        seg2_detrend = map(seg_polyfit, seg2)

        # Step 4
        # Compute Variance (eq. 5 in Kantelhardt, et. al) then average
        segs_detrend = np.concatenate(seg1_detrend, seg2_detrend)



