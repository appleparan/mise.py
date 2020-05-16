import copy
import datetime as dt
import os
from pathlib import Path

from bokeh.palettes import Category20_4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
import tqdm
import statsmodels.api as sm

import data
from constants import SEOUL_STATIONS

seoultz = timezone('Asia/Seoul')
HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"


def stats_stl_acf(station_name="종로구"):
    print("Data loading start...")
    if Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
        df_d = data.load_imputed(
            "/input/python/input_jongro_imputed_daily_pandas.csv")
        df_h = data.load_imputed(
            "/input/python/input_jongro_imputed_hourly_pandas.csv")
    else:
        # load imputed result
        _df_d = data.load_imputed(DAILY_DATA_PATH)
        _df_h = data.load_imputed(HOURLY_DATA_PATH)
        df_d = _df_d.query('stationCode == "' +
                           str(SEOUL_STATIONS[station_name]) + '"')
        df_h = _df_h.query('stationCode == "' +
                           str(SEOUL_STATIONS[station_name]) + '"')

        df_d.to_csv("/input/python/input_jongro_imputed_daily_pandas.csv")
        df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

    print("Data loading complete")
    targets = ["PM10", "PM25"]
    sample_size = 48
    output_size = 24
    epoch_size = 300
    batch_size = 32

    #train_fdate = dt.datetime(2017, 1, 1, 0).astimezone(seoultz)
    train_fdate = dt.datetime(2017, 1, 1, 0).astimezone(seoultz)
    train_tdate = dt.datetime(2017, 12, 31, 23).astimezone(seoultz)
    test_fdate = dt.datetime(2018, 1, 1, 0).astimezone(seoultz)
    #test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(seoultz)
    test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(seoultz)

    for target in targets:
        dir_prefix = Path("/mnt/data/stl_acf/" +
                            station_name + "/" + target + "/")
        Path.mkdir(dir_prefix, parents=True, exist_ok=True)
        target_sea_d_path = Path("/input/python/stl_decomp/" + station_name +
                                 "/" + target + "/stl_" + target + "_d.csv")
        target_sea_h_path = Path("/input/python/stl_decomp/" + station_name +
                                 "/" + target + "/stl_" + target + "_h.csv")

        df_sea_d = pd.read_csv(target_sea_d_path,
                               index_col=[0],
                               parse_dates=[0])
        df_sea_h = pd.read_csv(target_sea_h_path,
                               index_col=[0],
                               parse_dates=[0])

        nlag = 24*10

        obser_acf = sm.tsa.acf(df_sea_h[["observed"]], nlags=nlag)
        trend_acf = sm.tsa.acf(df_sea_h[["trend"]], nlags=nlag)
        seaso_acf = sm.tsa.acf(df_sea_h[["seasonal"]], nlags=nlag)
        resid_acf = sm.tsa.acf(df_sea_h[["residual"]], nlags=nlag)

        obser_acf_sr = pd.Series(obser_acf, name="observed")
        trend_acf_sr = pd.Series(trend_acf, name="trend")
        seaso_acf_sr = pd.Series(seaso_acf, name="seasonal")
        resid_acf_sr = pd.Series(resid_acf, name="residual")

        acf_df = merge_acfs([
            obser_acf_sr,
            trend_acf_sr,
            seaso_acf_sr,
            resid_acf_sr], ["observed", "trend", "seasonal", "residual"])

        plot_acf(obser_acf_sr, target, nlag, "observed", "Observed", dir_prefix)
        plot_acf(trend_acf_sr, target, nlag, "trend", "Trend", dir_prefix)
        plot_acf(seaso_acf_sr, target, nlag, "seasonal", "Seasonality", dir_prefix)
        plot_acf(resid_acf_sr, target, nlag, "residual", "Residual", dir_prefix)
        plot_acfs(acf_df, ["observed", "trend", "seasonal", "residual"],
                target,  nlag, "Autocorrelation", dir_prefix)
        plot_acfs(acf_df, ["observed", "trend"],  target, nlag,"Autocorrelation",
                  dir_prefix)
        plot_acfs(acf_df, ["observed", "residual"], target, nlag, "Autocorrelation",
                  dir_prefix)

def merge_acfs(acfs, cols):
    df = pd.concat(acfs, axis=1).reset_index()
    return df

def plot_acfs(df_acf, cols, target, nlag, title, dir_prefix):
    png_fname = "acfs_" + target + "_" + "_".join(cols) + "_" + str(nlag) + ".png"
    csv_fname = "acfs_" + target + "_" + "_".join(cols) + "_" + str(nlag) + ".csv"

    colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]

    plt.figure()
    plt.title(title)
    plt.xlabel("lags")
    plt.ylabel("acf")
    for i, col in enumerate(cols):
        plt.plot(df_acf.index.tolist(), df_acf[[col]].to_numpy(), label=col, color=colors[i])
    plt.grid(b=True)
    plt.legend()
    ax = plt.gca()
    ax.set_ylim((min(0.0, df_acf[cols].min().min()), df_acf[cols].max().max()))
    plt.savefig(dir_prefix / png_fname)
    plt.close()

    df_acf.to_csv(dir_prefix / csv_fname)

def plot_acf(acf, target, nlag, part_name, title, dir_prefix):
    png_fname = "acf_" + target + "_" + part_name + "_" + str(nlag) + ".png"
    csv_fname = "acf_" + target + "_" + part_name + "_" + str(nlag) + ".csv"

    plt.figure()
    plt.title(title)
    plt.xlabel("lags")
    plt.ylabel("acf")
    plt.plot(acf.axes[0].tolist(), acf.to_numpy())
    plt.grid(b=True)
    ax = plt.gca()
    ax.set_ylim((min(0.0, acf.min()), acf.max()))
    plt.savefig(dir_prefix / png_fname)
    plt.close()

    acf.to_csv(dir_prefix / csv_fname)



