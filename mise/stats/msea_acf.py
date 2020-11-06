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


def stats_msea_acf(station_name="종로구"):
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

    for target in targets:
        dir_prefix = Path("/mnt/data/msea_acf/" +
                          station_name + "/" + target + "/")
        Path.mkdir(dir_prefix, parents=True, exist_ok=True)
        target_sea_h_path = Path("/input/msea/weekly/" + station_name +
                                 "/" + target + "/df_" + target + ".csv")

        df_sea_h = pd.read_csv(target_sea_h_path,
                               index_col=[0],
                               parse_dates=[0])

        nlag = 24*10

        raw_acf = sm.tsa.acf(df_sea_h[[target + "_raw"]], nlags=nlag)
        ys_acf = sm.tsa.acf(df_sea_h[[target + "_ys"]], nlags=nlag)
        yr_acf = sm.tsa.acf(df_sea_h[[target + "_yr"]], nlags=nlag)
        ds_acf = sm.tsa.acf(df_sea_h[[target + "_ds"]], nlags=nlag)
        dr_acf = sm.tsa.acf(df_sea_h[[target + "_dr"]], nlags=nlag)
        ws_acf = sm.tsa.acf(df_sea_h[[target + "_ws"]], nlags=nlag)
        wr_acf = sm.tsa.acf(df_sea_h[[target + "_wr"]], nlags=nlag)

        raw_acf_sr = pd.Series(raw_acf, name="raw")
        ys_acf_sr = pd.Series(ys_acf, name="ys")
        yr_acf_sr = pd.Series(yr_acf, name="yr")
        ds_acf_sr = pd.Series(ds_acf, name="ds")
        dr_acf_sr = pd.Series(dr_acf, name="dr")
        ws_acf_sr = pd.Series(ws_acf, name="ws")
        wr_acf_sr = pd.Series(wr_acf, name="wr")
        
        acf_df = merge_acfs([
            raw_acf_sr,
            ys_acf_sr, yr_acf_sr,
            ds_acf_sr, dr_acf_sr,
            ws_acf_sr, wr_acf_sr], ["raw", "ys", "yr", "ds", "dr", "ws", "wr"])

        plot_acf(ys_acf_sr, target, nlag,
                 "ys", "Annual Seasonality", dir_prefix)
        plot_acf(yr_acf_sr, target, nlag,
                 "yr", "Annual Residual", dir_prefix)
        plot_acf(ds_acf_sr, target, nlag,
                 "ds", "Daily Seasonality", dir_prefix)
        plot_acf(dr_acf_sr, target, nlag,
                 "dr", "Daily Residual", dir_prefix)
        plot_acf(ws_acf_sr, target, nlag,
                 "ws", "Weekly Seasonality", dir_prefix)
        plot_acf(wr_acf_sr, target, nlag,
                 "wr", "Weekly Residual", dir_prefix)
        plot_acfs(acf_df, ["raw", "yr", "dr", "wr"],
                  target,  nlag, "Autocorrelation", dir_prefix)
        plot_acfs(acf_df, ["raw", "yr"],
                  target,  nlag, "Autocorrelation", dir_prefix)
        plot_acfs(acf_df, ["raw", "dr"],
                  target,  nlag, "Autocorrelation", dir_prefix)
        plot_acfs(acf_df, ["raw", "wr"],
                  target,  nlag, "Autocorrelation", dir_prefix)

def merge_acfs(acfs, cols):
    df = pd.concat(acfs, axis=1).reset_index()
    return df

def plot_acfs(df_acf, cols, target, nlag, title, dir_prefix):
    png_fname = "acfs_" + target + "_" + \
        "_".join(cols) + "_" + str(nlag) + ".png"
    svg_fname = "acfs_" + target + "_" + \
        "_".join(cols) + "_" + str(nlag) + ".svg"
    csv_fname = "acfs_" + target + "_" + \
        "_".join(cols) + "_" + str(nlag) + ".csv"
    print(csv_fname)
    colors = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A"]

    plt.figure()
    plt.title(title)
    plt.xlabel("lags")
    plt.ylabel("acf")
    for i, col in enumerate(cols):
        plt.plot(df_acf.index.tolist(),
                 df_acf[[col]].to_numpy(), label=col, color=colors[i])
    plt.grid(b=True)
    plt.legend()
    ax = plt.gca()
    ax.set_ylim((min(0.0, df_acf[cols].min().min()), df_acf[cols].max().max()))
    plt.savefig(dir_prefix / png_fname)
    plt.savefig(dir_prefix / svg_fname)
    plt.close()

    df_acf.to_csv(dir_prefix / csv_fname)


def plot_acf(acf, target, nlag, part_name, title, dir_prefix):
    png_fname = "acf_" + target + "_" + part_name + "_" + str(nlag) + ".png"
    svg_fname = "acf_" + target + "_" + part_name + "_" + str(nlag) + ".svg"
    csv_fname = "acf_" + target + "_" + part_name + "_" + str(nlag) + ".csv"
    print(csv_fname)

    plt.figure()
    plt.title(title)
    plt.xlabel("lags")
    plt.ylabel("acf")
    plt.plot(acf.axes[0].tolist(), acf.to_numpy())
    plt.grid(b=True)
    ax = plt.gca()
    ax.set_ylim((min(0.0, acf.min()), acf.max()))
    plt.savefig(dir_prefix / png_fname)
    plt.savefig(dir_prefix / svg_fname)
    plt.close()

    acf.to_csv(dir_prefix / csv_fname)

