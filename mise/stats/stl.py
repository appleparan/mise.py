import copy
import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
import tqdm

import data
from constants import SEOUL_STATIONS

seoultz = timezone('Asia/Seoul')
HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

def stats_stl(station_name="종로구"):
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
    output_size = 24
    fdate = dt.datetime(2008, 1, 1, 0).astimezone(seoultz)
    tdate = dt.datetime(2017, 12, 31, 23).astimezone(seoultz)

    for target in targets:
        dir_prefix = Path("/mnt/data/stl/" +
                          station_name + "/" + target + "/")
        Path.mkdir(dir_prefix, parents=True, exist_ok=True)

        # hourly data -> daily seasonality
        target_h = df_h[[target]]
        target_h.reset_index(level='stationCode', drop=True, inplace=True)

        # daily data -> annual seasonality
        target_d = df_d[[target]]
        target_d.reset_index(level='stationCode', drop=True, inplace=True)

        # hourly data -> daily seasonality
        print("STL with " + target + "...")
        
        stl_h = STL(target_h)
        res_h = stl_h.fit()
        
        stl_d = STL(target_d)
        res_d = stl_d.fit()
        
        df_sea_h = pd.DataFrame.from_dict({
                            'date': target_h.index.tolist(),
                            'observed': list(res_h.observed[target].tolist()),
                            'residual': list(res_h.resid),
                            'seasonal': list(res_h.seasonal),
                            'trend': list(res_h.trend)})
        df_sea_h.set_index('date')
        df_sea_d=pd.DataFrame.from_dict({
                            'date': target_d.index.tolist(),
                            'observed': list(res_d.observed[target].tolist()),
                            'residual': list(res_d.resid),
                            'seasonal': list(res_d.seasonal),
                            'trend': list(res_d.trend)})
        df_sea_d.set_index('date')

        prefix_h = "stl_" + target + "_h"
        prefix_d = "stl_" + target + "_d"

        plt.figure()
        fig_h = res_h.plot()
        plt.savefig(dir_prefix / (prefix_h + ".png"))
        df_sea_h.to_csv(dir_prefix / (prefix_h + ".csv"))

        plt.figure()
        fig_d = res_d.plot()
        plt.savefig(dir_prefix / (prefix_d + ".png"))
        df_sea_d.to_csv(dir_prefix / (prefix_d + ".csv"))


