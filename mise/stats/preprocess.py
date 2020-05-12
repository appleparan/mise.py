import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.impute import KNNImputer
import tqdm

import data
from constants import SEOUL_STATIONS

seoultz = timezone('Asia/Seoul')

def stats_preprocess(station_name="종로구"):
    print("Data preprocessing(imputation) start...")

    raw_df = data.load()
    dfs_h = []
    dfs_d = []
    for station_name in tqdm.tqdm(SEOUL_STATIONS.keys(), total=len(SEOUL_STATIONS.keys())):
        sdf = data.load_station(raw_df, SEOUL_STATIONS[station_name])

        imputer = KNNImputer(
            n_neighbors=2, weights="uniform", missing_values=np.NaN)
        _df = pd.DataFrame(imputer.fit_transform(sdf))
        _df.columns = sdf.columns
        _df.index = sdf.index

        dfs_h.append(_df)

        # daily average
        _df.reset_index(level='stationCode', inplace=True)
        _df_avg = _df.resample('D').mean()
        _df_avg.set_index(keys='stationCode', append=True, inplace=True)
        dfs_d.append(_df_avg)

    df = pd.concat(dfs_h)
    df.to_csv("/input/input_seoul_imputed_hourly_pandas.csv")

    df = pd.concat(dfs_d)
    df.to_csv("/input/input_seoul_imputed_daily_pandas.csv")
