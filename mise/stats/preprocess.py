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

def stats_preprocess():
    print("Data preprocessing(imputation) start...")

    raw_df = data.load(datecol=[1])
    dfs_h = []
    for station_name in tqdm.tqdm(SEOUL_STATIONS.keys(), total=len(SEOUL_STATIONS.keys())):
        sdf = data.load_station(raw_df, SEOUL_STATIONS[station_name])

        imputer = KNNImputer(
            n_neighbors=5, weights="distance", missing_values=np.NaN)
        _df = pd.DataFrame(imputer.fit_transform(sdf))
        _df.columns = sdf.columns
        _df.index = sdf.index

        dfs_h.append(_df)

    df = pd.concat(dfs_h)
    df.to_csv("/input/python/input_seoul_imputed_hourly_pandas.csv")

