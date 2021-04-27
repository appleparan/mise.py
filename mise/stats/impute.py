import datetime as dt
import dateutil
import time
import glob
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pytz import timezone
from openpyxl import load_workbook
from sklearn.impute import KNNImputer
import statsmodels.api as sm
# from statsmodels.imputation.mice import MICE
import tqdm

import data

from constants import SEOUL_STATIONS, SEOULTZ

def parse_raw_aerosols(aes_dir, aes_fea, fdate, tdate, station_name='종로구'):
    re_aes_fn = r"**/([0-9]+)년\W*([0-9]+)(분기|월).csv"
    re_ext_fn = r".*.csv"
    date_str = "%Y%m%d%H"
    re_date = "^([0-9]{4})([0-9]{2})([0-9]{2})([0-9]{2})$"
    station_code = SEOUL_STATIONS[station_name]

    # for tqdm, conver to list
    aes_globs = list(aes_dir.rglob('*.csv'))

    dfs = []

    for aes_path in tqdm.tqdm(aes_globs):
        _df_raw = pd.read_csv(aes_path)
        _df_raw.rename(mapper={"지역": "region", "측정소코드": "stationCode",
                    "측정소명": "stationName", "측정일시": "date",
                    "주소": "addr"}, axis='columns', inplace=True)

        def _parse_date(dstr, date_str):
            m = re.match(re_date, str(dstr))
            yyyy = int(m.groups()[0])
            mm = int(m.groups()[1])
            dd = int(m.groups()[2])
            hh = int(m.groups()[3])

            if hh == 24:
                d = dt.datetime(yyyy, mm, dd, hh-1).astimezone(SEOULTZ) + \
                    dt.timedelta(hours=1)
            else:
                d = dt.datetime(yyyy, mm, dd, hh).astimezone(SEOULTZ)

            return d

        _dates = list(map(lambda d: _parse_date(str(d), date_str), _df_raw.loc[:, 'date'].tolist()))

        # imputed results to DataFrame
        _df = pd.DataFrame({
                'stationCode' : _df_raw.loc[:, 'stationCode'],
                'date' : _dates,
                'SO2' : _df_raw.loc[:, 'SO2'],
                'CO' : _df_raw.loc[:, 'CO'],
                'NO2' : _df_raw.loc[:, 'NO2'],
                'PM10' : _df_raw.loc[:, 'PM10'],
                'PM25' : _df_raw.loc[:, 'PM25']})

        # print(_df['stationCode'] == station_code
        _df = _df.loc[_df['stationCode'] == station_code, :].copy()
        _df.drop(labels='stationCode', axis='columns', inplace=True)

        _df.sort_values(by=['date'], inplace=True)
        dfs.append(_df)

    df = pd.concat(dfs)
    df.sort_values(by=['date'], inplace=True)
    df.set_index(['date'], inplace=True)

    return df.loc[fdate:tdate, aes_fea]

def parse_raw_weathers(wea_dir, wea_fea, fdate, tdate, seoul_stn_code=108):
    re_wea_fn = r"SURFACE_ASOS_$(string(wea_stn_code))_HR_([0-9]+)_([0-9]+)_([0-9]+).csv"
    date_str = "yyyy-mm-dd HH:MM"

    # for tqdm, conver to list
    wea_globs = list(wea_dir.rglob('*.csv'))

    dfs = []
    for wea_path in tqdm.tqdm(wea_globs):
        _df_raw = pd.read_csv(wea_path)
        _df_raw.rename(mapper={"일시": "date",
                    "기온(°C)": "temp",
                    "강수량(mm)": "prep",
                    "풍속(m/s)": "wind_spd",
                    "풍향(16방위)": "wind_dir",
                    "습도(%)": "humid",
                    "현지기압(hPa)": "pres",
                    "적설(cm)": "snow"}, axis='columns', inplace=True)

        _dates = [dateutil.parser.isoparse(str(d)).astimezone(SEOULTZ) for d in _df_raw.loc[:, 'date']]

        dict_wind_dir = {"0": 90,
            "360": 90 , "20" : 67.5 , "50" : 45 , "70" : 22.5 ,
            "90" : 0  , "110": 337.5, "140": 315, "160": 292.5,
            "180": 270, "200": 247.5, "230": 225, "250": 202.5,
            "270": 180, "290": 157.5, "320": 135, "340": 112.5}

        _df = pd.DataFrame({
                'date' : _dates,
                'temp' : _df_raw.loc[:, 'temp'],
                'wind_spd' : _df_raw.loc[:, 'wind_spd'],
                'wind_dir' : _df_raw.loc[:, 'wind_dir'],
                'pres' : _df_raw.loc[:, 'pres'],
                'humid' : _df_raw.loc[:, 'humid'],
                'prep' : _df_raw.loc[:, 'prep']})

        _df.sort_values(by=['date'], inplace=True)
        _df.set_index('date', inplace=True)

        # dict key may not match -> find nearest key
        def approx_winddir(w):
            wd = np.array([0, 20, 50, 70, 90, 110, 140, 160, 180, 200, 230, 250, 270, 290, 320, 340, 360])
            return wd[np.argmin(np.sqrt((w - wd)**2))]

        _wind_dir = np.array([approx_winddir(w) for w in _df.loc[:, 'wind_dir']])
        _df.loc[:, 'wind_dir'] = _wind_dir

        dfs.append(_df)

    df = pd.concat(dfs)
    df.sort_values(by=['date'], inplace=True)

    return df.loc[fdate:tdate, wea_fea]

def stats_imputation_stats():
    seoul_stn_code = 108
    station_name = '종로구'
    input_dir = Path("/input")
    aes_dir = input_dir / "aerosol"
    wea_dir = input_dir / "weather" / "seoul"

    data_dir = Path("/mnt/data/")
    stat_dir = Path("/mnt/data/impute_stats")
    Path.mkdir(stat_dir, parents=True, exist_ok=True)

    fdate = dt.datetime(2008, 1, 1, 1).astimezone(SEOULTZ)
    tdate = dt.datetime(2020, 10, 31, 23).astimezone(SEOULTZ)
    dates = pd.date_range(fdate, tdate, freq='1H', tz=SEOULTZ)
    aes_fea = ["SO2", "CO", "NO2", "PM10", "PM25"]
    wea_fea = ["temp", "pres", "wind_spd", "wind_dir", "humid", "prep"]

    print(f'Parsing Range from {fdate.strftime("%Y/%m/%d %H")}' \
          f' to {tdate.strftime("%Y/%m/%d %H")} ...', flush=True)
    print(f"Parsing Weather dataset...", flush=True)
    df_wea = parse_raw_weathers(wea_dir, wea_fea, fdate, tdate,
                                seoul_stn_code=seoul_stn_code)

    df_wea_isna = df_wea.isna().sum()
    df_wea_isnotna = df_wea.notna().sum()
    df_wea_isna.to_csv(stat_dir / f"stats_wea_isna.csv")
    df_wea_isnotna.to_csv(stat_dir / f"stats_wea_isnotna.csv")

    df_wea.to_csv(data_dir / ('df_wea.csv'))
    print(df_wea.head(5))
    print(df_wea.tail(5))

    print(f"Parsing Aerosol dataset - {station_name}...", flush=True)
    df_aes = parse_raw_aerosols(aes_dir, aes_fea, fdate, tdate,
                                station_name=station_name)

    df_aes_isna = df_aes.isna().sum()
    df_aes_isnotna = df_aes.notna().sum()
    df_aes_isna.to_csv(stat_dir / f"stats_aes_{station_name}_isna.csv")
    df_aes_isnotna.to_csv(stat_dir / f"stats_aes_{station_name}_isnotna.csv")

    df_aes.to_csv(data_dir / ('df_aes.csv'))
    print(df_aes.head(5))
    print(df_aes.tail(5))

    df = df_aes.join(df_wea, on='date', how='left')
    df.to_csv(data_dir / ('df_raw_no_impute.csv'))

    print(f'Length of weather dataframe : {len(df_wea.index)}')
    print(f'Length of aerosol dataframe : {len(df_aes.index)}')
    print(f'Length of dates : {len(dates)}')
    print(list(set(list(dates)) - set(list(df_wea.index))))
    print(list(set(list(dates)) - set(list(df_aes.index))))
    print(list(set(list(df_wea.index)) - set(list(df_aes.index))))
