import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn import preprocessing

from torch.utils.data.dataset import Dataset

from constants import SEOUL_STATIONS, SEOULTZ

def load(filepath="/input/input.csv", datecol=[1]):
    # date column in raw data : 1
    # date column in imputed data : 0
    df = pd.read_csv(filepath,
        index_col=[0, 1],
        parse_dates=datecol)

    # prints
    pd.set_option('display.max_rows', 10)
    pd.reset_option('display.max_rows')
    print(df.head(10))

    return df

def load_imputed(filepath="/input/input.csv", datecol=[0]):
    # date column in raw data : 1
    # date column in imputed data : 0
    df = pd.read_csv(filepath,
                     index_col=[0, 1],
                     parse_dates=datecol)

    # prints
    pd.set_option('display.max_rows', 10)
    pd.reset_option('display.max_rows')
    print(df.head(10))

    return df

def load_station(df, code=111123):
    #return df[df['stationCode'] == code]
    return df.query('stationCode == "' + str(code) + '"')

class DNNDataset(Dataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('taret', 'PM10')
        self.features = kwargs.get('features',
                              ["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp", "u", "v", "pres", "humid", "prep", "snow"])

        self.fdate = kwargs.get('fdate', dt.datetime(
            2008, 1, 1, 1).astimezone(SEOULTZ))
        self.tdate = kwargs.get('tdate', dt.datetime(
            2017, 12, 31, 23).astimezone(SEOULTZ))

        self.sample_size = kwargs.get('sample_size', 48)
        self.batch_size = kwargs.get('batch_size', 32)
        self.output_size = kwargs.get('output_size', 24)
        self._train_valid_ratio = kwargs.get('train_valid_ratio', 0.8)

        # load imputed data from filepath if not provided as dataframe
        filepath = kwargs.get('filepath', Path("/input/python/input_jongro_imputed_hourly_pandas.csv"))
        raw_df = pd.read_csv(filepath,
                         index_col=[0, 1],
                         parse_dates=[0])
        # filter by station_name
        self._df = raw_df.query('stationCode == "' +
                            str(SEOUL_STATIONS[self.station_name]) + '"')
        self._df.reset_index(level='stationCode', drop=True, inplace=True)
        # filter by date
        self._df = self._df[self.fdate:self.tdate]

        self._xs = self.df[self.features]
        self._ys = self.df[[self.target]]

        # self._xs must not be available when creating instance so no kwargs for scaler
        self._scaler = preprocessing.StandardScaler().fit(self._xs)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i:i+self.sample_size]
        y = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)]

        return x.to_numpy().astype('float32'), np.reshape(y.to_numpy(), len(y)).astype('float32')

    def __len__(self):
        """
        hours of train and test dates

        Returns:
            int: total hours 
        """
        duration  = self.tdate - self.fdate - dt.timedelta(hours=(self.output_size + self.sample_size))
        return duration.days * 24 + duration.seconds // 3600

    # getter only
    @property
    def xs(self):
        return self._xs

    # getter only
    @property
    def ys(self):
        return self._ys

    @property
    def scaler(self):
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler.fit(self._xs)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df

    @property
    def train_valid_ratio(self):
        return self._train_valid_ratio

    @train_valid_ratio.setter
    def train_valid_ratio(self, value):
        self._train_valid_ratio = value


