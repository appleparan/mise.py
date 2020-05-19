import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
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

        self.fdate = kwargs.get('train_fdate', dt.datetime(
            2012, 1, 1, 0).astimezone(SEOULTZ))
        self.tdate = kwargs.get('train_tdate', dt.datetime(
            2018, 12, 31, 23).astimezone(SEOULTZ))

        self.sample_size = kwargs.get('sample_size', 48)
        self.batch_size = kwargs.get('batch_size', 32)
        self.output_size = kwargs.get('output_size', 24)
        self._train_valid_ratio = kwargs.get('train_valid_ratio', 0.8)
        self._transform = None

        # load imputed data from filepath if not provided as dataframe
        filepath = kwargs.get('filepath', Path("/input/python/") / + self.station_name +
                              "/" + self.target + "/stl_" + self.target + "_h.csv")
        raw_df = pd.read_csv(filepath,
                         index_col=[0, 1],
                         parse_dates=[0])
        self._df = raw_df.query('stationCode == "' +
                            str(SEOUL_STATIONS[station_name]) + '"')
        self._df.reset_index(level='stationCode', drop=True, inplace=True)

        self.data = df[features]
        self.targets = df[[target]]

        self.stds = self.df.std(axis=0, numeric_only=True)
        self.means = self.df.mean(axis=0, numeric_only=True)

    def __getitem__(self, _di: dt.datetime):
        x = self.data.loc[_di-dt.timedelta(hours=self.sample_size):_di-dt.timedelta(hours=1)]
        y = self.targets.loc[_di:_di+dt.timedelta(hours=output_size-1)]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        """
            hours of train and test dates
        """
        return divmod((self.train_tdate - self.train_fdate).seconds, 3600)[0] + \
            divmod((self.test_tdate - self.test_fdate).secondds, 3600)[0]

    @property
    def df(self):
        return self._df

    @df
    def df(self, value):
        self._df = value

    @property
    def train_valid_ratio(self):
        return self._train_valid_ratio

    @train_valid_ratio
    def train_valid_ratio(self, value):
        self._train_valid_ratio = value

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

