import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL
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
        self.target = kwargs.get('target', 'PM10')
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

        self._dates = self._df.index.to_pydatetime()
        self._xs = self._df[self.features]
        self._ys = self._df[[self.target]]

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

        # return X, Y, Y_dates
        return self._scaler.transform(x).to_numpy().astype('float32'), \
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

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


class DNNMeanSeasonalityDataset(Dataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
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

        self._dict_avg_annual = kwargs.get('avg_annual', None)
        self._dict_avg_hourly = kwargs.get('avg_hourly', None)
        self._dict_avg_weekly = kwargs.get('avg_weekly', None)

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

        self._dates = self._df.index.to_pydatetime()

        self.decompose_seasonality()
        new_features = [f + "_yr" if f == self.target else f  for f in self.features]
        self.features = new_features
        #self._xs = self._df[self.features]
        # residual to _xs
        self._xs = self._df[self.features]
        self._ys = self._df[[self.target]]

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

        # return X, Y, Y_dates
        return self._scaler.transform(x).to_numpy().astype('float32'), \
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

    def __len__(self):
        """
        hours of train and test dates

        Returns:
            int: total hours
        """
        duration  = self.tdate - self.fdate - dt.timedelta(hours=(self.output_size + self.sample_size))
        return duration.days * 24 + duration.seconds // 3600

    def decompose_seasonality(self):
        # add keys
        self._df['key_day'] = self._df.index.hour.to_numpy()
        self._df['key_week'] = self._df.index.weekday.to_numpy()
        months = self._df.index.month.to_numpy()
        days = self._df.index.day.to_numpy()
        hours = self._df.index.hour.to_numpy()
        self._df['key_year'] = [
            str(m).zfill(2) + str(d).zfill(2) + str(h).zfill(2)
            for (m, d, h) in zip(months, days, hours)]
        self._df[self.target + "_raw"] = self._df[self.target]

        def datetime2key(dt1, prefix):
            if prefix == 'y':
                return str(dt1.month).zfill(2) + str(dt1.day).zfill(2) + str(dt1.hour).zfill(2)
            elif prefix == 'd':
                return dt1.hour
            elif prefix == 'w':
                return dt1.weekday
            
        def periodic_mean(key, prefix_period, prefix_input, dict_avg):        
            # periodic mean
            if dict_avg == None:
                # only compute on train/valid set
                # test set will use mean of train/valid set which is fed on __init__
                grp_annual = self._df.groupby(key).mean()
                _dict_avg = grp_annual.to_dict()
            self._df[self.target + "_" + prefix_period + "s"] = self._df[self.target]
            self._df[self.target + "_" + prefix_period + "r"] = self._df[self.target]

            for index, row in self._df.iterrows():
                # seasonality
                if prefix_input == None:
                    sea = _dict_avg[self.target][datetime2key(index, prefix_period)]
                else:
                    sea = _dict_avg[self.target + '_' +
                                    prefix_input][datetime2key(index, prefix_period)]
                # annual residual
                res = row[self.target] - sea
                self._df.at[index, self.target + '_' + prefix_period + 's'] = sea
                self._df.at[index, self.target + '_' + prefix_period + 'r'] = res

            return _dict_avg

        # remove (preassumed) seasonality
        self._dict_avg_annual = periodic_mean("key_year", "y", None, self._dict_avg_annual)
        self._dict_avg_daily = periodic_mean("key_day", "d", "yr", self._dict_avg_daily)
        self._dict_avg_weekly = periodic_mean(
            "key_week", "w", "dr", self._dict_avg_weekly)

    # getter only
    @property
    def xs(self):
        return self._xs

    # getter only
    @property
    def ys(self):
        return self._ys

    @property
    def dict_avg_hourly(self):
        return self._dict_avg_hourly

    @dict_avg_hourly.setter
    def dict_avg_hourly(self, value):
        self._dict_avg_hourly = value

    @property
    def dict_avg_annual(self):
        return self._dict_avg_annual

    @dict_avg_annual.setter
    def dict_avg_annual(self, value):
        self._dict_avg_annual = value

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

class DNNARIMAMLPDataset(Dataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
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

        self._dates = self._df.index.to_pydatetime()
        # AR Input
        self._xas = self._df[self.target]
        # ML Input
        self._xms = self._df[self.features]
        self._ys = self._df[[self.target]]

        # self._xms must not be available when creating instance so no kwargs for scaler
        self._scaler = preprocessing.StandardScaler().fit(self._xms)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            i: where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        xa = self._xas.iloc[i:i+self.sample_size]
        xm = self._xms.iloc[i:i+self.sample_size]
        y = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)]

        # target in xm must be replaced after fit to ARIMA
        # residual about to around zero, so I ignored scaling

        return xa.to_numpy().astype('float32'), \
            self._scaler.transform(xm), \
            xm.columns.tolist().index(self.target), \
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

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
    def observed(self):
        return self._observed

    @observed.setter
    def observed(self, observed):
        self._observed = observed

    @property
    def trend(self):
        return self._trend

    @trend.setter
    def trend(self, trend):
        self._trend = trend

    @property
    def seasonal(self):
        return self._seasonal

    @seasonal.setter
    def seasonal(self, seasonal):
        self._seasonal = seasonal

    @property
    def resid(self):
        return self._resid

    @resid.setter
    def resid(self, resid):
        self._resid = resid

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

