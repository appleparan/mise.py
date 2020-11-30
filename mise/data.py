import datetime as dt
from functools import reduce
import hashlib
import inspect
from pathlib import Path
import random
from typing import TypeVar, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.tsatools import detrend
from sklearn import preprocessing

from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png, export_svgs

from torch.utils.data.dataset import Dataset

import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tpl
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

from constants import SEOUL_STATIONS, SEOULTZ
import utils

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

    return df

def load_station(df, code=111123):
    #return df[df['stationCode'] == code]
    return df.query('stationCode == "' + str(code) + '"')

class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        self.station_name = kwargs.get('station_name', '종로구')
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features',
                                   ["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp", "u", "v", "pres", "humid", "prep", "snow"])

        # date when prediction starts, if I need to predict 2018/1/1 1:00 AM, I need more data with size 'sample_size'
        self.fdate = kwargs.get('fdate', dt.datetime(
            2009, 1, 1, 1).astimezone(SEOULTZ))
        # date where prediction starts
        self.tdate = kwargs.get('tdate', dt.datetime(
            2017, 12, 31, 23).astimezone(SEOULTZ))

        # MLP sample_size
        self.sample_size = kwargs.get('sample_size', 48)
        self.batch_size = kwargs.get('batch_size', 32)
        self.output_size = kwargs.get('output_size', 24)
        self._train_valid_ratio = kwargs.get('train_valid_ratio', 0.8)

        # load imputed data from filepath if not provided as dataframe
        filepath = kwargs.get('filepath', Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv"))
        raw_df = pd.read_csv(filepath,
                             index_col=[0, 1],
                             parse_dates=[0])
        # filter by station_name
        self._df = raw_df.query('stationCode == "' +
                                str(SEOUL_STATIONS[self.station_name]) + '"')
        self._df.reset_index(level='stationCode', drop=True, inplace=True)

        # filter by date range including train date
        # i is a start of output, so prepare sample_size
        self._df = self._df[self.fdate -
                            dt.timedelta(hours=self.sample_size):self.tdate]

        self._dates = self._df.index.to_pydatetime()
        self._xs = self._df[self.features]
        self._ys = self._df[[self.target]]

        # self._xs must not be available when creating instance so no kwargs for scaler
        self._scaler = preprocessing.StandardScaler().fit(self._xs)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`
        index i actually indicates start of training data
        but logically, this is a start of output

        why doing this? because the main focus is not where traininng start, is where output starts.

        __len__(df) == fdate - tdate - output_size - sample_size
        actual len(df) == fdate - tdate + sample_size

        so df[i]
        -> physically (i:i + sample_size), (i + sample_size:i + sample_size + output_size)
        -> logcially (i - sample_size:i), (i:i + output_size)

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i:i+self.sample_size]
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size)]

        # return X, Y, Y_dates
        return self._scaler.transform(x).astype('float32'), \
            np.squeeze(y).astype('float32'), \
            self._dates[(i+self.sample_size)
                         :(i+self.sample_size+self.output_size)]

    def __len__(self):
        """
        hours of train and test dates

        __len__(df) == fdate - tdate - output_size - sample_size

        Returns:
            int: total hours
        """
        duration = self.tdate - self.fdate - \
            dt.timedelta(hours=(self.output_size + self.sample_size))
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

class UnivariateDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = [self.target]
        self._xs = self._df[self.features]

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
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size)]

        # return X, Y, Y_dates
        return np.squeeze(x.to_numpy()).astype('float32'), \
            np.squeeze(y).astype('float32'), \
            self._dates[(i+self.sample_size)
                         :(i+self.sample_size+self.output_size)]

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

class UnivariateMeanSeasonalityDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df_raw = self._df.copy()
        # 2D, in Univariate -> (row x 1)
        self._xs = self._df[self.features]
        self._xs_raw = self._xs.copy()
        # 1D
        self._ys = self._df[self.target]
        self._ys.name = self.target
        # convert to DataFrame to apply ColumnTransformer easily
        self._ys = self._ys.to_frame()
        self._ys_raw = self._ys.copy()
        # self._xs must not be available when creating instance so no kwargs for scaler

        # mix ColumnTransformer & Pipeline
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        numeric_pipeline_X = Pipeline(
            [('seasonalitydecompositor', SeasonalityDecompositor_AH(smoothing=True, smoothingFrac=0.05))])

        numeric_pipeline_Y = Pipeline(
            [('seasonalitydecompositor', SeasonalityDecompositor_AH(smoothing=True, smoothingFrac=0.05))])

        # Univariate -> only tself.
        preprocessor_X = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline_X, self.features)])

        preprocessor_Y = numeric_pipeline_Y

        # univariate dataset only needs single pipeline
        self._scaler_X = kwargs.get('scaler_X', preprocessor_X)
        self._scaler_Y = kwargs.get('scaler_Y', preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i:i+self.sample_size, :]
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size), :]
        y_raw = self._ys_raw.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size), :]

        return np.squeeze(x.to_numpy()).astype('float32'), \
            np.squeeze(y.to_numpy()).astype('float32'), \
            np.squeeze(y_raw.to_numpy()).astype('float32'), \
            y.index.to_numpy()

    def preprocess(self, data_dir, png_dir, svg_dir):
        """Compute seasonality and transform by seasonality
        """
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys, y=self._ys)

        # plot
        self.plot_seasonality(data_dir, png_dir, svg_dir)

        self.transform()

    def transform(self):
        """transform xs and ys as a part of preprocess to reduce training time
        """
        self._xs = pd.DataFrame(data=self._scaler_X.transform(self._xs),
            index=self._xs.index, columns=self._xs.columns)
        self._ys = pd.DataFrame(data=self._scaler_Y.transform(self._ys),
            index=self._ys.index, columns=self._ys.columns)

    def inverse_transform(self, Ys: tuple, dates: tuple):
        """inverse_transform accepts DataFrame, but output is always ndarray
        so keep DataFrame structure

        Args:
            Ys(tuple): batched ndarray, the shape of each element is (output_size,)
            dates(tuple): batched ndarray, the shape of each element is (output_size,)

        Return
            Yhats(tuple): batched ndarray, the shape of each element is (output_size,)
        """
        # temporary DataFrame to pass dates
        dfs = list(map(lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
            zip(Ys, dates)))

        # execute pipeline's inverse transform
        _inv_transYs = tuple(map(lambda b: np.squeeze(self._scaler_Y.inverse_transform(b)), dfs))

        # numpy.ndarray
        return _inv_transYs
        # DataFrame: just alter data by transformed data
        #return tuple(map(lambda b: pd.DataFrame(data=b[0],
        #                    index=b[1],
        #                    columns=[self.target]), zip(_inv_transYs, dates)))
        #return _inv_transYs

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        p = self._scaler_X.named_transformers_['num']
        p['seasonalitydecompositor'].plot(self._xs, self.target,
            self.fdate, self.tdate, data_dir, png_dir, svg_dir)

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        return self._scaler_X

    @property
    def scaler_Y(self):
        return self._scaler_Y

class UnivariateRNNDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = [self.target]
        self._xs = self._df[self.features]

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
        y0 = self._xs.iloc[i+self.sample_size-1]
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size)]

        # return X, Y, Y_dates
        return np.squeeze(x).to_numpy().astype('float32'), \
            y0.astype('float32'), \
            np.squeeze(y).astype('float32'), \
            self._dates[(i+self.sample_size)
                         :(i+self.sample_size+self.output_size)]

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

class MultivariateDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xs = self._df[self.features]
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
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size)]

        # return X, Y, Y_dates
        return np.squeeze(self._scaler.transform(x.to_numpy())).astype('float32'), \
            np.squeeze(y).astype('float32'), \
            self._dates[(i+self.sample_size)
                         :(i+self.sample_size+self.output_size)]

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

class MultivariateMeanSeasonalityDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1-step transformer
        # 1. StandardScalerWrapper
        self.features_1 = kwargs.get('features_1',
                                   ["temp", "u", "v", "pres", "humid", "prep", "snow"])
        # 2-step transformer
        # 1. SeasonalityDecompositor
        # 2. StandardScalerWrapper
        self.features_2 = kwargs.get('features_2',
                                   ["SO2", "CO", "O3", "NO2", "PM10", "PM25"])

        self._df_raw = self._df.copy()
        # 2D, in Univariate -> (row x 1)
        self._xs = self._df[self.features]
        self._xs_raw = self._xs.copy()

        # 1D
        self._ys = self._df[self.target]
        self._ys.name = self.target
        # convert to DataFrame to apply ColumnTransformer easily
        self._ys = self._ys.to_frame()
        self._ys_raw = self._ys.copy()
        # self._xs must not be available when creating instance so no kwargs for scaler

        # mix ColumnTransformer & Pipeline
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        # pipeline for regression data

        numeric_pipeline_X_1 = Pipeline(
            [('standardscalerwrapper', StandardScalerWrapper(scaler=StandardScaler()))])

        numeric_pipeline_X_2 = Pipeline(
            [('seasonalitydecompositor',
                SeasonalityDecompositor_AH(smoothing=True, smoothingFrac=0.05)),
             ('standardscalerwrapper', StandardScalerWrapper(scaler=StandardScaler()))])

        numeric_pipeline_Y = Pipeline(
            [('seasonalitydecompositor',
                SeasonalityDecompositor_AH(smoothing=True, smoothingFrac=0.05)),
             ('standardscalerwrapper', StandardScalerWrapper(scaler=StandardScaler()))])

        # Univariate -> only pipline needed
        # Multivariate -> Need ColumnTransformer
        preprocessor_X = ColumnTransformer(
            transformers=[
                ('num_2', numeric_pipeline_X_2, self.features_2),
                ('num_1', numeric_pipeline_X_1, self.features_1)])

        # y is always 1D, so doesn't need ColumnTransfomer
        preprocessor_Y = numeric_pipeline_Y

        self._scaler_X = kwargs.get('scaler_X', preprocessor_X)
        self._scaler_Y = kwargs.get('scaler_Y', preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i:i+self.sample_size, :]
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size), :]
        y_raw = self._ys_raw.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size), :]

        return np.squeeze(x.to_numpy()).astype('float32'), \
            np.squeeze(y.to_numpy()).astype('float32'), \
            np.squeeze(y_raw.to_numpy()).astype('float32'), \
            y.index.to_numpy()

    def preprocess(self, data_dir, png_dir, svg_dir):
        """Compute seasonality and transform by seasonality
        """
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys, y=self._ys)
        # plot
        self.plot_seasonality(data_dir, png_dir, svg_dir)

        self.transform()

    def transform(self):
        self._xs = pd.DataFrame(data=self._scaler_X.transform(self._xs),
            index=self._xs.index, columns=self._xs.columns)
        self._ys = pd.DataFrame(data=self._scaler_Y.transform(self._ys),
            index=self._ys.index, columns=self._ys.columns)

    def inverse_transform(self, Ys: tuple, dates: tuple):
        """inverse_transform accepts DataFrame, but output is always ndarray
        so keep DataFrame structure

        Args:
            Ys(tuple): batched ndarray, the shape of each element is (output_size,)
            dates(tuple): batched ndarray, the shape of each element is (output_size,)

        Return
            Yhats(tuple): batched ndarray, the shape of each element is (output_size,)
        """
        # temporary DataFrame to pass dates
        dfs = list(map(lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
            zip(Ys, dates)))

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) -> StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler -> (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(map(lambda b: np.squeeze(self._scaler_Y.inverse_transform(b)), dfs))

        # numpy.ndarray
        return _inv_transYs
        # DataFrame: just alter data by transformed data
        #return tuple(map(lambda b: pd.DataFrame(data=b[0],
        #                    index=b[1],
        #                    columns=[self.target]), zip(_inv_transYs, dates)))
        #return _inv_transYs

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        p = self._scaler_Y
        p['seasonalitydecompositor'].plot(self._xs, self.target,
            self.fdate, self.tdate, data_dir, png_dir, svg_dir)

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        return self._scaler_X

    @property
    def scaler_Y(self):
        return self._scaler_Y

class MultivariateRNNDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._xs = self._df[self.features]

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
        x_1d = self._ys.iloc[i:(i+self.sample_size)]
        # save initial input as target variable input at last step (single step)
        y0 = self._ys.iloc[i+self.sample_size-1]
        y = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)]

        # return X, Y, Y_dates
        return np.squeeze(x).to_numpy().astype('float32'), \
            np.squeeze(x_1d).to_numpy().astype('float32'), \
            y0.astype('float32'), \
            np.squeeze(y).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

class MultivariateRNNMeanSeasonalityDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1-step transformer
        # 1. StandardScalerWrapper
        self.features = kwargs.get('features',
                        ["temp", "u", "v", "pres", "humid", "prep", "snow",
                        "SO2", "CO", "O3", "NO2", "PM10", "PM25"])
        self.features_1 = kwargs.get('features_1',
                                   ["temp", "u", "v", "pres", "humid", "prep", "snow"])
        # 2-step transformer
        # 1. SeasonalityDecompositor
        # 2. StandardScalerWrapper
        self.features_2 = kwargs.get('features_2',
                                   ["SO2", "CO", "O3", "NO2", "PM10", "PM25"])
        self.features_sea_embed = kwargs.get('features_sea_embed',
                                        ["PM10", "PM25"])

        self._df_raw = self._df.copy()
        # 2D, in Univariate -> (row x 1)
        self._xs = self._df[self.features]
        self._xs_raw = self._xs.copy()
        # 1D
        self._ys = self._df[self.target]
        self._ys.name = self.target
        # convert to DataFrame to apply ColumnTransformer easily
        self._ys = self._ys.to_frame()
        self._ys_raw = self._ys.copy()
        # self._xs must not be available when creating instance so no kwargs for scaler

        # mix ColumnTransformer & Pipeline
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

        # pipeline for regression data
        numeric_pipeline_X_1 = Pipeline(
            [('standardscalerwrapper', StandardScalerWrapper(scaler=StandardScaler()))])

        numeric_pipeline_X_2 = Pipeline(
            [('seasonalitydecompositor',
                SeasonalityDecompositor_AH(smoothing=True, smoothingFrac=0.05)),
             ('standardscalerwrapper', StandardScalerWrapper(scaler=StandardScaler()))])

        numeric_pipeline_Y = Pipeline(
            [('seasonalitydecompositor',
                SeasonalityDecompositor_AH(smoothing=True, smoothingFrac=0.05)),
             ('standardscalerwrapper', StandardScalerWrapper(scaler=StandardScaler()))])

        # Univariate -> only pipline needed
        # Multivariate -> Need ColumnTransformer
        preprocessor_X = ColumnTransformer(
            transformers=[
                ('num_2', numeric_pipeline_X_2, self.features_2),
                ('num_1', numeric_pipeline_X_1, self.features_1)])

        preprocessor_Y = numeric_pipeline_Y

        # univariate dataset only needs single pipeline
        self._scaler_X = kwargs.get('scaler_X', preprocessor_X)
        self._scaler_Y = kwargs.get('scaler_Y', preprocessor_Y)

        self.embed_sea_annual = None
        self.embed_sea_hourly = None

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i:i+self.sample_size, :]
        # save initial input as target variable input at last step (single step)
        y0 = self._ys.iloc[i+self.sample_size-1]
        y = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)]
        y_raw = self._ys_raw.iloc[(i+self.sample_size)
                                 :(i+self.sample_size+self.output_size), :]

        # To embed dates, x and y is DataFrame
        return np.squeeze(x.to_numpy()).astype('float32'), \
            y0.astype('float32'), \
            np.squeeze(y.to_numpy()).astype('float32'), \
            np.squeeze(y_raw.to_numpy()).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

    def preprocess(self, data_dir, png_dir, svg_dir):
        """Compute seasonality and transform by seasonality
        """
        # compute seasonality
        self._scaler_X.fit(self._xs, y=self._xs)
        self._scaler_Y.fit(self._ys, y=self._ys)

        # plot
        self.plot_seasonality(data_dir, png_dir, svg_dir)

        self.transform()

    def transform(self):
        self._xs = pd.DataFrame(data=self._scaler_X.transform(self._xs),
            index=self._xs.index, columns=self._xs.columns)
        self._ys = pd.DataFrame(data=self._scaler_Y.transform(self._ys),
            index=self._ys.index, columns=self._ys.columns)

    def inverse_transform(self, Ys: tuple, dates: tuple):
        """transform accepts DataFrame, but output is always ndarray
        so keep DataFrame structure

        Args:
            Ys(tuple): batched ndarray, the shape of each element is (output_size,)
            dates(tuple): batched ndarray, the shape of each element is (output_size,)

        Return
            Yhats(tuple): batched DataFrames, the shape of each element is (output_size,)
        """
        # temporary DataFrame to pass dates
        dfs = list(map(lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                       zip(Ys, dates)))

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) -> StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler -> (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)

        _inv_transYs = tuple(map(lambda b: np.squeeze(
            self._scaler_Y.inverse_transform(b)), dfs))

        return _inv_transYs

    def build_seasonality(self, X: pd.DataFrame):
        """
            build seasonalities by given DataFrame's DateTimeIndex
        """

        # test
        p = self._scaler_X.named_transformers_['num_2']
        sea_annual = p['seasonalitydecompositor'].build_seasonality(X[self.features_sea_embed], 'y')
        sea_hourly = p['seasonalitydecompositor'].build_seasonality(X[self.features_sea_embed], 'h')

        return sea_annual, sea_hourly

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        p = self._scaler_X.named_transformers_['num_2']
        p['seasonalitydecompositor'].plot(self._xs, self.target,
                                          self.fdate, self.tdate, data_dir, png_dir, svg_dir)

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        return self._scaler_X

    @property
    def scaler_Y(self):
        return self._scaler_Y

class SeasonalityDecompositor_AWH(TransformerMixin, BaseEstimator):
    """Decompose Seasonality

    Attributes
    ----------
    sea_annual : list of dicts or None, shape (n_features,)
        Annual Seasonality
    sea_weekly : list of dicts or None, shape (n_features,)
        Weekly Seasonality
    sea_hourly : list of dicts or None, shape (n_features,)
        Hourly Seasonality
    """
    def __init__(self, sea_annual=None,
        sea_weekly=None, sea_hourly=None, smoothing=True, smoothingFrac=0.05):
        """seasonality data initialization for test data (key-value structure)

        Args:
            sea_annual: list of dicts or None, shape (n_features,)
            sea_weekly: list of dicts or None, shape (n_features,)
            sea_hourly: list of dicts or None, shape (n_features,)
            smoothing: bool

            * test data get input from train data rather than fit itself

        * key format of seasonality elements
            * sea_annual : YYMMDD: string (zero-padded)
            * sea_week : W: integer
            * sea_hour : HH: string (zero-padded)
        """
        # http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
        # Use inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_sesaonality(self, sea_annual, sea_weekly, sea_hourly):
        self.sea_annual = sea_annual
        self.sea_weekly = sea_weekly
        self.sea_hourly = sea_hourly

    def build_seasonality(self, xs: pd.DataFrame, period: str):
        """
        Build seasonality with given DataFrame's DatetimeIndex
        Similar job as `inverse_transform`,
        but instead of sum, just return seasonality
        """
        def _build_seasonality(_xs):
            # to subscript seasonality, get name (feature)
            name = _xs.name

            # method for row
            def _get_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                if period == 'y':
                    return self.sea_annual[name][utils.parse_ykey(idx)]
                elif period == 'w':
                    return self.sea_weekly[name][utils.parse_wkey(idx)]
                elif period == 'h':
                    return self.sea_hourly[name][utils.parse_hkey(idx)]
                else:
                    raise Exception("Invalid period ('y', 'w', 'h'): ", period)

            # index to sum of seasonality
            sea = _xs.index.map(_get_seasonality)
            _xs_df = _xs.to_frame()
            _xs_df['seas'] = sea

            return sea

        #= Y.apply(get_seasonality, axis=1)
        seas = xs.apply(_build_seasonality, 0)

        #raw = Y.apply(add_seasonality, 0)
        return seas

    def compute_seasonality(self, X, return_resid=False):
        """Decompose seasonality of single column of DataFrame

        There are two method to compute hourly residuals from daily another residuals
        1. Subtract First (Wrong)
           * Residual population makes hourly seasonality flat, which is wrong
        2. Seasonality population first (Right)
           * Hourly seasonality comptued correctly
        """
        # for key, convert series to dataframe
        X_h = X.copy().to_frame()
        X_h.columns = ['raw']

        if X_h.isnull().values.any():
            raise Exception(
                'Invalid data in {} column, NULL value found'.format(self.target))
        X_d = X_h.resample('D').mean()

        # 1. Annual Seasonality (mmdd: 0101 ~ 1231)
        # dictionary for key (mmdd) to value (daily average)
        sea_annual, df_sea_annual, resid_annual = utils.periodic_mean(
            X_d, 'raw', 'y', smoothing=self.smoothing, smoothingFrac=self.smoothingFrac)

        if '0229' not in sea_annual:
            raise KeyError("You must include leap year in train data")

        # 2. Weekly Seasonality (w: 0 ~ 6)
        # dictionary for key (w: 0~6) to value (daily average)
        # Weekly seasonality computes seasonality from annual residaul
        sea_weekly, df_sea_weekly, resid_weekly = utils.periodic_mean(
            resid_annual.copy(), 'resid', 'w')

        # 3. Hourly Seasonality (hh: 00 ~ 23)
        # join seasonality to original X_h DataFrame
        # populate seasonality from daily averaged data to hourly data

        ## method to generate key column from index
        def key_md(idx): return ''.join((str(idx.month).zfill(2),
                                         str(idx.day).zfill(2)))
        def key_w(idx): return str(idx.dayofweek)
        def key_h(idx): return str(idx.hour).zfill(2)

        ## Add column from index for key
        df_sea_weekly['key_w'] = df_sea_weekly.index
        df_sea_annual['key_md'] = df_sea_annual.index

        X_h['key_md'] = X_h.index.map(key_md)
        X_h['key_w'] = X_h.index.map(key_w)
        X_h['key_h'] = X_h.index.map(key_h)

        ## compute hourly populated seasonality from daily residuals
        X_h_spop = df_sea_annual.merge(X_h, how='left',
                                         on='key_md', left_index=True, validate="1:m").dropna()
        X_h_spop = X_h_spop.rename(columns={'sea': 'sea_annual'})
        X_h_spop = df_sea_weekly.merge(X_h_spop, how='left',
                                         on='key_w', left_index=True, validate="1:m").dropna()
        X_h_spop = X_h_spop.rename(columns={'sea': 'sea_weekly'})

        ## new hourly residual column from daily residuals
        X_h_spop['resid_d'] = X_h_spop['raw'] - X_h_spop['sea_annual'] - X_h_spop['sea_weekly']

        ## Check no missing values
        if len(X_h_spop.index) != len(X_h.index):
            raise Exception(
                "Merge Error: something missing when populate daily averaged DataFrame")

        ## Compute hourly seasonality
        sea_hourly, df_sea_hourly, resid_hourly = utils.periodic_mean(
            X_h_spop.copy(), 'resid_d', 'h')

        ## Add column from index for key
        df_sea_hourly['key_h'] = df_sea_hourly.index

        ## merge hourly seasonality to orignal hourly DataFram
        X_h_hourly = df_sea_hourly.merge(X_h, how='left',
                                         on='key_h', left_index=True, validate="1:m").dropna()
        X_h_hourly = X_h_hourly.rename(columns={'sea': 'sea_hourly'})
        ## Subtract annual and weekly seasonality
        X_h_spop['sea_hourly']= X_h_hourly['sea_hourly']
        X_h_spop['resid'] = X_h_spop['resid_d'] - X_h_hourly['sea_hourly']
        ## Sort by DateTimeIndex
        X_h_spop = X_h_spop.sort_index()

        # 3.2 drop key columns and intermediate residual column
        X_h_spop.drop('resid_d', axis='columns')
        X_h_spop.drop('key_md', axis='columns')
        X_h_spop.drop('key_w', axis='columns')
        X_h_spop.drop('key_h', axis='columns')

        if return_resid == True:
            return resid_annual, resid_weekly, X_h_spop, resid_hourly

        return sea_annual, sea_weekly, sea_hourly

    def fit(self, X: pd.DataFrame, y=None):
        """Compute seasonality,
        Computed residuals in `fit` will be dropped.
        In  `transform` residuals are computed again from seasonality from `fit`

        y is not used, but added for Pipeline compatibility
        ref: https://scikit-learn.org/stable/developers/develop.html

        Args:
            X_h: DataFrame which have datetime as index

        Return:
            None
        """
        # Decompose seasonality if there is no predefined seasonality (train/valid set)
        if self.sea_annual != None and self.sea_weekly != None and \
            self.sea_hourly != None:
            # if test set
            return self

        # apply function to column-wise
        #self.sea_annual, self.sea_weekly, self.sea_hourly = \
        seas = X.apply(self.compute_seasonality, 0).to_dict(orient='records')
        self.sea_annual = seas[0]
        self.sea_weekly = seas[1]
        self.sea_hourly = seas[2]

        return self

    def transform(self, X: pd.DataFrame):
        """Recompute residual by subtracting seasonality from X
        shape of X is (sample_size, num_features)

        Args:
            X (pd.DataFrame):
                must have DataTimeIndex to get dates

        Returns:
            resid (np.ndarray):
                dates now doesn't be needed anymore
        """
        def sub_seasonality(_X):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _X.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)] + \
                    self.sea_weekly[name][utils.parse_wkey(idx)] + \
                    self.sea_hourly[name][utils.parse_hkey(idx)]
            # index to sum of seasonality
            seas = _X.index.map(_sum_seasonality)
            _X_df = _X.to_frame()
            _X_df['seas'] = -seas

            _X_sum = _X_df.sum(axis = 'columns')
            _X_sum.name = name

            return _X_sum

        resid = X.apply(sub_seasonality, 0)

        return resid.to_numpy()

    def inverse_transform(self, Y: pd.DataFrame):
        """Compose value from residuals
        If smoothed annual seasonality is used,
        compsed value might be different with original value

        shape of Y is (output_size, 1) because self.target is always one target

        Args:
            Y (pd.DataFrame):
                pd.DataFrame must have DataTimeIndex to get dates

        Returns:
            raw (ndarray):
        """

        def add_seasonality(_Y):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _Y.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)] + \
                    self.sea_weekly[name][utils.parse_wkey(idx)] + \
                    self.sea_hourly[name][utils.parse_hkey(idx)]

            # index to sum of seasonality
            seas = _Y.index.map(_sum_seasonality)
            _Y_df = _Y.to_frame()
            _Y_df['seas'] = seas

            _Y_sum = _Y_df.sum(axis = 'columns')
            _Y_sum.name = name

            return _Y_sum

        # there is no way to pass row object of Series,
        # so I pass index and use Series.get() with closure
        #raw = Y.index.map(add_seasonality)
        raw = Y.apply(add_seasonality, 0)

        return raw.to_numpy()

    # utility functions
    def ydt2key(self, d): return str(d.astimezone(SEOULTZ).month).zfill(
            2) + str(d.astimezone(SEOULTZ).day).zfill(2)
    def wdt2key(self, d): return str(d.astimezone(SEOULTZ).weekday())
    def hdt2key(self, d): return str(d.astimezone(SEOULTZ).hour).zfill(2)
    def confint_idx(self, acf, confint):
        for i, (a, [c1, c2]) in enumerate(zip(acf, confint)):
            if c1 - a < a and a < c2 - a:
                return i

    def plot_raw_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        nlags=15):

        #yr = [df.loc[y, target] -
        #      self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        ys = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        # 95% confidance intervals
        ys_acf, confint, qstat, pvalues = sm.tsa.acf(
            ys, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(ys_acf[0:self.confint_idx(ys_acf, confint)])
        print("Raw Conf Int  : ", self.confint_idx(ys_acf, confint))
        print("Raw Int Scale : ", intscale)
        #print("Raw qstat : ", qstat)
        #print("Raw pvalues : ", pvalues)

        csv_path = data_dir / ("acf_raw_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': [i for i in range(len(ys_acf))], 'yr_acf': ys_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        p1 = figure(title="Autocorrelation of " + target)
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(ys_acf)), 1.1)
        p1.line([i*24 for i in range(len(ys_acf))], ys_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(ys, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(ys_acf, confint), intscale

    def plot_annual(self, df, target, data_dir, png_dir, svg_dir,
        target_year=2016, smoothing=True):
        # annual (pick 1 year)
        dt1 = dt.datetime(year=target_year, month=1, day=1, hour=0)
        dt2 = dt.datetime(year=target_year, month=12, day=31, hour=23)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])

        # if not smoothing, just use predecomposed seasonality
        ys = [self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        if smoothing:
            _sea_annual_nonsmooth, _, _ = \
                utils.periodic_mean(df, target, 'y', smoothing=False)
            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            ys_vanilla = [_sea_annual_nonsmooth[self.ydt2key(y)] for y in year_range]
            ys_smooth = [self.sea_annual[target][self.ydt2key(y)] for y in year_range]

        csv_path = data_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".csv")
        df_year = pd.DataFrame.from_dict(
            {'date': year_range, 'ys': ys})
        if smoothing:
            df_year = pd.DataFrame.from_dict(
                {'date': year_range, 'ys_vanilla': ys_vanilla, 'ys_smooth': ys_smooth})
        df_year.set_index('date', inplace=True)
        df_year.to_csv(csv_path)

        png_path = png_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".svg")

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                   months="%m/%d %H:%M",
                                                   hours="%m/%d %H:%M",
                                                   minutes="%m/%d %H:%M")
        p1.line(year_range_plt, ys, line_color="dodgerblue", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        if smoothing:
            # 1. smooth
            png_path = png_dir / ("annual_seasonality_(smooth)_" +
                                   dt1.strftime("%Y%m%d") + "_" +
                                   dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(smooth)_" +
                                   dt1.strftime("%Y%m%d") + "_" +
                                   dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth, line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 2. vanila
            png_path = png_dir / ("annual_seasonality_(vanilla)_" +
                                   dt1.strftime("%Y%m%d") + "_" +
                                   dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(vanilla)_" +
                                   dt1.strftime("%Y%m%d") + "_" +
                                   dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Raw)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_vanilla, line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 3. smooth + vanila
            png_path = png_dir / ("annual_seasonality_(both)_" +
                                   dt1.strftime("%Y%m%d") + "_" +
                                   dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(both)_" +
                                   dt1.strftime("%Y%m%d") + "_" +
                                   dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth & Vanilla)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth, line_color="dodgerblue",
                    line_width=2, legend_label="smooth")
            p1.line(year_range_plt, ys_vanilla, line_color="lightcoral",
                    line_width=2, legend_label="vanilla")
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

    def plot_annual_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
        target_year=2016, nlags=15, smoothing=True):
        ## residual autocorrelation by tsa.acf
        year_range = pd.date_range(start=fdate, end=tdate, freq='D').tz_convert(SEOULTZ).tolist()
        # set hour to 0
        year_range = [y.replace(hour=0) for y in year_range]

        #yr = [df.loc[y, target] -
        #      self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        yr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()

        # 95% confidance intervals
        yr_acf, confint, qstat, pvalues = sm.tsa.acf(yr, qstat=True, alpha=0.05, nlags=nlags)

        # I need "hours" as unit, so multiply 24
        intscale = sum(yr_acf[0:self.confint_idx(yr_acf, confint)]) * 24
        print("Annual Conf Int  : ", self.confint_idx(yr_acf, confint))
        print("Annual Int Scale : ", intscale)
        print("Annual qstat : ", qstat)
        print("Annual pvalues : ", pvalues)

        csv_path = data_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': [i*24 for i in range(len(yr_acf))], 'yr_acf': yr_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".svg")
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line([i*24 for i in range(len(yr_acf))], yr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf(tpl)_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".svg")
        fig = tpl.plot_acf(yr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(yr_acf, confint), intscale

    def plot_weekly(self, df, target, data_dir, png_dir, svg_dir,
        target_year=2016, target_week=10):

        # Set datetime range
        target_dt_str = str(target_year) + ' ' + str(target_week)
        ## ISO 8601 starts weekday from Monday (1)
        dt1 = dt.datetime.strptime(target_dt_str + ' 1', "%Y %W %w")
        ## ISO 8601 ends weekday to Sunday (0)
        dt2 = dt.datetime.strptime(target_dt_str + ' 0', "%Y %W %w")
        ## set dt2 time as Sunday 23:59
        dt2 = dt2.replace(hour=23, minute=59)
        week_range = pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ).tolist()
        week_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])

        # Compute Weekly seasonality
        ws = [self.sea_weekly[target][self.wdt2key(w)] for w in week_range]

        csv_path = data_dir / ("weekly_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".csv")
        df_week = pd.DataFrame.from_dict(
            {'day': week_range, 'ws': ws})
        df_week.set_index('day', inplace=True)
        df_week.to_csv(csv_path)

        png_path = png_dir / ("weekly_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("weekly_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".svg")

        p2 = figure(title="Weekly Seasonality")
        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.xaxis.axis_label = "day"
        p2.xaxis.formatter = DatetimeTickFormatter(
            days="%a")
        p2.line(week_range_plt, ws, line_color="dodgerblue", line_width=2)
        export_png(p2, filename=png_path)
        p2.output_backend = "svg"
        export_svgs(p2, filename=str(svg_path))

    def plot_weekly_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
        target_year=2016, target_week=10, nlags=15):
        """
            Residual autocorrelation by tsa.acf
        """
        # Set datetime range
        target_dt_str = str(target_year) + ' ' + str(target_week)
        week_range = pd.date_range(
            start=fdate, end=tdate, freq='D').tz_convert(SEOULTZ).tolist()
        # set hour to 0
        week_range = [w.replace(hour=0) for w in week_range]

        #wr = [df.loc[w, target] -
        #      self.sea_weekly[target][self.wdt2key(w)] for w in week_range]
        wr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        wr_acf, confint, qstat, pvalues = sm.tsa.acf(wr, qstat=True, alpha=0.05, nlags=nlags)

        # I need "hours" as unit, so multiply 24
        intscale = sum(wr_acf[0:self.confint_idx(wr_acf, confint)]) * 24
        print("Weekly Conf Int  : ", self.confint_idx(wr_acf, confint))
        print("Weekly Int Scale : ", intscale)
        print("Weekly qstat : ", qstat)
        print("Weekly pvalues : ", pvalues)

        csv_path = data_dir / ("acf_weekly_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".csv")
        df_week_acf = pd.DataFrame.from_dict(
            {'lags': [i*24 for i in range(len(wr_acf))], 'wr_acf': wr_acf})
        df_week_acf.set_index('lags', inplace=True)
        df_week_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_weekly_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf_weekly_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".svg")
        p2 = figure(title="Autocorrelation of Weekly Residual")
        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.xaxis.axis_label = "lags"
        p2.yaxis.bounds = (min(0, min(wr_acf)), 1.1)
        p2.line([i*24 for i in range(len(wr_acf))], wr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p2, filename=png_path)
        p2.output_backend = "svg"
        export_svgs(p2, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_weekly_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf(tpl)_weekly_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".svg")
        fig = tpl.plot_acf(wr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(wr_acf, confint), intscale

    def plot_hourly(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, target_month=5, target_day=1):

        dt1 = dt.datetime(year=target_year, month=target_month, day=target_day,
            hour=0)
        dt2 = dt.datetime(year=target_year, month=target_month, day=target_day,
            hour=23)
        hour_range = pd.date_range(start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        hour_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()

        # Compute Hourly seasonality
        hs = [self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]

        csv_path = data_dir / ("hourly_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_hour = pd.DataFrame.from_dict(
            {'hour': hour_range, 'hs': hs})
        df_hour.set_index('hour', inplace=True)
        df_hour.to_csv(csv_path)

        png_path = png_dir / ("hourly_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("hourly_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".svg")

        p3 = figure(title="Hourly Seasonality")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "hour"
        p3.xaxis.formatter = DatetimeTickFormatter(hours="%H")
        p3.line(hour_range_plt, hs, line_color="dodgerblue", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

    def plot_hourly_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, target_month=5, target_day=1, nlags=24*15):

        hour_range = pd.date_range(
            start=fdate, end=tdate, freq="1H").tz_convert(SEOULTZ).tolist()

        #hr = [df.loc[h, target] -
        #      self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]
        hr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        hr_acf, confint, qstat, pvalues = sm.tsa.acf(hr, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(hr_acf[0:self.confint_idx(hr_acf, confint)])
        print("Hourly Conf Int  : ", self.confint_idx(hr_acf, confint))
        print("Hourly Int Scale : ", intscale)
        #print("Hourly qstat : ", qstat)
        #print("Hourly pvalues : ", pvalues)

        csv_path = data_dir / ("acf_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".csv")
        df_hour_acf = pd.DataFrame.from_dict(
            {'lags': range(len(hr_acf)), 'hr_acf': hr_acf})
        df_hour_acf.set_index('lags', inplace=True)
        df_hour_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".svg")
        p3 = figure(title="Autocorrelation of Hourly Residual")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(hr_acf)), 1.1)
        p3.line(range(len(hr_acf)), hr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(hr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(hr_acf, confint), intscale

    def plot(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
             target_year=2016, target_week=10, target_month=5, target_day=1,
             nlags=7, smoothing=True):
        """
        Plot seasonality itself and autocorrelation of residuals

        Args:
            df (DataFrame):
                Hourly DataFrame that includes whole data

            target (str):
                column name of DataFrame

            fdate (DateTime):
                acf plot from range between fdate ~ tdate

            tdate (DateTime):
                acf plot from range between fdate ~ tdate

            data_dir (Path):
                Directory location to save csv file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            target_year (int):
                specified year for plot

            target_week (int):
                specified week number in target_year for plot

            nlags (int):
                nlags for autocorrelation (days)

            smoothing (bool):
                is smoothing used?

        Return:
            None
        """

        # filter by dates
        df = df[(df.index > fdate) & (df.index < tdate)]
        # daily averaged
        df_d = df.resample('D').mean()

        df_resid_annual, df_resid_weekly, df_resid_weekly_pop, df_resid_hourly = \
            self.compute_seasonality(df.loc[:, target], return_resid=True)
        df_resid_annual.rename(columns={'resid': target}, inplace=True)
        df_resid_weekly.rename(columns={'resid': target}, inplace=True)
        df_resid_weekly_pop.rename(columns={'resid': target}, inplace=True)
        df_resid_hourly.rename(columns={'resid': target}, inplace=True)

        dict_corr_dist = {}

        r_confint, r_intscale = self.plot_raw_acf(df, target, df.index[0], df.index[-1],
                                                  data_dir, png_dir, svg_dir, nlags=nlags*24)

        # raw seasonality plot needs to re-compute, so pass original df
        self.plot_annual(df, target, data_dir, png_dir, svg_dir,
            target_year=2016, smoothing=self.smoothing)
        a_confint, a_intscale = self.plot_annual_acf(df_resid_annual, target, df.index[0], df.index[-1],
            data_dir, png_dir, svg_dir,
            nlags=nlags, smoothing=self.smoothing)

        # df in weekly and hourly seasonality plot is a just dummy variable
        # to make consistent with other plot methods
        self.plot_weekly(df_resid_annual, target, data_dir, png_dir, svg_dir,
            target_year=target_year, target_week=target_week)
        w_confint, w_intscale = self.plot_weekly_acf(df_resid_weekly, target, df.index[0], df.index[-1],
            data_dir, png_dir, svg_dir,
            target_year=target_year, target_week=target_week, nlags=nlags)

        # df in weekly and hourly seasonality plot is a just dummy variable
        # to make consistent with other plot methods
        self.plot_hourly(df_resid_weekly_pop, target, data_dir, png_dir, svg_dir,
            target_year=target_year, target_month=target_month, target_day=target_day)
        h_confint, h_intscale = self.plot_hourly_acf(df_resid_hourly, target, df.index[0], df.index[-1],
                            data_dir, png_dir, svg_dir,
                            target_year=target_year, target_month=target_month, target_day=target_day,
                            nlags=nlags*24)

        dict_corr_dist = {
            "annual": {
                "confint": a_confint,
                "intscale": a_intscale},
            "weekly": {
                "confint": w_confint,
                "intscale": w_intscale},
            "hourly": {
                "confint": h_confint,
                "intscale": h_intscale}
        }

        with open(data_dir / 'intscale.json', 'w') as f:
            print(dict_corr_dist, file=f)

class SeasonalityDecompositor_AH(TransformerMixin, BaseEstimator):
    """Decompose Seasonality

    Attributes
    ----------
    sea_annual : list of dicts or None, shape (n_features,)
        Annual Seasonality
    sea_weekly : list of dicts or None, shape (n_features,)
        Weekly Seasonality
    sea_hourly : list of dicts or None, shape (n_features,)
        Hourly Seasonality
    """

    def __init__(self, sea_annual=None,
                 sea_hourly=None, smoothing=True, smoothingFrac=0.05):
        """seasonality data initialization for test data (key-value structure)

        Args:
            sea_annual: list of dicts or None, shape (n_features,)
            sea_weekly: list of dicts or None, shape (n_features,)
            sea_hourly: list of dicts or None, shape (n_features,)
            smoothing: bool

            * test data get input from train data rather than fit itself

        * key format of seasonality elements
            * sea_annual : YYMMDD: string (zero-padded)
            * sea_week : W: integer
            * sea_hour : HH: string (zero-padded)
        """
        # http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
        # Use inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_sesaonality(self, sea_annual, sea_hourly):
        self.sea_annual = sea_annual
        self.sea_hourly = sea_hourly

    def build_seasonality(self, xs: pd.DataFrame, period: str):
        """
        Build seasonality with given DataFrame's DatetimeIndex
        Similar job as `inverse_transform`,
        but instead of sum, just return seasonality
        """
        def _build_seasonality(_xs):
            # to subscript seasonality, get name (feature)
            name = _xs.name

            # method for row
            def _get_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                if period == 'y':
                    return self.sea_annual[name][utils.parse_ykey(idx)]
                elif period == 'h':
                    return self.sea_hourly[name][utils.parse_hkey(idx)]
                else:
                    raise Exception("Invalid period ('y', 'h'): ", period)

            # index to sum of seasonality
            sea = _xs.index.map(_get_seasonality)
            _xs_df = _xs.to_frame()
            _xs_df['seas'] = sea

            return sea

        #= Y.apply(get_seasonality, axis=1)
        seas = xs.apply(_build_seasonality, 0)

        #raw = Y.apply(add_seasonality, 0)
        return seas

    def compute_seasonality(self, X, return_resid=False):
        """Decompose seasonality of single column of DataFrame

        There are two method to compute hourly residuals from daily another residuals
        1. Subtract First (Wrong)
           * Residual population makes hourly seasonality flat, which is wrong
        2. Seasonality population first (Right)
           * Hourly seasonality comptued correctly
        """
        # for key, convert series to dataframe
        X_h = X.copy().to_frame()
        X_h.columns = ['raw']

        if X_h.isnull().values.any():
            raise Exception(
                'Invalid data in {} column, NULL value found'.format(self.target))
        X_d = X_h.resample('D').mean()

        # 1. Annual Seasonality (mmdd: 0101 ~ 1231)
        # dictionary for key (mmdd) to value (daily average)
        sea_annual, df_sea_annual, resid_annual = utils.periodic_mean(
            X_d, 'raw', 'y', smoothing=self.smoothing, smoothingFrac=self.smoothingFrac)

        if '0229' not in sea_annual:
            raise KeyError("You must include leap year in train data")

        # 2. Hourly Seasonality (hh: 00 ~ 23)
        # join seasonality to original X_h DataFrame
        # populate seasonality from daily averaged data to hourly data

        ## method to generate key column from index
        def key_md(idx): return ''.join((str(idx.month).zfill(2),
                                         str(idx.day).zfill(2)))
        def key_h(idx): return str(idx.hour).zfill(2)

        # Add column for key
        df_sea_annual['key_md'] = df_sea_annual.index
        X_h['key_md'] = X_h.index.map(key_md)
        X_h['key_h'] = X_h.index.map(key_h)

        ## compute hourly populated seasonality from daily residuals
        X_h_spop = df_sea_annual.merge(X_h, how='left',
                                on='key_md', left_index=True, validate="1:m").dropna()
        X_h_spop = X_h_spop.rename(columns={'sea': 'sea_annual'})

        ## Check no missing values
        if len(X_h_spop.index) != len(X_h.index):
            raise Exception(
                "Merge Error: something missing when populate daily averaged DataFrame")

        # Subtract annual seasonality
        X_h_spop['resid_d'] = X_h_spop['raw'] - X_h_spop['sea_annual']

        # Compute seasonality, use .copy not to violate targeting DataFrame
        sea_hourly, df_sea_hourly, resid_hourly = utils.periodic_mean(
            X_h_spop.copy(), 'resid_d', 'h')

        ## Add column from index for key
        df_sea_hourly['key_h'] = df_sea_hourly.index

        ## merge hourly seasonality to orignal hourly DataFram
        X_h_hourly = df_sea_hourly.merge(X_h, how='left',
                                         on='key_h', left_index=True, validate="1:m").dropna()
        X_h_spop['sea_hourly'] = X_h_hourly['sea']
        # Subtract annual seasonality
        X_h_spop['resid'] = X_h_spop['resid_d'] - X_h_hourly['sea']

        # 3.2 drop key columns and intermediate residual column
        X_h_spop.drop('resid_d', axis='columns')
        X_h_spop.drop('key_md', axis='columns')
        X_h_spop.drop('key_h', axis='columns')

        if return_resid == True:
            return resid_annual, X_h_spop, resid_hourly

        return sea_annual, sea_hourly

    def fit(self, X: pd.DataFrame, y=None):
        """Compute seasonality,
        Computed residuals in `fit` will be dropped.
        In  `transform` residuals are computed again from seasonality from `fit`

        y is not used, but added for Pipeline compatibility
        ref: https://scikit-learn.org/stable/developers/develop.html

        Args:
            X_h: DataFrame which have datetime as index

        Return:
            None
        """
        # Decompose seasonality if there is no predefined seasonality (train/valid set)
        if self.sea_annual != None and self.sea_hourly != None:
            # if test set
            return self

        # apply function to column-wise
        #self.sea_annual, self.sea_weekly, self.sea_hourly = \
        seas = X.apply(self.compute_seasonality, 0).to_dict(orient='records')
        self.sea_annual = seas[0]
        self.sea_hourly = seas[1]

        return self

    def transform(self, X: pd.DataFrame):
        """Recompute residual by subtracting seasonality from X
        shape of X is (sample_size, num_features)

        Args:
            X (pd.DataFrame):
                must have DataTimeIndex to get dates

        Returns:
            resid (np.ndarray):
                dates now doesn't be needed anymore
        """
        def sub_seasonality(_X):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _X.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)] + \
                    self.sea_hourly[name][utils.parse_hkey(idx)]
            # index to sum of seasonality
            seas = _X.index.map(_sum_seasonality)
            _X_df = _X.to_frame()
            _X_df['seas'] = -seas

            _X_sum = _X_df.sum(axis='columns')
            _X_sum.name = name

            return _X_sum

        resid = X.apply(sub_seasonality, 0)

        return resid.to_numpy()

    def inverse_transform(self, Y: pd.DataFrame):
        """Compose value from residuals
        If smoothed annual seasonality is used,
        compsed value might be different with original value

        shape of Y is (output_size, 1) because self.target is always one target

        Args:
            Y (pd.DataFrame):
                pd.DataFrame must have DataTimeIndex to get dates

        Returns:
            raw (ndarray):
        """

        def add_seasonality(_Y):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _Y.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """

                return \
                    self.sea_annual[name][utils.parse_ykey(idx)] + \
                    self.sea_hourly[name][utils.parse_hkey(idx)]

            # index to sum of seasonality
            seas = _Y.index.map(_sum_seasonality)
            _Y_df = _Y.to_frame()
            _Y_df['seas'] = seas

            _Y_sum = _Y_df.sum(axis='columns')
            _Y_sum.name = name

            return _Y_sum

        # there is no way to pass row object of Series,
        # so I pass index and use Series.get() with closure
        #raw = Y.index.map(add_seasonality)
        raw = Y.apply(add_seasonality, 0)

        return raw.to_numpy()

    # utility functions
    def ydt2key(self, d): return str(d.astimezone(SEOULTZ).month).zfill(
        2) + str(d.astimezone(SEOULTZ).day).zfill(2)

    def hdt2key(self, d): return str(d.astimezone(SEOULTZ).hour).zfill(2)

    def confint_idx(self, acf, confint):
        for i, (a, [c1, c2]) in enumerate(zip(acf, confint)):
            if c1 - a < a and a < c2 - a:
                return i

    def plot_raw_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        nlags=15):

        #yr = [df.loc[y, target] -
        #      self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        ys = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        # 95% confidance intervals
        ys_acf, confint, qstat, pvalues = sm.tsa.acf(
            ys, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(ys_acf[0:self.confint_idx(ys_acf, confint)])
        print("Raw Conf Int  : ", self.confint_idx(ys_acf, confint))
        print("Raw Int Scale : ", intscale)
        #print("Raw qstat : ", qstat)
        #print("Raw pvalues : ", pvalues)

        csv_path = data_dir / ("acf_raw_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': [i for i in range(len(ys_acf))], 'yr_acf': ys_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        p1 = figure(title="Autocorrelation of " + target)
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(ys_acf)), 1.1)
        p1.line([i for i in range(len(ys_acf))], ys_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_raw_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(ys, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(ys_acf, confint), intscale

    def plot_annual(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, smoothing=True):
        # annual (pick 1 year)
        dt1 = dt.datetime(year=target_year, month=1, day=1, hour=0)
        dt2 = dt.datetime(year=target_year, month=12, day=31, hour=23)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])

        # if not smoothing, just use predecomposed seasonality
        ys = [self.sea_annual[target][self.ydt2key(y)] for y in year_range]

        if smoothing:
            _sea_annual_nonsmooth, _, _ = \
                utils.periodic_mean(df, target, 'y', smoothing=False)

            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            ys_vanilla = [
                _sea_annual_nonsmooth[self.ydt2key(y)] for y in year_range]
            ys_smooth = [self.sea_annual[target]
                         [self.ydt2key(y)] for y in year_range]

        csv_path = data_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".csv")
        df_year = pd.DataFrame.from_dict(
            {'date': year_range, 'ys': ys})
        if smoothing:
            df_year = pd.DataFrame.from_dict(
                {'date': year_range, 'ys_vanilla': ys_vanilla, 'ys_smooth': ys_smooth})
        df_year.set_index('date', inplace=True)
        df_year.to_csv(csv_path)

        png_path = png_dir / ("annual_seasonality_" +
                              dt1.strftime("%Y%m%d") + "_" +
                              dt2.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("annual_seasonality_" +
                              dt1.strftime("%Y%m%d") + "_" +
                              dt2.strftime("%Y%m%d") + ".svg")

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                   months="%m/%d %H:%M",
                                                   hours="%m/%d %H:%M",
                                                   minutes="%m/%d %H:%M")
        p1.line(year_range_plt, ys, line_color="dodgerblue", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        if smoothing:
            # 1. smooth
            png_path = png_dir / ("annual_seasonality_(smooth)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(smooth)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 2. vanila
            png_path = png_dir / ("annual_seasonality_(vanilla)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(vanilla)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Raw)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_vanilla,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 3. smooth + vanila
            png_path = png_dir / ("annual_seasonality_(both)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(both)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth & Vanilla)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth, line_color="dodgerblue",
                    line_width=2, legend_label="smooth")
            p1.line(year_range_plt, ys_vanilla, line_color="lightcoral",
                    line_width=2, legend_label="vanilla")
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

    def plot_annual_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, nlags=15, smoothing=True):
        ## residual autocorrelation by tsa.acf
        year_range = pd.date_range(
            start=fdate, end=tdate, freq='D').tz_convert(SEOULTZ).tolist()
        # set hour to 0
        year_range = [y.replace(hour=0) for y in year_range]

        #yr = [df.loc[y, target] -
        #      self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        yr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        # 95% confidance intervals
        yr_acf, confint, qstat, pvalues = sm.tsa.acf(
            yr, qstat=True, alpha=0.05, nlags=nlags)

        # I need "hours" as unit, so multiply 24
        intscale = sum(yr_acf[0:self.confint_idx(yr_acf, confint)]) * 24
        print("Annual Conf Int  : ", self.confint_idx(yr_acf, confint))
        print("Annual Int Scale : ", intscale)
        print("Annual qstat : ", qstat)
        print("Annual pvalues : ", pvalues)

        csv_path = data_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': [i*24 for i in range(len(yr_acf))], 'yr_acf': yr_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".svg")
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line([i*24 for i in range(len(yr_acf))], yr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf(tpl)_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".svg")
        fig = tpl.plot_acf(yr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(yr_acf, confint), intscale

    def plot_hourly(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, target_month=5, target_day=1):

        dt1 = dt.datetime(year=target_year, month=target_month, day=target_day,
                          hour=0)
        dt2 = dt.datetime(year=target_year, month=target_month, day=target_day,
                          hour=23)
        hour_range = pd.date_range(
            start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        hour_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()

        # Compute Hourly seasonality
        hs = [self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]

        csv_path = data_dir / ("hourly_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_hour = pd.DataFrame.from_dict(
            {'hour': hour_range, 'hs': hs})
        df_hour.set_index('hour', inplace=True)
        df_hour.to_csv(csv_path)

        png_path = png_dir / ("hourly_seasonality_" +
                              dt1.strftime("%Y%m%d%H") + "_" +
                              dt2.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("hourly_seasonality_" +
                              dt1.strftime("%Y%m%d%H") + "_" +
                              dt2.strftime("%Y%m%d%H") + ".svg")

        p3 = figure(title="Hourly Seasonality")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "hour"
        p3.xaxis.formatter = DatetimeTickFormatter(hours="%H")
        p3.line(hour_range_plt, hs, line_color="dodgerblue", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

    def plot_hourly_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, target_month=5, target_day=1, nlags=24*15):

        hour_range = pd.date_range(
            start=fdate, end=tdate, freq="1H").tz_convert(SEOULTZ).tolist()

        #hr = [df.loc[h, target] -
        #      self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]
        hr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        hr_acf, confint, qstat, pvalues = sm.tsa.acf(
            hr, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(hr_acf[0:self.confint_idx(hr_acf, confint)])
        print("Hourly Conf Int  : ", self.confint_idx(hr_acf, confint))
        print("Hourly Int Scale : ", intscale)
        #print("Hourly qstat : ", qstat)
        #print("Hourly pvalues : ", pvalues)

        csv_path = data_dir / ("acf_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".csv")
        df_hour_acf = pd.DataFrame.from_dict(
            {'lags': range(len(hr_acf)), 'hr_acf': hr_acf})
        df_hour_acf.set_index('lags', inplace=True)
        df_hour_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        p3 = figure(title="Autocorrelation of Hourly Residual")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(hr_acf)), 1.1)
        p3.line(range(len(hr_acf)), hr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(hr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(hr_acf, confint), intscale

    def plot(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
             target_year=2016, target_week=10, target_month=5, target_day=1,
             nlags=7, smoothing=True):
        """
        Plot seasonality itself and autocorrelation of residuals
        Plot should not affected by other transformers, so recompute every seasonality when needed

        Args:
            df (DataFrame):
                Hourly DataFrame that includes whole data

            target (str):
                column name of DataFrame

            fdate (DateTime):
                acf plot from range between fdate ~ tdate

            tdate (DateTime):
                acf plot from range between fdate ~ tdate

            data_dir (Path):
                Directory location to save csv file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            target_year (int):
                specified year for plot

            target_week (int):
                specified week number in target_year for plot

            nlags (int):
                nlags for autocorrelation (days)

            smoothing (bool):
                is smoothing used?

        Return:
            None
        """

        # filter by dates
        df = df[(df.index > fdate) & (df.index < tdate)]
        # daily averaged
        df_d = df.resample('D').mean()

        df_resid_annual, df_resid_annual_pop, df_resid_hourly = \
            self.compute_seasonality(df.loc[:, target], return_resid=True)
        df_resid_annual.rename(columns={'resid': target}, inplace=True)
        df_resid_annual_pop.rename(columns={'resid': target}, inplace=True)
        df_resid_hourly.rename(columns={'resid': target}, inplace=True)

        dict_corr_dist = {}

        r_confint, r_intscale = self.plot_raw_acf(df, target, df.index[0], df.index[-1],
                                                  data_dir, png_dir, svg_dir, nlags=nlags*24)

        # raw seasonality plot needs to re-compute, so pass original df
        self.plot_annual(df, target, data_dir, png_dir, svg_dir,
                         target_year=2016, smoothing=self.smoothing)
        a_confint, a_intscale = self.plot_annual_acf(df_resid_annual, target, df.index[0], df.index[-1],
                                                     data_dir, png_dir, svg_dir,
                                                     nlags=nlags, smoothing=self.smoothing)

        # df in weekly and hourly seasonality plot is a just dummy variable
        # to make consistent with other plot methods
        self.plot_hourly(df_resid_annual_pop, target, data_dir, png_dir, svg_dir,
                         target_year=target_year, target_month=target_month, target_day=target_day)
        h_confint, h_intscale = self.plot_hourly_acf(df_resid_hourly, target, df.index[0], df.index[-1],
                                                     data_dir, png_dir, svg_dir,
                                                     target_year=target_year, target_month=target_month, target_day=target_day,
                                                     nlags=nlags*24)

        dict_corr_dist = {
            "raw": {
                "confint": r_confint,
                "intscale": r_intscale},
            "annual": {
                "confint": a_confint,
                "intscale": a_intscale},
            "hourly": {
                "confint": h_confint,
                "intscale": h_intscale}
        }

        with open(data_dir / 'intscale.json', 'w') as f:
            print(dict_corr_dist, file=f)

class SeasonalityDecompositor_HA(TransformerMixin, BaseEstimator):
    """Decompose Seasonality

    Attributes
    ----------
    sea_hourly : list of dicts or None, shape (n_features,)
        Hourly Seasonality
    sea_annual : list of dicts or None, shape (n_features,)
        Annual Seasonality
    """

    def __init__(self, sea_hourly=None, sea_annual=None,
                 smoothing=True, smoothingFrac=0.05):
        """seasonality data initialization for test data (key-value structure)

        Args:
            sea_hourly: list of dicts or None, shape (n_features,)
            sea_annual: list of dicts or None, shape (n_features,)
            smoothing: bool
            smoothingFrac: float

            * test data get input from train data rather than fit itself

        * key format of seasonality elements
            * sea_annual : YYMMDD: string (zero-padded)
            * sea_week : W: integer
            * sea_hour : HH: string (zero-padded)
        """
        # http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
        # Use inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_sesaonality(self, sea_hourly, sea_annual):
        self.sea_hourly = sea_hourly
        self.sea_annual = sea_annual

    def build_seasonality(self, xs: pd.DataFrame, period: str):
        """
        Build seasonality with given DataFrame's DatetimeIndex
        Similar job as `inverse_transform`,
        but instead of sum, just return seasonality
        """
        def _build_seasonality(_xs):
            # to subscript seasonality, get name (feature)
            name = _xs.name

            # method for row
            def _get_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                if period == 'y':
                    return self.sea_annual[name][utils.parse_ykey(idx)]
                elif period == 'h':
                    return self.sea_hourly[name][utils.parse_hkey(idx)]
                else:
                    raise Exception("Invalid period ('y', 'h'): ", period)

            # index to sum of seasonality
            sea = _xs.index.map(_get_seasonality)
            _xs_df = _xs.to_frame()
            _xs_df['seas'] = sea

            return sea

        #= Y.apply(get_seasonality, axis=1)
        seas = xs.apply(_build_seasonality, 0)

        #raw = Y.apply(add_seasonality, 0)
        return seas

    def compute_seasonality(self, X, return_resid=False):
        """Decompose seasonality single column of DataFrame
        """
        # for key, convert series to dataframe
        X_h = X.copy().to_frame()
        X_h.columns = ['raw']

        if X_h.isnull().values.any():
            raise Exception(
                'Invalid data in {} column, NULL value found'.format(self.target))

        sea_hourly, df_sea_hourly, resid_hourly = utils.periodic_mean(
            X_h, 'raw', 'h', smoothing=False)

        # 1. Hourly Seasonality (hh: 00 ~ 23)
        # join seasonality to original X_h DataFrame
        # populate residuals from daily averaged data to hourly data

        # residuals are daily index, so populate to hourly
        resid_hourly_d = resid_hourly.resample('D').mean()

        sea_annual, df_sea_annual, resid_annual_d = utils.periodic_mean(
            resid_hourly_d.copy(), 'resid', 'y', smoothing=self.smoothing, smoothingFrac=self.smoothingFrac)

        if '0229' not in sea_annual:
            raise KeyError("You must include leap year in train data")

        ## method to generate key column from index
        def key_md(idx): return ''.join((str(idx.month).zfill(2),
                                         str(idx.day).zfill(2)))
        def key_h(idx): return str(idx.hour).zfill(2)

        ## Add column from index for key
        resid_hourly['key_md'] = resid_hourly.index.map(key_md)
        resid_apop = resid_hourly.copy()

        def get_res(key): return sea_annual[key_md(key.name)]

        resid_apop['resid'] = resid_hourly['resid'] - \
            resid_hourly['resid'].to_frame().apply(get_res, axis=1)

        # 3.2. check no missing values
        if len(resid_apop.index) != len(resid_hourly.index):
            raise Exception(
                "Merge Error: something missing when populate daily averaged DataFrame")

        if return_resid == True:
            return resid_hourly, resid_apop, resid_annual_d

        return sea_hourly, sea_annual

    def fit(self, X: pd.DataFrame, y=None):
        """Compute seasonality,
        Computed residuals in `fit` will be dropped.
        In  `transform` residuals are computed again from seasonality from `fit`

        y is not used, but added for Pipeline compatibility
        ref: https://scikit-learn.org/stable/developers/develop.html

        Args:
            X_h: DataFrame which have datetime as index

        Return:
            None
        """
        # Decompose seasonality if there is no predefined seasonality (train/valid set)
        if self.sea_annual != None and self.sea_hourly != None:
            # if test set
            return self

        # apply function to column-wise
        #self.sea_annual, self.sea_weekly, self.sea_hourly = \
        seas = X.apply(self.compute_seasonality, 0).to_dict(orient='records')
        self.sea_hourly = seas[0]
        self.sea_annual = seas[1]

        return self

    def transform(self, X: pd.DataFrame):
        """Recompute residual by subtracting seasonality from X
        shape of X is (sample_size, num_features)

        Args:
            X (pd.DataFrame):
                must have DataTimeIndex to get dates

        Returns:
            resid (np.ndarray):
                dates now doesn't be needed anymore
        """
        def sub_seasonality(_X):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _X.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)] + \
                    self.sea_hourly[name][utils.parse_hkey(idx)]
            # index to sum of seasonality
            seas = _X.index.map(_sum_seasonality)
            _X_df = _X.to_frame()
            _X_df['seas'] = -seas

            _X_sum = _X_df.sum(axis='columns')
            _X_sum.name = name

            return _X_sum

        resid = X.apply(sub_seasonality, 0)

        return resid.to_numpy()

    def inverse_transform(self, Y: pd.DataFrame):
        """Compose value from residuals
        If smoothed annual seasonality is used,
        compsed value might be different with original value

        shape of Y is (output_size, 1) because self.target is always one target

        Args:
            Y (pd.DataFrame):
                pd.DataFrame must have DataTimeIndex to get dates

        Returns:
            raw (ndarray):
        """

        def add_seasonality(_Y):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _Y.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)] + \
                    self.sea_hourly[name][utils.parse_hkey(idx)]

            # index to sum of seasonality
            seas = _Y.index.map(_sum_seasonality)
            _Y_df = _Y.to_frame()
            _Y_df['seas'] = seas

            _Y_sum = _Y_df.sum(axis='columns')
            _Y_sum.name = name

            return _Y_sum

        # there is no way to pass row object of Series,
        # so I pass index and use Series.get() with closure
        #raw = Y.index.map(add_seasonality)
        raw = Y.apply(add_seasonality, 0)

        return raw.to_numpy()

    # utility functions
    def ydt2key(self, d): return str(d.astimezone(SEOULTZ).month).zfill(
        2) + str(d.astimezone(SEOULTZ).day).zfill(2)

    def hdt2key(self, d): return str(d.astimezone(SEOULTZ).hour).zfill(2)

    def confint_idx(self, acf, confint):
        for i, (a, [c1, c2]) in enumerate(zip(acf, confint)):
            if c1 - a < a and a < c2 - a:
                return i

    def plot_hourly(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, target_month=5, target_day=1):
        """

        df : Residual dataframe
        """

        dt1 = dt.datetime(year=target_year, month=target_month, day=target_day,
                          hour=0)
        dt2 = dt.datetime(year=target_year, month=target_month, day=target_day,
                          hour=23)
        hour_range = pd.date_range(
            start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        hour_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()

        # Compute Hourly seasonality
        hs = [self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]

        csv_path = data_dir / ("hourly_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_hour = pd.DataFrame.from_dict(
            {'date': hour_range, 'hs': hs})
        df_hour.set_index('date', inplace=True)
        df_hour.to_csv(csv_path)

        png_path = png_dir / ("hourly_seasonality_" +
                              dt1.strftime("%Y%m%d%H") + "_" +
                              dt2.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("hourly_seasonality_" +
                              dt1.strftime("%Y%m%d%H") + "_" +
                              dt2.strftime("%Y%m%d%H") + ".svg")

        p3 = figure(title="Hourly Seasonality")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "dates"
        p3.xaxis.formatter = DatetimeTickFormatter(hours="%H")
        p3.line(hour_range_plt, hs, line_color="dodgerblue", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

    def plot_hourly_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, target_month=5, target_day=1, nlags=24*15):

        """
        df : Original dataframe
        """

        hour_range = pd.date_range(
            start=fdate, end=tdate, freq="1H").tz_convert(SEOULTZ).tolist()

        #hr = [df.loc[h, target] -
        #      self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]

        # from already comptued residuals
        hr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        hr_acf, confint, qstat, pvalues = sm.tsa.acf(
            hr, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(hr_acf[0:self.confint_idx(hr_acf, confint)])
        print("Hourly Conf Int  : ", self.confint_idx(hr_acf, confint))
        print("Hourly Int Scale : ", intscale)
        #print("Hourly qstat : ", qstat)
        #print("Hourly pvalues : ", pvalues)

        csv_path = data_dir / ("acf_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".csv")
        df_hour_acf = pd.DataFrame.from_dict(
            {'lags': range(len(hr_acf)), 'hr_acf': hr_acf})
        df_hour_acf.set_index('lags', inplace=True)
        df_hour_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        p3 = figure(title="Autocorrelation of Hourly Residual")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(hr_acf)), 1.1)
        p3.line(range(len(hr_acf)), hr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(hr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(hr_acf, confint), intscale

    def plot_annual(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, smoothing=True):
        # annual (pick 1 year)
        dt1 = dt.datetime(year=target_year, month=1, day=1, hour=0)
        dt2 = dt.datetime(year=target_year, month=12, day=31, hour=23)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])

        # if not smoothing, just use predecomposed seasonality
        ys = [self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        if smoothing:
            _sea_annual_nonsmooth, _, _ = \
                utils.periodic_mean(df, target, 'y', smoothing=False)
            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            ys_vanilla = [
                _sea_annual_nonsmooth[self.ydt2key(y)] for y in year_range]
            ys_smooth = [self.sea_annual[target]
                         [self.ydt2key(y)] for y in year_range]

        csv_path = data_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d") + "_" +
                               dt2.strftime("%Y%m%d") + ".csv")
        df_year = pd.DataFrame.from_dict(
            {'date': year_range, 'ys': ys})
        if smoothing:
            df_year = pd.DataFrame.from_dict(
                {'date': year_range, 'ys_vanilla': ys_vanilla, 'ys_smooth': ys_smooth})
        df_year.set_index('date', inplace=True)
        df_year.to_csv(csv_path)

        png_path = png_dir / ("annual_seasonality_" +
                              dt1.strftime("%Y%m%d") + "_" +
                              dt2.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("annual_seasonality_" +
                              dt1.strftime("%Y%m%d") + "_" +
                              dt2.strftime("%Y%m%d") + ".svg")

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                   months="%m/%d %H:%M",
                                                   hours="%m/%d %H:%M",
                                                   minutes="%m/%d %H:%M")
        p1.line(year_range_plt, ys, line_color="dodgerblue", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        if smoothing:
            # 1. smooth
            png_path = png_dir / ("annual_seasonality_(smooth)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(smooth)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 2. vanila
            png_path = png_dir / ("annual_seasonality_(vanilla)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(vanilla)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Raw)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_vanilla,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 3. smooth + vanila
            png_path = png_dir / ("annual_seasonality_(both)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(both)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth & Vanilla)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth, line_color="dodgerblue",
                    line_width=2, legend_label="smooth")
            p1.line(year_range_plt, ys_vanilla, line_color="lightcoral",
                    line_width=2, legend_label="vanilla")
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

    def plot_annual_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, nlags=15, smoothing=True):
        ## residual autocorrelation by tsa.acf
        year_range = pd.date_range(
            start=fdate, end=tdate, freq='D').tz_convert(SEOULTZ).tolist()
        # set hour to 0
        year_range = [y.replace(hour=0) for y in year_range]

        #yr = [df.loc[y, target] -
        #      self.sea_annual[target][self.ydt2key(y)] for y in year_range]

        # from already comptued residuals
        yr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()

        # 95% confidance intervals
        yr_acf, confint, qstat, pvalues = sm.tsa.acf(
            yr, qstat=True, alpha=0.05, nlags=nlags)

        # I need "hours" as unit, so multiply 24
        intscale = sum(yr_acf[0:self.confint_idx(yr_acf, confint)])
        print("Annual Conf Int  : ", self.confint_idx(yr_acf, confint))
        print("Annual Int Scale : ", intscale)
        print("Annual qstat : ", qstat)
        print("Annual pvalues : ", pvalues)

        csv_path = data_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d") + "_" +
                               tdate.strftime("%Y%m%d") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': [i*24 for i in range(len(yr_acf))], 'yr_acf': yr_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".svg")
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line([i*24 for i in range(len(yr_acf))], yr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".png")
        svg_path = svg_dir / ("acf(tpl)_annual_seasonality_" +
                              fdate.strftime("%Y%m%d") + "_" +
                              tdate.strftime("%Y%m%d") + ".svg")
        fig = tpl.plot_acf(yr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(yr_acf, confint), intscale

    def plot(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
             target_year=2016, target_week=10, target_month=5, target_day=1,
             nlags=15, smoothing=True):
        """
        Plot seasonality itself and autocorrelation of residuals

        Args:
            df (DataFrame):
                Hourly DataFrame that includes whole data

            target (str):
                column name of DataFrame

            fdate (DateTime):
                acf plot from range between fdate ~ tdate

            tdate (DateTime):
                acf plot from range between fdate ~ tdate

            data_dir (Path):
                Directory location to save csv file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            target_year (int):
                specified year for plot

            target_week (int):
                specified week number in target_year for plot

            nlags (int):
                nlags for autocorrelation (days)

            smoothing (bool):
                is smoothing used?

        Return:
            None
        """

        # filter by dates
        df = df[(df.index > fdate) & (df.index < tdate)]
        # daily averaged
        df_d = df.resample('D').mean()

        df_resid_hourly, df_resid_annual_pop, df_resid_annual = \
            self.compute_seasonality(df.loc[:, target], return_resid=True)
        df_resid_hourly.rename(columns={'resid': target}, inplace=True)
        df_resid_annual_pop.rename(columns={'resid': target}, inplace=True)
        df_resid_annual.rename(columns={'resid': target}, inplace=True)

        dict_corr_dist = {}

        self.plot_hourly(df_resid_hourly, target, data_dir, png_dir, svg_dir,
                         target_year=target_year, target_month=target_month, target_day=target_day)
        h_confint, h_intscale = self.plot_hourly_acf(df_resid_hourly, target, df.index[0], df.index[-1],
                                                     data_dir, png_dir, svg_dir,
                                                     target_year=target_year, target_month=target_month, target_day=target_day,
                                                     nlags=nlags*24)

        self.plot_annual(df_resid_annual, target, data_dir, png_dir, svg_dir,
                         target_year=2016, smoothing=self.smoothing)
        a_confint, a_intscale = self.plot_annual_acf(df_resid_annual_pop, target, df.index[0], df.index[-1],
                                                     data_dir, png_dir, svg_dir,
                                                     nlags=nlags, smoothing=self.smoothing)

        dict_corr_dist = {
            "annual": {
                "confint": a_confint,
                "intscale": a_intscale},
            "hourly": {
                "confint": h_confint,
                "intscale": h_intscale}
        }

        with open(data_dir / 'intscale.json', 'w') as f:
            print(dict_corr_dist, file=f)

class SeasonalityDecompositor_A(TransformerMixin, BaseEstimator):
    """Decompose Seasonality only to annual seasonality

    Attributes
    ----------
    sea_annual : list of dicts or None, shape (n_features,)
        Annual Seasonality
    """

    def __init__(self, sea_annual=None, smoothing=True, smoothingFrac=0.05):
        """seasonality data initialization for test data (key-value structure)

        Args:
            sea_annual: list of dicts or None, shape (n_features,)
            smoothing: bool

            * test data get input from train data rather than fit itself

        * key format of seasonality elements
            * sea_annual : YYMMDD: string (zero-padded)
        """
        # http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
        # Use inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_sesaonality(self, sea_annual):
        self.sea_annual = sea_annual

    def build_seasonality(self, xs: pd.DataFrame, period: str):
        """
        Build seasonality with given DataFrame's DatetimeIndex
        Similar job as `inverse_transform`,
        but instead of sum, just return seasonality
        """
        def _build_seasonality(_xs):
            # to subscript seasonality, get name (feature)
            name = _xs.name

            # method for row
            def _get_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                if period == 'y':
                    return self.sea_annual[name][utils.parse_ykey(idx)]
                else:
                    raise Exception("Invalid period ('y'): ", period)

            # index to sum of seasonality
            sea = _xs.index.map(_get_seasonality)
            _xs_df = _xs.to_frame()
            _xs_df['seas'] = sea

            return sea

        #= Y.apply(get_seasonality, axis=1)
        seas = xs.apply(_build_seasonality, 0)

        #raw = Y.apply(add_seasonality, 0)
        return seas

    def compute_seasonality(self, X, return_resid=False):
        """Decompose seasonality single column of DataFrame
        """
        # for key, convert series to dataframe
        X_h = X.copy().to_frame()
        X_h.columns = ['raw']

        if X_h.isnull().values.any():
            raise Exception(
                'Invalid data in {} column, NULL value found'.format(self.target))
        X_d = X_h.resample('D').mean()

        # 1. Annual Seasonality (mmdd: 0101 ~ 1231)
        # dictionary for key (mmdd) to value (daily average)
        sea_annual, df_sea_annual, resid_annual = utils.periodic_mean(
            X_d, 'raw', 'y', smoothing=self.smoothing)

        if '0229' not in sea_annual:
            raise KeyError("You must include leap year in train data")

        # 2. Daily Seasonality (hh: 00 ~ 23)
        # join seasonality to original X_h DataFrame
        # populate seasonality from daily averaged data to hourly data

        ## method to generate key column from index
        def key_md(idx): return ''.join((str(idx.month).zfill(2),
                                         str(idx.day).zfill(2)))

        ## Add column from index for key
        df_sea_annual['key_md'] = df_sea_annual.index
        X_h['key_md'] = X_h.index.map(key_md)

        ## compute hourly populated seasonality from daily residuals
        X_h_spop = df_sea_annual.merge(X_h, how='left',
                                       on='key_md', left_index=True, validate="1:m").dropna()
        X_h_spop = X_h_spop.rename(columns={'sea': 'sea_annual'})

        ## new hourly residual column from daily residuals
        X_h_spop['resid'] = X_h_spop['raw'] - X_h_spop['sea_annual']

        # 3.2. check no missing values
        if len(X_h_spop.index) != len(X_h.index):
            raise Exception(
                "Merge Error: something missing when populate daily averaged DataFrame")

        # 3.4 drop key columns
        X_h.drop('key_md', axis='columns')

        # To make consistent output in fit, return dummy variables
        if return_resid == True:
            return resid_annual, X_h_spop

        # To return with column names, return dummy value
        return sea_annual, None

    def fit(self, X: pd.DataFrame, y=None):
        """Compute seasonality,
        Computed residuals in `fit` will be dropped.
        In  `transform` residuals are computed again from seasonality from `fit`

        y is not used, but added for Pipeline compatibility
        ref: https://scikit-learn.org/stable/developers/develop.html

        Args:
            X_h: DataFrame which have datetime as index

        Return:
            None
        """
        # Decompose seasonality if there is no predefined seasonality (train/valid set)
        if self.sea_annual != None:
            # if test set
            return self

        # apply function to column-wise
        seas = X.apply(self.compute_seasonality, 0).to_dict(orient='records')
        self.sea_annual = seas[0]

        return self

    def transform(self, X: pd.DataFrame):
        """Recompute residual by subtracting seasonality from X
        shape of X is (sample_size, num_features)

        Args:
            X (pd.DataFrame):
                must have DataTimeIndex to get dates

        Returns:
            resid (np.ndarray):
                dates now doesn't be needed anymore
        """
        def sub_seasonality(_X):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _X.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)]
            # index to sum of seasonality

            seas = _X.index.map(_sum_seasonality)
            _X_df = _X.to_frame()
            _X_df['seas'] = -seas

            _X_sum = _X_df.sum(axis='columns')
            _X_sum.name = name

            return _X_sum

        resid = X.apply(sub_seasonality, 0)

        return resid.to_numpy()

    def inverse_transform(self, Y: pd.DataFrame):
        """Compose value from residuals
        If smoothed annual seasonality is used,
        compsed value might be different with original value

        shape of Y is (output_size, 1) because self.target is always one target

        Args:
            Y (pd.DataFrame):
                pd.DataFrame must have DataTimeIndex to get dates

        Returns:
            raw (ndarray):
        """

        def add_seasonality(_Y):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _Y.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_annual[name][utils.parse_ykey(idx)]

            # index to sum of seasonality
            seas = _Y.index.map(_sum_seasonality)

            _Y_df = _Y.to_frame()
            _Y_df['seas'] = seas

            _Y_sum = _Y_df.sum(axis='columns')
            _Y_sum.name = name

            return _Y_sum

        # there is no way to pass row object of Series,
        # so I pass index and use Series.get() with closure
        raw = Y.apply(add_seasonality, 0)

        return raw.to_numpy()

    # utility functions
    def ydt2key(self, d): return str(d.astimezone(SEOULTZ).month).zfill(
        2) + str(d.astimezone(SEOULTZ).day).zfill(2)

    def confint_idx(self, acf, confint):
        for i, (a, [c1, c2]) in enumerate(zip(acf, confint)):
            if c1 - a < a and a < c2 - a:
                return i

    def plot_annual(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, smoothing=True, smoothingFrac=0.05):
        # annual (pick 1 year)
        dt1 = dt.datetime(year=target_year, month=1, day=1, hour=0)
        dt2 = dt.datetime(year=target_year, month=12, day=31, hour=23)
        year_range = pd.date_range(
            start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)])

        # if not smoothing, just use predecomposed seasonality
        ys = [self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        if smoothing:
            _sea_annual_nonsmooth, _, _ = \
                utils.periodic_mean(df, target, 'y', smoothing=False)
            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            ys_vanilla = [
                _sea_annual_nonsmooth[self.ydt2key(y)] for y in year_range]
            ys_smooth = [self.sea_annual[target]
                         [self.ydt2key(y)] for y in year_range]

        csv_path = data_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_year = pd.DataFrame.from_dict(
            {'date': year_range, 'ys': ys})
        if smoothing:
            df_year = pd.DataFrame.from_dict(
                {'date': year_range, 'ys_vanilla': ys_vanilla, 'ys_smooth': ys_smooth})
        df_year.set_index('date', inplace=True)
        df_year.to_csv(csv_path)

        png_path = png_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("annual_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".svg")

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                   months="%m/%d %H:%M",
                                                   hours="%m/%d %H:%M",
                                                   minutes="%m/%d %H:%M")
        p1.line(year_range_plt, ys, line_color="dodgerblue", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        if smoothing:
            # 1. smooth
            png_path = png_dir / ("annual_seasonality_(smooth)_" +
                                   dt1.strftime("%Y%m%d%H") + "_" +
                                   dt2.strftime("%Y%m%d%H") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(smooth)_" +
                                   dt1.strftime("%Y%m%d%H") + "_" +
                                   dt2.strftime("%Y%m%d%H") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 2. vanila
            png_path = png_dir / ("annual_seasonality_(vanilla)_" +
                                   dt1.strftime("%Y%m%d%H") + "_" +
                                   dt2.strftime("%Y%m%d%H") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(vanilla)_" +
                                   dt1.strftime("%Y%m%d%H") + "_" +
                                   dt2.strftime("%Y%m%d%H") + ".svg")
            p1 = figure(title="Annual Seasonality(Raw)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_vanilla,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 3. smooth + vanila
            png_path = png_dir / ("annual_seasonality_(both)_" +
                                   dt1.strftime("%Y%m%d%H") + "_" +
                                   dt2.strftime("%Y%m%d%H") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(both)_" +
                                   dt1.strftime("%Y%m%d%H") + "_" +
                                   dt2.strftime("%Y%m%d%H") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth & Vanilla)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(year_range_plt, ys_smooth, line_color="dodgerblue",
                    line_width=2, legend_label="smooth")
            p1.line(year_range_plt, ys_vanilla, line_color="lightcoral",
                    line_width=2, legend_label="vanilla")
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

    def plot_annual_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, nlags=15*24, smoothing=True):
        ## residual autocorrelation by tsa.acf
        year_range = pd.date_range(
            start=fdate, end=tdate, freq='1H').tz_convert(SEOULTZ).tolist()

        #yr = [df.loc[y, target] -
        #      self.sea_annual[target][self.ydt2key(y)] for y in year_range]
        yr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()

        # 95% confidance intervals
        yr_acf, confint, qstat, pvalues = sm.tsa.acf(
            yr, qstat=True, alpha=0.05, nlags=nlags)

        # I need "hours" as unit, so multiply 24
        intscale = sum(yr_acf[0:self.confint_idx(yr_acf, confint)])
        print("Annual Conf Int  : ", self.confint_idx(yr_acf, confint))
        print("Annual Int Scale : ", intscale)
        #print("Annual qstat : ", qstat)
        #print("Annual pvalues : ", pvalues)

        csv_path = data_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': [i for i in range(len(yr_acf))], 'yr_acf': yr_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_annual_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".svg")
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line([i*24 for i in range(len(yr_acf))], yr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_annual_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_annual_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(yr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(yr_acf, confint), intscale

    def plot(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
             target_year=2016, target_week=10, target_month=5, target_day=1,
             nlags=15*24, smoothing=True):
        """
        Plot seasonality itself and autocorrelation of residuals

        Args:
            df (DataFrame):
                Hourly DataFrame that includes whole data

            target (str):
                column name of DataFrame

            fdate (DateTime):
                acf plot from range between fdate ~ tdate

            tdate (DateTime):
                acf plot from range between fdate ~ tdate

            data_dir (Path):
                Directory location to save csv file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            target_year (int):
                specified year for plot

            target_week (int):
                specified week number in target_year for plot

            nlags (int):
                nlags for autocorrelation (days)

            smoothing (bool):
                is smoothing used?

        Return:
            None
        """

        # filter by dates
        df = df[(df.index > fdate) & (df.index < tdate)]
        # daily averaged
        df_d = df.resample('D').mean()

        df_resid_annual, df_resid_annual_pop = \
            self.compute_seasonality(df.loc[:, target], return_resid=True)
        df_resid_annual.rename(columns={'resid': target}, inplace=True)
        df_resid_annual_pop.rename(columns={'resid': target}, inplace=True)

        dict_corr_dist = {}

        self.plot_annual(df, target, data_dir, png_dir, svg_dir,
                         target_year=2016, smoothing=smoothing)
        a_confint, a_intscale = self.plot_annual_acf(df, target, df.index[0], df.index[-1],
                             data_dir, png_dir, svg_dir,
                             nlags=nlags, smoothing=smoothing)

        dict_corr_dist = {
            "annual": {
                "confint": a_confint,
                "intscale": a_intscale}
        }

        with open(data_dir / 'intscale.json', 'w') as f:
            print(dict_corr_dist, file=f)

class SeasonalityDecompositor_H(TransformerMixin, BaseEstimator):
    """Decompose Seasonality

    Attributes
    ----------
    sea_annual : list of dicts or None, shape (n_features,)
        Annual Seasonality
    sea_weekly : list of dicts or None, shape (n_features,)
        Weekly Seasonality
    sea_hourly : list of dicts or None, shape (n_features,)
        Hourly Seasonality
    """

    def __init__(self, sea_hourly=None, smoothing=True, smoothingFrac=0.05):
        """seasonality data initialization for test data (key-value structure)

        Args:
            sea_annual: list of dicts or None, shape (n_features,)
            sea_weekly: list of dicts or None, shape (n_features,)
            sea_hourly: list of dicts or None, shape (n_features,)
            smoothing: bool

            * test data get input from train data rather than fit itself

        * key format of seasonality elements
            * sea_annual : YYMMDD: string (zero-padded)
            * sea_week : W: integer
            * sea_hour : HH: string (zero-padded)
        """
        # http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
        # Use inspect

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_sesaonality(self, sea_hourly):
        self.sea_hourly = sea_hourly

    def build_seasonality(self, xs: pd.DataFrame, period: str):
        """
        Build seasonality with given DataFrame's DatetimeIndex
        Similar job as `inverse_transform`,
        but instead of sum, just return seasonality
        """
        def _build_seasonality(_xs):
            # to subscript seasonality, get name (feature)
            name = _xs.name

            # method for row
            def _get_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                if period == 'h':
                    return self.sea_hourly[name][utils.parse_hkey(idx)]
                else:
                    raise Exception("Invalid period ('h'): ", period)

            # index to sum of seasonality
            sea = _xs.index.map(_get_seasonality)
            _xs_df = _xs.to_frame()
            _xs_df['seas'] = sea

            return sea

        #= Y.apply(get_seasonality, axis=1)
        seas = xs.apply(_build_seasonality, 0)

        #raw = Y.apply(add_seasonality, 0)
        return seas

    def compute_seasonality(self, X, return_resid=False):
        """Decompose seasonality single column of DataFrame
        """
        # for key, convert series to dataframe
        X_h = X.copy().to_frame()
        X_h.columns = ['raw']

        if X_h.isnull().values.any():
            raise Exception(
                'Invalid data in {} column, NULL value found'.format(self.target))
        # 3.3 Compute seasonality
        sea_hourly, df_sea_hourly, resid_hourly = utils.periodic_mean(
            X_h, 'raw', 'h', smoothing=self.smoothing, smoothingFrac=self.smoothingFrac)

        # To make consistent output in fit, return dummy variables
        if return_resid == True:
            return resid_hourly, resid_hourly

        return sea_hourly, None

    def fit(self, X: pd.DataFrame, y=None):
        """Compute seasonality,
        Computed residuals in `fit` will be dropped.
        In  `transform` residuals are computed again from seasonality from `fit`

        y is not used, but added for Pipeline compatibility
        ref: https://scikit-learn.org/stable/developers/develop.html

        Args:
            X_h: DataFrame which have datetime as index

        Return:
            None
        """
        # Decompose seasonality if there is no predefined seasonality (train/valid set)
        if self.sea_hourly != None:
            # if test set
            return self

        # apply function to column-wise
        #self.sea_annual, self.sea_weekly, self.sea_hourly = \
        seas = X.apply(self.compute_seasonality, 0).to_dict(orient='records')
        self.sea_hourly = seas[0]

        return self

    def transform(self, X: pd.DataFrame):
        """Recompute residual by subtracting seasonality from X
        shape of X is (sample_size, num_features)

        Args:
            X (pd.DataFrame):
                must have DataTimeIndex to get dates

        Returns:
            resid (np.ndarray):
                dates now doesn't be needed anymore
        """
        def sub_seasonality(_X):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _X.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_hourly[name][utils.parse_hkey(idx)]
            # index to sum of seasonality
            seas = _X.index.map(_sum_seasonality)
            _X_df = _X.to_frame()
            _X_df['seas'] = -seas

            _X_sum = _X_df.sum(axis='columns')
            _X_sum.name = name

            return _X_sum

        resid = X.apply(sub_seasonality, 0)

        return resid.to_numpy()

    def inverse_transform(self, Y: pd.DataFrame):
        """Compose value from residuals
        If smoothed annual seasonality is used,
        compsed value might be different with original value

        shape of Y is (output_size, 1) because self.target is always one target

        Args:
            Y (pd.DataFrame):
                pd.DataFrame must have DataTimeIndex to get dates

        Returns:
            raw (ndarray):
        """

        def add_seasonality(_Y):
            """Method for columnwise operation
            """
            # to subscript seasonality, get name (feature)
            name = _Y.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation
                """
                return \
                    self.sea_hourly[name][utils.parse_hkey(idx)]

            # index to sum of seasonality
            seas = _Y.index.map(_sum_seasonality)
            _Y_df = _Y.to_frame()
            _Y_df['seas'] = seas

            _Y_sum = _Y_df.sum(axis='columns')
            _Y_sum.name = name

            return _Y_sum

        # there is no way to pass row object of Series,
        # so I pass index and use Series.get() with closure
        #raw = Y.index.map(add_seasonality)
        raw = Y.apply(add_seasonality, 0)

        return raw.to_numpy()

    # utility functions
    def hdt2key(self, d): return str(d.astimezone(SEOULTZ).hour).zfill(2)

    def confint_idx(self, acf, confint):
        for i, (a, [c1, c2]) in enumerate(zip(acf, confint)):
            if c1 - a < a and a < c2 - a:
                return i

    def plot_hourly(self, df, target, data_dir, png_dir, svg_dir,
                    target_year=2016, target_month=5, target_day=1,
                    smoothing=True, smoothingFrac=0.05):

        dt1 = dt.datetime(year=target_year, month=target_month, day=target_day,
                          hour=0)
        dt2 = dt.datetime(year=target_year, month=target_month, day=target_day,
                          hour=23)
        hour_range = pd.date_range(
            start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        hour_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()

        # Compute Hourly seasonality
        hs = [self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]
        if smoothing:
            _sea_hourly_nonsmooth, _, _ = \
                utils.periodic_mean(df, target, 'h', smoothing=False)
            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            hs_vanilla = [
                _sea_hourly_nonsmooth[self.hdt2key(h)] for h in hour_range]
            hs_smooth = [self.sea_hourly[target]
                         [self.hdt2key(h)] for h in hour_range]

        csv_path = data_dir / ("hourly_seasonality_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_hour = pd.DataFrame.from_dict(
            {'hour': hour_range, 'hs': hs})
        df_hour.set_index('hour', inplace=True)
        df_hour.to_csv(csv_path)

        png_path = png_dir / ("hourly_seasonality_" +
                              dt1.strftime("%Y%m%d%H") + "_" +
                              dt2.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("hourly_seasonality_" +
                              dt1.strftime("%Y%m%d%H") + "_" +
                              dt2.strftime("%Y%m%d%H") + ".svg")

        p3 = figure(title="Hourly Seasonality")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "hour"
        p3.xaxis.formatter = DatetimeTickFormatter(hours="%H")
        p3.line(hour_range_plt, hs, line_color="dodgerblue", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

        if smoothing:
            # 1. smooth
            png_path = png_dir / ("hourly_seasonality_(smooth)_" +
                                  dt1.strftime("%Y%m%d%H") + "_" +
                                  dt2.strftime("%Y%m%d%H") + ".png")
            svg_path = svg_dir / ("hourly_seasonality_(smooth)_" +
                                  dt1.strftime("%Y%m%d%H") + "_" +
                                  dt2.strftime("%Y%m%d%H") + ".svg")
            p1 = figure(title="Hourly Seasonality(Smooth)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(hour_range_plt, hs_smooth,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 2. vanila
            png_path = png_dir / ("hourly_seasonality_(vanilla)_" +
                                  dt1.strftime("%Y%m%d%H") + "_" +
                                  dt2.strftime("%Y%m%d%H") + ".png")
            svg_path = svg_dir / ("hourly_seasonality_(vanilla)_" +
                                  dt1.strftime("%Y%m%d%H") + "_" +
                                  dt2.strftime("%Y%m%d%H") + ".svg")
            p1 = figure(title="Hourly Seasonality(Raw)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(hour_range_plt, hs_vanilla,
                    line_color="dodgerblue", line_width=2)
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 3. smooth + vanila
            png_path = png_dir / ("annual_seasonality_(both)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".png")
            svg_path = svg_dir / ("annual_seasonality_(both)_" +
                                  dt1.strftime("%Y%m%d") + "_" +
                                  dt2.strftime("%Y%m%d") + ".svg")
            p1 = figure(title="Annual Seasonality(Smooth & Vanilla)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                       months="%m/%d %H:%M",
                                                       hours="%m/%d %H:%M",
                                                       minutes="%m/%d %H:%M")
            p1.line(hour_range_plt, hs_smooth, line_color="dodgerblue",
                    line_width=2, legend_label="smooth")
            p1.line(hour_range_plt, hs_vanilla, line_color="lightcoral",
                    line_width=2, legend_label="vanilla")
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

    def plot_hourly_acf(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
                        target_year=2016, target_month=5, target_day=1, nlags=24*15):

        hour_range = pd.date_range(
            start=fdate, end=tdate, freq="1H").tz_convert(SEOULTZ).tolist()

        # from already comptued residuals
        hr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()

        #hr = [df.loc[h, target] -
        #      self.sea_hourly[target][self.hdt2key(h)] for h in hour_range]

        # filter df by hour_range
        hr_acf, confint, qstat, pvalues = sm.tsa.acf(
            hr, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(hr_acf[0:self.confint_idx(hr_acf, confint)])
        print("Hourly Conf Int  : ", self.confint_idx(hr_acf, confint))
        print("Hourly Int Scale : ", intscale)
        #print("Hourly qstat : ", qstat)
        #print("Hourly pvalues : ", pvalues)

        csv_path = data_dir / ("acf_hourly_seasonality_" +
                               fdate.strftime("%Y%m%d%H") + "_" +
                               tdate.strftime("%Y%m%d%H") + ".csv")
        df_hour_acf = pd.DataFrame.from_dict(
            {'lags': range(len(hr_acf)), 'hr_acf': hr_acf})
        df_hour_acf.set_index('lags', inplace=True)
        df_hour_acf.to_csv(csv_path)

        png_path = png_dir / ("acf_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        p3 = figure(title="Autocorrelation of Hourly Residual")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(hr_acf)), 1.1)
        p3.line(range(len(hr_acf)), hr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / ("acf(tpl)_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".png")
        svg_path = svg_dir / ("acf(tpl)_hourly_seasonality_" +
                              fdate.strftime("%Y%m%d%H") + "_" +
                              tdate.strftime("%Y%m%d%H") + ".svg")
        fig = tpl.plot_acf(hr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return self.confint_idx(hr_acf, confint), intscale

    def plot(self, df, target, fdate, tdate, data_dir, png_dir, svg_dir,
             target_year=2016, target_week=10, target_month=5, target_day=1,
             nlags=15, smoothing=True):
        """
        Plot seasonality itself and autocorrelation of residuals

        Args:
            df (DataFrame):
                Hourly DataFrame that includes whole data

            target (str):
                column name of DataFrame

            fdate (DateTime):
                acf plot from range between fdate ~ tdate

            tdate (DateTime):
                acf plot from range between fdate ~ tdate

            data_dir (Path):
                Directory location to save csv file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality

            target_year (int):
                specified year for plot

            target_week (int):
                specified week number in target_year for plot

            nlags (int):
                nlags for autocorrelation (days)

            smoothing (bool):
                is smoothing used?

        Return:
            None
        """

        # filter by dates
        df = df[(df.index > fdate) & (df.index < tdate)]

        df_resid_hourly, _ = \
            self.compute_seasonality(df.loc[:, target], return_resid=True)
        df_resid_hourly.rename(columns={'resid': target}, inplace=True)

        dict_corr_dist = {}

        self.plot_hourly(df_resid_hourly, target, data_dir, png_dir, svg_dir,
                         target_year=target_year, target_month=target_month, target_day=target_day)
        h_confint, h_intscale = self.plot_hourly_acf(df_resid_hourly, target, df.index[0], df.index[-1],
                                                     data_dir, png_dir, svg_dir,
                                                     target_year=target_year, target_month=target_month, target_day=target_day,
                                                     nlags=nlags*24)

        dict_corr_dist = {
            "hourly": {
                "confint": h_confint,
                "intscale": h_intscale}
        }

        with open(data_dir / 'intscale.json', 'w') as f:
            print(dict_corr_dist, file=f)

class StandardScalerWrapper(StandardScaler):
    """Convert type as Series, not ndarray
    """
    def __init__(self, scaler):
        self.scaler = scaler

    def __getattr__(self, attr):
        return getattr(self.scaler, attr)

    # how to improve code?
    # 1. remove duplicated structure
    # 2. implement genarator? without knowledge of parent class
    def partial_fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            #self.scaler.partial_fit(X.iloc[:, 0].to_numpy().reshape(-1, 1),
            #    y=X.iloc[:, 0].to_numpy().reshape(-1, 1))
            self.scaler.partial_fit(X, y=X)
            return self
        elif isinstance(X, np.ndarray):
            self.scaler.partial_fit(X, y=X)
            return self
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            #self.scaler.fit(X.iloc[:, 0].to_numpy().reshape(-1, 1),
            #    y=X.iloc[:, 0].to_numpy().reshape(-1, 1))
            #return self
            self.scaler.fit(X, y=X)
            return self
        elif isinstance(X, np.ndarray):
            self.scaler.fit(X, y=X)
            return self
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return self.scaler.transform(X)
        elif isinstance(X, np.ndarray):
            return self.scaler.transform(X)
        else:
            raise TypeError("Type should be Pandas Series or Numpy Array")

    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            #return self.scaler.inverse_transform(X.iloc[:, 0].to_numpy().reshape(-1, 1))
            _invX = self.scaler.inverse_transform(X)
            return pd.DataFrame(data=_invX,
                index=X.index,
                columns=X.columns)
        elif isinstance(X, np.ndarray):
            return self.scaler.inverse_transform(X)
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

class MultivariateGeneralDataset(Dataset):
    def __init__(self, df,
                 sample_size, output_size,
                 **kwargs):
        super().__init__()
        self.target = kwargs.get('target', 'PM10')
        self.features = kwargs.get('features', ['PM10'])
        self.normalize = kwargs.get('normalize', False)

        # date when prediction starts, if I need to predict 2018/1/1 1:00 AM, I need more data with size 'sample_size'
        self.fdate = kwargs.get('fdate', dt.datetime(
            2009, 1, 1, 1).astimezone(SEOULTZ))
        # date where prediction starts
        self.tdate = kwargs.get('tdate', dt.datetime(
            2017, 12, 31, 23).astimezone(SEOULTZ))

        #self._df = df
        self._df = df[self.fdate:self.tdate]
        self._dates = self._df.index.to_pydatetime()
        self._xs = self._df[self.features]
        self._ys = self._df[[self.target]]
        self._scaler = preprocessing.StandardScaler().fit(self._xs)

        self.sample_size = sample_size
        self.output_size = output_size

        self._train_valid_ratio = kwargs.get('train_valid_ratio', 0.8)

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
        x_1d = self._ys.iloc[i:(i+self.sample_size)]
        # save initial input as target variable input at last step (single step)
        y0 = self._ys.iloc[i+self.sample_size-1]
        y = self._ys.iloc[(i+self.sample_size)
                           :(i+self.sample_size+self.output_size)]

        # return X, Y, Y_dates
        if self.normalize == True:
            x_ = self._scaler.transform(x.to_numpy())
        else:
            x_ = x.to_numpy()

        return np.squeeze(x_).astype('float32'), \
            np.squeeze(x_1d).to_numpy().astype('float32'), \
            y0.astype('float32'), \
            np.squeeze(y).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

    def __len__(self):
        """
        hours of train and test dates

        __len__(df) == fdate - tdate - output_size - sample_size

        Returns:
            int: total hours
        """

        return self._ys.shape[0] - self.output_size - self.sample_size

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

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
