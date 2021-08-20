import copy
import datetime as dt
import string
from pathlib import Path

import bokeh
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tpl
import statsmodels.tsa.stattools as tsast
from bokeh.io import export_png, export_svgs
from bokeh.models import DatetimeTickFormatter
from bokeh.plotting import figure
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data.dataset import Dataset

from mise.constants import SEOUL_STATIONS, SEOULTZ
from mise.utils import parse_hkey, parse_wkey, parse_ykey, periodic_mean


def load(datecol, filepath="/input/input.csv"):
    """load input file

    Args:
        filepath (str, optional):
            file path. Defaults to "/input/input.csv".

        datecol (list, optional):
            indicate which column has date format. Defaults to [1].

    Returns:
        pd.pandas.DataFrame: parsed DataFrame
    """
    # date column in raw data : 1
    # date column in imputed data : 0
    df = pd.read_csv(filepath, index_col=[0, 1], parse_dates=datecol)

    # prints
    pd.set_option("display.max_rows", 10)
    pd.reset_option("display.max_rows")
    print(df.head(10))

    return df


def load_imputed(datecol, filepath="/input/input.csv"):
    """load imputed input file

    Args:
        filepath (str, optional):
            file path. Defaults to "/input/input.csv".

        datecol (list, optional):
            indicate which column has date format. Defaults to [1].

    Returns:
        pd.pandas.DataFrame: parsed DataFrame
    """
    # station code & date
    df = pd.read_csv(filepath, index_col=[0, 1], parse_dates=datecol)

    # prints
    pd.set_option("display.max_rows", 10)
    pd.reset_option("display.max_rows")

    return df


def load_station(df, code=111123):
    """filter dataframe by station code

    Args:
        df (pd.DataFrmae): DataFrame to filter
        code (int, optional): station code. Defaults to 111123.

    Returns:
        pd.pandas.DataFrame: filtered DataFrame
    """
    # return df[df['stationCode'] == code]
    return df.query(f"stationCode == {code}")


class BaseDataset(Dataset):
    """Base Dataset"""

    def __init__(self, **kwargs):
        # args -- tuple of anonymous arguments
        # kwargs -- dictionary of named arguments
        self.station_name = kwargs.get("station_name", "종로구")
        self.target = kwargs.get("target", "PM10")
        self.features = kwargs.get(
            "features",
            [
                "SO2",
                "CO",
                "O3",
                "NO2",
                "PM10",
                "PM25",
                "temp",
                "wind_dir",
                "wind_spd",
                "pres",
                "humid",
                "prep",
                "snow",
            ],
        )

        # date when prediction starts,
        # #if I need to predict 2018/1/1 1:00 AM,
        # I need more data with size 'sample_size'
        self.fdate = kwargs.get("fdate", dt.datetime(2009, 1, 1, 1).astimezone(SEOULTZ))
        # date where prediction starts
        self.tdate = kwargs.get(
            "tdate", dt.datetime(2017, 12, 31, 23).astimezone(SEOULTZ)
        )

        # MLP sample_size
        self.sample_size = kwargs.get("sample_size", 48)
        self.batch_size = kwargs.get("batch_size", 32)
        self.output_size = kwargs.get("output_size", 24)
        self._train_valid_ratio = kwargs.get("train_valid_ratio", 0.8)

        # load imputed data from filepath if not provided as dataframe
        # 1. If Jongno ->
        #   1.1 -> Check imputed -> load preimputed data
        # 2. If other station -> load input.csv
        #   2.1 -> Impute and load
        filepath = kwargs.get(
            "filepath", Path("/input/python/input_seoul_imputed_hourly_pandas.csv")
        )
        if self.station_name == "종로구":
            # TODO : parse re? or whatever make logic
            #          for loading precomputed file if jongno,
            #           load default file otherwise
            filepath = kwargs.get(
                "filepath", Path("/input/python/input_jongno_imputed_hourly_pandas.csv")
            )

        raw_df = pd.read_csv(filepath, index_col=[0, 1], parse_dates=[1])
        # filter by station_name
        self._df = raw_df.query(f"stationCode == {SEOUL_STATIONS[self.station_name]}")
        self._df.reset_index(level="stationCode", drop=True, inplace=True)

        # filter by date range including train date
        # i is a start of output, so prepare sample_size
        self._df = self._df[
            self.fdate - dt.timedelta(hours=self.sample_size) : self.tdate
        ]

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
        x = self._xs.iloc[i : i + self.sample_size]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]

        # return X, Y, Y_dates
        return (
            self._scaler.transform(x).astype("float32"),
            np.squeeze(y).astype("float32"),
            self._dates[
                (i + self.sample_size) : (i + self.sample_size + self.output_size)
            ],
        )

    def __len__(self):
        """
        hours of train and test dates

        __len__(df) == fdate - tdate + output_size

        Returns:
            int: total hours
        """
        duration = self.tdate - self.fdate - dt.timedelta(hours=(self.output_size) - 2)
        return duration.days * 24 + duration.seconds // 3600

    # getter only
    @property
    def xs(self):
        """xs getter

        Returns:
            pandas.DataFrame: xs
        """
        return self._xs

    # getter only
    @property
    def ys(self):
        """ys getter

        Returns:
            pandas.DataFrame: ys
        """
        return self._ys

    @property
    def scaler(self):
        """scaler getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler
        """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        """scaler setter

        Args:
            scaler (sklearn.preprocessing.StandardScaler): Z score scaler
        """
        self._scaler = scaler.fit(self._xs)

    @property
    def df(self):
        """df getter

        Returns:
            pd.pandas.DataFrame: whole data
        """
        return self._df

    @df.setter
    def df(self, df):
        """df setter

        Args:
            df (pd.DataFrame): whole data
        """
        self._df = df

    @property
    def train_valid_ratio(self):
        """train_valid_ratio getter

        Returns:
            float: ratio of train/valid set
        """
        return self._train_valid_ratio

    @train_valid_ratio.setter
    def train_valid_ratio(self, value):
        """train_valid_ratio setter

        Args:
            value (float): ratio of traini/valid set
        """
        self._train_valid_ratio = value


class UnivariateDataset(BaseDataset):
    """Serialized Univariate Dataset without Seasonality Decomposition"""

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
        x = self._xs.iloc[i : i + self.sample_size]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]

        # return X, Y, Y_dates
        return (
            np.squeeze(x.to_numpy()).astype("float32"),
            np.squeeze(y).astype("float32"),
            self._dates[
                (i + self.sample_size) : (i + self.sample_size + self.output_size)
            ],
        )

    def to_csv(self, fpath):
        """Svae dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)


class UnivariateMeanSeasonalityDataset(BaseDataset):
    """Serialized Univariate Dataset with Seasonaltiy Decomposition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = kwargs.get("features", [self.target])
        self._df_raw = self._df.copy()
        # 2D, in Univariate -> (row x 1)
        self._xs = self._df[self.features]
        self._xs_raw = self._xs.copy()
        self._xs_sea = pd.DataFrame(index=self._xs.index)
        # 1D
        self._ys = self._df[self.target]
        self._ys.name = self.target
        # convert to DataFrame to apply ColumnTransformer easily
        self._ys = self._ys.to_frame()
        self._ys_raw = self._ys.copy()
        self._ys_sea = pd.DataFrame(index=self._ys.index)
        # self._xs must not be available when creating instance so no kwargs for scaler

        # mix ColumnTransformer & Pipeline
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        numeric_pipeline_X = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        numeric_pipeline_Y = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        # Univariate -> only tself.
        preprocessor_X = ColumnTransformer(
            transformers=[("num_sea", numeric_pipeline_X, self.features)]
        )

        preprocessor_Y = ColumnTransformer(
            transformers=[("num", numeric_pipeline_Y, [self.target])]
        )

        # univariate dataset only needs single pipeline
        self._scaler_X = kwargs.get("scaler_X", preprocessor_X)
        self._scaler_Y = kwargs.get("scaler_Y", preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i : i + self.sample_size, :]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]
        y_raw = self._ys_raw.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]

        return (
            np.squeeze(x.to_numpy()).astype("float32"),
            np.squeeze(y.to_numpy()).astype("float32"),
            np.squeeze(y_raw.to_numpy()).astype("float32"),
            y.index.to_numpy(),
        )

    def preprocess(self):
        """Compute seasonality and transform by seasonality"""
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys, y=self._ys)

        self.transform()

    def transform(self):
        """transform xs and ys as a part of preprocess to reduce training time"""
        self._xs = pd.DataFrame(
            data=self._scaler_X.transform(self._xs),
            index=self._xs.index,
            columns=self._xs.columns,
        )
        self._ys = pd.DataFrame(
            data=self._scaler_Y.transform(self._ys),
            index=self._ys.index,
            columns=self._ys.columns,
        )

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
        dfs = list(
            map(
                lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                zip(Ys, dates),
            )
        )

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) ->
        #   StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler ->
        #   (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(
            map(
                lambda b: np.squeeze(
                    self._scaler_Y.named_transformers_["num"].inverse_transform(b)
                ),
                dfs,
            )
        )

        # numpy.ndarray
        return _inv_transYs
        # pandas.DataFrame: just alter data by transformed data
        # return tuple(map(lambda b: pd.DataFrame(data=b[0],
        #                    index=b[1],
        #                    columns=[self.target]), zip(_inv_transYs, dates)))
        # return _inv_transYs

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        """Plot seasonality

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
        """
        p = self._scaler_X.named_transformers_["num"]
        p["seasonalitydecompositor"].plot(
            self._xs, self.target, self.fdate, self.tdate, data_dir, png_dir, svg_dir
        )

    def to_csv(self, fpath):
        """Svae dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        """scaler_X getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for X
        """
        return self._scaler_X

    @property
    def scaler_Y(self):
        """scaelr_Y getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for Y
        """
        return self._scaler_Y

    @property
    def ys(self):
        """ys getter

        Returns:
            pandas.DataFrame: ys
        """
        return self._ys

    @property
    def ys_raw(self):
        """ys_raw getter

        Returns:
            pandas.DataFrame: ys_raw
        """
        return self._ys_raw


class UnivariateRNNDataset(BaseDataset):
    """Non-Serialized Univariate Dataset without Seasonaltiy Decomposition"""

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
        x = self._xs.iloc[i : i + self.sample_size]
        y0 = self._xs.iloc[i + self.sample_size - 1]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]

        # return X, Y, Y_dates
        return (
            np.squeeze(x).to_numpy().astype("float32"),
            y0.astype("float32"),
            np.squeeze(y).astype("float32"),
            self._dates[
                (i + self.sample_size) : (i + self.sample_size + self.output_size)
            ],
        )

    def to_csv(self, fpath):
        """Save dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)


class UnivariateRNNMeanSeasonalityDataset(BaseDataset):
    """Non-Serialized Univariate Dataset with Seasonaltiy Decomposition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = kwargs.get("features", [self.target])
        self._df_raw = self._df.copy()
        # 2D, in Univariate -> (row x 1)
        self._xs = self._df[self.features]
        self._xs_raw = self._xs.copy()
        self._xs_sea = pd.DataFrame(index=self._xs.index)
        # 1D
        self._ys = self._df[self.target]
        self._ys.name = self.target
        self._ys = self._ys.to_frame()
        self._ys_raw = self._ys.copy()
        self._ys_sea = pd.DataFrame(index=self._ys.index)

        numeric_pipeline_X = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        numeric_pipeline_Y = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        preprocessor_X = ColumnTransformer(
            transformers=[("num_sea", numeric_pipeline_X, self.features)]
        )
        preprocessor_Y = ColumnTransformer(
            transformers=[("num", numeric_pipeline_Y, [self.target])]
        )

        # univariate dataset only needs single pipeline
        self._scaler_X = kwargs.get("scaler_X", preprocessor_X)
        self._scaler_Y = kwargs.get("scaler_Y", preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `i`

        Args:
            i(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i : i + self.sample_size]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y0 = self._xs.iloc[i + self.sample_size - 1]
        y_raw = self._ys_raw.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]

        # return X, Y, Y_dates
        return (
            np.squeeze(x.to_numpy()).astype("float32"),
            np.squeeze(y.to_numpy()).astype("float32"),
            y0.astype("float32"),
            np.squeeze(y_raw.to_numpy()).astype("float32"),
            y.index.to_numpy(),
        )

    def preprocess(self):
        """Compute seasonality and transform by seasonality"""
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys)

        self.transform()

    def transform(self):
        """Transform dataset and convert to DataFrame"""
        self._xs = pd.DataFrame(
            data=self._scaler_X.transform(self._xs),
            index=self._xs.index,
            columns=self._xs.columns,
        )
        self._ys = pd.DataFrame(
            data=self._scaler_Y.transform(self._ys),
            index=self._ys.index,
            columns=self._ys.columns,
        )

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
        dfs = list(
            map(
                lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                zip(Ys, dates),
            )
        )

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) ->
        #   StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler ->
        #   (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(
            map(
                lambda b: np.squeeze(
                    self._scaler_Y.named_transformers_["num"].inverse_transform(b)
                ),
                dfs,
            )
        )

        return _inv_transYs

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        """Plot seasonality

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
        """
        p = self._scaler_Y.named_transformers_["num"]
        p["seasonalitydecompositor"].plot(
            self._xs_raw,
            self.target,
            self.fdate,
            self.tdate,
            data_dir,
            png_dir,
            svg_dir,
        )

    def to_csv(self, fpath):
        """Save dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        """scaler_X getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for X
        """
        return self._scaler_X

    @property
    def scaler_Y(self):
        """scaelr_Y getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for Y
        """
        return self._scaler_Y

    @property
    def ys(self):
        """ys getter

        Returns:
            pandas.DataFrame: ys
        """
        return self._ys

    @property
    def ys_raw(self):
        """ys_raw getter

        Returns:
            pandas.DataFrame: ys_raw
        """
        return self._ys_raw

    @property
    def xs(self):
        """xs getter

        Returns:
            pandas.DataFrame: xs
        """
        return self._xs

    @property
    def xs_raw(self):
        """xs_raw getter

        Returns:
            pandas.DataFrame: xs_raw
        """
        return self._xs_raw


class MultivariateDataset(BaseDataset):
    """Serialized Multivariate Dataset without Seasonality Decomposition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1-step transformer
        # 1. StandardScalerWrapper
        self.features_1 = kwargs.get(
            "features_1", ["temp", "u", "v", "pres", "humid", "prep", "snow"]
        )
        # 2-step transformer
        # 1. SeasonalityDecompositor
        # 2. StandardScalerWrapper
        self.features_2 = kwargs.get(
            "features_2", ["SO2", "CO", "O3", "NO2", "PM10", "PM25"]
        )

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

        # mix ColumnTransformer & Pipeline
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        # pipeline for regression data

        numeric_pipeline_X_1 = Pipeline(
            [
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        numeric_pipeline_X_2 = Pipeline(
            [
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        numeric_pipeline_Y = Pipeline(
            [
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        # Univariate -> only pipeline needed
        # Multivariate -> Need ColumnTransformer
        preprocessor_X = ColumnTransformer(
            transformers=[
                ("num_2", numeric_pipeline_X_2, self.features_2),
                ("num_1", numeric_pipeline_X_1, self.features_1),
            ]
        )

        preprocessor_Y = ColumnTransformer(
            transformers=[("num", numeric_pipeline_Y, [self.target])]
        )

        self._scaler_X = kwargs.get("scaler_X", preprocessor_X)
        self._scaler_Y = kwargs.get("scaler_Y", preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i : i + self.sample_size]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y_raw = self._ys_raw.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]

        # return X, Y, Y_dates
        return (
            np.squeeze(self._scaler.transform(x.to_numpy())).astype("float32"),
            np.squeeze(y).astype("float32"),
            np.squeeze(y_raw.to_numpy()).astype("float32"),
            self._dates[
                (i + self.sample_size) : (i + self.sample_size + self.output_size)
            ],
        )

    def preprocess(self):
        """Fit and transform for input data"""
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys, y=self._ys)

        self.transform()

    def plot_pdf(self, png_dir, svg_dir, suffix):
        """Plot probability density function

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (Svg)
            suffix (str): filename suffix
        """
        fig = plt.figure()
        plt.title(self.target + " raw X input")
        sns.distplot(
            self._xs_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " raw Y input")
        sns.distplot(
            self._ys_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_Y_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed X input")
        sns.distplot(
            self._xs[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed Y input")
        sns.distplot(
            self._ys[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_Y_" + suffix + ".svg"))
        plt.close(fig)

    def transform(self):
        """Transform dataset and convert to DataFrame"""
        self._xs = pd.DataFrame(
            data=self._scaler_X.transform(self._xs),
            index=self._xs.index,
            columns=self._xs.columns,
        )
        self._ys = pd.DataFrame(
            data=self._scaler_Y.transform(self._ys),
            index=self._ys.index,
            columns=self._ys.columns,
        )

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
        dfs = list(
            map(
                lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                zip(Ys, dates),
            )
        )

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) ->
        #   StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler ->
        #   (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(
            map(
                lambda b: np.squeeze(
                    self._scaler_Y.named_transformers_["num"].inverse_transform(b)
                ),
                dfs,
            )
        )

        # numpy.ndarray
        return _inv_transYs

    def to_csv(self, fpath):
        """Save dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        """scaler_X getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for X
        """
        return self._scaler_X

    @property
    def scaler_Y(self):
        """scaler_Y getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for Y
        """
        return self._scaler_Y


class MultivariateMeanSeasonalityDataset(BaseDataset):
    """Serialized Multivariate Dataset with Seasonality Decomposition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1-step transformer
        # 1. StandardScalerWrapper
        self.features_1 = kwargs.get(
            "features_1", ["temp", "u", "v", "pres", "humid", "prep", "snow"]
        )
        # 2-step transformer
        # 1. SeasonalityDecompositor
        # 2. StandardScalerWrapper
        self.features_2 = kwargs.get(
            "features_2", ["SO2", "CO", "O3", "NO2", "PM10", "PM25"]
        )

        # complement of features
        self.features_c = copy.deepcopy(self.features)
        self.features_c.remove(self.target)

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
            [
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        numeric_pipeline_X_2 = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        numeric_pipeline_Y = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        # Univariate -> only pipline needed
        # Multivariate -> Need ColumnTransformer
        preprocessor_X = ColumnTransformer(
            transformers=[
                ("num_2", numeric_pipeline_X_2, self.features_2),
                ("num_1", numeric_pipeline_X_1, self.features_1),
            ]
        )

        preprocessor_Y = ColumnTransformer(
            transformers=[("num", numeric_pipeline_Y, [self.target])]
        )

        self._scaler_X = kwargs.get("scaler_X", preprocessor_X)
        self._scaler_Y = kwargs.get("scaler_Y", preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#combining-positional-and-label-based-indexing
        # get_loc -> series, get_indexer -> 1D DataFrame
        # x = self._xs.iloc[i:i+self.sample_size, self._xs.columns.get_indexer(self.features_c)]
        x = self._xs.iloc[i : i + self.sample_size, :]
        x1d = self._xs.iloc[
            i : i + self.sample_size, self._xs.columns.get_indexer([self.target])
        ]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]
        y_raw = self._ys_raw.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]

        return (
            np.squeeze(x.to_numpy()).astype("float32"),
            np.squeeze(x1d.to_numpy()).astype("float32"),
            np.squeeze(y.to_numpy()).astype("float32"),
            np.squeeze(y_raw.to_numpy()).astype("float32"),
            y.index.to_numpy(),
        )

    def preprocess(self):
        """Compute seasonality and transform by seasonality"""
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys, y=self._ys)

        self.transform()

    def plot_pdf(self, png_dir, svg_dir, suffix):
        """Plot probability density function

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (Svg)
            suffix (str): filename suffix
        """
        fig = plt.figure()
        plt.title(self.target + " raw X input")
        sns.distplot(
            self._xs_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " raw Y input")
        sns.distplot(
            self._ys_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_Y_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed X input")
        sns.distplot(
            self._xs[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed Y input")
        sns.distplot(
            self._ys[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_Y_" + suffix + ".svg"))
        plt.close(fig)

    def transform(self):
        """Transform dataset and convert to DataFrame"""
        self._xs = pd.DataFrame(
            data=self._scaler_X.transform(self._xs),
            index=self._xs.index,
            columns=self._xs.columns,
        )
        self._ys = pd.DataFrame(
            data=self._scaler_Y.transform(self._ys),
            index=self._ys.index,
            columns=self._ys.columns,
        )

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
        dfs = list(
            map(
                lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                zip(Ys, dates),
            )
        )

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) ->
        #   StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler ->
        #   (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(
            map(
                lambda b: np.squeeze(
                    self._scaler_Y.named_transformers_["num"].inverse_transform(b)
                ),
                dfs,
            )
        )

        # numpy.ndarray
        return _inv_transYs

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        """Plot seasonality

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
        """
        p = self._scaler_Y.named_transformers_["num"]

        p["seasonalitydecompositor"].plot(
            self._xs, self.target, self.fdate, self.tdate, data_dir, png_dir, svg_dir
        )

    def to_csv(self, fpath):
        """Save dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        """scaler_X getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for X
        """
        return self._scaler_X

    @property
    def scaler_Y(self):
        """scaler_Y getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for Y
        """
        return self._scaler_Y

    @property
    def xs(self):
        """xs getter

        Returns:
            pandas.DataFrame: xs
        """
        return self._xs

    @property
    def xs_raw(self):
        """xs_raw getter

        Returns:
            pandas.DataFrame: xs_raw
        """
        return self._xs_raw

    @property
    def ys(self):
        """ys getter

        Returns:
            pandas.DataFrame: ys
        """
        return self._ys

    @property
    def ys_raw(self):
        """ys_raw getter

        Returns:
            pandas.DataFrame: ys_raw
        """
        return self._ys_raw


class MultivariateRNNDataset(BaseDataset):
    """Non-Serialized Multivariate Dataset without Seasonaltiy Decomposition"""

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

        # pipeline for regression data
        numeric_pipeline_X = Pipeline(
            [
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        numeric_pipeline_Y = Pipeline(
            [
                (
                    "standardscalerwrapper",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        # Univariate -> only pipline needed
        # Multivariate -> Need ColumnTransformer
        preprocessor_X = ColumnTransformer(
            transformers=[("num", numeric_pipeline_X, self.features)]
        )

        preprocessor_Y = ColumnTransformer(
            transformers=[("num", numeric_pipeline_Y, [self.target])]
        )

        # univariate dataset only needs single pipeline
        self._scaler_X = kwargs.get("scaler_X", preprocessor_X)
        self._scaler_Y = kwargs.get("scaler_Y", preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`

        Args:
            di(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i : i + self.sample_size]
        x_1d = self._ys.iloc[i : (i + self.sample_size)]
        # save initial input as target variable input at last step (single step)
        y0 = self._ys.iloc[i + self.sample_size - 1]
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y_raw = self._ys_raw.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]

        # return X, Y, Y_dates
        return (
            np.squeeze(x).to_numpy().astype("float32"),
            np.squeeze(x_1d).to_numpy().astype("float32"),
            y0.astype("float32"),
            np.squeeze(y).astype("float32"),
            np.squeeze(y_raw).astype("float32"),
            self._dates[
                (i + self.sample_size) : (i + self.sample_size + self.output_size)
            ],
        )

    def preprocess(self):
        """Transform by scaler"""
        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys)

        self.transform()

    def plot_pdf(self, png_dir, svg_dir, suffix):
        """Plot probability density function

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (Svg)
            suffix (str): filename suffix
        """
        fig = plt.figure()
        plt.title(self.target + " raw X input")
        sns.distplot(
            self._xs_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " raw Y input")
        sns.distplot(
            self._ys_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_Y_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed X input")
        sns.distplot(
            self._xs[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed Y input")
        sns.distplot(
            self._ys[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_Y_" + suffix + ".svg"))
        plt.close(fig)

    def transform(self):
        """Transform dataset and convert to DataFrame"""
        self._xs = pd.DataFrame(
            data=self._scaler_X.transform(self._xs),
            index=self._xs.index,
            columns=self._xs.columns,
        )
        self._ys = pd.DataFrame(
            data=self._scaler_Y.transform(self._ys),
            index=self._ys.index,
            columns=self._ys.columns,
        )

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
        dfs = list(
            map(
                lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                zip(Ys, dates),
            )
        )

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) ->
        #   StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler ->
        #   (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(
            map(
                lambda b: np.squeeze(
                    self._scaler_Y.named_transformers_["num"].inverse_transform(b)
                ),
                dfs,
            )
        )

        return _inv_transYs

    def to_csv(self, fpath):
        """Save dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        """scaler_X getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for X
        """
        return self._scaler_X

    @property
    def scaler_Y(self):
        """scaler_Y getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for Y
        """
        return self._scaler_Y

    @property
    def ys(self):
        """ys getter

        Returns:
            pandas.DataFrame: ys
        """
        return self._ys


class MultivariateRNNMeanSeasonalityDataset(BaseDataset):
    """Non-Serialized Multivariate Dataset with Seasonaltiy Decomposition"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1-step transformer
        # 1. StandardScalerWrapper
        self.features = kwargs.get(
            "features",
            [
                "temp",
                "u",
                "v",
                "pres",
                "humid",
                "prep",
                "snow",
                "SO2",
                "CO",
                "O3",
                "NO2",
                "PM10",
                "PM25",
            ],
        )
        self.features_1 = kwargs.get(
            "features_1", ["temp", "u", "v", "pres", "humid", "prep", "snow"]
        )
        # 2-step transformer
        # 1. SeasonalityDecompositor
        # 2. StandardScalerWrapper
        self.features_2 = kwargs.get(
            "features_2", ["SO2", "CO", "O3", "NO2", "PM10", "PM25"]
        )
        self.features_sea_embed = kwargs.get("features_sea_embed", [self.target])

        self._df_raw = self._df.copy()
        # 2D, in Univariate -> (row x 1)
        self._xs = self._df[self.features]
        self._xs_raw = self._xs.copy()
        self._xs_sea = pd.DataFrame(index=self._xs.index)
        # 1D
        self._ys = self._df[self.target]
        self._ys.name = self.target
        # convert to DataFrame to apply ColumnTransformer easily
        self._ys = self._ys.to_frame()
        self._ys_raw = self._ys.copy()
        self._ys_sea = pd.DataFrame(index=self._ys.index)

        # mix ColumnTransformer & Pipeline
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

        # numeric_pipeline_X_std = Pipeline(
        #     [('powertransformer', PowerTransformerWrapper(scaler=PowerTransformer()))])

        # numeric_pipeline_X_sea = Pipeline(
        #     [('seasonalitydecompositor',
        #         SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05)),
        #      ('powertransformer', PowerTransformerWrapper(scaler=PowerTransformer()))])

        # numeric_pipeline_Y = Pipeline(
        #     [('seasonalitydecompositor',
        #         SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05)),
        #      ('powertransformer', PowerTransformerWrapper(scaler=PowerTransformer()))])

        numeric_pipeline_X_std = Pipeline(
            [
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                )
            ]
        )

        numeric_pipeline_X_sea = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        numeric_pipeline_Y = Pipeline(
            [
                (
                    "seasonalitydecompositor",
                    SeasonalityDecompositor_AWH(smoothing=True, smoothing_frac=0.05),
                ),
                (
                    "standardtransformer",
                    StandardScalerWrapper(scaler=StandardScaler(with_std=False)),
                ),
            ]
        )

        # Univariate -> only pipline needed
        # Multivariate -> Need ColumnTransformer
        preprocessor_X = ColumnTransformer(
            transformers=[
                ("num_sea", numeric_pipeline_X_sea, self.features_2),
                ("num_std", numeric_pipeline_X_std, self.features_1),
            ]
        )

        preprocessor_Y = ColumnTransformer(
            transformers=[("num", numeric_pipeline_Y, [self.target])]
        )

        # univariate dataset only needs single pipeline
        self._scaler_X = kwargs.get("scaler_X", preprocessor_X)
        self._scaler_Y = kwargs.get("scaler_Y", preprocessor_Y)

    def __getitem__(self, i: int):
        """
        get X, Y for given index `i`

        Args:
            i(datetime): datetime where output starts

        Returns:
            Tensor: transformed input (might be normalized)
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        x = self._xs.iloc[i : i + self.sample_size, :]
        # want 1D array of target column, so use ys instead
        x_1d = self._ys.iloc[i : (i + self.sample_size)]
        x_sa = self._xs_sea["annual"].iloc[i : (i + self.sample_size)]
        x_sw = self._xs_sea["weekly"].iloc[i : (i + self.sample_size)]
        x_sh = self._xs_sea["annual"].iloc[i : (i + self.sample_size)]
        # save initial input as target variable input at last step (single step)
        y = self._ys.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y_sa = self._ys_sea["annual"].iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y_sw = self._ys_sea["weekly"].iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y_sh = self._ys_sea["annual"].iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size)
        ]
        y_raw = self._ys_raw.iloc[
            (i + self.sample_size) : (i + self.sample_size + self.output_size), :
        ]

        # To embed dates, x and y is DataFrame
        return (
            np.squeeze(x.to_numpy()).astype("float32"),
            np.squeeze(x_1d.to_numpy()).astype("float32"),
            np.squeeze(x_sa.to_numpy()).astype("float32"),
            np.squeeze(x_sw.to_numpy()).astype("float32"),
            np.squeeze(x_sh.to_numpy()).astype("float32"),
            np.squeeze(y.to_numpy()).astype("float32"),
            np.squeeze(y_raw.to_numpy()).astype("float32"),
            np.squeeze(y_sa.to_numpy()).astype("float32"),
            np.squeeze(y_sw.to_numpy()).astype("float32"),
            np.squeeze(y_sh.to_numpy()).astype("float32"),
            self._dates[
                (i + self.sample_size) : (i + self.sample_size + self.output_size)
            ],
        )

    def preprocess(self):
        """Compute seasonality and transform by seasonality"""
        # plot correlation matrix

        # compute seasonality
        self._scaler_X.fit(self._xs)
        self._scaler_Y.fit(self._ys)

        self.transform()

    def plot_pdf(self, png_dir, svg_dir, suffix):
        """Plot probability density function

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (Svg)
            suffix (str): filename suffix
        """
        fig = plt.figure()
        plt.title(self.target + " raw X input")
        sns.distplot(
            self._xs_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " raw Y input")
        sns.distplot(
            self._ys_raw[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_raw_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_raw_Y_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed X input")
        sns.distplot(
            self._xs[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_X_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_X_" + suffix + ".svg"))
        plt.close(fig)

        fig = plt.figure()
        plt.title(self.target + " transformed Y input")
        sns.distplot(
            self._ys[self.target],
            hist=False,
            kde=True,
            kde_kws={"linewidth": 3},
            label=self.target,
        )
        plt.savefig(png_dir / (self.target + "_tf_Y_" + suffix + ".png"))
        plt.savefig(svg_dir / (self.target + "_tf_Y_" + suffix + ".svg"))
        plt.close(fig)

    def transform(self):
        """Transform dataset and convert to DataFrame"""
        self._xs = pd.DataFrame(
            data=self._scaler_X.transform(self._xs),
            index=self._xs.index,
            columns=self._xs.columns,
        )
        self._ys = pd.DataFrame(
            data=self._scaler_Y.transform(self._ys),
            index=self._ys.index,
            columns=self._ys.columns,
        )

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
        dfs = list(
            map(
                lambda b: pd.DataFrame(data=b[0], index=b[1], columns=[self.target]),
                zip(Ys, dates),
            )
        )

        # execute pipeline's inverse transform
        # transform : (DataFrame) -> SeasonalityDecompositor (needs date) ->
        #   StandardScaler -> (ndarray)
        # inverse_transform : (ndarray) -> StandardScaler ->
        #   (ndarray) -> (DataFrame) -> SeasonaltyDecompositor -> (ndarray)
        _inv_transYs = tuple(
            map(
                lambda b: np.squeeze(
                    self._scaler_Y.named_transformers_["num"].inverse_transform(b)
                ),
                dfs,
            )
        )

        return _inv_transYs

    def broadcast_seasonality(self):
        """Build seasonality DataFrame by DateTimeIndex"""
        p = self._scaler_X.named_transformers_["num_sea"]

        # extract seasonality from seasonalitydecompositor
        _sea_annual = p.get_params()["seasonalitydecompositor__sea_annual"]
        _sea_weekly = p.get_params()["seasonalitydecompositor__sea_weekly"]
        _sea_hourly = p.get_params()["seasonalitydecompositor__sea_hourly"]

        # filter target
        sea_annual = _sea_annual[self.target]
        sea_weekly = _sea_weekly[self.target]
        sea_hourly = _sea_hourly[self.target]

        def key_md(idx): return "".join((str(idx.month).zfill(2), str(idx.day).zfill(2)))
        def key_w(idx): return str(idx.dayofweek)
        def key_h(idx): return str(idx.hour).zfill(2)

        def get_sea_annual(idx): return sea_annual[key_md(idx)]
        def get_sea_weekly(idx): return sea_weekly[key_w(idx)]
        def get_sea_hourly(idx): return sea_hourly[key_h(idx)]

        self._xs_sea["annual"] = self._xs_sea.index.map(get_sea_annual)
        self._xs_sea["weekly"] = self._xs_sea.index.map(get_sea_weekly)
        self._xs_sea["hourly"] = self._xs_sea.index.map(get_sea_hourly)

        self._ys_sea["annual"] = self._ys_sea.index.map(get_sea_annual)
        self._ys_sea["weekly"] = self._ys_sea.index.map(get_sea_weekly)
        self._ys_sea["hourly"] = self._ys_sea.index.map(get_sea_hourly)

        return sea_annual, sea_weekly, sea_hourly

    def plot_seasonality(self, data_dir, png_dir, svg_dir):
        """Plot seasonality

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
        """
        p = self._scaler_Y.named_transformers_["num"]
        p["seasonalitydecompositor"].plot(
            self._xs_raw,
            self.target,
            self.fdate,
            self.tdate,
            data_dir,
            png_dir,
            svg_dir,
        )

    def plot_fused_seasonality(self, data_dir, png_dir, svg_dir):
        """Plot seasonality in single plot

        Args:
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
        """
        Path.mkdir(png_dir, parents=True, exist_ok=True)
        Path.mkdir(svg_dir, parents=True, exist_ok=True)
        Path.mkdir(data_dir, parents=True, exist_ok=True)
        p = self._scaler_Y.named_transformers_["num"]
        p["seasonalitydecompositor"].plot_fused(
            self._xs_raw,
            self.target,
            self.fdate,
            self.tdate,
            data_dir,
            png_dir,
            svg_dir,
        )

    def to_csv(self, fpath):
        """Save dataset to csv

        Args:
            fpath (Path): csv file path
        """
        self._df.to_csv(fpath)

    @property
    def scaler_X(self):
        """scaler_X getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for X
        """
        return self._scaler_X

    @property
    def scaler_Y(self):
        """scaler_Y getter

        Returns:
            sklearn.preprocessing.StandardScaler: scaler for Y
        """
        return self._scaler_Y

    @property
    def ys(self):
        """ys getter

        Returns:
            pandas.DataFrame: ys
        """
        return self._ys

    @property
    def ys_raw(self):
        """ys_raw getter

        Returns:
            pandas.DataFrame: ys_raw
        """
        return self._ys_raw

    @property
    def xs(self):
        """xs getter

        Returns:
            pandas.DataFrame: xs
        """
        return self._xs

    @property
    def xs_raw(self):
        """xs_raw getter

        Returns:
            pandas.DataFrame: xs_raw
        """
        return self._xs_raw


# utility functions
def ydt2key(d):
    """parse date and generate key for annual seasonality

    Args:
        d (datetime): datetime object

    Returns:
        str: `mmdd` date string
    """
    return str(d.astimezone(SEOULTZ).month).zfill(2) + str(
        d.astimezone(SEOULTZ).day
    ).zfill(2)


def wdt2key(d):
    """parse date and generate key for weekly seasonality

    Args:
        d (datetime): datetime object

    Returns:
        str: `w` date string
    """
    return str(d.astimezone(SEOULTZ).weekday())


def hdt2key(d):
    """parse date and generate key for hourly seasonality

    Args:
        d (datetime): datetime object

    Returns:
        str: `hh` date string
    """
    return str(d.astimezone(SEOULTZ).hour).zfill(2)


def confint_idx(acf, confint: float):
    """Find confidence interval index

    If acf is below than confidence interval, return index of acf array

    Args:
        acf (numpy.array): autocorrelation function array
        confint (float): confidence interval given by statsmodels

    Returns:
        int: index of acf array where acf is below confidence interval
    """
    for i, (a, [c1, c2]) in enumerate(zip(acf, confint)):
        if c1 - a < a and a < c2 - a:
            return i

    return len(acf) - 1


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

    def __init__(
        self,
        sea_annual=None,
        sea_weekly=None,
        sea_hourly=None,
        smoothing=True,
        smoothing_frac=0.05,
    ):
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

        # args, _, _, values = inspect.getargvalues(inspect.currentframe())
        # values.pop("self")

        self.sea_annual = sea_annual
        self.sea_weekly = sea_weekly
        self.sea_hourly = sea_hourly
        self.smoothing = smoothing
        self.smoothing_frac = smoothing_frac

    def set_sesaonality(self, sea_annual, sea_weekly, sea_hourly):
        """seasonality setter

        Args:
            sea_annual (List): Daliy averaged annual seasonality (365d)
            sea_weekly (List): Daily averaged weekly seasonality (7d)
            sea_hourly (List): Hourly averaged daily seasonality (24h)
        """
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
                """Method for rowwise operation"""
                if period == "y":
                    return self.sea_annual[name][parse_ykey(idx)]
                elif period == "w":
                    return self.sea_weekly[name][parse_wkey(idx)]
                elif period == "h":
                    return self.sea_hourly[name][parse_hkey(idx)]
                else:
                    raise Exception("Invalid period ('y', 'w', 'h'): ", period)

            # index to sum of seasonality
            sea = _xs.index.map(_get_seasonality)
            _xs_df = _xs.to_frame()
            _xs_df["seas"] = sea

            return sea

        # = Y.apply(get_seasonality, axis=1)
        seas = xs.apply(_build_seasonality, 0)

        # raw = Y.apply(add_seasonality, 0)
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
        X_h.columns = ["raw"]

        if X_h.isnull().values.any():
            raise Exception("Invalid data, NULL value found")
        X_d = X_h.resample("D").mean()

        # 1. Annual Seasonality (mmdd: 0101 ~ 1231)
        # dictionary for key (mmdd) to value (daily average)
        sea_annual, df_sea_annual, df_resid_annual = periodic_mean(
            X_d,
            "raw",
            "y",
            smoothing=self.smoothing,
            smoothing_frac=self.smoothing_frac,
        )

        if "0229" not in sea_annual:
            raise KeyError("You must include leap year in train data")

        # 2. Weekly Seasonality (w: 0 ~ 6)
        # dictionary for key (w: 0~6) to value (daily average)
        # Weekly seasonality computes seasonality from annual residaul
        sea_weekly, df_sea_weekly, _ = periodic_mean(
            df_resid_annual.copy(), "resid", "w"
        )

        # 3. Hourly Seasonality (hh: 00 ~ 23)
        # join seasonality to original X_h DataFrame
        # populate seasonality from daily averaged data to hourly data

        ## method to generate key column from index
        def key_md(idx): return "".join((str(idx.month).zfill(2), str(idx.day).zfill(2)))
        def key_w(idx): return str(idx.dayofweek)
        def key_h(idx): return str(idx.hour).zfill(2)

        ## Add column from index for key
        df_sea_weekly["key_w"] = df_sea_weekly.index
        df_sea_annual["key_md"] = df_sea_annual.index

        X_h["key_md"] = X_h.index.map(key_md)
        X_h["key_w"] = X_h.index.map(key_w)
        X_h["key_h"] = X_h.index.map(key_h)

        ## compute hourly populated seasonality from daily residuals
        X_h_spop = df_sea_annual.merge(
            X_h, how="left", on="key_md", validate="1:m"
        ).dropna()
        X_h_spop.set_index(X_h.index, inplace=True)
        X_h_spop = X_h_spop.rename(columns={"sea": "sea_annual"})
        X_h_spop = df_sea_weekly.merge(
            X_h_spop, how="left", on="key_w", validate="1:m"
        ).dropna()
        X_h_spop = X_h_spop.rename(columns={"sea": "sea_weekly"})
        X_h_spop.set_index(X_h.index, inplace=True)

        ## new hourly residual column from daily residuals
        X_h_spop["resid_y"] = X_h_spop["raw"] - X_h_spop["sea_annual"]
        X_h_spop["resid_w"] = (
            X_h_spop["raw"] - X_h_spop["sea_annual"] - X_h_spop["sea_weekly"]
        )
        X_h_spop["resid_d"] = (
            X_h_spop["raw"] - X_h_spop["sea_annual"] - X_h_spop["sea_weekly"]
        )

        ## Check no missing values
        if len(X_h_spop.index) != len(X_h.index):
            raise Exception(
                "Merge Error: something missing when populate daily averaged DataFrame"
            )

        ## Compute hourly seasonality
        sea_hourly, df_sea_hourly, _ = periodic_mean(X_h_spop.copy(), "resid_d", "h")

        ## Add column from index for key
        df_sea_hourly["key_h"] = df_sea_hourly.index

        ## merge hourly seasonality to orignal hourly DataFram
        X_h_hourly = df_sea_hourly.merge(
            X_h_spop, how="left", on="key_h", validate="1:m"
        ).dropna()
        X_h_hourly = X_h_hourly.rename(columns={"sea": "sea_hourly"})
        X_h_spop.set_index(X_h.index, inplace=True)
        ## Subtract annual and weekly seasonality
        X_h_spop["sea_hourly"] = X_h_hourly["sea_hourly"]
        X_h_spop["resid"] = X_h_spop["resid_d"] - X_h_hourly["sea_hourly"]
        X_h_spop["resid_h"] = X_h_spop["resid_d"] - X_h_hourly["sea_hourly"]
        ## Sort by DateTimeIndex
        X_h_spop = X_h_spop.sort_index()

        # 3.2 drop key columns and intermediate residual column
        X_h_spop.drop("resid_d", axis="columns")
        X_h_spop.drop("key_md", axis="columns")
        X_h_spop.drop("key_w", axis="columns")
        X_h_spop.drop("key_h", axis="columns")

        if return_resid is True:
            return (
                X_h_spop["resid_y"],
                X_h_spop["resid_w"],
                X_h_spop,
                X_h_spop["resid_h"],
            )

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
        if (
            self.sea_annual is not None
            and self.sea_weekly is not None
            and self.sea_hourly is not None
        ):
            # if test set
            return self

        # apply function to column-wise
        # self.sea_annual, self.sea_weekly, self.sea_hourly = \
        seas = X.apply(self.compute_seasonality, 0).to_dict(orient="records")
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
            """Method for columnwise operation"""
            # to subscript seasonality, get name (feature)
            name = _X.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation"""
                return (
                    self.sea_annual[name][parse_ykey(idx)]
                    + self.sea_weekly[name][parse_wkey(idx)]
                    + self.sea_hourly[name][parse_hkey(idx)]
                )

            # index to sum of seasonality
            seas = _X.index.map(_sum_seasonality)
            _X_df = _X.to_frame()
            _X_df["seas"] = -seas

            _X_sum = _X_df.sum(axis="columns")
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
            """Method for columnwise operation"""
            # to subscript seasonality, get name (feature)
            name = _Y.name

            # method for row
            def _sum_seasonality(idx: pd.DatetimeIndex):
                """Method for rowwise operation"""
                return (
                    self.sea_annual[name][parse_ykey(idx)]
                    + self.sea_weekly[name][parse_wkey(idx)]
                    + self.sea_hourly[name][parse_hkey(idx)]
                )

            # index to sum of seasonality
            seas = _Y.index.map(_sum_seasonality)
            _Y_df = _Y.to_frame()
            _Y_df["seas"] = seas

            _Y_sum = _Y_df.sum(axis="columns")
            _Y_sum.name = name

            return _Y_sum

        # there is no way to pass row object of Series,
        # so I pass index and use Series.get() with closure
        # raw = Y.index.map(add_seasonality)
        raw = Y.apply(add_seasonality, 0)

        return raw.to_numpy()

    def plot_raw_acf(
        self, df, target, fdate, tdate, data_dir, png_dir, svg_dir, nlags=15
    ):
        """Plot acf without seasonality decomposition

        Args:
            df (pandas.DataFrame): DataFrame that holds total data
            target (str): DataFrame column to compute acf
            fdate (datetime): start date of target range
            tdate (datetime): end date of target range
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            nlags (int, optional): number of lags to compute acf.
                Defaults to 15.
        """
        # yr = [df.loc[y, target] -
        #      self.sea_annual[target][ydt2key(y)] for y in year_range]
        ys = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        # 95% confidance intervals
        ys_acf, confint, _, _ = sm.tsa.acf(ys, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(ys_acf[0 : confint_idx(ys_acf, confint) + 1])
        print("Raw Conf Int  : ", confint_idx(ys_acf, confint))
        print("Raw Int Scale : ", intscale)
        # print("Raw qstat : ", qstat)
        # print("Raw pvalues : ", pvalues)

        csv_path = data_dir / (
            "acf_raw_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".csv"
        )
        df_year_acf = pd.DataFrame.from_dict(
            {"lags": [i for i in range(len(ys_acf))], "yr_acf": ys_acf}
        )
        df_year_acf.set_index("lags", inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / (
            "acf_raw_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf_raw_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".svg"
        )
        p1 = figure(title="Autocorrelation of " + target)
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(ys_acf)), 1.1)
        p1.line(
            x=[i * 24 for i in range(len(ys_acf))],
            y=ys_acf,
            line_color="lightcoral",
            line_width=2,
        )
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / (
            "acf(tpl)_raw_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf(tpl)_raw_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".svg"
        )
        fig = tpl.plot_acf(ys, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return confint_idx(ys_acf, confint), intscale

    def plot_annual(
        self, df, target, data_dir, png_dir, svg_dir, target_year=2016, smoothing=True
    ):
        """Plot annual seasonality

        Args:
            df (pandas.DataFrame): DataFrame that holds total data
            target (str): DataFrame column to compute acf
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            nlags (int, optional): number of lags to compute acf Defaults to 15.
            target_year (int, optional): year to plot. Defaults to 2016.
            smoothing (bool, optional): whether plot smoothed seasonality together.
                Defaults to True.
        """

        # annual (pick 1 year)
        dt1 = dt.datetime(year=target_year, month=1, day=1, hour=0)
        dt2 = dt.datetime(year=target_year, month=12, day=31, hour=23)
        year_range = pd.date_range(start=dt1, end=dt2, freq="D", tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex(
            [
                i.replace(tzinfo=None)
                for i in pd.date_range(start=dt1, end=dt2, freq="D", tz=SEOULTZ)
            ]
        )

        # if not smoothing, just use predecomposed seasonality
        ys = [self.sea_annual[target][ydt2key(y)] for y in year_range]
        if smoothing:
            _sea_annual_nonsmooth, _, _ = periodic_mean(
                df, target, "y", smoothing=False
            )
            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            ys_vanilla = [_sea_annual_nonsmooth[ydt2key(y)] for y in year_range]
            ys_smooth = [self.sea_annual[target][ydt2key(y)] for y in year_range]

        csv_path = data_dir / (
            "annual_seasonality_"
            + dt1.strftime("%Y%m%d")
            + "_"
            + dt2.strftime("%Y%m%d")
            + ".csv"
        )
        df_year = pd.DataFrame.from_dict({"date": year_range, "ys": ys})
        if smoothing:
            df_year = pd.DataFrame.from_dict(
                {"date": year_range, "ys_vanilla": ys_vanilla, "ys_smooth": ys_smooth}
            )
        df_year.set_index("date", inplace=True)
        df_year.to_csv(csv_path)

        png_path = png_dir / (
            "annual_seasonality_"
            + dt1.strftime("%Y%m%d")
            + "_"
            + dt2.strftime("%Y%m%d")
            + ".png"
        )
        svg_path = svg_dir / (
            "annual_seasonality_"
            + dt1.strftime("%Y%m%d")
            + "_"
            + dt2.strftime("%Y%m%d")
            + ".svg"
        )

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.formatter = DatetimeTickFormatter(
            days="%m/%d", months="%b", hours="%H:%M"
        )
        p1.xaxis.ticker = bokeh.models.DatetimeTicker()
        p1.line(x=year_range_plt, y=ys, line_color="dodgerblue", line_width=2)
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        if smoothing:
            # 1. smooth
            png_path = png_dir / (
                "annual_seasonality_(smooth)_"
                + dt1.strftime("%Y%m%d")
                + "_"
                + dt2.strftime("%Y%m%d")
                + ".png"
            )
            svg_path = svg_dir / (
                "annual_seasonality_(smooth)_"
                + dt1.strftime("%Y%m%d")
                + "_"
                + dt2.strftime("%Y%m%d")
                + ".svg"
            )
            p1 = figure(title="Annual Seasonality(Smooth)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(
                days="%m/%d", months="%b", hours="%H:%M"
            )
            p1.xaxis.ticker = bokeh.models.DatetimeTicker()
            p1.line(
                x=year_range_plt, y=ys_smooth, line_color="dodgerblue", line_width=2
            )
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 2. vanila
            png_path = png_dir / (
                "annual_seasonality_(vanilla)_"
                + dt1.strftime("%Y%m%d")
                + "_"
                + dt2.strftime("%Y%m%d")
                + ".png"
            )
            svg_path = svg_dir / (
                "annual_seasonality_(vanilla)_"
                + dt1.strftime("%Y%m%d")
                + "_"
                + dt2.strftime("%Y%m%d")
                + ".svg"
            )
            p1 = figure(title="Annual Seasonality(Raw)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(
                days="%m/%d", months="%b", hours="%H:%M"
            )
            p1.xaxis.ticker = bokeh.models.DatetimeTicker()
            p1.line(
                x=year_range_plt, y=ys_vanilla, line_color="dodgerblue", line_width=2
            )
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

            # 3. smooth + vanila
            png_path = png_dir / (
                "annual_seasonality_(both)_"
                + dt1.strftime("%Y%m%d")
                + "_"
                + dt2.strftime("%Y%m%d")
                + ".png"
            )
            svg_path = svg_dir / (
                "annual_seasonality_(both)_"
                + dt1.strftime("%Y%m%d")
                + "_"
                + dt2.strftime("%Y%m%d")
                + ".svg"
            )
            p1 = figure(title="Annual Seasonality(Smooth & Vanilla)")
            p1.xaxis.axis_label = "dates"
            p1.toolbar.logo = None
            p1.toolbar_location = None
            p1.xaxis.formatter = DatetimeTickFormatter(
                days="%m/%d", months="%b", hours="%H:%M"
            )
            p1.xaxis.ticker = bokeh.models.DatetimeTicker()
            p1.line(
                x=year_range_plt,
                y=ys_smooth,
                line_color="dodgerblue",
                line_width=2,
                legend_label="smooth",
            )
            p1.line(
                x=year_range_plt,
                y=ys_vanilla,
                line_color="lightcoral",
                line_width=2,
                legend_label="vanilla",
            )
            export_png(p1, filename=png_path)
            p1.output_backend = "svg"
            export_svgs(p1, filename=str(svg_path))

    def plot_annual_acf(
        self, df, target, fdate, tdate, data_dir, png_dir, svg_dir, nlags=15
    ):
        """Plot acf of residuals which removed annual seasonality

        Args:
            df (pandas.DataFrame): DataFrame that holds total data
            target (str): DataFrame column to compute acf
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            nlags (int, optional): number of lags to compute acf.
                Defaults to 15.
            target_year (int, optional): year to plot. Defaults to 2016.
            smoothing (bool, optional): whether plot smoothed seasonality together.
                Defaults to True.

        Returns:
            None
        """
        ## residual autocorrelation by tsa.acf
        year_range = (
            pd.date_range(start=fdate, end=tdate, freq="D").tz_convert(SEOULTZ).tolist()
        )
        # set hour to 0
        year_range = [y.replace(hour=0) for y in year_range]

        # yr = [df.loc[y, target] -
        #      self.sea_annual[target][ydt2key(y)] for y in year_range]
        yr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()

        # 95% confidance intervals
        yr_acf, confint, qstat, pvalues = sm.tsa.acf(
            yr, qstat=True, alpha=0.05, nlags=nlags
        )

        # I need "hours" as unit, so multiply 24
        intscale = sum(yr_acf[0 : confint_idx(yr_acf, confint) + 1]) * 24
        print("Annual Conf Int  : ", confint_idx(yr_acf, confint))
        print("Annual Int Scale : ", intscale)
        print("Annual qstat : ", qstat)
        print("Annual pvalues : ", pvalues)

        csv_path = data_dir / (
            "acf_annual_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".csv"
        )
        df_year_acf = pd.DataFrame.from_dict(
            {"lags": [i * 24 for i in range(len(yr_acf))], "yr_acf": yr_acf}
        )
        df_year_acf.set_index("lags", inplace=True)
        df_year_acf.to_csv(csv_path)

        png_path = png_dir / (
            "acf_annual_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf_annual_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".svg"
        )
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.toolbar.logo = None
        p1.toolbar_location = None
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line(
            x=[i * 24 for i in range(len(yr_acf))],
            y=yr_acf,
            line_color="lightcoral",
            line_width=2,
        )
        export_png(p1, filename=png_path)
        p1.output_backend = "svg"
        export_svgs(p1, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / (
            "acf(tpl)_annual_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf(tpl)_annual_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".svg"
        )
        fig = tpl.plot_acf(yr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return confint_idx(yr_acf, confint), intscale

    def plot_weekly(
        self, target, data_dir, png_dir, svg_dir, target_year=2016, target_week=10
    ):
        """Plot weekly seasonality

        Args:
            target (str): DataFrame column to compute acf
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            target_year (int, optional): year to plot. Defaults to 2016.
            target_week (int, optional): week to plot. Defaults to 10.

        Returns:
            None
        """
        # Set datetime range
        target_dt_str = str(target_year) + " " + str(target_week)
        ## ISO 8601 starts weekday from Monday (1)
        dt1 = dt.datetime.strptime(target_dt_str + " 1", "%Y %W %w")
        ## ISO 8601 ends weekday to Sunday (0)
        dt2 = dt.datetime.strptime(target_dt_str + " 0", "%Y %W %w")
        ## set dt2 time as Sunday 23:59
        dt2 = dt2.replace(hour=23, minute=59)
        week_range = pd.date_range(start=dt1, end=dt2, freq="D", tz=SEOULTZ).tolist()
        week_range_plt = pd.DatetimeIndex(
            [
                i.replace(tzinfo=None)
                for i in pd.date_range(start=dt1, end=dt2, freq="D", tz=SEOULTZ)
            ]
        )

        # Compute Weekly seasonality
        ws = [self.sea_weekly[target][wdt2key(w)] for w in week_range]

        csv_path = data_dir / (
            "weekly_seasonality_"
            + dt1.strftime("%Y%m%d")
            + "_"
            + dt2.strftime("%Y%m%d")
            + ".csv"
        )
        df_week = pd.DataFrame.from_dict({"day": week_range, "ws": ws})
        df_week.set_index("day", inplace=True)
        df_week.to_csv(csv_path)

        png_path = png_dir / (
            "weekly_seasonality_"
            + dt1.strftime("%Y%m%d")
            + "_"
            + dt2.strftime("%Y%m%d")
            + ".png"
        )
        svg_path = svg_dir / (
            "weekly_seasonality_"
            + dt1.strftime("%Y%m%d")
            + "_"
            + dt2.strftime("%Y%m%d")
            + ".svg"
        )

        p2 = figure(title="Weekly Seasonality")
        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.xaxis.axis_label = "day"
        p2.xaxis.formatter = DatetimeTickFormatter(days="%a")
        p2.xaxis.ticker = bokeh.models.DaysTicker(days=[w.day for w in week_range_plt])
        p2.line(x=week_range_plt, y=ws, line_color="dodgerblue", line_width=2)
        export_png(p2, filename=png_path)
        p2.output_backend = "svg"
        export_svgs(p2, filename=str(svg_path))

    def plot_weekly_acf(
        self, df, target, fdate, tdate, data_dir, png_dir, svg_dir, nlags=15
    ):
        """Plot acf of residuals which removed weekly seasonality

        Args:
            df (pandas.DataFrame): DataFrame that holds residuals
            target (str): DataFrame column to compute acf
            fdate (datetime): start date of target range
            tdate (datetime): end date of target range
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            nlags (int, optional): number of lags to compute acf.
                Defaults to 15.

        Returns:
            None
        """
        # Set datetime range
        # target_dt_str = str(target_year) + ' ' + str(target_week)
        week_range = (
            pd.date_range(start=fdate, end=tdate, freq="D").tz_convert(SEOULTZ).tolist()
        )
        # set hour to 0
        week_range = [w.replace(hour=0) for w in week_range]

        # wr = [df.loc[w, target] -
        #      self.sea_weekly[target][wdt2key(w)] for w in week_range]
        wr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        wr_acf, confint, qstat, pvalues = sm.tsa.acf(
            wr, qstat=True, alpha=0.05, nlags=nlags
        )

        # I need "hours" as unit, so multiply 24
        intscale = sum(wr_acf[0 : confint_idx(wr_acf, confint) + 1]) * 24
        print("Weekly Conf Int  : ", confint_idx(wr_acf, confint))
        print("Weekly Int Scale : ", intscale)
        print("Weekly qstat : ", qstat)
        print("Weekly pvalues : ", pvalues)

        csv_path = data_dir / (
            "acf_weekly_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".csv"
        )
        df_week_acf = pd.DataFrame.from_dict(
            {"lags": [i * 24 for i in range(len(wr_acf))], "wr_acf": wr_acf}
        )
        df_week_acf.set_index("lags", inplace=True)
        df_week_acf.to_csv(csv_path)

        png_path = png_dir / (
            "acf_weekly_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf_weekly_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".svg"
        )
        p2 = figure(title="Autocorrelation of Weekly Residual")
        p2.toolbar.logo = None
        p2.toolbar_location = None
        p2.xaxis.axis_label = "lags"
        p2.yaxis.bounds = (min(0, min(wr_acf)), 1.1)
        p2.line(
            x=[i * 24 for i in range(len(wr_acf))],
            y=wr_acf,
            line_color="lightcoral",
            line_width=2,
        )
        export_png(p2, filename=png_path)
        p2.output_backend = "svg"
        export_svgs(p2, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / (
            "acf(tpl)_weekly_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf(tpl)_weekly_seasonality_"
            + fdate.strftime("%Y%m%d")
            + "_"
            + tdate.strftime("%Y%m%d")
            + ".svg"
        )
        fig = tpl.plot_acf(wr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return confint_idx(wr_acf, confint), intscale

    def plot_hourly(
        self,
        target,
        data_dir,
        png_dir,
        svg_dir,
        target_year=2016,
        target_month=5,
        target_day=1,
    ):
        """Plot hourly seasonality

        Args:
            target (str): DataFrame column to compute acf
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            target_year (int, optional): year to plot. Defaults to 2016.
            target_week (int, optional): week to plot. Defaults to 10.
            target_day (int, optional): day to plot. Defaults to 1.

        Returns:
            None
        """
        dt1 = dt.datetime(year=target_year, month=target_month, day=target_day, hour=0)
        dt2 = dt.datetime(year=target_year, month=target_month, day=target_day, hour=23)
        hour_range = pd.date_range(start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        hour_range_plt = pd.DatetimeIndex(
            [
                i.replace(tzinfo=None)
                for i in pd.date_range(start=dt1, end=dt2, freq="1H", tz=SEOULTZ)
            ]
        ).tolist()

        # Compute Hourly seasonality
        hs = [self.sea_hourly[target][hdt2key(h)] for h in hour_range]

        csv_path = data_dir / (
            "hourly_seasonality_"
            + dt1.strftime("%Y%m%d%H")
            + "_"
            + dt2.strftime("%Y%m%d%H")
            + ".csv"
        )
        df_hour = pd.DataFrame.from_dict({"hour": hour_range, "hs": hs})
        df_hour.set_index("hour", inplace=True)
        df_hour.to_csv(csv_path)

        png_path = png_dir / (
            "hourly_seasonality_"
            + dt1.strftime("%Y%m%d%H")
            + "_"
            + dt2.strftime("%Y%m%d%H")
            + ".png"
        )
        svg_path = svg_dir / (
            "hourly_seasonality_"
            + dt1.strftime("%Y%m%d%H")
            + "_"
            + dt2.strftime("%Y%m%d%H")
            + ".svg"
        )

        p3 = figure(title="Hourly Seasonality")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "hour"
        p3.xaxis.formatter = DatetimeTickFormatter(days="%H", hours="%H")
        p3.xaxis.ticker = bokeh.models.DatetimeTicker()
        p3.line(x=hour_range_plt, y=hs, line_color="dodgerblue", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

    def plot_hourly_acf(
        self, df, target, fdate, tdate, data_dir, png_dir, svg_dir, nlags=24 * 15
    ):
        """Plot acf of residuals which removed hourly seasonality

        Args:
            df (pandas.DataFrame): DataFrame that holds residuals
            target (str): DataFrame column to compute acf
            fdate (datetime): start date of target range
            tdate (datetime): end date of target range
            data_dir (Path): data path
            png_dir (Path): seasonality plot path (png)
            svg_dir (Path): seasonality plot path (svg)
            nlags (int, optional): number of lags to compute acf.
                Defaults to 15.

        Returns:
            None
        """
        # hour_range = pd.date_range(
        #     start=fdate, end=tdate, freq="1H").tz_convert(SEOULTZ).tolist()

        # hr = [df.loc[h, target] -
        #      self.sea_hourly[target][hdt2key(h)] for h in hour_range]
        hr = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        hr_acf, confint, _, _ = sm.tsa.acf(hr, qstat=True, alpha=0.05, nlags=nlags)

        intscale = sum(hr_acf[0 : confint_idx(hr_acf, confint) + 1])
        print("Hourly Conf Int  : ", confint_idx(hr_acf, confint))
        print("Hourly Int Scale : ", intscale)
        # print("Hourly qstat : ", qstat)
        # print("Hourly pvalues : ", pvalues)

        csv_path = data_dir / (
            "acf_hourly_seasonality_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".csv"
        )
        df_hour_acf = pd.DataFrame.from_dict(
            {"lags": range(len(hr_acf)), "hr_acf": hr_acf}
        )
        df_hour_acf.set_index("lags", inplace=True)
        df_hour_acf.to_csv(csv_path)

        png_path = png_dir / (
            "acf_hourly_seasonality_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf_hourly_seasonality_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".svg"
        )
        p3 = figure(title="Autocorrelation of Hourly Residual")
        p3.toolbar.logo = None
        p3.toolbar_location = None
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(hr_acf)), 1.1)
        p3.line(x=range(len(hr_acf)), y=hr_acf, line_color="lightcoral", line_width=2)
        export_png(p3, filename=png_path)
        p3.output_backend = "svg"
        export_svgs(p3, filename=str(svg_path))

        ## residual autocorrelation by plot_acf
        png_path = png_dir / (
            "acf(tpl)_hourly_seasonality_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".png"
        )
        svg_path = svg_dir / (
            "acf(tpl)_hourly_seasonality_"
            + fdate.strftime("%Y%m%d%H")
            + "_"
            + tdate.strftime("%Y%m%d%H")
            + ".svg"
        )
        fig = tpl.plot_acf(hr, lags=nlags)
        fig.savefig(png_path)
        fig.savefig(svg_path)

        return confint_idx(hr_acf, confint), intscale

    def plot(
        self,
        df,
        target,
        fdate,
        tdate,
        data_dir,
        png_dir,
        svg_dir,
        target_year=2016,
        target_week=10,
        target_month=5,
        target_day=1,
        nlags=7,
    ):
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
                type(e.g. raw) /
                    Simulation name (e.g. MLP) /
                    station name (종로구) /
                    target name (e.g. PM10) /
                    'seasonality'
            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) /
                    Simulation name (e.g. MLP) /
                    station name (종로구) /
                    target name (e.g. PM10) /
                    'seasonality'
            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) /
                    Simulation name (e.g. MLP) /
                    station name (종로구) /
                    target name (e.g. PM10) /
                    'seasonality'
            target_year (int):
                specified year for plot
            target_week (int):
                specified week number in target_year for plot
            nlags (int):
                nlags for autocorrelation (days)

        Return:
            None
        """

        # filter by dates
        df = df[(df.index > fdate) & (df.index < tdate)]
        # daily averaged
        # df_d = df.resample('D').mean()

        df_resid_annual, df_resid_weekly, _, df_resid_hourly = self.compute_seasonality(
            df.loc[:, target], return_resid=True
        )
        df_resid_annual = df_resid_annual.to_frame().rename(columns={"resid_y": target})
        df_resid_weekly = df_resid_weekly.to_frame().rename(columns={"resid_w": target})
        # df_resid_weekly_pop = df_resid_weekly_pop.to_frame().rename(columns={'resid': target})
        df_resid_hourly = df_resid_hourly.to_frame().rename(columns={"resid_h": target})

        dict_corr_dist = {}

        _, _ = self.plot_raw_acf(
            df,
            target,
            df.index[0],
            df.index[-1],
            data_dir,
            png_dir,
            svg_dir,
            nlags=nlags * 24,
        )

        # raw seasonality plot needs to re-compute, so pass original df
        self.plot_annual(
            df,
            target,
            data_dir,
            png_dir,
            svg_dir,
            target_year=2016,
            smoothing=self.smoothing,
        )
        a_confint, a_intscale = self.plot_annual_acf(
            df_resid_annual,
            target,
            df.index[0],
            df.index[-1],
            data_dir,
            png_dir,
            svg_dir,
            nlags=nlags,
        )

        # df in weekly and hourly seasonality plot is a just dummy variable
        # to make consistent with other plot methods
        self.plot_weekly(
            target,
            data_dir,
            png_dir,
            svg_dir,
            target_year=target_year,
            target_week=target_week,
        )
        w_confint, w_intscale = self.plot_weekly_acf(
            df_resid_weekly,
            target,
            df.index[0],
            df.index[-1],
            data_dir,
            png_dir,
            svg_dir,
            nlags=nlags,
        )

        # df in weekly and hourly seasonality plot is a just dummy variable
        # to make consistent with other plot methods
        self.plot_hourly(
            target,
            data_dir,
            png_dir,
            svg_dir,
            target_year=target_year,
            target_month=target_month,
            target_day=target_day,
        )
        h_confint, h_intscale = self.plot_hourly_acf(
            df_resid_hourly,
            target,
            df.index[0],
            df.index[-1],
            data_dir,
            png_dir,
            svg_dir,
            nlags=nlags * 24,
        )

        dict_corr_dist = {
            "annual": {"confint": a_confint, "intscale": a_intscale},
            "weekly": {"confint": w_confint, "intscale": w_intscale},
            "hourly": {"confint": h_confint, "intscale": h_intscale},
        }

        with open(data_dir / "intscale.json", "w") as f:
            print(dict_corr_dist, file=f)

    def plot_fused(
        self,
        df,
        target,
        fdate,
        tdate,
        data_dir,
        png_dir,
        svg_dir,
        target_year=2016,
        target_week=10,
        target_month=5,
        target_day=1,
        nlags=15,
        smoothing=True,
    ):
        """Plot acf and seasonality for paper

        1. Seasonality Plot : Annual (+Smoothed) & Weekly & Hourly
        2. ACF Plot : Raw & Residuals (15 days)

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
                type(e.g. raw) /
                    Simulation name (e.g. MLP) /
                    station name (종로구) /
                    target name (e.g. PM10) /
                    'seasonality'
            png_dir (Path):
                Directory location to save png file
                type(e.g. raw) /
                    Simulation name (e.g. MLP) /
                    station name (종로구) /
                    target name (e.g. PM10) /
                    'seasonality'
            svg_dir (Path):
                Directory location to save Svg file
                type(e.g. raw) /
                    Simulation name (e.g. MLP) /
                    station name (종로구) /
                    target name (e.g. PM10) /
                    'seasonality'
            target_year (int, optional):
                specified year for plot. Defaults to 2016.
            target_week (int, optional):
                specified week number in target_year for plot. Defaults to 10.
            target_month (int, optional):
                specified month in target_year for plot. Defaults to 5.
            target_day (int):
                specified day in target_year for plot. Defaults to 1.
            nlags (int, optional): nlags for autocorrelation (days).
                Defaults to 15.
            smoothing (bool, optional): whether plot smoothed seasonality together.
                Defaults to True.

        Returns:
            None
        """
        df = df[(df.index > fdate) & (df.index < tdate)]
        # daily averaged
        df_d = df.resample("D").mean()

        df_resid_annual, df_resid_weekly, _, df_resid_hourly = self.compute_seasonality(
            df.loc[:, target], return_resid=True
        )
        df_resid_annual = df_resid_annual.to_frame().rename(columns={"resid_y": target})
        df_resid_weekly = df_resid_weekly.to_frame().rename(columns={"resid_w": target})
        # df_resid_weekly_pop = df_resid_weekly_pop.to_frame().rename(columns={'resid': target})
        df_resid_hourly = df_resid_hourly.to_frame().rename(columns={"resid_h": target})

        df_resid_annual.to_csv(data_dir / ("resid_annual.csv"))
        df_resid_weekly.to_csv(data_dir / ("resid_weekly.csv"))
        df_resid_hourly.to_csv(data_dir / ("resid_hourly.csv"))

        # RAW data
        raws = df[(df.index >= fdate) & (df.index <= tdate)][target].to_numpy()
        df_resid_raw = pd.DataFrame.from_dict({"date": df.index, target: df[target]})
        df_resid_raw_d = pd.DataFrame.from_dict(
            {"date": df_d.index, target: df_d[target]}
        )
        df_resid_raw.to_csv(data_dir / ("resid_raw.csv"))
        df_resid_raw_d.to_csv(data_dir / ("resid_raw_d.csv"))

        # ANNUAL
        dt1 = dt.datetime(year=target_year, month=1, day=1, hour=0)
        dt2 = dt.datetime(year=target_year, month=12, day=31, hour=23)
        yx = pd.date_range(start=dt1, end=dt2, freq="D", tz=SEOULTZ).tolist()
        # yx_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
        #     start=dt1, end=dt2, freq='D', tz=SEOULTZ)])

        ys = [self.sea_annual[target][ydt2key(y)] for y in yx]
        # yr = df_resid_annual[(df_resid_annual.index >= fdate) & (
        #     df_resid_annual.index <= tdate)][target].to_numpy()
        if self.smoothing:
            _sea_annual_nonsmooth, _, _ = periodic_mean(
                df_d, target, "y", smoothing=False
            )
            # if smoothing, predecomposed seasonality is already smoothed, so redecompose
            ys_vanilla = [_sea_annual_nonsmooth[ydt2key(y)] for y in yx]
            ys_smooth = [self.sea_annual[target][ydt2key(y)] for y in yx]

        df_sea_year = pd.DataFrame.from_dict({"date": yx, "ys": ys})
        if self.smoothing:
            df_sea_year = pd.DataFrame.from_dict(
                {"date": yx, "ys_vanilla": ys_vanilla, "ys_smooth": ys_smooth}
            )
        df_sea_year.to_csv(data_dir / ("sea_annual.csv"))

        # WEEKLY
        # Set datetime range
        target_dt_str = str(target_year) + " " + str(target_week)
        ## ISO 8601 starts weekday from Monday (1)
        dt1 = dt.datetime.strptime(target_dt_str + " 1", "%Y %W %w")
        ## ISO 8601 ends weekday to Sunday (0)
        dt2 = dt.datetime.strptime(target_dt_str + " 0", "%Y %W %w")
        ## set dt2 time as Sunday 23:59
        dt2 = dt2.replace(hour=23, minute=59)

        wx = pd.date_range(start=dt1, end=dt2, freq="D", tz=SEOULTZ).tolist()
        # wx_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
        #     start=dt1, end=dt2, freq='D', tz=SEOULTZ)])

        # Compute Weekly seasonality
        ws = [self.sea_weekly[target][wdt2key(w)] for w in wx]
        # wr = df_resid_weekly[(df_resid_weekly.index >= fdate) 
        #   & (df_resid_weekly.index <= tdate)][target].to_numpy()
        df_sea_week = pd.DataFrame.from_dict({"date": wx, "ws": ws})
        df_sea_week.set_index("date", inplace=True)
        df_sea_week.to_csv(data_dir / ("sea_weekly.csv"))

        # HOURLY
        dt1 = dt.datetime(year=target_year, month=target_month, day=target_day, hour=0)
        dt2 = dt.datetime(year=target_year, month=target_month, day=target_day, hour=23)
        hx = pd.date_range(start=dt1, end=dt2, freq="1H", tz=SEOULTZ).tolist()
        # hx_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
        #     start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()

        # Compute Hourly seasonality
        hs = [self.sea_hourly[target][hdt2key(h)] for h in hx]
        # hr = df_resid_hourly[(df_resid_hourly.index >= fdate) & 
        #   (df_resid_hourly.index <= tdate)][target].to_numpy()

        df_sea_hour = pd.DataFrame.from_dict({"date": hx, "hs": hs})
        df_sea_hour.set_index("date", inplace=True)
        df_sea_hour.to_csv(data_dir / ("sea_hourly.csv"))

        # ingradient
        # raws
        # ys, yr or ys_vanila, ys_smooth, yr
        # ws, wr
        # hs, hr

        # set 95% confidance intervals and compute correlation distance(integral length scale)
        def compute_cd(_acfs, _confint):
            # find index where acf value are in confidance interval
            _confint_idx = confint_idx(_acfs, _confint)
            if _confint_idx is None:
                return 0
            else:
                return sum(_acfs[0 : _confint_idx + 1])

        raw_acf, raw_confint = tsast.acf(raws, alpha=0.05, nlags=nlags)
        raw_cd = compute_cd(raw_acf, raw_confint) * 24
        df_acf_raw = pd.DataFrame.from_dict(
            {
                "lags": list(range(nlags + 1)),
                "acf": raw_acf,
                "confint0": raw_confint[:, 0],
                "confint1": raw_confint[:, 1],
            }
        )
        df_acf_raw.to_csv(data_dir / ("acf_raw.csv"))
        yr_acf, yr_confint = tsast.acf(df_resid_annual[target], alpha=0.05, nlags=nlags)
        y_cd = compute_cd(yr_acf, yr_confint) * 24
        df_acf_yr = pd.DataFrame.from_dict(
            {
                "lags": list(range(nlags + 1)),
                "acf": yr_acf,
                "confint0": yr_confint[:, 0],
                "confint1": yr_confint[:, 1],
            }
        )
        df_acf_yr.to_csv(data_dir / ("acf_annual.csv"))
        wr_acf, wr_confint = tsast.acf(df_resid_weekly[target], alpha=0.05, nlags=nlags)
        w_cd = compute_cd(wr_acf, wr_confint) * 24
        df_acf_wr = pd.DataFrame.from_dict(
            {
                "lags": list(range(nlags + 1)),
                "acf": wr_acf,
                "confint0": wr_confint[:, 0],
                "confint1": wr_confint[:, 1],
            }
        )
        df_acf_wr.to_csv(data_dir / ("acf_weekly.csv"))
        hr_acf, hr_confint = tsast.acf(
            df_resid_hourly[target], alpha=0.05, nlags=nlags * 24
        )
        h_cd = compute_cd(hr_acf, hr_confint)
        df_acf_hr = pd.DataFrame.from_dict(
            {
                "lags": list(range((nlags) * 24 + 1)),
                "acf": hr_acf,
                "confint0": hr_confint[:, 0],
                "confint1": hr_confint[:, 1],
            }
        )
        df_acf_hr.to_csv(data_dir / ("acf_hourly.csv"))

        print("Int scale (raw, annual, weekly, hourly) : ", raw_cd, y_cd, w_cd, h_cd)

        sns.color_palette("tab10")

        # Plot seasonality (Annual, Weekly, Hourly)
        nrows, ncols = 1, 3
        # rough figure size
        # wspace, hspace = 0.2, 0.2
        # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
        ax_size = min(7.22 / ncols, 9.45 / nrows)
        # fig_size_w = ax_size*ncols
        # fig_size_h = ax_size*nrows

        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols]).reshape(
            ncols
        )

        fig, axs = plt.subplots(
            nrows, ncols, figsize=(ax_size * ncols, ax_size * nrows), dpi=600
        )
        fig.tight_layout(pad=0.15)
        fig.subplots_adjust(left=0.2, bottom=0.2)

        # ax[0] == annual seasoanlity
        if smoothing is True:
            sns.lineplot(
                x="date",
                y="value",
                hue="variable",
                data=pd.melt(df_sea_year, ["date"]),
                ax=axs[0],
            )

            # add legend
            leg_handles, leg_labels = axs[0].get_legend_handles_labels()
            dic = {"ys_vanilla": "vanilla", "ys_smooth": "smooth"}
            leg_labels = [dic.get(l, l) for l in leg_labels]
            # remove legend title
            axs[0].legend(
                leg_handles[1:], leg_labels[1:], fancybox=True, fontsize="x-small"
            )
        else:
            sns.lineplot(x="date", y="ys", data=df_sea_year, ax=axs[0], legend=False)

        # ax[1] == weekly seasoanlity
        sns.lineplot(data=df_sea_week, ax=axs[1], legend=False)
        # ax[2] == hourly seasoanlity
        sns.lineplot(data=df_sea_hour, ax=axs[2], legend=False)

        # mpl.rcParams['timezone'] = SEOULTZ

        axs[0].xaxis.set_major_locator(
            mdates.MonthLocator(bymonth=[1, 4, 7, 10], tz=SEOULTZ)
        )
        axs[0].xaxis.set_minor_locator(
            mdates.MonthLocator(bymonth=range(1, 13), tz=SEOULTZ)
        )
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%m", tz=SEOULTZ))
        axs[0].set_xlabel("a year", fontsize="small")
        axs[0].set_ylabel(target, fontsize="small")
        axs[1].xaxis.set_major_locator(mdates.HourLocator(byhour=0, tz=SEOULTZ))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%a", tz=SEOULTZ))
        axs[1].set_xlabel("a week", fontsize="small")
        axs[2].xaxis.set_major_locator(
            mdates.HourLocator(byhour=[0, 4, 8, 12, 16, 20], tz=SEOULTZ)
        )
        axs[2].xaxis.set_minor_locator(mdates.HourLocator(tz=SEOULTZ))
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H", tz=SEOULTZ))
        axs[2].set_xlabel("a day", fontsize="small")

        for i in range(ncols):
            # axs[i].set_ylabel(target, fontsize='small')
            axs[i].xaxis.grid(True, visible=True, which="major")
            for tick in axs[i].xaxis.get_major_ticks():
                tick.label.set_fontsize("x-small")
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize("x-small")

            axs[i].annotate(
                multipanel_labels[i],
                (-0.13, 1.02),
                xycoords="axes fraction",
                fontsize="medium",
                fontweight="bold",
            )
        output_name = target + "_fused_seasonality"
        png_path = png_dir / (output_name + ".png")
        svg_path = svg_dir / (output_name + ".svg")
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

        # Plot ACF (Raw, Hourly Residual)
        nrows, ncols = 1, 2
        # rough figure size
        # wspace, hspace = 0.2, 0.2
        # inch/1pt (=1.0inch / 72pt) * 10pt/row * 8row (6 row + margins)
        ax_size = min(7.22 / ncols, 9.45 / nrows)
        # fig_size_w = ax_size*ncols
        # fig_size_h = ax_size*nrows

        multipanel_labels = np.array(list(string.ascii_uppercase)[:ncols]).reshape(
            ncols
        )

        fig, axs = plt.subplots(
            nrows, ncols, figsize=(ax_size * ncols, ax_size * nrows), dpi=600
        )
        fig.tight_layout(pad=0.15)
        fig.subplots_adjust(left=0.1, bottom=0.2)

        tpl.plot_acf(raws, ax=axs[0], lags=nlags)
        axs[0].set_xlabel("day", fontsize="small")
        axs[0].set_ylabel("corr", fontsize="small")
        tpl.plot_acf(df_resid_hourly[target], ax=axs[1], lags=nlags * 24)
        axs[1].set_xlabel("hour", fontsize="small")

        # axs[0].annotate("Correlation Distance \nof Raw Values : \n" + '{0:.3g}'.format(raw_cd),
        #                     (0.65, 0.85), xycoords='axes fraction', fontsize='x-small')
        # axs[1].annotate("Correlation Distance \nof Residuals : \n" + '{0:.3g}'.format(h_cd),
        #                     (0.65, 0.85), xycoords='axes fraction', fontsize='x-small')
        for i in range(2):
            axs[i].annotate(
                multipanel_labels[i],
                (-0.13, 1.05),
                xycoords="axes fraction",
                fontsize="medium",
                fontweight="bold",
            )
            for tick in axs[i].xaxis.get_major_ticks():
                tick.label.set_fontsize("x-small")
            for tick in axs[i].yaxis.get_major_ticks():
                tick.label.set_fontsize("x-small")

        output_name = target + "_fused_acf"
        png_path = png_dir / (output_name + ".png")
        svg_path = svg_dir / (output_name + ".svg")
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

        df_cd = pd.DataFrame.from_dict(
            {"raw": raw_cd, "y": y_cd, "w": w_cd, "h": h_cd}, orient="index"
        )
        df_cd.to_csv(data_dir / ("correlation_distance.csv"))


class StandardScalerWrapper(TransformerMixin, BaseEstimator):
    """Convert type as Series, not ndarray"""

    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler

    def __getattr__(self, attr):
        return getattr(self.scaler, attr)

    def partial_fit(self, X, y=None):
        """Call partial_fit method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X
            y (pandas.DataFrame or numpy.ndarray, optional):
                Y. Defaults to None.

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            self.scaler.partial_fit(X, y=X)
            return self
        elif isinstance(X, np.ndarray):
            self.scaler.partial_fit(X, y=X)
            return self
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def fit(self, X, y=None):
        """Call fit method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X
            y (pandas.DataFrame or numpy.ndarray, optional):
                Y. Defaults to None.

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            self.scaler.fit(X, y=X)
            return self
        elif isinstance(X, np.ndarray):
            self.scaler.fit(X, y=X)
            return self
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def transform(self, X):
        """Call transform method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            return self.scaler.transform(X)
        elif isinstance(X, np.ndarray):
            return self.scaler.transform(X)
        else:
            raise TypeError("Type should be Pandas Series or Numpy Array")

    def inverse_transform(self, X):
        """Call inverse_transform method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            # return self.scaler.inverse_transform(X.iloc[:, 0].to_numpy().reshape(-1, 1))
            _invX = self.scaler.inverse_transform(X)
            return pd.DataFrame(data=_invX, index=X.index, columns=X.columns)
        elif isinstance(X, np.ndarray):
            return self.scaler.inverse_transform(X)
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")


class PowerTransformerWrapper(TransformerMixin, BaseEstimator):
    """Convert type as Series, not ndarray"""

    def __init__(self, scaler=PowerTransformer()):
        self.scaler = scaler

    def __getattr__(self, attr):
        return getattr(self.scaler, attr)

    def partial_fit(self, X, y=None):
        """Call partial_fit method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X
            y (pandas.DataFrame or numpy.ndarray, optional):
                Y. Defaults to None.

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            self.scaler.partial_fit(X, y=X)
            return self
        elif isinstance(X, np.ndarray):
            self.scaler.partial_fit(X, y=X)
            return self
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def fit(self, X, y=None):
        """Call fit method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X
            y (pandas.DataFrame or numpy.ndarray, optional):
                Y. Defaults to None.

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            print(self.scaler.get_params())
            self.scaler.fit(X, y=y)
            return self
        elif isinstance(X, np.ndarray):
            print(self.scaler.get_params())
            self.scaler.fit(X, y=y)
            return self
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def transform(self, X):
        """Call transform method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            return self.scaler.transform(X)
        elif isinstance(X, np.ndarray):
            return self.scaler.transform(X)
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")

    def inverse_transform(self, X):
        """Call inverse_transform method of scaler depends on type of input

        Args:
            X (pandas.DataFrame or numpy.ndarray): X

        Raises:
            TypeError: If X is not pandas.DataFrame or numpy.ndarray
        """
        if isinstance(X, pd.DataFrame):
            _invX = self.scaler.inverse_transform(X)
            return pd.DataFrame(data=_invX, index=X.index, columns=X.columns)
        elif isinstance(X, np.ndarray):
            return self.scaler.inverse_transform(X)
        else:
            raise TypeError("Type should be Pandas DataFrame or Numpy Array")
