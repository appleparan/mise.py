import datetime as dt
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import preprocessing

from bokeh.models import DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png

from torch.utils.data.dataset import Dataset

import statsmodels.api as sm

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
        # filter by date
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
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
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

class MultivariateDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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
        return self._scaler.transform(x.to_numpy()).astype('float32'), \
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
            self._dates[(i+self.sample_size)
                         :(i+self.sample_size+self.output_size)]

    def to_csv(self, fpath):
        self._df.to_csv(fpath)


class MultivariateMeanSeasonalityDataset(MultivariateDataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        super().__init__(*args, **kwargs)

        self._dict_avg_annual = kwargs.get('avg_annual', {})
        self._dict_avg_daily = kwargs.get('avg_daily', {})
        self._dict_avg_weekly = kwargs.get('avg_weekly', {})

        # load imputed data from filepath if not provided as dataframe
        filepath = kwargs.get('filepath', Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv"))
        raw_df = pd.read_csv(filepath,
                             index_col=[0, 1],
                             parse_dates=[0])
        # filter by station_name
        self._df_h = raw_df.query('stationCode == "' +
                            str(SEOUL_STATIONS[self.station_name]) + '"')
        self._df_h.reset_index(level='stationCode', drop=True, inplace=True)
        # filter by date
        self._df_h = self._df_h[self.fdate -
                                dt.timedelta(hours=self.sample_size):self.tdate]
        # filter by date
        self._df_d = self._df_h.copy()
        self._df_d = self._df_d.resample('D').mean()

        self._dates_h = self._df_h.index.to_pydatetime()
        self._dates_d = self._df_d.index.to_pydatetime()

        self.decompose_seasonality()
        new_features = [f + "_dr" if f == self.target else f  for f in self.features]
        self.features = new_features
        #self._xs = self._df[self.features]
        # residual to _xs
        # if len(self.features) == 1 -> univariate, else -> multivariate
        self._xs = self._df_h[self.features]
        self._ys = self._df_h[[self.target]]

        # self._xs must not be available when creating instance so no kwargs for scaler
        self._scaler = preprocessing.StandardScaler().fit(self._xs)

    def decompose_seasonality(self):
        # add keys to _d and _h
        self._df_h['key_day'] = self._df_h.index.hour.to_numpy()
        self._df_d['key_day'] = self._df_d.index.hour.to_numpy()
        self._df_h['key_week'] = self._df_h.index.weekday.to_numpy()
        self._df_d['key_week'] = self._df_d.index.weekday.to_numpy()

        months = self._df_d.index.month.to_numpy()
        days = self._df_d.index.day.to_numpy()
        hours = self._df_d.index.hour.to_numpy()
        self._df_d['key_year'] = [
            str(m).zfill(2) + str(d).zfill(2) + str(h).zfill(2)
            for (m, d, h) in zip(months, days, hours)]
        months = self._df_h.index.month.to_numpy()
        days = self._df_h.index.day.to_numpy()
        hours = self._df_h.index.hour.to_numpy()
        self._df_h['key_year'] = [
            str(m).zfill(2) + str(d).zfill(2) + str(h).zfill(2)
            for (m, d, h) in zip(months, days, hours)]

        self._df_h[self.target + "_raw"] = self._df_h[self.target]
        self._df_d[self.target + "_raw"] = self._df_d[self.target]

        def periodic_mean(df, target, key, prefix_period, prefix_input, _dict_avg):
            # if dictionary is empty, create new one
            if not _dict_avg:
                # only compute on train/valid set
                # test set will use mean of train/valid set which is fed on __init__
                grp_sea = df.groupby(key).mean()
                _dict_avg = grp_sea.to_dict()

            # set initial value to create column
            df[target + "_" + prefix_period + "s"] = df[target]
            df[target + "_" + prefix_period + "r"] = df[target]

            if prefix_period == 'y':
                def datetime2key(d): return str(d.month).zfill(
                    2) + str(d.day).zfill(2) + str(d.hour).zfill(2)
            elif prefix_period == 'd':
                def datetime2key(d): return d.hour
            elif prefix_period == 'w':
                def datetime2key(d): return d.weekday()

            for index, row in df.iterrows():
                # seasonality
                if prefix_input == None:
                    target_key = self.target
                else:
                    target_key = self.target + '_' + prefix_input

                sea = _dict_avg[target_key][datetime2key(index)]
                res = row[target_key] - sea
                df.at[index, target + '_' + prefix_period + 's'] = sea
                df.at[index, target + '_' + prefix_period + 'r'] = res

            return _dict_avg

        def populate(key, target):
            """
            Populate from self._df_d to self._df_h by freq yearly and weekly
            yearly:
                freq = 'Y'
                move '_ys', '_yr' from self._df_d to self._df_h
            weekly:
                freq = 'W'
                move '_ws', '_wr' from self._df_d to self._df_h

            """
            if key == 'key_year':
                key_s = target + '_ys'
                key_r = target + '_yr'
                dictionary = self._dict_avg_annual
                def datetime2key(d): return str(d.month).zfill(
                    2) + str(d.day).zfill(2) + str(d.hour).zfill(2)
            elif key == 'key_week':
                key_s = target + '_ws'
                key_r = target + '_wr'
                dictionary = self._dict_avg_weekly
                def datetime2key(d): return d.weekday()
            else:
                raise KeyError("Wrong Key")

            for index, row in self._df_h.iterrows():
                # check row's day key is same as _df_d's key
                _date_day = index
                # get data from df_d
                _date_day = _date_day.replace(
                    hour=0, minute=0, second=0, microsecond=0)
                self._df_h.loc[index, key_s] = self._df_d.loc[_date_day, key_s]
                self._df_h.loc[index, key_r] = self._df_d.loc[_date_day, key_r]

        # remove (preassumed) seasonality
        # 1. Annual Seasonality (yymmdd: 010100 ~ 123123)
        self._dict_avg_annual = periodic_mean(
            self._df_d, self.target, "key_year", "y", None, self._dict_avg_annual)
        # 2. Weekly Seasonality (w: 0 ~ 6)
        self._dict_avg_weekly = periodic_mean(
            self._df_d, self.target, "key_week", "w", "yr", self._dict_avg_weekly)

        # now populate seasonality (_ys or _ws) and residual (_yr or _wr) from _d to _h
        # TODO : how to populate with join?
        populate('key_year', self.target)
        populate('key_week', self.target)

        # above just populate residual of daily averaged
        for index, row in self._df_h.iterrows():
            self._df_h.loc[index, self.target + '_yr'] = self._df_h.loc[index, self.target + '_raw'] - \
                self._df_h.loc[index, self.target + '_ys']
            self._df_h.loc[index, self.target + '_wr'] = self._df_h.loc[index, self.target + '_yr'] - \
                self._df_h.loc[index, self.target + '_ws']

        # 3. Daily Seasonality (hh: 00 ~ 23)
        self._dict_avg_daily = periodic_mean(
            self._df_h, self.target, "key_day", "d", "wr", self._dict_avg_daily)

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

    def plot_seasonality(self, data_dir, plot_dir, nlags=24*30):
        """
        Plot seasonality and residual, and their autocorrealtions

        output_dir: Path :
        type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality
        """
        def nextMonday(_date): return (_date + dt.timedelta(days=-_date.weekday(),
                                                            weeks=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        def nextMidnight(_date): return (_date + dt.timedelta(hours=24)
                                         ).replace(hour=0, minute=0, second=0, microsecond=0)
        def nextNewyearDay(_date): return _date.replace(
            year=_date.year+1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        # annual
        dt1 = nextNewyearDay(self.fdate)
        dt2 = dt1.replace(year=dt1.year+1) - dt.timedelta(hours=1)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])
        ys = [self._df_d.loc[y, self.target + '_ys'] for y in year_range]
        yr = [self._df_d.loc[y, self.target + '_yr'] for y in year_range]

        csv_path = data_dir / ("annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_year = pd.DataFrame.from_dict(
            {'date': year_range, 'ys': ys, 'yr': yr})
        df_year.set_index('date', inplace=True)
        df_year.to_csv(csv_path)

        plt_path = plot_dir / ("annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                   months="%m/%d %H:%M",
                                                   hours="%m/%d %H:%M",
                                                   minutes="%m/%d %H:%M")
        p1.line(year_range_plt, ys, line_color="dodgerblue", line_width=2,
                legend_label="seasonality")
        p1.line(year_range_plt, yr, line_color="lightcoral", line_width=2,
                legend_label="residual")
        export_png(p1, filename=plt_path)

        ## autocorrelation
        dt2 = dt1 + dt.timedelta(hours=nlags)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])
        ys = [self._df_d.loc[y, self.target + '_ys'] for y in year_range]
        yr = [self._df_d.loc[y, self.target + '_yr'] for y in year_range]
        ys_acf = sm.tsa.acf(ys, nlags=nlags)
        yr_acf = sm.tsa.acf(yr, nlags=nlags)

        csv_path = data_dir / ("acf_annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': range(len(yr_acf)),  'ys_acf': ys_acf, 'yr_acf': yr_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        plt_path = plot_dir / ("acf_annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line(range(len(yr_acf)), yr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p1, filename=plt_path)

        # weekly
        dt1 = nextMonday(self.fdate)
        dt2 = dt1 + dt.timedelta(days=6)
        week_range = pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ).tolist()
        week_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ)]).tolist()
        wr = [self._df_d.loc[w, self.target + '_wr'] for w in week_range]
        ws = [self._df_d.loc[w, self.target + '_ws'] for w in week_range]

        csv_path = data_dir / ("weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_week = pd.DataFrame.from_dict(
            {'date': week_range, 'ws': ws, 'wr': wr})
        df_week.set_index('date', inplace=True)
        df_week.to_csv(csv_path)

        plt_path = plot_dir / ("weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")

        p2 = figure(title="Weekly Seasonality")
        p2.xaxis.axis_label = "dates"
        p2.xaxis.formatter = DatetimeTickFormatter(
            days="%w")
        p2.line(week_range_plt, ws, line_color="dodgerblue", line_width=2,
                legend_label="seasonality")
        p2.line(week_range_plt, wr, line_color="lightcoral", line_width=2,
                legend_label="residual")
        export_png(p2, filename=plt_path)

        ## autocorrelation
        dt2 = dt1 + dt.timedelta(hours=nlags)
        week_range = pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ).tolist()
        week_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ)]).tolist()
        wr = [self._df_d.loc[w, self.target + '_wr'] for w in week_range]
        ws = [self._df_d.loc[w, self.target + '_ws'] for w in week_range]
        ws_acf = sm.tsa.acf(ws, nlags=nlags)
        wr_acf = sm.tsa.acf(wr, nlags=nlags)

        csv_path = data_dir / ("acf_weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_week_acf = pd.DataFrame.from_dict(
            {'lags': range(len(wr_acf)),  'ws_acf': ws_acf, 'wr_acf': wr_acf})
        df_week_acf.set_index('lags', inplace=True)
        df_week_acf.to_csv(csv_path)

        plt_path = plot_dir / ("acf_weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        p2 = figure(title="Autocorrelation of Weekly Residual")
        p2.xaxis.axis_label = "lags"
        p2.yaxis.bounds = (min(0, min(wr_acf)), 1.1)
        p2.line(range(len(wr_acf)), wr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p2, filename=plt_path)

        # daily
        dt1 = nextMidnight(self.fdate)
        dt2 = dt1 + dt.timedelta(hours=23)
        day_range = pd.date_range(dt1, dt2, freq="1H", tz=SEOULTZ).tolist()
        day_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()
        ds = [self._df_h.loc[d, self.target + '_ds'] for d in day_range]
        dr = [self._df_h.loc[d, self.target + '_dr'] for d in day_range]

        csv_path = data_dir / ("daily_seasonality_hourly_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_day = pd.DataFrame.from_dict(
            {'date': day_range, 'ds': ds, 'dr': dr})
        df_day.set_index('date', inplace=True)
        df_day.to_csv(csv_path)

        plt_path = plot_dir / ("daily_seasonality_hourly_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")

        p3 = figure(title="Daily Seasonality")
        p3.xaxis.axis_label = "dates"
        p3.xaxis.formatter = DatetimeTickFormatter(hours="%H")
        p3.line(day_range_plt, ds, line_color="dodgerblue", line_width=2,
                legend_label="seasonality")
        p3.line(day_range_plt, dr, line_color="lightcoral", line_width=2,
                legend_label="residual")
        export_png(p3, filename=plt_path)

        ## autocorrelation
        dt2 = dt1 + dt.timedelta(hours=nlags)
        day_range = pd.date_range(dt1, dt2, freq="1H", tz=SEOULTZ).tolist()
        day_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()
        ds = [self._df_h.loc[d, self.target + '_ds'] for d in day_range]
        dr = [self._df_h.loc[d, self.target + '_dr'] for d in day_range]
        ds_acf = sm.tsa.acf(ds, nlags=len(ds), fft=False)
        dr_acf = sm.tsa.acf(dr, nlags=len(dr), fft=False)

        csv_path = data_dir / ("acf_daily_seasonality_hourly_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_day_acf = pd.DataFrame.from_dict(
            {'lags': range(len(dr_acf)),  'ds_acf': ds_acf, 'dr_acf': dr_acf})
        df_day_acf.set_index('lags', inplace=True)
        df_day_acf.to_csv(csv_path)

        plt_path = plot_dir / ("acf_daily_seasonality_hourly_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        p3 = figure(title="Autocorrelation of Daily Residual")
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(dr_acf)), 1.1)
        p3.line(range(len(dr_acf)), dr_acf,
                line_color="lightcoral", line_width=2)
        export_png(p3, filename=plt_path)

    @property
    def dict_avg_annual(self):
        return self._dict_avg_annual

    @dict_avg_annual.setter
    def dict_avg_annual(self, dicts):
        self._dict_avg_annual = dicts

    @property
    def dict_avg_weekly(self):
        return self._dict_avg_weekly

    @dict_avg_weekly.setter
    def dict_avg_weekly(self, dicts):
        self._dict_avg_weekly = dicts

    @property
    def dict_avg_daily(self):
        return self._dict_avg_daily

    @dict_avg_daily.setter
    def dict_avg_daily(self, dicts):
        self._dict_avg_daily = dicts

    @property
    def df_h(self):
        return self._df_h

    @property
    def df_d(self):
        return self._df_d

class MultivariateAutoRegressiveDataset(MultivariateDataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        super().__init__(*args, **kwargs)

        # MLP smaple_size
        self.sample_size_m = kwargs.get('sample_size_m', 48)
        # ARIMA sample_size
        self.sample_size_a = kwargs.get('sample_size_a', 24*30)
        self.sample_size = max(self.sample_size_m, self.sample_size_a)
        # i.e.
        # sample_size_a = 12
        # sample_size_m = 5
        # sample_size = 12
        # sample_size_ao = 0 (range(0:12))
        # sample_size_mo = 5 (range(5:12))
        if self.sample_size == self.sample_size_a:
            # i
            # | --               sample_size            -- | -- output_size -- |
            # | --              sample_size_a           -- | -- output_size -- |
            # | -- sample_size_mo -- | -- sample_size_m -- | -- output_size -- |
            # ARIMA sample_size is longer (most case)
            # difference as offset
            self.sample_size_ao = 0
            self.sample_size_mo = self.sample_size - self.sample_size_m
        # MLP sample_size is longer
        else:
            # i
            # | --               sample_size            -- | -- output_size -- |
            # | --              sample_size_m           -- | -- output_size -- |
            # | -- sample_size_ao -- | -- sample_size_a -- | -- output_size -- |
            # ARIMA sample_size is longer (most case)
            # difference as offset
            self.sample_size_ao = self.sample_size - self.sample_size_a
            self.sample_size_mo = 0

        # load imputed data from filepath if not provided as dataframe
        filepath = kwargs.get('filepath', Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv"))
        raw_df = pd.read_csv(filepath,
                             index_col=[0, 1],
                             parse_dates=[0])
        # filter by station_name
        self._df = raw_df.query('stationCode == "' +
                            str(SEOUL_STATIONS[self.station_name]) + '"')
        # filter by date
        self._df = self._df[self.fdate -
                            dt.timedelta(hours=self.sample_size):self.tdate]

        self._df.reset_index(level='stationCode', drop=True, inplace=True)

        self._dates = self._df.index.to_pydatetime()
        self._arima_o = (1, 0, 1)
        self._arima_so = (0, 0, 0, 24)
        # AR Input
        self._xas = self._df[self.target]
        # ML Input
        self._xms = self._df[self.features]
        self._ys = self._df[self.target]

        # self._xms must not be available when creating instance so no kwargs for scaler
        self._scaler = preprocessing.StandardScaler().fit(self._xms)
        print("Construct AR part with SARIMX")
        self.arima_x, self.arima_y = self.preprocess_arima_table()

    def __getitem__(self, i: int):
        """
        get X, Y for given index `di`
        For ARIMA + MLP Hybrid model, original Time series is considered as

            Z = L + N

        Then ARIMA predicts L, MLP Predicts Z - L
        Due to stationarity, ARIMA(actually SARIMAX) needs longer data than MLP
        This means we need two sample_size

        1. ARIMA predicts 'target' from its `sample_size_a` series
        2. Get In-sample prediction (for residual) & out-sample prediction (for loss function)
        3. Compute Residual of Input (E = Z - L)
        4. Return Residual and Output
        5. (MLP Model part) Train E to get Residual prediction
        6. Compute Loss function from summing MLP trained prediction & ARIMA trained prediction

        Args:
            i: where output starts

        Returns:
            Ndarray:
            Tensor: transformed input (might be normalized)
            Tensor: output without transform
        """
        xa = self.arima_x[self.arima_x.index.get_level_values('key').isin([i])]['xa'].to_numpy()
        ya = self.arima_y[self.arima_y.index.get_level_values(
            'key').isin([i])]['ya'].to_numpy()
        xm = self._xms.iloc[i+self.sample_size_mo:i+self.sample_size]
        y = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)]

        # target in xm must be replaced after fit to ARIMA
        # residual about to around zero, so I ignored scaling
        return xa.astype('float32'), ya.astype('float32'), \
            self._scaler.transform(xm), \
            xm.columns.tolist().index(self.target), \
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
            self._dates[(i+self.sample_size):(i+self.sample_size+self.output_size)]

    def preprocess_arima_table(self):
        """
        construct arima table

        1. iterate len(dataset)
        """

        df_xs = []
        df_ys = []
        for i in tqdm(range(self.__len__())):
            # where output starts
            key_idx = i
            # ARIMA is precomputed
            _xa = self._xas.iloc[i+self.sample_size_ao:i+self.sample_size]
            _xa.index.freq = 'H'
            model = SARIMAX(_xa, order=self._arima_o,
                            seasonal_order=self._arima_so, freq='H')
            model_fit = model.fit(disp=False)

            if self.sample_size == self.sample_size_a:
                # in-sample & out-sample prediction
                # its size is sample_size_m + output_size
                _xa_pred = model_fit.predict(
                    start=self.sample_size_mo, end=self.sample_size+self.output_size-1)
                xa = self._xas.iloc[i+self.sample_size_mo:i+self.sample_size].to_numpy() - \
                    _xa_pred[0:self.sample_size_m]
                ya = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)].to_numpy() - \
                    _xa_pred[self.sample_size_m:self.sample_size_m+self.output_size]
            else:
                # in-sample & out-sample prediction
                # its size is sample_size_m + output_size
                _xa_pred = model_fit.predict(
                    start=self.sample_size_ao, stop=self.sample_size+self.output_size)
                xa = self._xas.iloc[i+self.sample_size_mo:i+self.sample_size].to_numpy() - \
                    _xa_pred[0:self.sample_size_m]
                ya = self._ys.iloc[(i+self.sample_size):(i+self.sample_size+self.output_size)].to_numpy() - \
                    _xa_pred[self.sample_size_m:self.sample_size_m+self.output_size]
            # index for outer
            keys = np.repeat(i, self.sample_size_m)
            # index for innner
            dates_x = self._dates[i+self.sample_size_mo:i+self.sample_size]
            dates_y = self._dates[i+self.sample_size +
                                  1:i+self.sample_size+self.output_size+1]

            # same key -> get x & y

            _df_x = pd.DataFrame({
                    'key': np.repeat(i, self.sample_size_m),
                    'date': dates_x,
                    'xa': xa})
            _df_x.set_index(['key', 'date'], inplace=True)
            _df_x.sort_index(inplace=True)
            _df_y = pd.DataFrame(pd.DataFrame({
                    'key': np.repeat(i, self.output_size),
                    'date': dates_y,
                    'ya': ya}))
            _df_y.set_index(['key', 'date'], inplace=True)
            _df_y.sort_index(inplace=True)
            df_xs.append(_df_x)
            df_ys.append(_df_y)
        df_x = pd.concat(df_xs)
        df_y = pd.concat(df_ys)

        # if you want to filter by key
        # df[df.index.get_level_values('key').isin(['some_key'])]

        return df_x, df_y

    def to_csv(self, fpath):
        self._df.to_csv(fpath)


class UnivariateMeanSeasonalityDataset(UnivariateDataset):
    def __init__(self, *args, **kwargs):
        #args -- tuple of anonymous arguments
        #kwargs -- dictionary of named arguments
        super().__init__(*args, **kwargs)

        self._dict_avg_annual = kwargs.get('avg_annual', {})
        self._dict_avg_daily = kwargs.get('avg_daily', {})
        self._dict_avg_weekly = kwargs.get('avg_weekly', {})

        # load imputed data from filepath if not provided as dataframe
        filepath = kwargs.get('filepath', Path(
            "/input/python/input_jongro_imputed_hourly_pandas.csv"))
        raw_df = pd.read_csv(filepath,
                             index_col=[0, 1],
                             parse_dates=[0])
        # filter by station_name
        self._df_h = raw_df.query('stationCode == "' +
                                  str(SEOUL_STATIONS[self.station_name]) + '"')
        self._df_h.reset_index(level='stationCode', drop=True, inplace=True)
        # filter by date
        self._df_h = self._df_h[self.fdate -
                                dt.timedelta(hours=self.sample_size):self.tdate]
        # filter by date
        self._df_d = self._df_h.copy()
        self._df_d = self._df_d.resample('D').mean()

        self._dates_h = self._df_h.index.to_pydatetime()
        self._dates_d = self._df_d.index.to_pydatetime()

        self.decompose_seasonality()
        new_features = [self.target + "_dr"]
        self.features = new_features

        # residual to _xs
        # if len(self.features) == 1 -> univariate, else -> multivariate        
        self._xs = self._df_h[self.features]
        self._ys = self._df_h[[self.target]]

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
        return self._scaler.transform(x.to_numpy()).astype('float32'), \
            np.reshape(y.to_numpy(), len(y)).astype('float32'), \
            self._dates[(i+self.sample_size)
                         :(i+self.sample_size+self.output_size)]

    def decompose_seasonality(self):
        # add keys to _d and _h
        self._df_h['key_day'] = self._df_h.index.hour.to_numpy()
        self._df_d['key_day'] = self._df_d.index.hour.to_numpy()
        self._df_h['key_week'] = self._df_h.index.weekday.to_numpy()
        self._df_d['key_week'] = self._df_d.index.weekday.to_numpy()

        months = self._df_d.index.month.to_numpy()
        days = self._df_d.index.day.to_numpy()
        hours = self._df_d.index.hour.to_numpy()
        self._df_d['key_year'] = [
            str(m).zfill(2) + str(d).zfill(2) + str(h).zfill(2)
            for (m, d, h) in zip(months, days, hours)]
        months = self._df_h.index.month.to_numpy()
        days = self._df_h.index.day.to_numpy()
        hours = self._df_h.index.hour.to_numpy()
        self._df_h['key_year'] = [
            str(m).zfill(2) + str(d).zfill(2) + str(h).zfill(2)
            for (m, d, h) in zip(months, days, hours)]

        self._df_h[self.target + "_raw"] = self._df_h[self.target]
        self._df_d[self.target + "_raw"] = self._df_d[self.target]

        def periodic_mean(df, target, key, prefix_period, prefix_input, _dict_avg):
            # if dictionary is empty, create new one
            if not _dict_avg:
                # only compute on train/valid set
                # test set will use mean of train/valid set which is fed on __init__
                grp_sea = df.groupby(key).mean()
                _dict_avg = grp_sea.to_dict()

            # set initial value to create column
            df[target + "_" + prefix_period + "s"] = df[target]
            df[target + "_" + prefix_period + "r"] = df[target]

            if prefix_period == 'y':
                def datetime2key(d): return str(d.month).zfill(
                    2) + str(d.day).zfill(2) + str(d.hour).zfill(2)
            elif prefix_period == 'd':
                def datetime2key(d): return d.hour
            elif prefix_period == 'w':
                def datetime2key(d): return d.weekday()

            for index, row in df.iterrows():
                # seasonality
                if prefix_input == None:
                    target_key = self.target
                else:
                    target_key = self.target + '_' + prefix_input

                sea = _dict_avg[target_key][datetime2key(index)]
                res = row[target_key] - sea
                df.at[index, target + '_' + prefix_period + 's'] = sea
                df.at[index, target + '_' + prefix_period + 'r'] = res

            return _dict_avg

        def populate(key, target):
            """
            Populate from self._df_d to self._df_h by freq yearly and weekly
            yearly:
                freq = 'Y'
                move '_ys', '_yr' from self._df_d to self._df_h
            weekly:
                freq = 'W'
                move '_ws', '_wr' from self._df_d to self._df_h

            """
            if key == 'key_year':
                key_s = target + '_ys'
                key_r = target + '_yr'
                dictionary = self._dict_avg_annual
                def datetime2key(d): return str(d.month).zfill(
                    2) + str(d.day).zfill(2) + str(d.hour).zfill(2)
            elif key == 'key_week':
                key_s = target + '_ws'
                key_r = target + '_wr'
                dictionary = self._dict_avg_weekly
                def datetime2key(d): return d.weekday()
            else:
                raise KeyError("Wrong Key")

            for index, row in self._df_h.iterrows():
                # check row's day key is same as _df_d's key
                _date_day = index
                # get data from df_d
                _date_day = _date_day.replace(hour=0, minute=0, second=0, microsecond=0)
                self._df_h.loc[index, key_s] = self._df_d.loc[_date_day, key_s]
                self._df_h.loc[index, key_r] = self._df_d.loc[_date_day, key_r]


        # remove (preassumed) seasonality
        # 1. Annual Seasonality (yymmdd: 010100 ~ 123123)
        self._dict_avg_annual = periodic_mean(
            self._df_d, self.target, "key_year", "y", None, self._dict_avg_annual)
        # 2. Weekly Seasonality (w: 0 ~ 6)
        self._dict_avg_weekly = periodic_mean(
            self._df_d, self.target, "key_week", "w", "yr", self._dict_avg_weekly)

        # now populate seasonality (_ys or _ws) and residual (_yr or _wr) from _d to _h
        # TODO : how to populate with join?
        populate('key_year', self.target)
        populate('key_week', self.target)

        # above just populate residual of daily averaged
        for index, row in self._df_h.iterrows():
            self._df_h.loc[index, self.target + '_yr'] = self._df_h.loc[index, self.target + '_raw'] - \
                self._df_h.loc[index, self.target + '_ys']
            self._df_h.loc[index, self.target + '_wr'] = self._df_h.loc[index, self.target + '_yr'] - \
                self._df_h.loc[index, self.target + '_ws']

        # 3. Daily Seasonality (hh: 00 ~ 23)
        self._dict_avg_daily = periodic_mean(
            self._df_h, self.target, "key_day", "d", "wr", self._dict_avg_daily)

    def to_csv(self, fpath):
        self._df.to_csv(fpath)

    def plot_seasonality(self, data_dir, plot_dir, nlags=24*30):
        """
        Plot seasonality and residual, and their autocorrealtions

        output_dir: Path :
        type(e.g. raw) / Simulation name (e.g. MLP) / station name (종로구) / target name (e.g. PM10) / seasonality
        """
        nextMonday = lambda _date: (_date + dt.timedelta(days=-_date.weekday(), weeks=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        nextMidnight = lambda _date: (_date + dt.timedelta(hours=24)).replace(hour=0, minute=0, second=0, microsecond=0)
        nextNewyearDay = lambda _date: _date.replace(year=_date.year+1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        # annual
        dt1 = nextNewyearDay(self.fdate)
        dt2 = dt1.replace(year=dt1.year+1) - dt.timedelta(hours=1)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])
        ys = [self._df_d.loc[y, self.target + '_ys'] for y in year_range]
        yr = [self._df_d.loc[y, self.target + '_yr'] for y in year_range]

        csv_path = data_dir / ("annual_seasonality_daily_avg_" + \
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_year = pd.DataFrame.from_dict(
            {'date': year_range, 'ys': ys, 'yr': yr})
        df_year.set_index('date', inplace=True)
        df_year.to_csv(csv_path)

        plt_path = plot_dir / ("annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")

        p1 = figure(title="Annual Seasonality")
        p1.xaxis.axis_label = "dates"
        p1.xaxis.formatter = DatetimeTickFormatter(days="%m/%d %H:%M",
                                                   months="%m/%d %H:%M",
                                                   hours="%m/%d %H:%M",
                                                   minutes="%m/%d %H:%M")
        p1.line(year_range_plt, ys, line_color="dodgerblue", line_width=2,
                legend_label="seasonality")
        p1.line(year_range_plt, yr, line_color="lightcoral", line_width=2,
                legend_label="residual")
        export_png(p1, filename=plt_path)

        ## autocorrelation
        dt2 = dt1 + dt.timedelta(hours=nlags)
        year_range = pd.date_range(start=dt1, end=dt2, tz=SEOULTZ).tolist()
        year_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, tz=SEOULTZ)])
        ys = [self._df_d.loc[y, self.target + '_ys'] for y in year_range]
        yr = [self._df_d.loc[y, self.target + '_yr'] for y in year_range]
        ys_acf = sm.tsa.acf(ys, nlags=nlags)
        yr_acf = sm.tsa.acf(yr, nlags=nlags)

        csv_path = data_dir / ("acf_annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_year_acf = pd.DataFrame.from_dict(
            {'lags': range(len(yr_acf)),  'ys_acf': ys_acf, 'yr_acf': yr_acf})
        df_year_acf.set_index('lags', inplace=True)
        df_year_acf.to_csv(csv_path)

        plt_path = plot_dir / ("acf_annual_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        p1 = figure(title="Autocorrelation of Annual Residual")
        p1.xaxis.axis_label = "lags"
        p1.yaxis.bounds = (min(0, min(yr_acf)), 1.1)
        p1.line(range(len(yr_acf)), yr_acf, line_color="lightcoral", line_width=2)
        export_png(p1, filename=plt_path)

        # weekly
        dt1 = nextMonday(self.fdate)
        dt2 = dt1 + dt.timedelta(days=6)
        week_range = pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ).tolist()
        week_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ)]).tolist()
        wr = [self._df_d.loc[w, self.target + '_wr'] for w in week_range]
        ws = [self._df_d.loc[w, self.target + '_ws'] for w in week_range]

        csv_path = data_dir / ("weekly_seasonality_daily_avg_" + \
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_week = pd.DataFrame.from_dict(
            {'date': week_range, 'ws': ws, 'wr': wr})
        df_week.set_index('date', inplace=True)
        df_week.to_csv(csv_path)

        plt_path = plot_dir / ("weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")

        p2 = figure(title="Weekly Seasonality")
        p2.xaxis.axis_label = "dates"
        p2.xaxis.formatter = DatetimeTickFormatter(
            days="%w")
        p2.line(week_range_plt, ws, line_color="dodgerblue", line_width=2,
                legend_label="seasonality")
        p2.line(week_range_plt, wr, line_color="lightcoral", line_width=2,
                legend_label="residual")
        export_png(p2, filename=plt_path)
        
        ## autocorrelation
        dt2 = dt1 + dt.timedelta(hours=nlags)
        week_range = pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ).tolist()
        week_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='D', tz=SEOULTZ)]).tolist()
        wr = [self._df_d.loc[w, self.target + '_wr'] for w in week_range]
        ws = [self._df_d.loc[w, self.target + '_ws'] for w in week_range]
        ws_acf = sm.tsa.acf(ws, nlags=nlags)
        wr_acf = sm.tsa.acf(wr, nlags=nlags)

        csv_path = data_dir / ("acf_weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_week_acf = pd.DataFrame.from_dict(
            {'lags': range(len(wr_acf)),  'ws_acf': ws_acf, 'wr_acf': wr_acf})
        df_week_acf.set_index('lags', inplace=True)
        df_week_acf.to_csv(csv_path)

        plt_path = plot_dir / ("acf_weekly_seasonality_daily_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        p2 = figure(title="Autocorrelation of Weekly Residual")
        p2.xaxis.axis_label = "lags"
        p2.yaxis.bounds = (min(0, min(wr_acf)), 1.1)
        p2.line(range(len(wr_acf)), wr_acf, line_color="lightcoral", line_width=2)
        export_png(p2, filename=plt_path)

        # daily
        dt1 = nextMidnight(self.fdate)
        dt2 = dt1 + dt.timedelta(hours=23)
        day_range = pd.date_range(dt1, dt2, freq="1H", tz=SEOULTZ).tolist()
        day_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()
        ds = [self._df_h.loc[d, self.target + '_ds'] for d in day_range]
        dr = [self._df_h.loc[d, self.target + '_dr'] for d in day_range]

        csv_path = data_dir / ("daily_seasonality_hourly_avg_" + \
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_day = pd.DataFrame.from_dict(
            {'date': day_range, 'ds': ds, 'dr': dr})
        df_day.set_index('date', inplace=True)
        df_day.to_csv(csv_path)

        plt_path = plot_dir / ("daily_seasonality_hourly_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")

        p3 = figure(title="Daily Seasonality")
        p3.xaxis.axis_label = "dates"
        p3.xaxis.formatter = DatetimeTickFormatter(hours="%H")
        p3.line(day_range_plt, ds, line_color="dodgerblue", line_width=2,
                legend_label="seasonality")
        p3.line(day_range_plt, dr, line_color="lightcoral", line_width=2,
                legend_label="residual")
        export_png(p3, filename=plt_path)

        ## autocorrelation
        dt2 = dt1 + dt.timedelta(hours=nlags)
        day_range = pd.date_range(dt1, dt2, freq="1H", tz=SEOULTZ).tolist()
        day_range_plt = pd.DatetimeIndex([i.replace(tzinfo=None) for i in pd.date_range(
            start=dt1, end=dt2, freq='1H', tz=SEOULTZ)]).tolist()
        ds = [self._df_h.loc[d, self.target + '_ds'] for d in day_range]
        dr = [self._df_h.loc[d, self.target + '_dr'] for d in day_range]
        ds_acf = sm.tsa.acf(ds, nlags=len(ds), fft=False)
        dr_acf = sm.tsa.acf(dr, nlags=len(dr), fft=False)

        csv_path = data_dir / ("acf_daily_seasonality_hourly_avg_" + \
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".csv")
        df_day_acf = pd.DataFrame.from_dict(
            {'lags': range(len(dr_acf)),  'ds_acf': ds_acf, 'dr_acf': dr_acf})
        df_day_acf.set_index('lags', inplace=True)
        df_day_acf.to_csv(csv_path)

        plt_path = plot_dir / ("acf_daily_seasonality_hourly_avg_" +
                               dt1.strftime("%Y%m%d%H") + "_" +
                               dt2.strftime("%Y%m%d%H") + ".png")
        p3 = figure(title="Autocorrelation of Daily Residual")
        p3.xaxis.axis_label = "lags"
        p3.yaxis.bounds = (min(0, min(dr_acf)), 1.1)
        p3.line(range(len(dr_acf)), dr_acf, line_color="lightcoral", line_width=2)
        export_png(p3, filename=plt_path)

    @property
    def dict_avg_annual(self):
        return self._dict_avg_annual

    @dict_avg_annual.setter
    def dict_avg_annual(self, dicts):
        self._dict_avg_annual = dicts

    @property
    def dict_avg_weekly(self):
        return self._dict_avg_weekly

    @dict_avg_weekly.setter
    def dict_avg_weekly(self, dicts):
        self._dict_avg_weekly = dicts

    @property
    def dict_avg_daily(self):
        return self._dict_avg_daily

    @dict_avg_daily.setter
    def dict_avg_daily(self, dicts):
        self._dict_avg_daily = dicts

    @property
    def df_h(self):
        return self._df_h

    @property
    def df_d(self):
        return self._df_d

