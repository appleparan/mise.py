import copy
import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.impute import KNNImputer
from scipy.stats import boxcox
import statsmodels.tsa.arima_model as arm
import statsmodels.tsa.stattools as stls
import statsmodels.graphics.tsaplots as tpl
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.tsatools import detrend
import tqdm

from bokeh.models import Range1d, DatetimeTickFormatter
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png, export_svgs

import data
from constants import SEOUL_STATIONS

seoultz = timezone('Asia/Seoul')
HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

def stats_arima(station_name = "종로구"):
    print("Data loading start...")
    if Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
        df_d = data.load_imputed("/input/python/input_jongro_imputed_daily_pandas.csv")
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
    orders = [(1, 0, 0), (1, 0, 1)]
    output_size = 24
    train_fdate = dt.datetime(2018, 1, 1, 0).astimezone(seoultz)
    train_tdate = dt.datetime(2018, 12, 31, 23).astimezone(seoultz)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(seoultz)
    test_tdate = dt.datetime(2019, 12, 31, 23).astimezone(seoultz)
    # consective dates between train and test
    assert train_tdate + dt.timedelta(hours=1) == test_fdate

    for target in targets:
        for order in orders:
            output_dir = Path('/mnt/data/ARIMA_' + str(order) + '/' +
                            station_name + "/" + target + "/")
            png_dir = output_dir / Path('png/')
            svg_dir = output_dir / Path('svg/')
            data_dir = output_dir / Path('csv/')
            Path.mkdir(data_dir, parents=True, exist_ok=True)
            Path.mkdir(png_dir, parents=True, exist_ok=True)
            Path.mkdir(svg_dir, parents=True, exist_ok=True)
            norm_values, norm_maxlog = boxcox(df_h[target])
            norm_target = "norm_" + target

            df_ta = df_h[[target]].copy()
            df_ta[norm_target] = norm_values

            df_train = df_ta.loc[(df_ta.index.get_level_values(level='date') >= train_fdate) & \
                            (df_ta.index.get_level_values(level='date') <= train_tdate)]
            df_test = df_ta.loc[(df_ta.index.get_level_values(level='date') >= test_fdate) &
                            (df_ta.index.get_level_values(level='date') <= test_tdate)]

            print("ACF analysis with " + target + "...")
            # plot acf and pacf
            nlags_acf = 24*10
            _acf = acf(df_train[target], nlags=nlags_acf)
            _pacf = pacf(df_train[target])
            plot_acf(df_train[target], nlags_acf, _acf, _pacf, data_dir, png_dir, svg_dir)

            print("ARIMA " + str(order) + " of " + target + "...")
            def run_arima(order):
                df_obs = mw_df(df_ta, target, output_size,
                            test_fdate, test_tdate)
                df_sim = sim_arima(df_train, df_test,
                                norm_target, order, test_fdate, test_tdate,
                                norm_maxlog, output_size)

                # join df
                plot_arima(df_sim, df_obs, target, order, data_dir, png_dir, svg_dir, test_fdate,
                        test_tdate, station_name, output_size)
                # save to csv
                csv_fname = "df_test_obs.csv"
                df_obs.to_csv(data_dir / csv_fname)

                csv_fname = "df_test_sim.csv"
                df_sim.to_csv(data_dir / csv_fname)

            print("ARIMA " + str(order) + " ...")
            run_arima(order)


def mw_df(df_org, target, output_size, fdate, tdate):
    """
    moving window
    """
    # get indices from df_sim
    df = df_org[(df_org.index.get_level_values(level='date') >= fdate) &
                (df_org.index.get_level_values(level='date') <= tdate)]
    df.reset_index(level='stationCode', drop=True, inplace=True)
    cols = [str(i) for i in range(output_size)]
    df_obs = pd.DataFrame(columns=cols)
    _dict = {}

    dates = [fdate + dt.timedelta(hours=x)
             for x in range(len(df) - output_size + 1)]
    values = np.zeros((len(dates), output_size), dtype=df[target].dtype)
    assert len(dates) == len(df) - output_size + 1

    for i, (_index, _row) in enumerate(df.iterrows()):
        # findex ~ tindex = output_size
        findex = _index
        tindex = _index + dt.timedelta(hours=(output_size - 1))
        if tindex > tdate:
            break
        _df = df[(df.index.get_level_values(level='date') >= findex) &
                 (df.index.get_level_values(level='date') <= tindex)]

        # get observation value from original datafram and construct df_obs
        value = _df[target].to_numpy()[:]
        _date = _index

        assert dates[i] == _date
        values[i, :] = value

    df_obs = pd.DataFrame(data=values, index=dates, columns=cols)
    df_obs.index.name = 'date'

    return df_obs


def sim_arima(_df_train, _df_test, target, order, fdate, tdate, norm_maxlog, output_size):
    # columns are offset to datetime
    cols = [str(i) for i in range(output_size)]
    df_train = _df_train.reset_index(level='stationCode', drop=True).asfreq('1H')
    df_test = _df_test.reset_index(level='stationCode', drop=True).asfreq('1H')
    df_sim = pd.DataFrame(columns=cols)

    # train -> initial data
    detr_x = detrend(df_train[target], order=2)
    endog = detr_x
    #endog = df_train[target].tolist()
    sz = len(endog)
    duration = tdate - fdate
    dates_hours = duration.days * 24 + duration.seconds // 3600
    dates = [fdate + dt.timedelta(hours=x)
             for x in range(dates_hours - output_size + 1)]
    values = np.zeros((len(dates), output_size), dtype=df_train[target].dtype)
    assert len(dates) == dates_hours - output_size + 1

    # initial endog
    index0 = df_test.index[0]
    endog = list(df_train.loc[index0-sz*index0.freq:index0, target])

    for i, (index, row) in tqdm.tqdm(enumerate(df_test.iterrows()), total=len(dates)-1):
        if i > len(dates) - 1:
            break

        model = arm.ARIMA(np.array(endog[-sz:]), order)
        model_fit = model.fit(disp = 0)

        out_raw, err_arr, conf_ints = model_fit.forecast(steps = output_size)
        # recover box-cox transformation
        value = [np.exp(np.log(norm_maxlog * o + 1.0) / norm_maxlog) for o in out_raw]

        _date = index
        assert dates[i] == _date
        values[i, :] = value

        endog.append(row[target])

    df_sim = pd.DataFrame(data=values, index=dates, columns=cols)
    df_sim.index.name = 'date'
    return df_sim


def plot_arima(df_sim, df_obs, target, order, data_dir, png_dir, svg_dir, _test_fdate, _test_tdate, station_name, output_size):
    dir_prefix = Path("/mnt/data/ARIMA/" + station_name + "/" + target + "/")

    times = list(range(0, output_size+1))
    corrs = [1.0]

    test_fdate = _test_fdate
    test_tdate = _test_tdate - dt.timedelta(hours=output_size)

    _obs = df_obs[(df_obs.index.get_level_values(level='date') >= test_fdate) &
                    (df_obs.index.get_level_values(level='date') <= test_tdate)]
    # simulation result might have exceed our observation
    _sim = df_sim[(df_sim.index.get_level_values(level='date') >= test_fdate) &
                  (df_sim.index.get_level_values(level='date') <= test_tdate)]

    for t in range(output_size):
        # zero-padded directory name
        Path.mkdir(data_dir / Path(str(t).zfill(2)),
                   parents=True, exist_ok=True)
        Path.mkdir(svg_dir / Path(str(t).zfill(2)),
                   parents=True, exist_ok=True)
        Path.mkdir(png_dir / Path(str(t).zfill(2)),
                   parents=True, exist_ok=True)

        # get column per each time
        obs = _obs[str(t)].to_numpy()
        sim = _sim[str(t)].to_numpy()

        scatter_fname = "scatter_" + str(t).zfill(2) + "h"
        plot_scatter(obs, sim, data_dir / Path(str(t).zfill(2)),
                     png_dir / Path(str(t).zfill(2)),
                     svg_dir / Path(str(t).zfill(2)), scatter_fname)
        # plot line
        line_fname = "line_" + str(t).zfill(2) + "h"
        plot_dates = plot_line(obs, sim, test_fdate, test_tdate, target,
                               data_dir / Path(str(t).zfill(2)),
                               png_dir / Path(str(t).zfill(2)),
                               svg_dir / Path(str(t).zfill(2)), line_fname)

        csv_fname = "data_" + str(t).zfill(2) + "h.csv"
        df_obs_sim = pd.DataFrame({'obs': obs, 'sim': sim}, index=plot_dates)
        df_obs_sim.to_csv(data_dir / Path(str(t).zfill(2)) / csv_fname)

        # np.corrcoef -> [[1.0, corr], [corr, 1]]
        corrs.append(np.corrcoef(obs, sim)[0, 1])

    # plot corr for all times
    corr_fname = "corr_time"

    plot_corr(times, corrs, data_dir, png_dir, svg_dir, corr_fname)


def plot_scatter(obs, sim, data_dir, png_dir, svg_dir, output_name):
    png_path = png_dir / (output_name + ".png")
    svg_path = svg_dir / (output_name + ".svg")

    df_scatter = pd.DataFrame({'obs': obs, 'sim': sim})
    df_scatter.to_csv(data_dir / (output_name + ".csv"))

    p = figure(title="Model/OBS")
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.axis_label = "OBS"
    p.yaxis.axis_label = "Model"
    maxval = np.nanmax([np.nanmax(obs), np.nanmax(sim)])
    p.xaxis.bounds = (0.0, maxval)
    p.yaxis.bounds = (0.0, maxval)
    p.x_range = Range1d(0.0, maxval)
    p.y_range = Range1d(0.0, maxval)
    p.scatter(obs, sim)
    export_png(p, filename=png_path)
    p.output_backend = "svg"
    export_svgs(p, filename=str(svg_path))


def plot_line(obs, sim, test_fdate, test_tdate, target, data_dir, png_dir, svg_dir, output_name):
    png_path = png_dir / (output_name + ".png")
    svg_path = svg_dir / (output_name + ".svg")

    dates = np.array([test_fdate + dt.timedelta(hours=i)
                    for i in range(len(obs))])

    df_line = pd.DataFrame({'dates': dates, 'obs': obs, 'sim': sim})
    df_line.to_csv(data_dir / (output_name + ".csv"))

    p = figure(title="OBS & Model")
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.axis_label = "dates"
    p.xaxis.formatter = DatetimeTickFormatter()
    p.yaxis.axis_label = target
    p.line(dates, obs, line_color="dodgerblue", legend_label="obs")
    p.line(dates, sim, line_color="lightcoral", legend_label="sim")
    export_png(p, filename=png_path)
    p.output_backend = "svg"
    export_svgs(p, filename=str(svg_path))

    return dates


def plot_corr(times, corrs, data_dir, png_dir, svg_dir, output_name):
    png_path = png_dir / (output_name + ".png")
    svg_path = svg_dir / (output_name + ".svg")

    df_corr = pd.DataFrame({'lags': times, 'corr': corrs})
    df_corr.to_csv(data_dir / (output_name + ".csv"))

    p = figure(title="Correlation of OBS & Model")
    p.toolbar.logo = None
    p.toolbar_location = None
    p.xaxis.axis_label = "lags"
    p.yaxis.axis_label = "corr"
    p.yaxis.bounds = (0.0, 1.0)
    p.y_range = Range1d(0.0, 1.0)
    p.line(times, corrs)
    export_png(p, filename=png_path)
    p.output_backend = "svg"
    export_svgs(p, filename=str(svg_path))

def plot_acf(x, nlags, _acf, _pacf, data_dir, png_dir, svg_dir):
    lags_acf = range(len(_acf))
    lags_pacf = range(len(_pacf))

    png_path = png_dir / ("acf.png")
    svg_path = svg_dir / ("acf.svg")
    plt.figure()
    fig = tpl.plot_acf(x, lags=nlags)
    fig.savefig(png_path)
    fig.savefig(svg_path)

    csv_path = data_dir / ("acf.csv")
    df_acf = pd.DataFrame({'lags': lags_acf, 'acf': _acf})
    df_acf.set_index('lags', inplace=True)
    df_acf.to_csv(csv_path)

    png_path = png_dir / ("acf_default_lag.png")
    svg_path = svg_dir / ("acf_default_lag.svg")
    plt.figure()
    fig = tpl.plot_acf(x)
    fig.savefig(png_path)
    fig.savefig(svg_path)

    png_path = png_dir / ("pacf.png")
    svg_path = svg_dir / ("pacf.svg")
    plt.figure()
    fig = tpl.plot_pacf(x)
    fig.savefig(png_path)
    fig.savefig(svg_path)

    csv_path = data_dir / ("pacf.csv")
    df_pacf = pd.DataFrame({'lags': lags_pacf, 'acf': _pacf})
    df_pacf.set_index('lags', inplace=True)
    df_pacf.to_csv(csv_path)

    # detrened
    detr_x = detrend(x, order=2)
    png_path = png_dir / ("detrend_acf.png")
    svg_path = svg_dir / ("detrend_acf.svg")
    plt.figure()
    fig = tpl.plot_acf(detr_x, lags=nlags)
    fig.savefig(png_path)
    fig.savefig(svg_path)

    png_path = png_dir / ("detrend_acf_default_lag.png")
    svg_path = svg_dir / ("detrend_acf_default_lag.svg")
    plt.figure()
    fig = tpl.plot_acf(detr_x)
    fig.savefig(png_path)
    fig.savefig(svg_path)

    png_path = png_dir / ("detrend_pacf.png")
    svg_path = svg_dir / ("detrend_pacf.svg")
    plt.figure()
    fig = tpl.plot_pacf(detr_x)
    fig.savefig(png_path)
    fig.savefig(svg_path)



