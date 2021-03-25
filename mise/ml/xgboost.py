from argparse import Namespace
import copy
import datetime as dt
from math import sqrt
import os
from pathlib import Path
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
import tqdm

import statsmodels.graphics.tsaplots as tpl
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.tsatools import detrend
import xgboost
import shap

from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

import data
from constants import SEOUL_STATIONS, SEOULTZ
import utils

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

def ml_xgboost(station_name="종로구"):
    print("Start Multivariate XGBoost", flush=True)
    _df_h = data.load_imputed(HOURLY_DATA_PATH)
    df_h = _df_h.query('stationCode == "' +
                        str(SEOUL_STATIONS[station_name]) + '"')

    if station_name == '종로구' and \
        not Path("/input/python/input_jongno_imputed_hourly_pandas.csv").is_file():
        # load imputed result

        df_h.to_csv("/input/python/input_jongno_imputed_hourly_pandas.csv")

    print("Data loading complete", flush=True)
    targets = ["PM10", "PM25"]

    features=["SO2", "CO", "O3", "NO2", "PM10", "PM25",
            "temp", "wind_spd", "wind_cdir", "wind_sdir",
            "pres", "humid", "prep", "snow"]
    features_periodic=["SO2", "CO", "O3", "NO2", "PM10", "PM25", "temp",
                        "wind_spd", "wind_cdir", "wind_sdir", "pres", "humid"]
    features_nonperiodic=["prep", "snow"]

    # use one step input
    sample_size = 1
    output_size = 24
    train_fdate = dt.datetime(2015, 1, 3, 0).astimezone(SEOULTZ)
    train_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2020, 10, 31, 23).astimezone(SEOULTZ)
    # consective dates between train and test
    assert train_tdate + dt.timedelta(hours=1) == test_fdate

    # check date range assumption
    assert test_tdate > train_fdate
    assert test_fdate > train_tdate

    for target in targets:
        train_set = data.MultivariateMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=HOURLY_DATA_PATH,
            features=features,
            features_1=features_nonperiodic,
            features_2=features_periodic,
            fdate=train_fdate,
            tdate=train_tdate,
            sample_size=sample_size,
            output_size=output_size,
            train_valid_ratio=0.8)

        train_set.preprocess()

        # set fdate=test_fdate,
        test_set = data.MultivariateMeanSeasonalityDataset(
            station_name=station_name,
            target=target,
            filepath=HOURLY_DATA_PATH,
            features=features,
            features_1=features_nonperiodic,
            features_2=features_periodic,
            fdate=test_fdate,
            tdate=test_tdate,
            sample_size=sample_size,
            output_size=output_size,
            scaler_X=train_set.scaler_X,
            scaler_Y=train_set.scaler_Y)

        test_set.transform()

        df_train = train_set.ys.loc[train_fdate:train_tdate, :].copy()
        df_test = test_set.ys.loc[test_fdate:test_tdate, :].copy()
        df_test_org = test_set.ys_raw.loc[test_fdate:test_tdate, :].copy()

        # for lag in range(23, 24):
        input_lag = 0
        output_dir = Path(f'/mnt/data/XGBoost/' + station_name + "/" + target + "/")
        png_dir = output_dir / Path('png/')
        svg_dir = output_dir / Path('svg/')
        data_dir = output_dir / Path('csv/')
        Path.mkdir(data_dir, parents=True, exist_ok=True)
        Path.mkdir(png_dir, parents=True, exist_ok=True)
        Path.mkdir(svg_dir, parents=True, exist_ok=True)
        # prepare dataset
        print("Dataset conversion start..", flush=True)
        X_train, Y_train, train_dates = dataset2svinput(train_set, lag=input_lag)
        X_test, Y_test, test_dates = dataset2svinput(test_set, lag=input_lag)

        print("Dataset conversion complete..", flush=True)

        print("XGBoost " + target + "...", flush=True)
        df_obs = mw_df(df_test_org, target, output_size, input_lag,
                    test_fdate, test_tdate)

        # prediction
        df_sim = sim_xgboost(X_train, Y_train, X_test, Y_test, test_dates,
                features, target, sample_size, output_size, test_set.scaler_Y,
                test_fdate, test_tdate, data_dir, png_dir, svg_dir)

        assert df_obs.shape == df_sim.shape

        # join df
        plot_xgboost(df_sim, df_obs, target, \
            data_dir, png_dir, svg_dir, test_fdate, test_tdate, station_name, output_size)
        # save to csv
        csv_fname = "df_test_obs.csv"
        df_obs.to_csv(data_dir / csv_fname)

        csv_fname = "df_test_sim.csv"
        df_sim.to_csv(data_dir / csv_fname)

def dataset2svinput(dataset, lag=0):
    """Iterate dataset then separate it to X and Y 

    X: single-step input
    Y: lagged multi-step output

    if lag == 0, X + 1 hour => 1st item of Y
    """
    # single step
    _Xset = [dataset[i][0] for i in range(len(dataset)) if i + lag < len(dataset)]
    # lagged multi step 
    _Yset = [dataset[i + lag][1] for i in range(len(dataset)) if i + lag < len(dataset)]
    # index of single step -> 1 step
    x_dates = [dataset.xs.index[i] for i in range(len(dataset)) if i + lag < len(dataset)]
    # starting index of multi step -> 1 step
    # dataset[i + lag][3] : total dates of prediction result of single step
    y_dates = [dataset[i + lag][3][0] for i in range(len(dataset)) if i + lag < len(dataset)]

    ycols = range(len(_Yset[0]))
    # 1D inputs -> total time steps x features DataFrame
    Xset = pd.DataFrame(data=_Xset, index=x_dates, columns=dataset.xs.columns)
    # 1D inputs -> total time steps x predition time steps
    Yset = pd.DataFrame(data=_Yset, index=y_dates, columns=ycols)

    return Xset, Yset, y_dates


def mw_df(df_org, target, output_size, lag, fdate, tdate):
    """
    moving window
    """
    cols = [str(i) for i in range(output_size)]
    df_obs = pd.DataFrame(columns=cols)
    _dict = {}

    df = df_org.loc[fdate:tdate, :]
    cols = [str(t) for t in range(output_size)]
    df_obs = pd.DataFrame(columns=cols)

    values, indicies = [], []

    for i, (index, row) in enumerate(df.iterrows()):
        # skip prediction before fdate
        if index + dt.timedelta(hours=lag) < fdate:
            continue

        # findex ~ tindex = output_size
        findex = index + dt.timedelta(hours=lag)
        tindex = index + dt.timedelta(hours=lag) + dt.timedelta(hours=(output_size - 1))
        if tindex > tdate - dt.timedelta(hours=output_size):
            break

        _df = df.loc[findex:tindex, :]

        df_obs.loc[findex] = _df.to_numpy().reshape(-1)

    df_obs.index.name = 'date'

    return df_obs


def sim_xgboost(X_train, Y_train, X_test, Y_test, dates,
    features, target, sample_size, output_size, scaler,
    test_fdate, test_tdate, data_dir, png_dir, svg_dir):
    # columns are offset to datetime
    cols = [str(i) for i in range(output_size)]
    df_sim = pd.DataFrame(columns=cols)

    # output shape and dtype is same as Y_test (observation)
    values = np.zeros(Y_test.shape, dtype=Y_test.dtypes[0])

    # create model and fit to X_train and Y_train
    models = []
    models.append(xgboost.XGBRegressor(objective='reg:squarederror',
        n_estimators=1000))
    print("Models are created!", flush=True)
    # multioutputregressor = MultiOutputRegressor(model.fit(X_train, Y_train.loc[:, lag], verbose=True))
    for l in tqdm.tqdm(range(output_size)):
        models[l].fit(X_train, Y_train.loc[:, l], verbose=True)

        # feature importance by XGBOost Feature importance
        plt.figure()
        plt.barh(features, models[l].feature_importances_)
        output_to_plot = 'xgb_feature_importance_' + str(l + 1).zfill(2) +"h"
        data_path = data_dir / (output_to_plot + '.csv')
        png_path = png_dir / (output_to_plot + '.png')
        svg_path = svg_dir / (output_to_plot + '.svg')
        pd.DataFrame(data=[models[l].feature_importances_], columns=features).to_csv(data_path)
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()

        # feature importance by SHAP values
        explainer = shap.Explainer(models[l])
        shap_values = explainer(X_test)

        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_name)
        output_to_plot = 'shap_values_' + str(l + 1).zfill(2) +"h"
        data_path = data_dir / (output_to_plot + '.csv')
        png_path = png_dir / (output_to_plot + '.png')
        svg_path = svg_dir / (output_to_plot + '.svg')
        pd.DataFrame(data=[shap_values], columns=features).to_csv(data_path)
        plt.savefig(png_path, dpi=600)
        plt.savefig(svg_path)
        plt.close()


    print("Models are fitted", flush=True)

    for i, (index, row) in tqdm.tqdm(enumerate(X_test.iterrows())):
        _dates = pd.date_range(
                index, index + dt.timedelta(hours=(output_size - 1)), freq='1H')
        ys = np.zeros(output_size)
        for l in range(output_size):
            # out-of-sample forecast
            ys[l] = models[l].predict([row])

        # inverse_transform
        value = scaler.named_transformers_['num'].inverse_transform(
            pd.DataFrame(data=ys, index=_dates, columns=[target]))

        values[i, :] = value.squeeze()

    df_sim = pd.DataFrame(data=values, index=Y_test.index, columns=cols)
    df_sim.index.name = 'date'

    return df_sim

def plot_xgboost(df_sim, df_obs, target, order, data_dir, png_dir, svg_dir, _test_fdate, _test_tdate, station_name, output_size):
    dir_prefix = Path("/mnt/data/XGBoost/" + station_name + "/" + target + "/")

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

