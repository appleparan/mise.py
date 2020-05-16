import datetime as dt
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytz import timezone
from sklearn.impute import KNNImputer
from scipy.stats import boxcox
import statsmodels.tsa.stattools as stls
from tbats import TBATS
import tqdm

import data
from constants import SEOUL_STATIONS

seoultz = timezone('Asia/Seoul')
HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

def stats_tbats(station_name="종로구"):
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
    output_size = 24
    #train_fdate = dt.datetime(2017, 1, 1, 0).astimezone(seoultz)
    train_fdate = dt.datetime(2017, 1, 1, 0).astimezone(seoultz)
    train_tdate = dt.datetime(2017, 12, 31, 23).astimezone(seoultz)
    test_fdate = dt.datetime(2018, 1, 1, 0).astimezone(seoultz)
    #test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(seoultz)
    test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(seoultz)

    for target in targets:
        dir_prefix = Path("/mnt/data/tbats/" + station_name + "/" + target + "/")
        Path.mkdir(dir_prefix, parents=True, exist_ok=True)

        norm_values, norm_maxlog = boxcox(df_h[target])
        norm_target = "norm_" + target

        df_ta = df_h[[target]].copy()
        df_ta[norm_target] = norm_values

        df_train = df_ta.loc[(df_ta.index.get_level_values(level='date') >= train_fdate) &
                             (df_ta.index.get_level_values(level='date') <= train_tdate)]
        df_test = df_ta.loc[(df_ta.index.get_level_values(level='date') >= test_fdate) &
                            (df_ta.index.get_level_values(level='date') <= test_tdate)]

        print("TBATS with " + target + "...")

        def run_tbats():
            df_obs = mw_df(df_ta, target, output_size,
                        test_fdate, test_tdate)
            df_sim, df_params = sim_tbats(df_train, df_test,
                                          norm_target, test_fdate, test_tdate,
                                          norm_maxlog, output_size)

            # join df
            plot_tbats(df_sim, df_obs, target, test_fdate,
                       test_tdate, station_name, output_size)
            # save to csv
            csv_fname = "obs_tbats_" + target + ".csv"
            df_obs.to_csv(dir_prefix / csv_fname)

            csv_fname = "sim_tbats_" + target + ".csv"
            df_sim.to_csv(dir_prefix / csv_fname)

            # save to csv
            csv_fname = "tbats_params_" + target + ".csv"
            df_params.to_csv(dir_prefix / csv_fname)
        run_tbats()


def mw_df(df_org, target, output_size, fdate, tdate):
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
        index = pd.Index([_date], dtype=type(_date), name='date')

        assert dates[i] == _date
        values[i, :] = value

    df_obs = pd.DataFrame(data=values, index=dates, columns=cols)
    df_obs.index.name = 'date'

    return df_obs


def sim_tbats(_df_train, _df_test, target, fdate, tdate, norm_maxlog, output_size):
    # columns are offset to datetime
    cols = [str(i) for i in range(output_size)]
    df_train = _df_train.reset_index(level='stationCode', drop=True)
    df_test = _df_test.reset_index(level='stationCode', drop=True)
    df_sim = pd.DataFrame(columns=cols)
    cols_params = ["alpha",  "beta", "phi", # coeffs.
        "prd_sea1", "prd_sea2",  # seasonal periods
        "hmn_sea1", "hmn_sea2",  # seasonal harmonics
        "gmm_sea1_0", "gmm_sea1_1",
        "gmm_sea2_0", "gmm_sea2_1"]  # smoothing parameters
    df_param = pd.DataFrame(columns=cols_params)

    # train -> initial data
    endog = df_train[target].tolist()
    sz = len(endog)
    duration = tdate - fdate
    dates_hours = duration.days * 24 + duration.seconds // 3600
    dates = [fdate + dt.timedelta(hours=x)
             for x in range(dates_hours - output_size + 1)]
    values = np.zeros((len(dates), output_size), dtype=df_train[target].dtype)
    values_params = np.zeros((len(cols_params), output_size))
    assert len(dates) == dates_hours - output_size + 1
    
    model = TBATS(seasonal_periods=[7*24, 365.25*24],
                use_arma_errors=False,
                use_box_cox=False,
                n_jobs=8)
    print("Fitting...")
    model_fit = model.fit(endog)
    print("FItting Complete!")

    for i, (index, row) in tqdm.tqdm(enumerate(df_test.iterrows()), total=len(dates)-1):
        if i > len(dates) - 1:
            break
        out_raw = model_fit.forecast(steps=output_size)
        value = [np.exp(np.log(norm_maxlog * o + 1.0) / norm_maxlog) for o in out_raw]

        _date = index
        # insert row to dataframe
        assert dates[i] == _date
        values[i, :] = value

        params = model_fit.params
        components = model_fit.params.components
        """
        print(_date)
        print(params.alpha, params.beta, params.phi)
        print(components.seasonal_periods)
        print(components.seasonal_harmonics)
        print(params.gamma_1())
        print(params.gamma_1())
        """
        values_param[i, :] = [
            params.alpha,
            params.beta,
            params.phi,
            components.seasonal_periods[0],
            components.seasonal_periods[1],
            components.seasonal_harmonics[0],
            components.seasonal_harmonics[1],
            params.gamma_1()[0],
            params.gamma_1()[1],
            params.gamma_2()[0],
            params.gamma_2()[1]
        ]

        # Rolling Forecast tbats Model
        endog.append(row[target])
    df_sim = pd.DataFrame(data=values, index=dates, columns=cols)
    df_param = pd.DataFrame(data=values_params, index=dates, columns=cols)
    df_sim.index.name = 'date'
    df_param.index.name = 'date'
    print(df_sim.head(5))

    return df_sim, df_param


def plot_tbats(df_sim, df_obs, target, _test_fdate, _test_tdate, station_name, output_size):
    dir_prefix = Path("/mnt/data/tbats/" + station_name + "/" + target + "/")

    times = list(range(0, output_size+1))
    corrs = [1.0]

    test_fdate = _test_fdate
    test_tdate = _test_tdate - dt.timedelta(hours=output_size)

    # filter by test dates
    _obs = df_obs[(df_obs.index.get_level_values(level='date') >= test_fdate) &
                  (df_obs.index.get_level_values(level='date') <= test_tdate)]
    # simulation result might have exceed our observation
    _sim = df_sim[(df_sim.index.get_level_values(level='date') >= test_fdate) &
                  (df_sim.index.get_level_values(level='date') <= test_tdate)]
    print(_obs.head(5))
    print(_sim.head(5))

    for t in range(output_size):
        # zero-padded directory name
        output_dir = dir_prefix / str(t).zfill(2)
        Path.mkdir(output_dir, parents=True, exist_ok=True)

        # get column per each time
        obs = _obs[str(t)].to_numpy()
        sim = _sim[str(t)].to_numpy()

        scatter_fname = "scatter_tbats" + \
            target + "_h" + str(t).zfill(2) + ".png"
        plot_scatter(obs, sim, output_dir / scatter_fname)
        # plot line
        line_fname = "line_tbats" + \
            target + "_h" + str(t).zfill(2) + ".png"
        plot_dates = plot_line(obs, sim, test_fdate, test_tdate, target,
                               output_size, output_dir / line_fname)

        csv_fname = "data_tbats" + \
            target + "_h" + str(t).zfill(2) + ".csv"
        df_obs_sim = pd.DataFrame({'obs': obs, 'sim': sim}, index=plot_dates)
        df_obs_sim.to_csv(output_dir / csv_fname)

        # np.corrcoef -> [[1.0, corr], [corr, 1]]
        corrs.append(np.corrcoef(obs, sim)[0, 1])
    output_dir = dir_prefix
    # plot corr for all times
    corr_fname = "corr_tbats_" + \
        target + "_" + str(t).zfill(2) + ".png"
    csv_fname = "corr_hourly_tbats_" + \
        target + "_h" + str(t).zfill(2) + ".csv"

    df_corrs = pd.DataFrame({'time': times, 'corr': corrs})
    df_corrs.to_csv(output_dir / csv_fname)

    plot_corr(times, corrs, output_dir / corr_fname)


def plot_scatter(obs, sim, output_name):
    plt.figure()
    plt.title("Model/OBS")
    plt.xlabel("OBS")
    plt.ylabel("Model")
    maxval = np.nanmax([np.nanmax(obs), np.nanmax(sim)])
    plt.xlim((0, maxval))
    plt.ylim((0, maxval))
    plt.scatter(obs, sim)
    plt.savefig(output_name)
    plt.close()


def plot_line(obs, sim, test_fdate, test_tdate, target, output_size, output_name):
    dates = np.array([test_fdate + dt.timedelta(hours=i)
                      for i in range(len(obs))])

    plt.figure()
    plt.title("OBS & Model")
    plt.xlabel("dates")
    plt.ylabel(target)
    plt.plot(dates, obs, "b", dates, sim, "r")
    plt.savefig(output_name)
    plt.close()

    return dates


def plot_corr(times, corrs, output_name):
    plt.figure()
    plt.title("Correlation of OBS & Model")
    plt.xlabel("lags")
    plt.ylabel("corr")
    plt.plot(times, corrs)
    plt.savefig(output_name)
    plt.close()
