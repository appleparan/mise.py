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
import tqdm

import data
from constants import SEOUL_STATIONS

seoultz = timezone('Asia/Seoul')

def stats_arima(station_name = "종로구"):
    print("Data loading start...")
    if Path("/input/input_jongro_pandas.csv").is_file():
        df = data.load("/input/input_jongro_pandas.csv")
    else:
        sdf = data.load_station(data.load(), SEOUL_STATIONS[station_name])
        print("Start KNN Imputation...")
        imputer = KNNImputer(
            n_neighbors=2, weights="uniform", missing_values=np.NaN)
        df = pd.DataFrame(imputer.fit_transform(sdf))
        df.columns = sdf.columns
        df.index = sdf.index
        df.to_csv("/input/input_jongro_pandas.csv")

    print("Data loading complete")
    targets = ["PM10", "PM25"]
    output_size = 24
    train_fdate = dt.datetime(2018,12,16,1).astimezone(seoultz)
    train_tdate = dt.datetime(2018,12,31,23).astimezone(seoultz)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(seoultz)
    test_tdate = dt.datetime(2019, 1, 15, 23).astimezone(seoultz)

    for target in targets:
        dir_prefix = Path("/mnt/arima/" + station_name + "/" + target + "/")
        Path.mkdir(dir_prefix, parents=True, exist_ok=True)

        norm_values, norm_maxlog = boxcox(df[target])
        norm_target = "norm_" + target

        df_ta = df[[target]].copy()
        df_ta[norm_target] = norm_values

        #train = df_ta.loc[pd.Timestamp(train_fdate):pd.Timestamp(train_tdate)]
        #test = df_ta.loc[pd.Timestamp(test_tdate):pd.Timestamp(test_tdate)]
        print(df_ta.head(10))
        df_train = df_ta.loc[(df_ta.index.get_level_values(level='date') >= train_fdate) & \
                          (df_ta.index.get_level_values(level='date') <= train_tdate)]
        df_test = df_ta.loc[(df_ta.index.get_level_values(level='date') >= test_fdate) &
                          (df_ta.index.get_level_values(level='date') <= test_tdate)]

        print("ARIMA with " + target + "...")
        def run_arima(order):
            _df_out = sim_arima(df_train, df_test, norm_target, order, norm_maxlog, output_size)
            plot_arima(_df_out, df_ta, target, order, test_fdate, test_tdate, station_name, output_size)
            # join df
            #df_out = mw_df(_df_out, target, output_size, test_fdate, test_tdate - dt.timedelta(hours=output_size))
            # save to csv
            #csv_fname = "csv_arima(" + \
            #    str(order[0]) + ", " + str(order[1]) + ", " + str(order[2]) + ")_" + \
            #    target + ".png"
            #df_out.to_csv(dir_prefix / csv_fname)
        print("ARIMA(1, 0, 0)...")
        run_arima((1, 0, 0))
        print("ARIMA(0, 0, 1)...")
        run_arima((0, 0, 1))
        print("ARIMA(1, 0, 1)...")
        run_arima((1, 0, 1))
        print("ARIMA(1, 1, 1)...")
        run_arima((1, 1, 1))

def mw_df(_df, target, output_size, fdate, tdate):
    df = _df[(fdate <= _df.date) & (_df.date <= tdate)]
    cols = [str(i) for i in range(output_size)]
    df_out = pd.DataFrame(columns=["date"].extend(cols))

    dates = pd.date_range(start=fdate, end=tdate, freq='1H')

    for index, row in df.iterrows():
        values = df[target].iloc[index:(index + output_size-1)].to_numpy()[:]
        _d = df[target].iloc[index:(index + output_size-1)].index[0]
        arr = [_d]
        arr.extend(values)
        df_out.append(arr)

    return df_out


def sim_arima(df_train, df_test, target, order, norm_maxlog, output_size):
    # columns are offset to datetime
    cols = [str(i) for i in range(output_size)]
    df_out = pd.DataFrame(columns=cols)

    # train -> initial data
    endog = df_train[target].tolist()
    sz = len(endog)

    for index, row in tqdm.tqdm(df_test.iterrows(), total=len(df_test)):
        model = arm.ARIMA(endog[-sz:], order)
        model_fit = model.fit(disp = 0)

        out_raw, err_arr, conf_ints = model_fit.forecast(steps = output_size)
        # recover box-cox transformation
        out_ibc = [np.exp(np.log(norm_maxlog * o + 1.0) / norm_maxlog) for o in out_raw]

        _date = index[1]
        _idx = pd.Index([_date], dtype=type(_date), name='date')
        # insert row to dataframe
        df_out = df_out.append(pd.DataFrame(dict(zip(cols, out_ibc)), index=[_date]))

        # Rolling Forecast ARIMA Model
        endog.append(row[target])
    df_out_idx = df_out.index.set_names('date')
    df_out.index = df_out_idx

    return df_out


def plot_arima(df_sim, df_obs, target, order, _test_fdate, _test_tdate, station_name, output_size):
    dir_prefix = Path("/mnt/arima/" + station_name + "/" + target + "/")

    times = list(range(0, output_size+1))
    corrs = [1.0]

    for t in range(output_size):
        # zero-padded directory name
        output_dir = dir_prefix / str(t).zfill(2)
        Path.mkdir(output_dir, parents=True, exist_ok = True)

        test_fdate = _test_fdate
        test_tdate = _test_tdate - dt.timedelta(hours=t)

        # plot scatter
        #obs = df_obs[test_fdate <= df_obs.loc[target] < test_tdate]
        _obs = df_obs[(df_obs.index.get_level_values(level='date') >= test_fdate) &
                      (df_obs.index.get_level_values(level='date') <= test_tdate)]
        _sim = df_sim[(df_sim.index.get_level_values(level='date') >= test_fdate) &
                      (df_sim.index.get_level_values(level='date') <= test_tdate)]

        obs = _obs[target].to_numpy()
        sim = _sim[str(t)].to_numpy()

        scatter_fname = "scatter_arima(" + \
            str(order[0]) + ", " + str(order[1]) + ", " + str(order[2]) + ")_" + \
            target + "_h" + str(t).zfill(2) + ".png"
        plot_scatter(obs, sim, output_dir / scatter_fname)
        # plot line
        line_fname = "line_arima(" + \
            str(order[0]) + ", " + str(order[1]) + ", " + str(order[2]) + ")_" + \
            target + "_h" + str(t).zfill(2) + ".png"
        plot_dates = plot_line(obs, sim, test_fdate, test_tdate, target,
                  output_size, output_dir / line_fname)

        csv_fname = "data_arima(" + \
            str(order[0]) + ", " + str(order[1]) + ", " + str(order[2]) + ")_" + \
            target + "_h" + str(t).zfill(2) + ".csv"
        df_obs_sim = pd.DataFrame({'obs': obs, 'sim': sim}, index=plot_dates)
        df_obs_sim.to_csv(output_dir / csv_fname)
        
        # np.corrcoef -> [[1.0, corr], [corr, 1]]
        corrs.append(np.corrcoef(obs, sim)[0, 1])
    output_dir = dir_prefix
    # plot corr for all times
    corr_fname = "corr_arima(" + \
        str(order[0]) + ", " + str(order[1]) + ", " + str(order[2]) + ")_" + \
        target + "_" + str(t).zfill(2) + ".png"
    csv_fname = "corr_hourly_arima(" + \
        str(order[0]) + ", " + str(order[1]) + ", " + str(order[2]) + ")_" + \
        target + "_h" + str(t).zfill(2) + ".csv"
    
    df_corrs = pd.DataFrame({'time': times, 'corr': corrs})
    df_corrs.to_csv(output_dir / csv_fname)

    plot_corr(times, corrs, output_dir / corr_fname)

def plot_scatter(obs, sim, output_name):
    plt.figure()
    plt.title("Model/OBS")
    plt.xlabel("OBS")
    plt.ylabel("Model")
    plt.scatter(obs, sim)
    plt.savefig(output_name)
    plt.close()


def plot_line(obs, sim, test_fdate, test_tdate, target, output_size, output_name):
    dates = np.array([test_fdate + dt.timedelta(hours=i) for i in range(len(obs))])

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

