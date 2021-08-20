import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import MFDFA
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import torch

from mise.constants import SEOUL_CODES, SEOUL_STATIONS, SEOULTZ
from mise.data import MultivariateRNNMeanSeasonalityDataset, load_imputed

HOURLY_DATA_PATH = "/input/python/input_seoul_imputed_hourly_pandas.csv"
DAILY_DATA_PATH = "/input/python/input_seoul_imputed_daily_pandas.csv"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stats_analysis(_station_name="종로구"):
    """
    References
    * Ghil, M., et al. "Extreme events: dynamics, statistics and prediction."
        Nonlinear Processes in Geophysics 18.3 (2011): 295-350.
    """
    print("Start Analysis of input")
    if not Path(HOURLY_DATA_PATH).is_file():
        query_str = "stationCode in " + str(SEOUL_CODES)
        print(query_str, flush=True)

        _df_h: pd.DataFrame = load_imputed([1])
        _df_h.to_csv("/input/python/input_imputed_hourly_pandas.csv")

        df_h: pd.DataFrame = _df_h.query(query_str)
        df_h.to_csv(HOURLY_DATA_PATH)

        # filter by seoul codes
        print("Imputed!", flush=True)

    # targets = ["PM10", "PM25"]
    # sea_targets = ["SO2", "CO", "O3", "NO2", "PM10", "PM25",
    #                "temp", "u", "v", "pres", "humid", "prep", "snow"]
    # sea_targets = ["prep", "snow"]
    # 24*14 = 336
    sample_size = 24 * 2
    output_size = 24

    train_fdate = dt.datetime(2015, 1, 5, 0).astimezone(SEOULTZ)
    train_fdate = dt.datetime(2008, 1, 3, 0).astimezone(SEOULTZ)
    train_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_fdate = dt.datetime(2019, 1, 1, 0).astimezone(SEOULTZ)
    # test_tdate = dt.datetime(2018, 12, 31, 23).astimezone(SEOULTZ)
    test_tdate = dt.datetime(2020, 10, 31, 23).astimezone(SEOULTZ)

    # check date range assumption
    assert test_tdate > train_fdate
    assert test_fdate > train_tdate

    train_features = [
        "SO2",
        "CO",
        "O3",
        "NO2",
        "PM10",
        "PM25",
        "temp",
        "u",
        "v",
        "pres",
        "humid",
        "prep",
    ]
    train_features_periodic = [
        "SO2",
        "CO",
        "O3",
        "NO2",
        "PM10",
        "PM25",
        "temp",
        "u",
        "v",
        "pres",
        "humid",
    ]
    train_features_nonperiodic = ["prep"]
    # station_names = ['종로구']
    station_names = ["종로구", "강서구", "서초구", "광진구"]

    # for target in targets:
    for station_name in station_names:
        for target in train_features_periodic:
            print("Analyze " + target + "...")

            # if not Path("/input/python/input_jongro_imputed_hourly_pandas.csv").is_file():
            #     # load imputed result
            #     _df_h = load_imputed(HOURLY_DATA_PATH)
            #     df_h = _df_h.query('stationCode == "' +
            #                     str(SEOUL_STATIONS[station_name]) + '"')
            #     df_h.to_csv("/input/python/input_jongro_imputed_hourly_pandas.csv")

            _df_seoul = pd.read_csv(HOURLY_DATA_PATH, index_col=[0, 1], parse_dates=[0])
            # filter by station_name
            _df_station = _df_seoul.query(
                'stationCode == "' + str(SEOUL_STATIONS[station_name]) + '"'
            )
            _df_station.reset_index(level="stationCode", drop=True, inplace=True)
            # df_sea_h = _df_station

            output_dir = Path(
                "/mnt/data/STATS_ANALYSIS_"
                + str(sample_size)
                + "/"
                + station_name
                + "/"
                + target
                + "/"
            )
            Path.mkdir(output_dir, parents=True, exist_ok=True)

            data_dir = output_dir / Path("csv/")
            png_dir = output_dir / Path("png/")
            svg_dir = output_dir / Path("svg/")
            Path.mkdir(data_dir, parents=True, exist_ok=True)
            Path.mkdir(png_dir, parents=True, exist_ok=True)
            Path.mkdir(svg_dir, parents=True, exist_ok=True)

            # prepare dataset
            train_valid_set = MultivariateRNNMeanSeasonalityDataset(
                station_name=station_name,
                target=target,
                filepath="/input/python/input_seoul_imputed_hourly_pandas.csv",
                features=train_features,
                features_1=train_features_nonperiodic,
                features_2=train_features_periodic,
                fdate=train_fdate,
                tdate=train_tdate,
                sample_size=sample_size,
                output_size=output_size,
                train_valid_ratio=0.8,
            )

            # first mkdir of seasonality
            Path.mkdir(png_dir / "seasonality", parents=True, exist_ok=True)
            Path.mkdir(svg_dir / "seasonality", parents=True, exist_ok=True)
            Path.mkdir(data_dir / "seasonality", parents=True, exist_ok=True)

            # fit & transform (seasonality)
            # without seasonality
            # train_valid_set.preprocess()
            # with seasonality
            train_valid_set.preprocess()
            # save seasonality index-wise
            train_valid_set.broadcast_seasonality()

            test_set = MultivariateRNNMeanSeasonalityDataset(
                station_name=station_name,
                target=target,
                filepath="/input/python/input_seoul_imputed_hourly_pandas.csv",
                features=train_features,
                features_1=train_features_nonperiodic,
                features_2=train_features_periodic,
                fdate=test_fdate,
                tdate=test_tdate,
                sample_size=sample_size,
                output_size=output_size,
                scaler_X=train_valid_set.scaler_X,
                scaler_Y=train_valid_set.scaler_Y,
            )

            test_set.transform()
            # save seasonality index-wise
            test_set.broadcast_seasonality()

            def run_02_MFDFA():
                """MFDFA Analysis"""
                print("MF-DFA..")
                _data_dir = data_dir / "02-LRD-MFDFA"
                _png_dir = png_dir / "02-LRD-MFDFA"
                _svg_dir = svg_dir / "02-LRD-MFDFA"
                Path.mkdir(_data_dir, parents=True, exist_ok=True)
                Path.mkdir(_png_dir, parents=True, exist_ok=True)
                Path.mkdir(_svg_dir, parents=True, exist_ok=True)

                # Define unbounded process
                Xs = train_valid_set.ys
                Xs_raw = train_valid_set.ys_raw

                n_lag = 100
                large_s = int(n_lag * 0.3)
                org_lag = np.unique(np.logspace(0.5, 3, n_lag).astype(int))

                # Select a list of powers q
                # if q == 2 -> standard square root based average
                q_list = [-6, -2, -3, 2, 3, 6]

                # The order of the polynomial fitting
                for order in [1, 2, 3]:
                    lag, dfa, _ = MFDFA.MFDFA(
                        Xs[target].to_numpy(), lag=org_lag, q=q_list, order=order
                    )
                    norm_dfa = np.zeros_like(dfa)

                    for i in range(dfa.shape[1]):
                        norm_dfa[:, i] = np.divide(dfa[:, i], np.sqrt(lag))

                    df = pd.DataFrame.from_dict(
                        {str(q_list[i]): dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df["s"] = lag

                    df_norm = pd.DataFrame.from_dict(
                        {str(q_list[i]): norm_dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df_norm["s"] = lag

                    # plot
                    fig = plt.figure()
                    plt.clf()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df_norm, id_vars=["s"], var_name="q"),
                    )
                    q0fit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, 0])[large_s:],
                        1,
                    )
                    q0fit_vals = np.polynomial.polynomial.polyval(
                        np.log10(lag), q0fit.coef
                    )
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[0], q0fit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    qnfit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, -1])[large_s:],
                        1,
                    )
                    # qnfit_vals = np.polynomial.polynomial.polyval(
                    #     np.log10(lag), qnfit.coef)
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[-1], qnfit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    for (i, _) in enumerate(q_list):
                        leg_labels[i] = r"h({{{0}}})".format(q_list[i])
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)/\sqrt{s}$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df_norm.set_index("s", inplace=True)
                    df_norm.to_csv(
                        _data_dir / ("MFDFA_norm_res_o" + str(order) + ".csv")
                    )
                    png_path = _png_dir / ("MFDFA_norm_res_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("MFDFA_norm_res_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

                    fig = plt.figure()
                    plt.clf()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df, id_vars=["s"], var_name="q"),
                    )
                    q0fit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, 0])[large_s:],
                        1,
                    )
                    q0fit_vals = np.polynomial.polynomial.polyval(
                        np.log10(lag), q0fit.coef
                    )
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[0], q0fit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    qnfit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, -1])[large_s:],
                        1,
                    )
                    # qnfit_vals = np.polynomial.polynomial.polyval(
                    #     np.log10(lag), qnfit.coef)
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[-1], qnfit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    for (i, _) in enumerate(q_list):
                        leg_labels[i] = r"h({{{0}}})".format(q_list[i])
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("MFDFA_res_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("MFDFA_res_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("MFDFA_res_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

                    lag, dfa, _ = MFDFA.MFDFA(
                        Xs_raw[target].to_numpy(), lag=org_lag, q=q_list, order=order
                    )
                    norm_dfa = np.zeros_like(dfa)

                    for i in range(dfa.shape[1]):
                        norm_dfa[:, i] = dfa[:, i] / np.sqrt(lag[i])

                    df = pd.DataFrame.from_dict(
                        {str(q_list[i]): dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df["s"] = lag

                    df_norm = pd.DataFrame.from_dict(
                        {str(q_list[i]): norm_dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df_norm["s"] = lag

                    # plot
                    fig = plt.figure()
                    plt.clf()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df_norm, id_vars=["s"], var_name="q"),
                    )
                    q0fit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, 0])[large_s:],
                        1,
                    )
                    q0fit_vals = np.polynomial.polynomial.polyval(
                        np.log10(lag), q0fit.coef
                    )
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[0], q0fit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    qnfit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, -1])[large_s:],
                        1,
                    )
                    # qnfit_vals = np.polynomial.polynomial.polyval(
                    #     np.log10(lag), qnfit.coef)
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[-1], qnfit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    for (i, _) in enumerate(q_list):
                        leg_labels[i] = r"h({{{0}}})".format(q_list[i])
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)/\sqrt{s}$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df_norm.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("MFDFA_norm_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("MFDFA_norm_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("MFDFA_norm_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

                    fig = plt.figure()
                    plt.clf()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df, id_vars=["s"], var_name="q"),
                    )
                    q0fit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, 0])[large_s:],
                        1,
                    )
                    q0fit_vals = np.polynomial.polynomial.polyval(
                        np.log10(lag), q0fit.coef
                    )
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[0], q0fit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    qnfit = np.polynomial.Polynomial.fit(
                        np.log10(lag)[large_s:],
                        np.log10(df.to_numpy()[:, -1])[large_s:],
                        1,
                    )
                    # qnfit_vals = np.polynomial.polynomial.polyval(
                    #     np.log10(lag), qnfit.coef)
                    plt.plot(
                        lag,
                        np.power(10, q0fit_vals),
                        label=r"$h({{{0}}}) = {{{1:.2f}}}$".format(
                            q_list[-1], qnfit.coef[1]
                        ),
                        alpha=0.7,
                        color="k",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    for (i, _) in enumerate(q_list):
                        leg_labels[i] = r"h({{{0}}})".format(q_list[i])
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("MFDFA_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("MFDFA_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("MFDFA_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

            def run_01_DFA():
                """DFA Analysis"""
                print("DFA..")
                _data_dir = data_dir / "01-LRD-DFA"
                _png_dir = png_dir / "01-LRD-DFA"
                _svg_dir = svg_dir / "01-LRD-DFA"
                Path.mkdir(_data_dir, parents=True, exist_ok=True)
                Path.mkdir(_png_dir, parents=True, exist_ok=True)
                Path.mkdir(_svg_dir, parents=True, exist_ok=True)

                # Define unbounded process
                Xs = train_valid_set.ys
                Xs_raw = train_valid_set.ys_raw

                n_lag = 100
                large_s = int(n_lag * 0.3)
                org_lag = np.unique(np.logspace(0.5, 3, n_lag).astype(int))

                # Select a list of powers q
                # if q == 2 -> standard square root based average
                q_list = [2]

                def model_func(x, A, B):
                    return A * np.power(x, B)

                # The order of the polynomial fitting
                for order in [1, 2, 3]:
                    # RESIDUALS
                    lag, dfa, _ = MFDFA.MFDFA(
                        Xs[target].to_numpy(), lag=org_lag, q=q_list, order=order
                    )
                    norm_dfa = np.zeros_like(dfa)

                    for i in range(dfa.shape[1]):
                        norm_dfa[:, i] = np.divide(dfa[:, i], np.sqrt(lag))

                    df = pd.DataFrame.from_dict(
                        {str(q_list[i]): dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df["s"] = lag

                    df_norm = pd.DataFrame.from_dict(
                        {str(q_list[i]): norm_dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df_norm["s"] = lag

                    # plot
                    fig = plt.figure()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df_norm, id_vars=["s"], var_name="q"),
                    )
                    base_lines = np.ones(len(lag)) * 10.0 ** (-2) * np.power(lag, 0.5)
                    plt.plot(
                        lag,
                        base_lines,
                        label=r"$h(2) = 0.5$",
                        alpha=0.7,
                        color="tab:green",
                        linestyle="dashed",
                    )
                    p0 = (1.0, 1.0e-5)
                    popt, _, _, _, _ = sp.optimize.curve_fit(
                        model_func,
                        lag[large_s:],
                        df_norm.to_numpy()[:, -1][large_s:],
                        p0,
                    )
                    coef_annot = popt[1]
                    gamma_annot = 2.0 * (1.0 - popt[1])
                    estimated = model_func(lag, popt[0], popt[1])
                    plt.plot(
                        lag,
                        estimated,
                        label=r"$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$".format(
                            coef_annot, gamma_annot
                        ),
                        alpha=0.7,
                        color="tab:orange",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    leg_labels[0] = r"$h(2)$"
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df_norm.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("DFA_norm_res_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("DFA_norm_res_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("DFA_norm_res_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

                    fig = plt.figure()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df, id_vars=["s"], var_name="q"),
                    )
                    base_lines = np.ones(len(lag)) * 10.0 ** (-2) * np.power(lag, 0.5)
                    plt.plot(
                        lag,
                        base_lines,
                        label=r"$h(2) = 0.5$",
                        alpha=0.7,
                        color="tab:green",
                        linestyle="dashed",
                    )
                    p0 = (1.0, 1.0e-5)
                    popt, _, _, _, _ = sp.optimize.curve_fit(
                        model_func, lag[large_s:], df.to_numpy()[:, -1][large_s:], p0
                    )
                    coef_annot = popt[1]
                    gamma_annot = 2.0 * (1.0 - popt[1])
                    estimated = model_func(lag, popt[0], popt[1])
                    plt.plot(
                        lag,
                        estimated,
                        label=r"$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$".format(
                            coef_annot, gamma_annot
                        ),
                        alpha=0.7,
                        color="tab:orange",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    leg_labels[0] = r"$h(2)$"
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("DFA_res_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("DFA_res_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("DFA_res_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

                    # RAW
                    lag, dfa, _ = MFDFA.MFDFA(
                        Xs_raw[target].to_numpy(), lag=org_lag, q=q_list, order=order
                    )
                    norm_dfa = np.zeros_like(dfa)

                    for i in range(dfa.shape[1]):
                        norm_dfa[:, i] = dfa[:, i] / np.sqrt(lag[i])

                    df = pd.DataFrame.from_dict(
                        {str(q_list[i]): dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df["s"] = lag

                    df_norm = pd.DataFrame.from_dict(
                        {str(q_list[i]): norm_dfa[:, i] for i in range(dfa.shape[1])}
                    )
                    df_norm["s"] = lag

                    # plot
                    fig = plt.figure()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df_norm, id_vars=["s"], var_name="q"),
                    )
                    base_lines = np.ones(len(lag)) * 10.0 ** (-2) * np.power(lag, 0.5)
                    plt.plot(
                        lag,
                        base_lines,
                        label=r"$h(2) = 0.5$",
                        alpha=0.7,
                        color="tab:green",
                        linestyle="dashed",
                    )
                    p0 = (1.0, 1.0e-5)
                    popt, _, _, _, _ = sp.optimize.curve_fit(
                        model_func,
                        lag[large_s:],
                        df_norm.to_numpy()[:, -1][large_s:],
                        p0,
                    )
                    coef_annot = popt[1]
                    gamma_annot = 2.0 * (1.0 - popt[1])
                    estimated = model_func(lag, popt[0], popt[1])
                    plt.plot(
                        lag,
                        estimated,
                        label=r"$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$".format(
                            coef_annot, gamma_annot
                        ),
                        alpha=0.7,
                        color="tab:orange",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    leg_labels[0] = r"$h(2)$"
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)/\sqrt{s}$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df_norm.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("DFA_norm_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("DFA_norm_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("DFA_norm_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

                    fig = plt.figure()
                    sns.color_palette("tab10")
                    sns.lineplot(
                        x="s",
                        y="value",
                        hue="q",
                        data=pd.melt(df, id_vars=["s"], var_name="q"),
                    )
                    base_lines = np.ones(len(lag)) * 10.0 ** (-2) * np.power(lag, 0.5)
                    plt.plot(
                        lag,
                        base_lines,
                        label=r"$h(2) = 0.5$",
                        alpha=0.7,
                        color="tab:green",
                        linestyle="dashed",
                    )
                    p0 = (1.0, 1.0e-5)
                    popt, _ = sp.optimize.curve_fit(
                        model_func, lag[large_s:], df.to_numpy()[:, -1][large_s:], p0
                    )
                    coef_annot = popt[1]
                    gamma_annot = 2.0 * (1.0 - popt[1])
                    estimated = model_func(lag, popt[0], popt[1])
                    plt.plot(
                        lag,
                        estimated,
                        label=r"$h(2) = {{{0:.2f}}}, \gamma = {{{1:.2f}}}$".format(
                            coef_annot, gamma_annot
                        ),
                        alpha=0.7,
                        color="tab:orange",
                        linestyle="dashed",
                    )
                    ax = plt.gca()
                    leg_handles, leg_labels = ax.get_legend_handles_labels()
                    leg_labels[0] = r"$h(2)$"
                    ax.legend(leg_handles, leg_labels)
                    ax.set_xlabel(r"$s$")
                    ax.set_ylabel(r"$F^{(n)}(s)$")
                    ax.set_xscale("log")
                    ax.set_yscale("log")

                    df.set_index("s", inplace=True)
                    df_norm.to_csv(_data_dir / ("DFA_o" + str(order) + ".csv"))
                    png_path = _png_dir / ("DFA_o" + str(order) + ".png")
                    svg_path = _svg_dir / ("DFA_o" + str(order) + ".svg")
                    fig.savefig(png_path, dpi=600)
                    fig.savefig(svg_path)
                    plt.close(fig)

            run_01_DFA()
            run_02_MFDFA()
