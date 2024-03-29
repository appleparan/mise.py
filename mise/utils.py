import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

from mise.constants import SEOULTZ


def parse_key(df, period):
    """parse DateTimeIndex to key by period

    Args:
        df (DataFrame or Series):
            pandas object that have DateTimeIndex

        period (str):
            'y', 'w', or 'd'

    Returns:
        List(str): list of parsed string
    """
    if period == "y":
        # %m%d%H
        return [
            "".join(row)
            for row in zip(
                np.char.zfill(df.index.month.to_numpy().astype(str), 2),
                np.char.zfill(df.index.day.to_numpy().astype(str), 2),
            )
        ]

    if period == "w":
        # %w, no padding
        return df.index.weekday.to_numpy().astype(str)

    if period == "h":
        # %H, zero-padded 24-hour clock
        return np.char.zfill(df.index.hour.to_numpy().astype(str), 2)
    else:
        raise KeyError("KeyError: wrong period! period should be in 'y', 'w', or 'h'")


def parse_ykey(idx: pd.DatetimeIndex):
    """parse index and convert to str

    Args:
        idx (pd.DatetimeIndex): datetime index

    Returns:
        str: zero padded month and day(%m%d)
    """
    return "".join((str(idx.month).zfill(2), str(idx.day).zfill(2)))


def parse_wkey(idx: pd.DatetimeIndex):
    """parse index and convert to str

    Args:
        idx (pd.DatetimeIndex): datetime index

    Returns:
        str: weekday, (Monday=0, Sunday=6)
    """
    return str(idx.weekday())


def parse_hkey(idx):
    """parse index and convert to str

    Args:
        idx (pd.DatetimeIndex): datetime index

    Returns:
        str: zero padded 24-clock hour(%H)
    """
    return str(idx.hour).zfill(2)


def periodic_mean(
    df, target, period, smoothing=False, smoothing_frac=0.05, smoothing_col="y"
):
    """compute periodic mean and residuals, annual and weekly means
        are daily average, hourly means are hourly raw data

    Args:
        df (pd.DataFrame or pd.Series):
            If DataFrame, original data.
            If Series, this means residuals
            DataFrame and Series both must have Single DatetimeIndex

        target (str): if target is not None, indicates column name in DataFrame.
            if None, this means DataFrame is from residuals with no column name

        period (str):
            Indicates periods, only paossible values, 'y', 'w', or 'd'

        smoothing  (bool, optional):
            Only works if period is "y", smooth seasonality with LOWESS.
            Defaults to False.

        smoothing_frac (float, optional):
            Between 0 and 1.
            The fraction of the data used
            when estimating each y-value.
            frac in `statsmodels.nonparametric.smoothers_lowess.lowess`
            Defaults to 0.05.

        smoothing_col (str, optional): set column to smooth. Defaults to 'y'.

    Raises:
        KeyError: if period is not 'y', 'w', or 'h'

    Returns:
        dict_sea (dict):
            Assign seasonality directly to SeasonalityDecompositor

        sea (pd.Series):
            Seasonality

        res (pd.Series):
            Residuals
    """
    # define function to convert datetime to key
    if period == "y":
        # %m%d
        # SettingWithCopyWarning:
        # A value is trying to be set on a copy of a slice from a DataFrame.
        # from pandas 1.1.4 but no issue, why?
        df.loc[:, "key"] = df.index.map(parse_ykey)

        def dt2key(d): return str(d.astimezone(SEOULTZ).month).zfill(2) + str(d.day).zfill(
            2
        )
    elif period == "w":
        # %w, no padding
        df.loc[:, "key"] = df.index.map(parse_wkey)
        # weekday() is a function, not property
        def dt2key(d): return str(d.astimezone(SEOULTZ).weekday())
    elif period == "h":
        # %H, zero-padded 24-hour clock
        df.loc[:, "key"] = df.index.map(parse_hkey)
        def dt2key(d): return str(d.astimezone(SEOULTZ).hour).zfill(2)
    else:
        # already raised in parse_key
        raise KeyError("Wrong period! period should be in 'y', 'w', and 'h'")

    # if average dictionary not defined, create new one
    # function periodic_mean is always executed in train/valid set (fit method)
    # if test set, dict_sea was given and will not execute this function
    # test set will use mean of train/valid set which is fed on __init__
    grp_sea = df.groupby(by="key").mean().loc[:, target]
    dict_sea = grp_sea.to_dict()

    def get_sea(key): return dict_sea[dt2key(key.name)]

    # convert dictionary to DataFrame
    sea = pd.DataFrame.from_dict(dict_sea, orient="index", columns=["sea"])
    # axis=1 in apply menas apply function `get_sea` to columns
    res = df.loc[:, target] - df.loc[:, target].to_frame().apply(get_sea, axis=1)

    # smoothing applies only to annual seasaonlity
    if smoothing and period == smoothing_col:
        sea_values = sea.loc[:, "sea"].to_numpy()
        sea_smoothed = lowess(
            sea_values, range(len(sea_values)), return_sorted=False, frac=smoothing_frac
        )
        dict_sea = dict(zip(sea.index, sea_smoothed))
        # redefine get function due to closure
        def get_sea(key): return dict_sea[dt2key(key.name)]

        sea = pd.DataFrame.from_dict(dict_sea, orient="index", columns=["sea"])
        res = df[target] - df[target].to_frame().apply(get_sea, axis=1)

    # convert residual from Series to DataFrame
    res = res.to_frame()
    res.columns = ["resid"]
    # res['key'] = parse_key(res, period)

    return dict_sea, sea, res
