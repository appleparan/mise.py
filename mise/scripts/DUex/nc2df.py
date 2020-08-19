from pathlib import Path

import pandas as pd
import netCDF4 as nc
import numpy as np
from pytz import timezone

# Seoul Latitude & Longtitude
SEOUL_LAT = 37.5665
SEOUL_LON = 126.9780

SEOULTZ = timezone('Asia/Seoul')

def ncs(lat=SEOUL_LAT, lon=SEOUL_LON):
    p = Path('.')
    # year paths
    yps = [x for x in p.iterdir() if x.is_dir()]
    dict_AOD550 = {}
    dict_AOD10000 = {}
    for yp in yps:
        # year/month paths
        ymps = [x for x in yp.iterdir() if x.is_dir()]
        for ymp in ymps:
            print(ymp)
            files = sorted(ymp.glob('**/*.nc'))
            for fp in files:
                date, AOD550, AOD10000 = nc2df(fp, lat=lat, lon=lon)
                dict_AOD550[date] = AOD550
                dict_AOD10000[date] = AOD10000

    df_AOD550 = pd.DataFrame.from_dict(dict_AOD550, orient='index', columns=['AOD550'])
    df_AOD550.index = pd.to_datetime(df_AOD550.index, format='%Y%m%d')
    df_AOD550 = df_AOD550.tz_localize(SEOULTZ)
    df_AOD550.sort_index(inplace=True)
    df_AOD10000 = pd.DataFrame.from_dict(dict_AOD10000, orient='index', columns=['AOD10000'])
    df_AOD10000.index = pd.to_datetime(df_AOD10000.index, format='%Y%m%d')
    df_AOD10000 = df_AOD10000.tz_localize(SEOULTZ)
    df_AOD10000.sort_index(inplace=True)

    # all values   : 4332
    # nan values   : 3067
    # no-na values : 1265
    df_AOD550.index.rename('date', inplace=True)
    df_AOD10000.index.rename('date', inplace=True)

    df = df_AOD550.merge(df_AOD10000, how='inner', on='date', sort=True)
    print(df.head(5))

def nc2df(fp, lat=SEOUL_LAT, lon=SEOUL_LON):
    """read daily 'fn' netCDF4 files and filter by lat/lon then convert to DataFrames
    """
    date = str(fp).split('/')[2].split('-')[0]
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    nc_fid = nc.Dataset(fp, 'r')

    lats = nc_fid.variables['latitude'][:]
    lons = nc_fid.variables['longitude'][:]

    abs_lat_diff = lambda list_value : abs(list_value - lat)
    abs_lon_diff = lambda list_value : abs(list_value - lon)

    clat = min(lats, key=abs_lat_diff)
    clon = min(lons, key=abs_lon_diff)

    lat_idx = np.where(lats == clat)[0][0]
    lon_idx = np.where(lons == clon)[0][0]

    AOD550 = nc_fid.variables['D_AOD550'][lat_idx, lon_idx].__float__()
    AOD10000 = nc_fid.variables['D_AOD10000'][lat_idx, lon_idx].__float__()

    nc_fid.close()
    return date, AOD550, AOD10000

if __name__ == '__main__':
    # simple script to convert netCDF4 dataset to DataFrame (select AOD500, AOD100000)
    # nc files
    ncs()
