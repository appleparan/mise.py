import pandas as pd

def load(filepath="/input/input.csv"):
    df = pd.read_csv(filepath,
        index_col = [0, 1],
        parse_dates = [1])

    # prints
    pd.set_option('display.max_rows', 10)
    pd.reset_option('display.max_rows')
    print(df.head(10))

    return df

def load_station(df, code=111123):
    #return df[df['stationCode'] == code]
    return df.query('stationCode == "' + str(code) + '"')
