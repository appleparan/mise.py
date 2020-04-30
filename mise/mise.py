import argparse

from stats.arima import stats_arima

"""
    plot(args)

Plot figures from args
"""
def plot(args):
    figs = args["plot"]

    if len(figs) == 0:
        # specify all simulation name
        max_figure = 2
        figs = list(map(str, list(range(1, max_figure))))

    print("PLOT FIGS: ", figs)

    funcs = ["plots" + fig for fig in figs]
    for f in funcs:
        globals()[f]()

    pass

"""
    stats(args)

Run statistical models
"""
def stats(args):
    sims = args['stats']

    if len(sims) == 0:
        # specify all simulation name
        # sims =
        pass

    print("STAT SIMS: ", sims)

    funcs = ["stats_" + sim for sim in sims]
    for f in funcs:
        globals()[f]()

    pass

"""
    ml(args)

Run machine learning models
"""
def ml(args):
    sims = args["ml"]

    if len(sims) == 0:
        # specify all simulation name
        # sims =
        pass

    print("ML   SIMS: ", sims)

    funcs = ["ml_" + sim for sim in sims]
    for f in funcs:
        globals()[f]()

    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", nargs='*', help="plot figures, must used with --figure")
    parser.add_argument("-s", "--stats", nargs='*', help="statistics simulations")
    parser.add_argument("-m", "--ml", nargs='*', help="machine learning simulations")

    args = vars(parser.parse_args())

    # statistical models
    if args["stats"] != None:
        stats(args)

    # machine learning
    if args["ml"] != None:
        ml(args)

    # plot
    if args["plot"] != None:
        plot(args)
