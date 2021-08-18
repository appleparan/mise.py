"""Main interface to other modules
"""
import argparse

# multivariate
from mise.ml.dt_xgboost import ml_xgboost
from mise.ml.mlp_mul_ms import ml_mlp_mul_ms
from mise.ml.mlp_mul_ms_mccr import ml_mlp_mul_ms_mccr
from mise.ml.mlp_mul_transformer import ml_mlp_mul_transformer
from mise.ml.mlp_mul_transformer_mccr import ml_mlp_mul_transformer_mccr

# machine learning models
# univariate
from mise.ml.mlp_uni_ms import ml_mlp_uni_ms
from mise.ml.mlp_uni_ms_mccr import ml_mlp_uni_ms_mccr
from mise.ml.rnn_mul_lstnet_skip import ml_rnn_mul_lstnet_skip
from mise.ml.rnn_mul_lstnet_skip_mccr import ml_rnn_mul_lstnet_skip_mccr
from mise.ml.rnn_uni_attn import ml_rnn_uni_attn
from mise.ml.rnn_uni_attn_mccr import ml_rnn_uni_attn_mccr
from mise.stats.analysis import stats_analysis

# statistical models
from mise.stats.ARIMA import stats_arima
from mise.stats.impute import stats_imputation_stats
from mise.stats.OU import stats_ou
from mise.stats.preprocess import stats_parse, stats_preprocess


def compute_plot(_args):
    """
        plot(_args)

    Plot figures from args
    """
    figs = _args["plot"]

    if len(figs) == 0:
        # specify all simulation name
        max_figure = 2
        figs = list(map(str, list(range(1, max_figure))))

    print("PLOT FIGS: ", figs)

    funcs = ["plots" + fig for fig in figs]
    for f in funcs:
        globals()[f]()


def compute_stats(_args):
    """
        stats(_args)

    Run statistical models
    """
    sims = _args["stats"]

    if len(sims) == 0:
        # specify all simulation name
        # sims =
        pass

    print("STAT SIMS: ", sims)

    funcs = ["stats_" + sim for sim in sims]
    for f in funcs:
        globals()[f]()


def compute_ml(_args):
    """
        ml(_args)

    Run machine learning models
    """
    sims = _args["ml"]

    if len(sims) == 0:
        # specify all simulation name
        # sims =
        pass

    print("ML   SIMS: ", sims)

    funcs = ["ml_" + sim for sim in sims]
    for f in funcs:
        globals()[f]()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--plot", nargs="*", help="plot figures, must used with --figure"
    )
    parser.add_argument("-s", "--stats", nargs="*", help="statistics simulations")
    parser.add_argument("-m", "--ml", nargs="*", help="machine learning simulations")

    args = vars(parser.parse_args())

    # statistical models
    if args["stats"] is not None:
        compute_stats(args)

    # machine learning
    if args["ml"] is not None:
        compute_ml(args)

    # plot
    if args["plot"] is not None:
        compute_plot(args)
