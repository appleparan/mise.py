"""Main interface to other modules
"""
import argparse

# multivariate
from mise.dl.dt_xgboost import dl_xgboost
from mise.dl.mlp_mul_ms import dl_mlp_mul_ms
from mise.dl.mlp_mul_ms_mccr import dl_mlp_mul_ms_mccr
from mise.dl.mlp_mul_transformer import dl_mlp_mul_transformer
from mise.dl.mlp_mul_transformer_mccr import dl_mlp_mul_transformer_mccr

# machine learning models
# univariate
from mise.dl.mlp_uni_ms import dl_mlp_uni_ms
from mise.dl.mlp_uni_ms_mccr import dl_mlp_uni_ms_mccr
from mise.dl.rnn_mul_lstnet_skip import dl_rnn_mul_lstnet_skip
from mise.dl.rnn_mul_lstnet_skip_mccr import dl_rnn_mul_lstnet_skip_mccr
from mise.dl.rnn_uni_attn import dl_rnn_uni_attn
from mise.dl.rnn_uni_attn_mccr import dl_rnn_uni_attn_mccr
from mise.stats.analysis import stats_analysis

# statistical models
from mise.stats.ARIMA import stats_arima
from mise.stats.impute import stats_imputation_stats
from mise.stats.OU import stats_ou
from mise.stats.preprocess import stats_parse, stats_preprocess


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


def compute_dl(_args):
    """
        dl(_args)

    Run deep learning models
    """
    sims = _args["dl"]

    if len(sims) == 0:
        # specify all simulation name
        # sims =
        pass

    print("DL   SIMS: ", sims)

    funcs = ["dl_" + sim for sim in sims]
    for f in funcs:
        globals()[f]()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stats", nargs="*", help="statistics simulations")
    parser.add_argument("-d", "--dl", nargs="*", help="deep learning simulations")

    args = vars(parser.parse_args())

    # statistical models
    if args["stats"] is not None:
        compute_stats(args)

    # machine learning
    if args["dl"] is not None:
        compute_dl(args)
