import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
from scipy.optimize import minimize
import warnings
from . import utility_functions as utility_functions
import cvxpy as cp


class BaseEstimator:
    def __init__(self, tickers) -> None:
        self.tickers = tickers

    def _get_logreturns(self, prices, period=1) -> pd.DataFrame:
        return np.log(prices.shift(-1)/prices).dropna()

    def _pairwise_exp_cov(self, X, Y, span=180) -> pd.DataFrame:
        pair_cov = (X - X.mean()) * (Y - Y.mean())

        return pair_cov.ewm(span=span).mean().iloc[-1]

    def _cov_to_corr(self, cov):
        Dinv = np.diag(1 / np.sqrt(np.diag(cov))) 
        corr = Dinv @ cov @ Dinv

        return corr

    def _corr_to_cov(self, corr, stds):
        return corr * np.outer(stds, stds)
