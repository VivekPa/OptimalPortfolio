"""
The ``risk`` module provides tools to calcalate the covariance of the assets. 

Note that all prices are assumed to be daily closing prices, and the estimated returns are annual returns. 
However, prices and horizons of other periods can be accommodated by changing the ``frequency`` parameter. 

Currently Implemented:

- Nonparametric Estimators
    - Historical Covariance
    - Historical Exponentially Weighted Covariance

- Parametric Estimators
    - Normal distribution   
    - Student-t distribution (computed using an expectation maximisation algorithm) 

- Shrinkage Estimators
    - Ledoit Wolf shrinkage
    - Exponentially weighted + parametric shrinkage

- Factor Models
    - Fama-French 3 factor covariance

"""

import pandas as pd 
import numpy as np 
from sklearn.covariance import LedoitWolf
from sklearn.covariance import ledoit_wolf
import warnings

# Nonparametric estimators

def sample_cov(prices, is_returns=False, frequency=252):
    """
    Calculates historical covariance of returns

    :param prices: sample data of asset prices
    :type prices: pd.Dataframe
    :param is_returns: whether data provided is returns or price data
    :type is_returns: boolean
    :param frequency: time horizon of projection
    :type frequency: int, optional
    :return: historical covariance matrix
    :rtype: pd.DataFrame
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not a pd.DataFrame", RuntimeWarning)
        prices = pd.DataFrame(prices)
    
    if is_returns:
        daily_returns = prices
    else:
        daily_returns = prices.pct_change().dropna()
    
    daily_cov = daily_returns.cov()

    return daily_cov*frequency

def _exp_cov_helper(X1, X2, span=180):
    simple_cov = (X1-X1.mean())*(X2-X2.mean())

    return simple_cov.ewm(span=180).mean().iloc[-1]

def exp_cov(prices, is_returns=False, span=180, frequency=252):
    """
    Calculates historical exponentially weighted covariance of returns

    :param prices: sample data of asset prices
    :type prices: pd.Dataframe
    :param is_returns: whether data provided is returns or price data
    :type is_returns: boolean
    :param frequency: time horizon of projection
    :type frequency: int, optional
    :return: historical exponentially weighted covariance matrix
    :rtype: pd.DataFrame
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not a pd.DataFrame", RuntimeWarning)
        prices = pd.DataFrame(prices)
    
    if is_returns:
        daily_returns = prices
    else:
        daily_returns = prices.pct_change().dropna()
    
    N = len(prices.columns)
    C = np.zeros([N, N])

    for i in range(N):
        for j in range(i, N):
            C[i, j] = C[j, i] = _exp_cov_helper(daily_returns.iloc[:, i], daily_returns.iloc[:, j], span=span)
    exponential_cov = pd.DataFrame(C, index=prices.columns, columns=prices.columns)

    return exponential_cov*frequency

# Shrinkage Estimators

class Shrinkage:
    """
    Provide methods to calculate shrinkage estimators for mean, covariance. Includes novel shrinkage estimator using
    nonparametric and maximum likelihood estimators.

    Instance variables:

    - ``prices`` (asset price data)
    - ``is_returns`` (whether dataframe is returns or prices)
    - ``frequency`` (investment time horizon, default set to 252 days)

    Public methods:

    - ``ledoit_wolf`` (calculates shrunk covariance using sample covariance and identity matrix)
    - ``exp_ledoit`` (calculates shrunk covariance using exponentially weighted covariance matrix and identity matrix)
    - ``param_mle`` (calculates manually shrunk covariance using nonparametric and maximum likelihood estimate of covariance matrix)

    """
    def __init__(self, prices, is_returns=False, frequency=252):
        if not isinstance(prices, pd.DataFrame):
            warnings.warn("prices is not a pd.DataFrame", RuntimeWarning)
        
        if is_returns:
            self.returns = prices
        else:
            self.returns = prices.pct_change().dropna()

        self.N = len(prices.columns)
        self.frequency = frequency
        self.is_returns = is_returns

    def _format_cov(self, raw_cov):
        assets = self.returns
        return pd.DataFrame(raw_cov, index=assets, columns=assets)*self.frequency

    def ledoit_wolf(self, block_size=1000):
        """
        Calculates the shrinkage of sample covariance using Ledoit-Wolf shrinkage estimate.

        :param block_size: block size for Ledoit-Wolf calculation
        :type block_size: int
        :return: shrunk covariance matrix
        :rtype: pd.DataFrame
        """

        cov = LedoitWolf().fit(self.returns).covariance_

        return self._format_cov(cov)

    def exp_ledoit(self, block_size=1000):
        """
        Calculates the shrinkage of exponentially weighted covariance using Ledoit-Wolf shrinkage estimate.

        :param block_size: block size for Ledoit-Wolf calculation
        :type block_size: int
        :return: shrunk covariance matrix
        :rtype: pd.DataFrame
        """

        ecov = exp_cov(self.returns, self.is_returns)
        shrunk_cov = ledoit_wolf(ecov, assume_centered=False, block_size=block_size)

        return self._format_cov(shrunk_cov)

    def param_mle(self, shrinkage=None, block_size=1000):
        pass

