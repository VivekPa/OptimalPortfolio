"""
The ``returns`` module provides tools to calcalate the expected returns of the assets. 
This module is intended to be extendable, allowing for users to input their own expected return estimations.

Note that all prices are assumed to be daily closing prices, and the estimated returns are annual returns. 
However, prices and horizons of other periods can be accommodated by changing the ``frequency`` parameter. 

Currently Implemented:

- Nonparametric Estimators
    - Historical Mean Returns
    - Historical Exponentially Weighted Returns
    - CAPM Expected Returns

- Parametric Estimators
    - Normal distribution   
    - Student-t distribution (computed using an expectation maximisation algorithm) 

- User given returns or prices

"""
import pandas as pd 
import numpy as np 
import warnings
from optimalportfolio.utility import exp_max_st


# Nonparametric Esimators

def hist_mean_ret(prices, is_returns=False, frequency=252):
    """
    Calculates historical mean returns

    :param prices: sample data of asset prices
    :type prices: pd.Dataframe
    :param is_returns: whether data provided is returns or price data
    :type is_returns: boolean
    :param frequency: time horizon of projection
    :type frequency: int, optional
    :return: sample exponentially weighted mean dataframe
    :rtype: pd.Series
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not a pd.DataFrame", RuntimeWarning)
        prices = pd.DataFrame(prices)
    
    if is_returns:
        daily_returns = prices
    else:
        daily_returns = prices.pct_change().dropna()

    daily_mean = daily_returns.mean()

    return daily_mean*frequency

def hist_ewma_ret(prices, is_returns=False, span=180, frequency=252):
    """
    Calculates historical exponentially weighted mean returns

    :param prices: sample data of asset prices
    :type prices: pd.Dataframe
    :param is_returns: whether data provided is returns or price data
    :type is_returns: boolean
    :param frequency: time horizon of projection
    :type frequency: int, optional
    :return: historical exponentially weighted mean dataframe
    :rtype: pd.Series
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not a pd.Dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if is_returns:
        daily_returns = prices
    else:
        daily_returns = prices.pct_change().dropna()

    daily_mean = daily_returns.ewm(span=span).mean()

    return daily_mean*frequency

def capm_ret(prices, market_prices, is_returns=False, cov_method='sample', risk_free_rate=0.02, frequency=252):
    """
    Calculates return estimates using Capital Asset Pricing Model (CAPM). 
    In short, CAPM states that asset returns are the sum of market returns 
    and the market excess returns weighted by market beta.

    :param prices: sample data of asset prices
    :type prices: pd.Dataframe
    :param market_prices: sample data of benchmark market prices
    :type market_prices: pd.DataFrame
    :param is_returns: whether data provided is returns or price data
    :type is_returns: boolean
    :param cov_method: method to estimate the covariance matrix. Uses same models as ``risk`` module:

        - ``sample`` 
        - ``exp_cov``
        - ``shrunk_ledoit``
        - ``shrunk_param``
    
    :type cov_method: str, optional
    :param risk_free_rate: current risk free rate of borrowing
    :type risk_free_rate: float, optional
    :param frequency: time horizon of projection
    :type frequency: int, optional 
    :return: historical exponentially weighted mean dataframe
    :rtype: pd.Series
    """

    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices not a pd.Dataframe", RuntimeWarning)
        prices = pd.DataFrame(prices)

    if not isinstance(market_prices, pd.DataFrame):
        warnings.warn("market_prices not a pd.Dataframe", RuntimeWarning)
        market_prices = pd.DataFrame(market_prices)

    if is_returns:
        daily_returns = prices
        daily_market_returns = market_prices
    else:
        daily_returns = prices.pct_change().dropna()
        daily_market_returns = market_prices.pct_change().dropna()

    daily_returns['Market'] = daily_market_returns
    # calculate the covariance of asset+market returns

    if cov_method == 'sample':
        return_cov = daily_returns.cov()
    elif cov_method == 'exp_cov':
        pass
    elif cov_method == 'shrunk_ledoit':
        pass
    elif cov_method == 'shrunk_param':
        pass
    else:
        raise NotImplementedError(f"Covariance estimation method {cov_method} not implemented")
    
    beta = (return_cov['Market']/return_cov.loc['Market', 'Market']).drop('Market')

    mean_market_returns = daily_market_returns.mean()*frequency

    return mean_market_returns + beta*(mean_market_returns - risk_free_rate)


def custom_returns(prices, predicted_prices, is_returns, frequency=252):
    """
    Calculates expected return from user fed predicted prices or returns. 

    :param prices: historical price data
    :type prices: pd.DataFrame
    :param predicted_prices: predicted data of asset prices or returns
    :type predicted_prices: pd.Series
    :param is_returns: whether data provided is returns or price data
    :type is_returns: boolean
    :param frequency: time horizon of projection
    :type frequency: int, optional
    :return: expected returns series
    :rtype: pd.Series
    """

    if is_returns:
        daily_returns = predicted_prices
    else:
        daily_returns = predicted_prices/prices.iloc[-1, :]

    return daily_returns*frequency

