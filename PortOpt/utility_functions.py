"""
The ``utility_functions`` module provides common utility functions, mainly used interally.

Currently implemented:

- Data Fetcher from yfinance
- Mean, Covariance, Skew and Kurtosis utility function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

def get_data(tickers, interval, start, end):
    """ 
    Fetch stock data from yfinance, for particular interval and start/end dates.
    :param tickers: list of tickers
    :type tickers: list or pd.Series or pd.Index
    :param interval: interval of data
    :type interval: str
    :param start: starting date
    :type start: str "%Y%m%d"
    :param end: ending date
    :type end: str "%Y%m%d"
    :return: dataset of stock data
    :rtype: pd.DataFrame
    """
    full_prices = pd.DataFrame()
    for i in tickers:
        df = yf.download(i, interval=interval, start=start, end=end)['Adj Close']
        print(f'{i} data downloaded...')

        full_prices = pd.concat([full_prices, df], axis=1, sort=False)

    full_prices.columns = tickers
      
    return full_prices

def mean(weights, mean):
    """
    Calculate the negative mean return of a portfolio
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param expected_returns: expected return of each asset
    :type expected_returns: pd.Series
    :return: negative mean return
    :rtype: float
    """
    return -weights.dot(mean)


def sharpe(weights, mean, cov, risk_free_rate=0.02):
    """
    Calculate the negative Sharpe ratio of a portfolio
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param mean: mean of market invariants
    :param cov: the covariance matrix of invariants
    :param risk_free_rate: risk-free rate of return, defaults to 0.02
    :type risk_free_rate: float, optional
    :return: negative Sharpe ratio
    :rtype: float
    """
    mu = weights.dot(mean)
    sigma = np.sqrt(np.dot(weights, np.dot(cov, weights.T)))
    return -(mu - risk_free_rate) / sigma


def volatility(weights, cov):
    """
    Calculate the volatility of a portfolio.
    :param weights: asset weights of the portfolio
    :param cov: the covariance of invariants
    :return: portfolio variance
    :rtype: float
    """
    portfolio_volatility = np.dot(weights.T, np.dot(cov, weights))
    return portfolio_volatility


def moment_utility(weights, mean, cov, skew, kurt, delta1, delta2, delta3, delta4):
    """
    Calculates the utility using mean, covariance, skew and kurtosis of data.
    :param weights: portfolio weights
    :param mean: mean of market invariants
    :param cov: covariance of market invariants
    :param skew: skew of market invariants
    :param kurt: kurtosis of market invariants
    :param delta1: coefficient of mean
    :param delta2: coefficient of covariance
    :param delta3: coefficient of skew
    :param delta4: coefficient of kurtosis
    :return: portfolio utility
    """
    utility = delta1 * (np.dot(np.transpose(weights), mean)) - \
              delta2 * (np.dot(np.dot(np.transpose(weights), cov), weights)) + \
              delta3 * (np.dot(np.dot(np.transpose(weights), skew), weights)) - \
              delta4 * (np.dot(np.dot(np.transpose(weights), kurt), weights))
    return -utility



