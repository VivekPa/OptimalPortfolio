"""
The ``utility_functions`` module provides common utility functions, mainly used interally.
Those wanting to extend must take note of the requirements of scipy.optimize to
not run into errors.

Currently implemented:

- mean, covariance, skew and kurtosis utility function
"""

import numpy as np


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


def comoment_utility(weights, mean, cov, coskew, cokurt, delta1, delta2, delta3, delta4):
    """
    Calculates the utility using mean, covariance, coskewness and cokurtosis of data.
    :param weights: portfolio weights
    :param mean: mean of market invariants
    :param cov: covariance matrix of market invariants
    :param skew: coskewness matrix of market invariants
    :param kurt: cokurtosis matrix of market invariants
    :param delta1: relative weight in optimization for mean
    :param delta2: relative weight in optimization for covariance
    :param delta3: relative weight in optimization for skew
    :param delta4: relative weight in optimization for kurtosis
    :return: portfolio utility
    """
    utility = delta1 * (np.dot(np.transpose(weights), mean)) - \
              delta2 * (np.dot(np.dot(np.transpose(weights), cov), weights)) + \
              delta3 * (np.dot(np.dot(np.transpose(x0), coskew), np.kron(x0,x0)))[0,0] - \
              delta4 * (np.dot(np.dot(np.transpose(x0), cokurt), np.kron(np.kron(x0,x0),x0)))[0,0]
    return -utility



