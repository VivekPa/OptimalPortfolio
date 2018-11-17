"""
The ``utility_functions`` module provides common utility functions.
These methods are primarily designed for internal use during optimisation (via
scipy.optimize), and each requires a certain signature (which is why they have not been
factored into a class). The utility function must accept ``weights``
as an argument, and must also have at least one of ``mean`` or ``cov``.
Because scipy.optimize only minimises, any objectives that we want to maximise must be
made negative.
Currently implemented:
- negative mean return
- (regularised) negative Sharpe ratio
- (regularised) volatility
- empirically distributed CVaR (expected shortfall)
- mean, covariance, skew and kurtosis utility function
"""

import numpy as np
import scipy.stats


def mean_return(weights, expected_returns):
    """
    Calculate the negative mean return of a portfolio
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param expected_returns: expected return of each asset
    :type expected_returns: pd.Series
    :return: negative mean return
    :rtype: float
    """
    return -weights.dot(expected_returns)


def sharpe(
    weights, expected_returns, cov_matrix, gamma=0, risk_free_rate=0.02
):
    """
    Calculate the negative Sharpe ratio of a portfolio
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param expected_returns: expected return of each asset
    :type expected_returns: pd.Series
    :param cov_matrix: the covariance matrix of asset returns
    :type cov_matrix: pd.DataFrame
    :param gamma: L2 regularisation parameter, defaults to 0. Increase if you want more
                    non-negligible weights
    :type gamma: float, optional
    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
    :type risk_free_rate: float, optional
    :return: negative Sharpe ratio
    :rtype: float
    """
    mu = weights.dot(expected_returns)
    sigma = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights.T)))
    L2_reg = gamma * (weights ** 2).sum()
    return -(mu - risk_free_rate) / sigma + L2_reg


def volatility(weights, cov_matrix, gamma=0):
    """
    Calculate the volatility of a portfolio.
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param cov_matrix: the covariance matrix of asset returns
    :type cov_matrix: pd.DataFrame
    :param gamma: L2 regularisation parameter, defaults to 0. Increase if you want more
                  non-negligible weights
    :type gamma: float, optional
    :return: portfolio variance
    :rtype: float
    """
    L2_reg = gamma * (weights ** 2).sum()
    portfolio_volatility = np.dot(weights.T, np.dot(cov_matrix, weights))
    return portfolio_volatility + L2_reg


def moment_utility(weights, mean, cov, skew, kurt, delta1, delta2, delta3, delta4, gamma):
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
    :param gamma: coefficient of L2 regularisation
    :return: portfolio utility
    """
    L2_reg = gamma * (weights ** 2).sum()
    utility = delta1 * (np.dot(np.transpose(weights), mean)) - \
              delta2 * (np.dot(np.dot(np.transpose(weights), cov), weights)) + \
              delta3 * (np.dot(np.dot(np.transpose(weights), skew), weights)) - \
              delta4 * (np.dot(np.dot(np.transpose(weights), kurt), weights)) + L2_reg
    return -utility


def empirical_cvar(weights, returns, s=10000, beta=0.95, random_state=None):
    """
    Calculate the negative CVaR.
    :param weights: asset weights of the portfolio
    :type weights: np.ndarray
    :param returns: asset returns
    :type returns: pd.DataFrame or np.ndarray
    :param s: number of bootstrap draws, defaults to 10000
    :type s: int, optional
    :param beta: "significance level" (i. 1 - q), defaults to 0.95
    :type beta: float, optional
    :param random_state: seed for random sampling, defaults to None
    :type random_state: int, optional
    :return: negative CVaR
    :rtype: float
    """
    np.random.seed(seed=random_state)
    # Calcualte the returns given the weights
    portfolio_returns = (weights * returns).sum(axis=1)
    # Sample from the historical distribution
    dist = scipy.stats.gaussian_kde(portfolio_returns)
    sample = dist.resample(s)
    # Calculate the value at risk
    var = portfolio_returns.quantile(1 - beta)
    # Mean of all losses worse than the value at risk
    return -sample[sample < var].mean()
