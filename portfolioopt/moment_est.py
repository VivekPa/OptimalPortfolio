"""
The ``moment_est`` module estimates the moments of the distribution of market invariants using a number of methods.

Currently implemented:
- Exponentially weighted mean and covariance
- Nonparametric estimators of mean, covariance and higher moments
    - Gaussian kernel sample mean and covariance
- Maximum Likelihood estimators of mean, covariance and higher moments
    - Normal distribution
    - Student-t distribution
- Shrinkage estimators of mean, covariance and higher moments
    - Ledoit Wolf shrinkage
    - Oracle Approximating shrinkage
    - Exponentially weighted shrinkage
    - Novel nonparametric + MLE shrinkage
"""

import pandas as pd
import numpy as np
from sklearn import covariance
from sklearn.covariance.shrunk_covariance_ import ledoit_wolf_shrinkage
from portfolioopt.functions import *
from scipy.stats import moment
import warnings


def sample_mean(invariants, frequency=252):
    """
    Calculates sample mean
    :param invariants: sample data of market invariants
    :type invariants: pd.Dataframe
    :param frequency: time horizon of projection, default set to 252 days
    :type frequency: int
    :return: sample mean dataframe
    """
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    daily_mean = invariants.mean()
    return daily_mean*np.sqrt(frequency)


def sample_cov(invariants, frequency=252):
    """
    Calculates sample covariance
    :param invariants: sample data of market invariants
    :type invariants: pd.Dataframe
    :param frequency: time horizon of projection, default set to 252 days
    :type frequency: int
    :return: sample covariance dataframe
    """
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    daily_cov = invariants.cov()
    return daily_cov*frequency


def sample_skew(invariants, frequency=252):
    """
    Calculates sample skew
    :param invariants: sample data of market invariants
    :type invariants: pd.Dataframe
    :param frequency: time horizon of projection, default set ot 252 days
    :type frequency: int
    :return: sample skew dataframe
    """
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    daily_skew = moment(invariants, moment=3)
    return daily_skew*(frequency**1.5)


def sample_kurt(invariants, frequency=252):
    """
    Calculates sample kurtosis
    :param invariants: sample data of market invariants
    :type invariants: pd.Dataframe
    :param frequency: time horizon of projection, default set to 252 days
    :type frequency: int
    :return: sample kurtosis dataframe
    """
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    daily_kurt = moment(invariants, moment=4)
    return daily_kurt*(frequency**2)


def sample_moment(invariants, order, frequency=252):
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    daily_moment = moment(invariants, moment=order)
    return daily_moment*frequency


def exp_mean(invariants, span=180, frequency=252):
    """
    Calculates sample exponentially weighted mean
    :param invariants: sample data of market invariants
    :type invariants: pd.Dataframe
    :param frequency: time horizon of projection
    :type frequency: int
    :return: sample exponentially weighted mean dataframe
    """
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    daily_mean = invariants.ewm(span=span).mean()
    return daily_mean*np.sqrt(frequency)


def exp_cov(invariants, span=180, frequency=252):
    """
    Calculates sample exponentially weighted covariance
    :param invariants: sample data of market invariants
    :type invariants: pd.Dataframe
    :param frequency: time horizon of projection
    :type frequency: int
    :param span: the span for exponential weights
    :return: sample exponentially weighted covariance dataframe
    """
    if not isinstance(invariants, pd.DataFrame):
        warnings.warn("invariants not a pd.Dataframe", RuntimeWarning)
        invariants = pd.DataFrame(invariants)
    assets = invariants.columns
    daily_cov = invariants.ewm(span=span).cov().iloc[-len(assets):, -len(assets):]
    return pd.DataFrame(daily_cov*frequency)


class MLE:
    """
    Provide methods to calculate maximum likelihood estimators (MLE) of mean, covariance and higher moments. Currently
    implemented distributions:
    - Normal

    Instance variables:

    - ``invariants`` (market invariants data)
    - ``dist`` (distribution choice)
    - ``n`` (number of assets
    - ``mean`` (estimate of mean, initially None)
    - ``cov`` (estimate of covariance, initially None)
    - ``skew`` (estimate of skew, initially None)
    - ``kurt`` (estimate of kurtosis, initially None)

    Public methods:

    - ``norm_est`` (calculates the normally distributed maximum likelihood estimate of mean, covariance, skew and
    kurtosis)
    """
    def __init__(self, invariants, n, dist="normal"):
        self.invariants = invariants
        self.dist = dist
        self.n = n
        self.mean = None
        self.cov = None
        self.skew = None
        self.kurt = None

    def norm_est(self, invariants):
        if self.dist == "normal":
            self.mean = 1/self.n * np.sum(invariants)
            self.cov = 1/self.n * np.dot((invariants - self.mean), np.transpose(invariants - self.mean))
            self.skew = 0
            self.kurt = 0
        return self.mean, self.cov, self.skew, self.kurt


class Shrinkage:
    """
    Provide methods to calculate shrinkage estimators for mean, covariance. Includes novel shrinkage estimator using
    nonparametric and maximum likelihood estimators.

    Instance variables:
    - ``invariants`` (market invariants data)
    - ``n`` (number of assets)
    - ``frequency`` (investment time horizon, default set to 252 days)

    Public methods:
    - ``shrunk_covariance`` (calculates manually shrunk covariance matrix)
    - ``ledoit_wolf`` (calculates optimal shrinkage using Ledoit-Wolf method)
    - ``oracle_approximate`` (calculates optimal shrinkage using Oracle approximation)
    - ``exp_ledoit`` (calculates optimal shrinkage of exponentially weighted covariance using Ledoit-Wolf method)
    - ``param_mle`` (calculates manually shrunk covariance using nonparametric and maximum likelihood estimate of
    covariance matrix)
    """
    def __init__(self, invariants, n, frequency=252):
        if not isinstance(invariants, pd.DataFrame):
            warnings.warn("invariants is not pd.Dataframe", RuntimeWarning)
        self.invariants = invariants
        self.S = self.invariants.cov()
        self.frequency = frequency
        self.n = n

    def _format_cov(self, raw_cov):
        """
        Helper method which annualises the output of shrinkage covariance,
        and formats the result into a dataframe.
        :param raw_cov: raw covariance matrix of daily returns
        :type raw_cov: np.ndarray
        :return: annualised covariance matrix
        :rtype: pd.DataFrame
        """
        assets = self.invariants.columns
        return pd.DataFrame(raw_cov, index=assets, columns=assets) * self.frequency

    def shrunk_covariance(self, delta=0.2):
        """
        Shrink a sample covariance matrix to the identity matrix (scaled by the average
        sample variance). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.
        :param delta: shrinkage parameter, defaults to 0.2.
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: pd.Dataframe
        """
        self.delta = delta
        N = self.S.shape[1]
        # Shrinkage target
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_cov(shrunk_cov)

    def ledoit_wolf(self):
        """
        Calculates the Ledoit-Wolf shrinkage estimate.
        :return: shrunk sample covariance matrix
        :rtype: pd.Dataframe
        """
        X = np.nan_to_num(self.invariants.values)
        shrunk_cov, self.delta = covariance.ledoit_wolf(X)
        return self._format_cov(shrunk_cov)

    def oracle_approximation(self):
        """
        Calculates the Oracle Approximating Shrinkage estimate
        :return: shrunk sample covariance matrix
        :rtype: pd.Dataframe
        """
        X = np.nan_to_num(self.invariants.values)
        shrunk_cov, self.delta = covariance.oas(X)
        return self._format_cov(shrunk_cov)

    def exp_ledoit(self, X, block_size=1000):
        """
        Calculates the shrinkage of exponentially weighted covariance using Ledoit-Wolf shrinkage estimate.
        :param X: market invariants data
        :type X: pd.Dataframe
        :param block_size: block size for Ledoit-Wolf calculation
        :type block_size: int
        :return: shrunk covariance matrix
        """
        cov = exp_cov(X)
        shrinkage = ledoit_wolf_shrinkage(cov, block_size=block_size)
        shrunk_cov = (1 - shrinkage) * cov + shrinkage * (np.trace(cov)/self.n) * np.identity(self.n)
        return shrunk_cov

    def param_mle(self, X, n, shrinkage):
        """
        Calculates the shrinkage estimate of nonparametric and maximum likelihood estimate of covariance matrix
        :param X: market invariants data
        :type X: pd.Dataframe
        :param n: number of assets
        :type: n: int
        :param shrinkage: shrinkage coefficient
        :type shrinkage: int
        :return: shrunk covariance matrix
        """
        mle = MLE(X, n, dist="normal")
        mean, cov, skew, kurt = mle.norm_est(X)
        param_cov = exp_cov(X)
        shrunk_cov = (1 - shrinkage) * cov + shrinkage * param_cov
        return shrunk_cov
