"""
This ``eff_frontier`` module calculates the optimal portfolio weights given the mean, covariance, skew and kurtosis of
the data, for various utility functions. Currently implemented:

- Moment Optimisation
- Maximum Sharpe Ratio

"""

import numpy as np
import pandas as pd
import scipy.optimize as scop
import warnings
import portfolioopt.utility_functions as utility_functions


class OptimalAllocations:
    """
    This ``OptimalAllocations`` class optimises portfolio weights for given utility function.

    Instance variables:

    - Inputs:

        - ``n`` (number of assets)
        - ``mean`` (mean of market invariants)
        - ``cov`` (covariance of market invariants)
        - ``weight_bounds`` (the bounds of portfolio weights)
        - ``tickers`` (list of tickers of assets)

    - Optimisation parameters:

        - ``initial_guess`` (initial guess for portfolio weights, set to evenly distributed)
        - ``constraints`` (constraints for optimisation)

    - Outputs:

        - ``weights`` (portfolio weights, initially set to None)


    Public methods:

        - ``moment_optimisation`` (calculates portfolio weights that maximises utility function of higher moments)
        - ``max_sharpe()`` (calculates portfolio weights that maximises Sharpe Ratio)
        - ``portfolio_performance()`` (calculates portfolio performance and optionally prints it)

    """
    def __init__(self, n, mean, cov, tickers, gamma, weight_bounds=(0, 1)):
        """

        :param n: number of assets
        :type n: int
        :param mean: mean estimate of market invariants
        :type mean: pd.Dataframe
        :param cov: covariance estimate of market invariants
        :type cov: pd.Dataframe
        :param tickers: tickers of securities used
        :type tickers: list
        :param gamma: L2 regularisation coefficient
        :type gamma: float
        :param weight_bounds: bounds for portfolio weights. Change to (-1,1) for shorting
        :type weight_bounds: tuple
        """
        self.n = n
        self.mean = mean
        self.cov = cov
        self.weight_bounds = weight_bounds
        self.initial_guess = np.array([1 / self.n] * self.n)
        self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}] # set constraint to 0 if market neutral
        self.gamma = gamma
        self.tickers = tickers
        self.weights = None
        self.skew = None
        self.kurt = None

    def _bounds(self, bounds):
        """
        Private method: processes input bounds for scipy.optimize.

        :param bounds: minimum and maximum weight of an asset
        :type bounds: tuple
        :return: a tuple of bounds, e.g ((0, 1), (0, 1), (0, 1) ...)
        :rtype: tuple of tuples
        """
        if len(bounds) != 2 or not isinstance(bounds, tuple):
            raise ValueError("bounds must be a tuple of (lower bound, upper bound)")
        return (bounds,) * self.n

    def moment_optimisation(self, skew, kurt, delta1, delta2, delta3, delta4):
        """
        Calculates the optimal portfolio weights for utility functions that uses mean, covariance, skew and kurtosis of
        market invariants.

        :param skew: skew of market invariants
        :param kurt: kurtosis of market invariants
        :param delta1: coefficient of mean, (i.e how much weight to give maximising mean)
        :param delta2: coefficient of covariance, (i.e how much weight to give minimising covariance)
        :param delta3: coefficient of skew, (i.e how much weight to give maximising skew)
        :param delta4: coefficient of kurtosis, (i.e how much weight to give minimising kurtosis)
        :return: dictionary of tickers and weights
        """
        self.skew = skew
        self.kurt = kurt
        args = (self.mean, self.cov, skew, kurt, delta1, delta2, delta3, delta4)
        result = scop.minimize(
            utility_functions.moment_utility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self._bounds(self.weight_bounds),
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def max_sharpe(self, risk_free_rate=0.02):
        """
        Maximise the Sharpe Ratio.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: dict
        """
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        args = (self.mean, self.cov,
                self.gamma, risk_free_rate)
        result = scop.minimize(
            utility_functions.sharpe,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self._bounds(self.weight_bounds),
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the return, volatility and Sharpe Ratio of the portfolio.

        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        if self.weights is None:
            raise ValueError("Weights not calculated yet")
        sigma = np.sqrt(utility_functions.volatility(
            self.weights, self.cov))
        mu = self.weights.dot(self.mean)

        sharpe = -utility_functions.sharpe(
            self.weights, self.mean, self.cov, risk_free_rate
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
