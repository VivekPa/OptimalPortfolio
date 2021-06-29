"""
This ``opt_allocations`` module calculates the optimal portfolio weights given the mean, covariance, skew and kurtosis of
the data, for various utility functions. Currently implemented:

- Moment Optimisation
- Maximum Sharpe Ratio

"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
# import portfolioopt.utility_functions as utility_functions
runfile('/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt/utility_functions.py', wdir='/home/sven/Documents/PyDox/OptimalPortfolio/portfolioopt')

class OptimalAllocations:
    def __init__(self, n, mean, cov, tickers, weight_bounds=(0, 1)):
        """
        :param n: number of assets
        :type n: int
        :param mean: mean estimate of market invariants
        :type mean: pd.Dataframe
        :param cov: covariance estimate of market invariants
        :type cov: pd.Dataframe
        :param tickers: tickers of securities used
        :type tickers: list
        :param weight_bounds: bounds for portfolio weights.
        :type weight_bounds: tuple
        """
        self.n = n
        self.mean = mean
        self.cov = cov
        self.weight_bounds = (weight_bounds,)*self.n
        self.x0 = np.array([1 / self.n] * self.n)
        self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        self.tickers = tickers
        self.weights = None
        self.skew = None
        self.kurt = None
        self.coskew = None
        self.cokurt = None

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
        result = minimize(moment_utility, x0=self.x0, args=args,
                               method="SLSQP", bounds=self.weight_bounds, constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))
    
    def comoment_optimisation(self, coskew, cokurt, delta1, delta2, delta3, delta4):
        """
        Calculates the optimal portfolio weights for utility functions that uses mean, covariance, coskewness and cokurtosis of
        market invariants.

        :param skew: coskewness matrix of market invariants
        :param kurt: cokurtosis matrix of market invariants
        :param delta1: relative optimization weight for mean
        :param delta2: relative optimization weight for covariance
        :param delta3: relative optimization weight for coskewness
        :param delta4: relative optimization weight for cokurtosis
        :return: dictionary of tickers and weights
        
        """
        self.coskew = coskew
        self.cokurt = cokurt
        args = (self.mean, self.cov, coskew, cokurt, delta1, delta2, delta3, delta4)
        result = minimize(comoment_utility, x0=self.x0, args=args,
                               method="SLSQP", bounds=self.weight_bounds, constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def sharpe_opt(self, risk_free_rate=0.02):
        """
        Maximise the Sharpe Ratio.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the Sharpe-maximising portfolio
        :rtype: dict
        """
        args = (self.mean, self.cov, risk_free_rate)
        result = minimize(sharpe, x0=self.x0, args=args,
                               method="SLSQP", bounds=self.weight_bounds, constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def portfolio_metrics(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the return, volatility and Sharpe Ratio of the portfolio.

        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        sigma = np.sqrt(utility_functions.volatility(
            self.weights, self.cov))
        mu = self.weights.dot(self.mean)

        sharpe = -sharpe(self.weights, self.mean, self.cov, risk_free_rate)
        print(f"Expected annual return: {100*mu}")
        print(f"Annual volatility: {100*sigma}")
        print(f"Sharpe Ratio: {sharpe}")
        return mu, sigma, sharpe
