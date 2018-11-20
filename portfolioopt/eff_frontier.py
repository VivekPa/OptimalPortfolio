"""
This ``eff_frontier`` module calculates the optimal portfolio weights given the mean, covariance, skew and kurtosis of
the data, for various utility functions. Currently implemented:

- Maximum Sharpe Ratio
- Minimum Volatility
- Efficient Risk
- Efficient Return
- Moment Optimisation
- Kelly Criterion

"""

import numpy as np
import pandas as pd
import scipy.optimize as scop
import warnings
import portfolioopt.utility_functions as utility_functions


class EfficientFrontier:
    """
    This ``EfficientFrontier`` class optimises portfolio weights for given utility function.

    Instance variables:

    - Inputs:

        - ``n`` (number of assets)
        - ``mean`` (mean of market invariants)
        - ``cov`` (covariance of market invariants)
        - ``weight_bounds`` (the bounds of portfolio weights)
        - ``gamma`` (L2 regularisation coefficient)
        - ``tickers`` (list of tickers of assets)

    - Optimisation parameters:

        - ``initial_guess`` (initial guess for portfolio weights, set to evenly distributed)
        - ``constraints`` (constraints for optimisation)

    - Outputs:

        - ``weights`` (portfolio weights, initially set to None)


    Public methods:

        - ``max_sharpe()`` (calculates portfolio weights that maximises Sharpe Ratio)
        - ``min_volatility()`` (calculates portfolio weights that minimises volatility)
        - ``efficient_risk()`` (calculates portfolio weights that minimises efficient risk)
        - ``efficient_return()`` (calculates portfolio weights that maximises efficient return)
        - ``custom_objective()`` (calculates portfolio weights for custom utility function)
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

    def moment_optimisation(self, skew, kurt, delta1, delta2, delta3, delta4, gamma=0.2):
        """
        Calculates the optimal portfolio weights for utility functions that uses mean, covariance, skew and kurtosis of
        market invariants.

        :param skew: skew of market invariants
        :param kurt: kurtosis of market invariants
        :param delta1: coefficient of mean, (i.e how much weight to give maximising mean)
        :param delta2: coefficient of covariance, (i.e how much weight to give minimising covariance)
        :param delta3: coefficient of skew, (i.e how much weight to give maximising skew)
        :param delta4: coefficient of kurtosis, (i.e how much weight to give minimising kurtosis)
        :param gamma: coefficient of L2 Regularisation (default set to 0.2)
        :return: dictionary of tickers and weights
        """
        self.skew = skew
        self.kurt = kurt
        args = (self.mean, self.cov, skew, kurt, delta1, delta2, delta3, delta4, gamma)
        result = scop.minimize(
            utility_functions.moment_utility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def kelly_criterion(self, risk_free_rate=0.02):
        """
        Calculates the optimal portfolio weights according to Kelly's Criterion.

        :param risk_free_rate: risk free rate of return
        :type: float
        :return: dictionary of tickers + weights
        :rtype: dict
        """
        weights = (1+risk_free_rate)*(np.dot(np.invert(self.cov), (self.mean-risk_free_rate)))
        return dict(zip(self.tickers, weights))
        
    def _create_bounds(self, test_bounds):
        """
        Private method: make sure bounds are valid and formats them

        :param test_bounds: minimum and maximum weight of an asset
        :type test_bounds: tuple
        :raises ValueError: if ``test_bounds`` is not a tuple of length two.
        :raises ValueError: if the lower bound is too high
        :return: a tuple of bounds
        :rtype: tuple of tuples
        """
        if len(test_bounds) != 2 or not isinstance(test_bounds, tuple):
            raise ValueError(
                "test_bounds must be a tuple of (lower bound, upper bound)"
            )
        if test_bounds[0] is not None:
            if test_bounds[0] * self.n > 1:
                raise ValueError("Lower bound is too high")
        return (test_bounds,) * self.n

    def maxmise_sharpe(self, risk_free_rate=0.02):
        """
        Maximise the Sharpe Ratio. Maximises the risk adjusted excess return of the portfolio.

        :param risk_free_rate: risk-free rate of return, defaults to 0.02
        :type risk_free_rate: float, optional
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
            bounds=self.weight_bounds,
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def min_vol(self):
        """
        Minimise volatility.

        :return: asset weights for the portfolio that minismises volatility
        :rtype: dict
        """
        args = (self.cov, self.gamma)
        result = scop.minimize(
            utility_functions.volatility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def custom_utility(self, utility_function, *args):
        """
        Optimise some utility function. The utility function must be able to be optimised via quadratic programming,
        any more complex might cause a failure to optimise.

        :param utility_function: function which maps (weight, args) -> cost
        :type utility_function: function with signature (np.ndarray, args) -> float
        :return: asset weights that optimise the custom objective
        :rtype: dict
        """
        result = scop.minimize(
            utility_function,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def eff_risk(self, target_risk, risk_free_rate=0.02, market_neutral=False):
        """
        Calculate the maximum Sharpe ratio portfolio for a given volatility (i.e max return
        for a target risk).

        :param target_risk: the desired volatility of the resulting portfolio.
        :type target_risk: float
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :param market_neutral: bool, optional
        :raises ValueError: if ``target_risk`` is not a positive float
        :raises ValueError: if ``risk_free_rate`` is non-numeric
        :return: asset weights for the efficient risk portfolio
        :rtype: dict
        """
        if not isinstance(target_risk, float) or target_risk < 0:
            raise ValueError("target_risk should be a positive float")
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError("risk_free_rate should be numeric")

        args = (self.mean, self.cov,
                self.gamma, risk_free_rate)
        target_constraint = {
            "type": "ineq",
            "fun": lambda w: target_risk
            - np.sqrt(utility_functions.volatility(w, self.cov))}
        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            if self.weight_bounds[0][0] is not None and self.weight_bounds[0][0] >= 0:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning)
                self.weight_bounds = self._make_bounds((-1, 1))
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x)},
                target_constraint]
        else:
            constraints = self.constraints + [target_constraint]

        result = scop.minimize(
            utility_functions.sharpe,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints=constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def eff_return(self, target_return, market_neutral=False):
        """
        Calculate the portfolio that minimises volatility for a given target return.

        :param target_return: the desired return of the resulting portfolio.
        :type target_return: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :type market_neutral: bool, optional
        :raises ValueError: if ``target_return`` is not a positive float
        :return: asset weights for portfolio
        :rtype: dict
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_risk should be a positive float")

        args = (self.cov, self.gamma)
        target_constraint = {
            "type": "eq",
            "fun": lambda w: w.dot(self.mean) - target_return}
        # The equality constraint is either "weights sum to 1" (default), or
        # "weights sum to 0" (market neutral).
        if market_neutral:
            if self.weight_bounds[0][0] is not None and self.weight_bounds[0][0] >= 0:
                warnings.warn(
                    "Market neutrality requires shorting - bounds have been amended",
                    RuntimeWarning,
                )
                self.weight_bounds = self._make_bounds((-1, 1))
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x)},
                target_constraint,
            ]
        else:
            constraints = self.constraints + [target_constraint]

        result = scop.minimize(
            utility_functions.volatility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints=constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))


    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        """
        After optimising, calculate (and optionally print) the performance of the optimal
        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.
        
        :param verbose: whether performance should be printed, defaults to False
        :type verbose: bool, optional
        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
        :type risk_free_rate: float, optional
        :raises ValueError: if weights have not been calculated yet
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
