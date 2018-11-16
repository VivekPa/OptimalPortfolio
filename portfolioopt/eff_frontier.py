import numpy as np
import pandas as pd
import scipy.optimize as scop
import warnings
import portfolioopt.utility_functions as utility_functions


class EfficientFrontier:
    def __init__(self, n, mean, cov, tickers, gamma, weight_bounds=(0, 1)):
        self.n = n
        self.mean = mean
        self.cov = cov
        self.weight_bounds = weight_bounds
        self.initial_guess = np.array([1 / self.n] * self.n)
        self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}] # set constraint to 0 if market neutral
        self.gamma = gamma
        self.tickers = tickers
        self.weights = None

    def max_sharpe(self, risk_free_rate=0.02):
        """
        Maximise the Sharpe Ratio. The result is also referred to as the tangency portfolio,
        as it is the tangent to the efficient frontier curve that intercepts the risk-free
        rate.
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
            utility_functions.negative_sharpe,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def min_volatility(self):
        """
        Minimise volatility.
        :return: asset weights for the volatility-minimising portfolio
        :rtype: dict
        """
        args = (self.cov, self.gamma)
        result = scop.minimize(
            utility_functions.volatility,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints,
        )
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def custom_objective(self, objective_function, *args):
        """
        Optimise some objective function. While an implicit requirement is that the function
        can be optimised via a quadratic optimiser, this is not enforced. Thus there is a
        decent chance of silent failure.
        :param objective_function: function which maps (weight, args) -> cost
        :type objective_function: function with signature (np.ndarray, args) -> float
        :return: asset weights that optimise the custom objective
        :rtype: dict
        """
        result = scop.minimize(
            objective_function,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=self.constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def efficient_risk(self, target_risk, risk_free_rate=0.02, market_neutral=False):
        """
        Calculate the Sharpe-maximising portfolio for a given volatility (i.e max return
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
            if self.bounds[0][0] is not None and self.bounds[0][0] >= 0:
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
            utility_functions.negative_sharpe,
            x0=self.initial_guess,
            args=args,
            method="SLSQP",
            bounds=self.bounds,
            constraints=constraints)
        self.weights = result["x"]
        return dict(zip(self.tickers, self.weights))

    def efficient_return(self, target_return, market_neutral=False):
        """
        Calculate the 'Markowitz portfolio', minimising volatility for a given target return.
        :param target_return: the desired return of the resulting portfolio.
        :type target_return: float
        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),
                               defaults to False. Requires negative lower weight bound.
        :type market_neutral: bool, optional
        :raises ValueError: if ``target_return`` is not a positive float
        :return: asset weights for the Markowitz portfolio
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
        :raises ValueError: if weights have not been calcualted yet
        :return: expected return, volatility, Sharpe ratio.
        :rtype: (float, float, float)
        """
        if self.weights is None:
            raise ValueError("Weights not calculated yet")
        sigma = np.sqrt(utility_functions.volatility(
            self.weights, self.cov))
        mu = self.weights.dot(self.mean)

        sharpe = -utility_functions.negative_sharpe(
            self.weights, self.mean, self.cov, risk_free_rate
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
