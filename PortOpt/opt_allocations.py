"""
This ``opt_allocations`` module calculates the optimal portfolio weights given the mean, covariance of
the data, for various utility functions. Currently implemented:

- Mean-Variance Optimisation (MVO)
- Minimum Volatility
- Maximum Sharpe Ratio

"""

from typing import Type
from cvxpy.problems import objective
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
from scipy.optimize import minimize
import warnings
from . import utility_functions as utils
import cvxpy as cp
from .base_opt import ConvexOptimiser
from . import cvx_functions as func


class Optimiser(ConvexOptimiser):
    def __init__(self, tickers, exp_ret, cov, weight_bounds=(0, 1)) -> None:
        # self.tickers = tickers
        self.exp_ret = exp_ret
        self.cov = cov

        super().__init__(len(tickers), tickers, weight_bounds=weight_bounds)

    # def solve(self, type='cvxpy', objective='max_sharpe', solver=None, solver_options=None, verbose=True):
    #     if type == 'cvxpy':
    #         if objective == 'max_sharpe':
    #             self.inter = self._cvx_solve(solver=solver, solver_options=solver_options, verbose=verbose)
    #             self.weights = self._fix_weights(self.inter.value/self.k.value)
    #         else:
    #             self.weights = self._cvx_solve(solver=solver, solver_options=solver_options, verbose=verbose)
    #     else:
    #         print("Solver not implemented...")

    #     return self.weights

    def add_constraints(self, constraints):
        for i in constraints:
            curr_const = constraints[i]

            self._add_constraint(curr_const)

    def add_objectives(self, objectives):
        for i in objectives:
            curr_obj = objectives[i]

            self._add_objective(curr_obj)

    def mean_variance(self, threshold, type='variance', solver=None, solver_options=None, verbose=True, risk_free_rate=0.02):
        if type == 'variance':
            self.objective += func.port_vol(self._w, self.cov)
            self.constraints += [
                self._w @ self.exp_ret >= threshold,
                cp.sum(self._w) == 1
            ]

        elif type == 'returns':
            self.objective += -func.port_ret(self._w, self.exp_ret)
            self.constraints += [
                func.port_vol(self._w, self.cov) <= threshold,
                cp.sum(self._w) == 1
            ]

        self.weights = self._cvx_solve(solver=solver, solver_options=solver_options, verbose=verbose)

        return self.weights

    def min_volatility(self, solver=None, solver_options=None, verbose=True, ):
        self.objective = func.port_vol(self._w, self.cov)

        self.constraints = [cp.sum(self._w) == 1]

        self.weights = self._cvx_solve(solver=solver, solver_options=solver_options, verbose=verbose)

        return self.weights

        # self.weights = self.solve()

    def max_sharpe(self, solver=None, solver_options=None, verbose=True, risk_free_rate=0.02):
        if solver_options == None:
            solver_options = {}

        self.y = cp.Variable(shape=self.n)
        self.k = cp.Variable()
        self.objective = cp.quad_form(self.y, self.cov)

        curr_constraints = self.constraints
        new_constraints = []

        for i in curr_constraints:
            if isinstance(i, cp.constraints.nonpos.Inequality):
                if isinstance(i.args[0], cp.expressions.constants.constant.Constant):
                    new_constraints += [i.args[1] >= i.args[0]*self.k]
                else:
                    new_constraints += [i.args[1] <= i.args[0]*self.k]
            elif isinstance(i, cp.constraints.zero.Equality):
                new_constraints += [i.args[1] == i.args[0]]
            else:
                raise TypeError("Constraint not in correct format...")

        new_constraints += [
            self.y @ (self.exp_ret - risk_free_rate) == 1,
            cp.sum(self.y) == self.k,
            self.k >= 0,
            self.y >= 0
        ]

        prob = cp.Problem(objective=cp.Minimize(self.objective), constraints=self.constraints)
        prob.solve(solver=solver, verbose=verbose, **solver_options)

        self.weights = self._fix_weights(self.y.value/self.k.value)

        return self.weights

    

        

        
            
        


# class OptimalAllocations:
#     def __init__(self, n, mean, cov, tickers, weight_bounds=(0, 1)):
#         """
#         :param n: number of assets
#         :type n: int
#         :param mean: mean estimate of market invariants
#         :type mean: pd.Dataframe
#         :param cov: covariance estimate of market invariants
#         :type cov: pd.Dataframe
#         :param tickers: tickers of securities used
#         :type tickers: list
#         :param weight_bounds: bounds for portfolio weights.
#         :type weight_bounds: tuple
#         """
#         self.n = n
#         self.mean = mean
#         self.cov = cov
#         self.weight_bounds = (weight_bounds,)*self.n
#         self.x0 = np.array([1 / self.n] * self.n)
#         self.constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
#         self.tickers = tickers
#         self.weights = None
#         self.skew = None
#         self.kurt = None

#     def moment_optimisation(self, skew, kurt, delta1, delta2, delta3, delta4):
#         """
#         Calculates the optimal portfolio weights for utility functions that uses mean, covariance, skew and kurtosis of
#         market invariants.

#         :param skew: skew of market invariants
#         :param kurt: kurtosis of market invariants
#         :param delta1: coefficient of mean, (i.e how much weight to give maximising mean)
#         :param delta2: coefficient of covariance, (i.e how much weight to give minimising covariance)
#         :param delta3: coefficient of skew, (i.e how much weight to give maximising skew)
#         :param delta4: coefficient of kurtosis, (i.e how much weight to give minimising kurtosis)
#         :return: dictionary of tickers and weights
#         """
#         self.skew = skew
#         self.kurt = kurt
#         args = (self.mean, self.cov, skew, kurt, delta1, delta2, delta3, delta4)
#         result = minimize(utility_functions.moment_utility, x0=self.x0, args=args,
#                                method="SLSQP", bounds=self.weight_bounds, constraints=self.constraints)
#         self.weights = result["x"]
#         return dict(zip(self.tickers, self.weights))

#     def sharpe_opt(self, risk_free_rate=0.02):
#         """
#         Maximise the Sharpe Ratio.

#         :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
#         :type risk_free_rate: float, optional
#         :raises ValueError: if ``risk_free_rate`` is non-numeric
#         :return: asset weights for the Sharpe-maximising portfolio
#         :rtype: dict
#         """
#         args = (self.mean, self.cov, risk_free_rate)
#         result = minimize(utility_functions.sharpe, x0=self.x0, args=args,
#                                method="SLSQP", bounds=self.weight_bounds, constraints=self.constraints)
#         self.weights = result["x"]
#         return dict(zip(self.tickers, self.weights))

#     def portfolio_metrics(self, verbose=False, risk_free_rate=0.02):
#         """
#         After optimising, calculate (and optionally print) the return, volatility and Sharpe Ratio of the portfolio.

#         :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02
#         :type risk_free_rate: float, optional
#         :return: expected return, volatility, Sharpe ratio.
#         :rtype: (float, float, float)
#         """
#         sigma = np.sqrt(utility_functions.volatility(
#             self.weights, self.cov))
#         mu = self.weights.dot(self.mean)

#         sharpe = -utility_functions.sharpe(self.weights, self.mean, self.cov, risk_free_rate)
#         print(f"Expected annual return: {100*mu}")
#         print(f"Annual volatility: {100*sigma}")
#         print(f"Sharpe Ratio: {sharpe}")
#         return mu, sigma, sharpe
