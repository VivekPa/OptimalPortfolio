"""
This ``opt_allocations`` module contains Optimiser class, which optimises portfolio weights
for a particular set of objective functions.

Currently implemented:

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
    """ 
    The Optimiser class (inheriting ConvexOptimiser class) contains various optimisation methods to be called 
    based on need. 
    
    Instance variables:
    
    - Inputs
        - ``tickers`` - list
        - ``exp_ret`` - pd.Series
        - ``cov`` - pd.DataFrame
        - ``weight_bounds`` - float tuple
        
    - Output
        - ``weights`` - pd.Series
        
    Public Methods:
    
    - ``add_constraints()`` add a list of constraints (following DCP)
    - ``add_objectives()`` add a list of objective functions to overall objective (convex)

    - ``mean_variance()`` optimises weights for either maximum returns with variance limit, or minimum variance with return limit
    - ``min_volatility()`` optimises weights for minimum volatility portfolio 
    - ``max_sharpe()`` optimises weights for maximum sharpe ratio

    - ``weight_tearsheet()`` generates annual returns, volatility and sharpe for a given set of weights
    """

    def __init__(self, tickers, exp_ret, cov, weight_bounds=(0, 1)) -> None:
        """ 
        :param tickers: list of tickers in portfolio
        :type tickers: np.ndarray or pd.Series or list
        :param exp_ret: expected return of each ticker
        :type exp_ret: pd.Series
        :param cov: covariance matrix of tickers
        :type cov: pd.DataFrame
        :param weight_bounds: lower and upper bounds on weights
        :type weight_bounds: float tuple
        """
        self.exp_ret = exp_ret
        self.cov = cov

        super().__init__(len(tickers), tickers, weight_bounds=weight_bounds)

    def add_constraints(self, constraints):
        """ 
        Function to add multiple constraints into an optimisation problem.

        :param constraints: list of constraints to be added
        :type constraints: list(expressions)
        """
        for i in constraints:
            curr_const = constraints[i]

            self._add_constraint(curr_const)

    def add_objectives(self, objectives):
        """ 
        Function to add multiple objectives into an optimisation problem.
        
        :param objectives: list of objectives to be added
        :type objectives: list(functions)
        """
        for i in objectives:
            curr_obj = objectives[i]

            self._add_objective(curr_obj)

    def mean_variance(self, threshold, type='variance', solver=None, solver_options=None, verbose=True, risk_free_rate=0.02):
        """ 
        Mean-Variance optimisation, with maximum returns (minimum volatility) with volatility (return) limit.

        :param threshold: threshold limit for variance or returns
        :type threshold: float
        :param type: type of optimisation, 'variance' or 'returns'
        :type type: str
        :param solver: type of cvxpy solver to use, default None
        :type solver: str
        :param solver_options: dict of solver options for cvxpy
        :type solver_options: dict()
        :param verbose: whether to print steps
        :type verbose: boolean
        :param risk_free_rate: risk free rate parameter, default to 0.02
        :type risk_free_rate: float
        :return: weights
        :rtype: pd.Series
        """
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

    def min_volatility(self, solver=None, solver_options=None, verbose=True):
        """
        Minimum volatility optimisation

        :param solver: type of cvxpy solver to use, default None
        :type solver: str
        :param solver_options: dict of solver options for cvxpy
        :type solver_options: dict()
        :param verbose: whether to print steps
        :type verbose: boolean
        :return: weights
        :rtype: pd.Series
        """
        self.objective = func.port_vol(self._w, self.cov)

        self.constraints = [cp.sum(self._w) == 1]

        self.weights = self._cvx_solve(solver=solver, solver_options=solver_options, verbose=verbose)

        return self.weights

        # self.weights = self.solve()

    def max_sharpe(self, solver=None, solver_options=None, verbose=True, risk_free_rate=0.02):
        """ 
        Maximum Sharpe Ratio Optimisation
        
        :param solver: type of cvxpy solver to use, default None
        :type solver: str
        :param solver_options: dict of solver options for cvxpy
        :type solver_options: dict()
        :param verbose: whether to print steps
        :type verbose: boolean
        :param risk_free_rate: risk free rate parameter, default to 0.02
        :type risk_free_rate: float
        :return: weights
        :rtype: pd.Series
        """
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

    def weight_tearsheet(self, weights, risk_free_rate=0.02, verbose=True):
        """ 
        Calculates annual returns, volatility and Sharpe ratio.
        
        :param weights: current (optimised) weights to use
        :type weights: pd.Series
        :param risk_free_rate: risk free rate parameter, default to 0.02
        :type risk_free_rate: float
        :param verbose: whether to print steps
        :type verbose: boolean
        """
        ret = weights @ self.exp_ret
        sigma = np.sqrt(cp.quad_form(weights, self.cov).value)
        sharpe = (ret - risk_free_rate)/sigma

        if verbose:
            print(f"Annual Return: {round(100*ret, 3)}")
            print(f"Annual Volatility: {round(100*sigma, 3)}")
            print(f"Annual Sharpe: {round(sharpe, 3)}")

    


