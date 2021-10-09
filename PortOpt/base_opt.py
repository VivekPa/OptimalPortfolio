import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
from scipy.optimize import minimize
import warnings
from . import utility_functions as utility_functions
import cvxpy as cp


class ConvexOptimiser:
    def __init__(self, n, tickers, weight_bounds=(0, 1)) -> None:
        self.n = n
        self.tickers = tickers

        self.weight_bounds = weight_bounds
        self.x0 = np.array([1 / self.n] * self.n)
        
        self._w = cp.Variable(shape=self.n)
        self.constraints = []
        self.objective = None
        self._problem = None

        self._make_weight_constraint(weight_bounds=weight_bounds)

    def _make_weight_constraint(self, weight_bounds=None):
        if weight_bounds == None:
            self._lower = self.weight_bounds[0]
            self._upper = self.weight_bounds[1]

            self.constraints += [self._w >= self._lower]
            self.constraints += [self._w <= self._upper]

        else:
            self._lower = weight_bounds[0]
            self._upper = weight_bounds[1]

            self.constraints += [self._w >= self._lower]
            self.constraints += [self._w <= self._upper]

    def _fix_weights(self, weights, threshold=1e-4):
        new_weights = pd.Series(weights, index=self.tickers)
        new_weights.loc[new_weights < threshold] = 0

        return new_weights

    def _add_constraint(self, constraint):
        self.constraints += [constraint(self._w)]

    def _add_objective(self, objective):
        self.objective += objective(self._w)

    def _cvx_solve(self, solver=None, solver_options=None, verbose=True):
        if solver_options == None:
            solver_options = {}
        try:
            self._problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

            if solver == None:
                self._problem.solve(verbose=verbose, **solver_options)
            else:
                self._problem.solve(solver=solver, verbose=verbose, **solver_options)

        except (ValueError, TypeError, cp.DCPError, cp.DGPError) as e:
            print(e)

        if self._problem.status not in {"optimal", "optimal_inaccurate"}:
            raise cp.SolverError("Problem was not solved...")

        self.weights = self._w.value.round(20)
        self.weights = self._fix_weights(self.weights)

        return self.weights





