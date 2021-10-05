from cvxpy.atoms.elementwise.exp import exp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
from . import utility_functions as utils
import cvxpy as cp

# Define Objective Functions

def sharpe_ratio(w, exp_ret, cov, risk_free_rate=0.02, neg=True):
    mu = w @ exp_ret
    sigma = cp.sqrt(cp.quad_form(w, cov))

    if neg:
        return -(mu - risk_free_rate)/sigma
    else:
        return (mu - risk_free_rate)/sigma

def port_vol(w, cov):
    sigma = cp.quad_form(w, cov)

    return sigma

def port_ret(w, exp_ret):
    ret = w.T @ exp_ret

    return ret

# Define Constraints

def LS_neutral(w, curr_price):
    return w @ curr_price == 0

def MaxPos(w, curr_price, threshold):
    return (cp.multiply(w, curr_price))/(w @ curr_price) <= threshold * np.ones(len(w))

