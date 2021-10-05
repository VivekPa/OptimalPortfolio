"""
The ``moment_est`` module estimates the moments of the distribution of market invariants using a number of methods.

Currently implemented:

- Expected Returns
    - Historical Mean
    - Exponentially Weighted Mean
    - Mean Shrinkage
        - James-Stein Mean Shrinkage

- Risk Models
    - Historical Covariance
    - Exponentially Weighted Covariance
    - Covariance Shrinkage
        - Identity Shrinkage
        - Scaled Variance Shrinkage
        - Ledoit-Wolf Single Index
        - Ledoit-Wolf Constant Correlation

- Maximum Likelihood estimators of mean, covariance and higher moments:

    - Normal distribution
    - Student-t distribution

"""

import pandas as pd
import numpy as np
from sklearn import covariance
from sklearn.covariance import ledoit_wolf
from scipy.stats import moment
import statsmodels.api as sm
import warnings
from .exp_max import expectation_max
from .base_est import BaseEstimator


class ExpReturns(BaseEstimator):
    def __init__(self, tickers) -> None:
        super.__init__(tickers)

    def avg_hist_ret(self, data, is_returns=False, period=1, frequency=252):
        if is_returns:
            returns = data
        else:
            returns = self._get_logreturns(data, period=period)
        
        return frequency * returns.mean(axis=0)

    def ema_hist_ret(self, data, is_returns=False, period=1, span=180, frequency=252): 
        if is_returns:
            returns = data
        else:
            returns = self._get_logreturns(data, period=period)

        return frequency * returns.ewm(span=span).mean().iloc[-1]

    def shrink_ret(self, data, target_mean, method, market_data=None, delta=None, period=1, is_returns=True):
        shrink = Shrinkage(self.tickers)

        shrunk_mean = shrink.shrink_mean(method=method, data=data, target_mean=target_mean, market_data=market_data, delta=delta, is_returns=is_returns)

        return shrunk_mean

    def CAPM(self, data, market_data, is_returns=False, period=1, frequency=252):
        pass


class RiskModel(BaseEstimator):
    def __init__(self, tickers) -> None:
        super.__init__(tickers)

    def avg_hist_cov(self, data, is_returns=False, period=1, frequency=252):
        if is_returns:
            returns = data
        else:
            returns = self._get_logreturns(data, period=period)

        cov = returns.cov()

        return frequency * cov

    def ema_hist_cov(self, data, is_returns=False, period=1, span=180, frequency=252):
        if is_returns:
            returns = data
        else:
            returns = self._get_logreturns(data, period=period)

        N = len(self.tickers)

        S = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                S[i, j] = S[j, i] = self._pairwise_exp_cov(returns.iloc[:, i], returns.iloc[:, j], span)
        cov = pd.DataFrame(S * frequency, columns=self.tickers, index=self.tickers)

        return cov

    def shrinkage_cov(self, data, method, market_data=None, delta=None, period=1, is_returns=True):
        shrink = Shrinkage(self.tickers)

        shrunk_cov = shrink.shrink_cov(method=method, data=data, market_data=market_data, delta=delta, is_returns=is_returns)

        return shrunk_cov


class Shrinkage(BaseEstimator):
    def __init__(self, tickers) -> None:
        self.delta = None
        self.n = len(tickers)

        super.__init__(tickers)

    def _identity_shrinkage(self, data, delta=0.1, is_returns=True):
        if is_returns:
            returns = data
        else:
            returns = self._get_logreturns(data, period=1)
        
        S = returns.cov()

        self.delta = delta
        I = np.identity(self.n)

        return self.delta*I + (1-self.delta)*S

    def _scaled_variance_shrinkage(self, data, delta=0.1, is_returns=True):
        if is_returns:
            returns = data
        else:
            returns = self._get_logreturns(data, period=1)
        
        S = returns.cov()

        self.delta = delta
        F = np.identity(self.n) * np.trace(S)/self.n

        return self.delta*F + (1-self.delta)*S

    def _ledoit_single_index(self, data, market_data, is_returns=True):
        if is_returns:
            stock_returns = data
            market_returns = market_data
        else:
            stock_returns = self._get_logreturns(data, period=1)
            market_returns = self._get_logreturns(market_data, period=1)
            
        S = stock_returns.cov()

        # Calculate shrinkage target
            
        s00 = market_returns.var()
            
        betas = []
        resid = []
        
        for i in stock_returns.columns:
            curr_returns = stock_returns[i]
            
            y = curr_returns
            X = sm.add_constant(market_returns)
            
            curr_model = sm.OLS(y, X)
            curr_results = curr_model.fit()
            
            curr_beta = curr_results.params[1]
            curr_resid = curr_results.resid.var()
            
            betas.append(curr_beta)
            resid.append(curr_resid)
        
        betas = pd.Series(betas, index=stock_returns.columns)
        resid = pd.Series(resid, index=stock_returns.columns)
        
        resid_var = pd.DataFrame(np.diag(resid), index=stock_returns.columns, columns=stock_returns.columns)
        
        F = s00 * np.outer(betas, betas) + resid_var
        
        # Calculate optimal shrinkage coefficient
        
        X_m = stock_returns - stock_returns.mean(axis=0)
        M_m = market_returns - market_returns.mean()
        
        s_0x = (stock_returns.corrwith(market_returns)*stock_returns.var())
        
        p = 0
        r = 0
        g = 0
        
        for i in stock_returns.columns:
            for j in stock_returns.columns:
                p_ij = (1/len(stock_returns)) * np.sum(((X_m[i])*(X_m[j]) - S.loc[i, j])**2)
                p += p_ij
                
                f_ij = s_0x.loc[i]*s_0x.loc[j]/s00 * S.loc[i, j]
                
                if i == j:
                    r_ij = p_ij
                else:
                    r_ijt = ((1/s00**2) * (s_0x.loc[j]*s00*X_m[i] + s_0x.loc[i]*s00*X_m[j] - s_0x.loc[i]*s_0x.loc[j]*M_m) * 
                            (X_m[i]*X_m[j]*M_m)) - f_ij
                    r_ij = (1/len(stock_returns)) * np.sum(r_ijt)
                    
                r += r_ij
                
                g_ij = (f_ij - S.loc[i, j])**2
                g += g_ij
                            
        k = (p - r)/g

        self.alpha = max(0, min(1, k/len(stock_returns)))

        return self.alpha*F + (1-self.alpha)*S

    def _ledoit_constant_corr(self, data, is_returns=True):
        if is_returns:
            stock_returns = data
        else:
            stock_returns = self._get_logreturns(data, period=1)

        # Calculate Shrinkage target

        S = stock_returns.cov()
        R = stock_returns.corr()
        r_bar = 1/len(R)**2 *np.sum(np.sum(R))

        F = pd.DataFrame(index=stock_returns.columns, columns=stock_returns.columns)

        for i in stock_returns.columns:
            for j in stock_returns.columns:
                if i == j:
                    F.loc[i, j] = S.loc[i, j]
                else:
                    F.loc[i, j] = r_bar*np.sqrt(S.loc[i, i]*S.loc[j, j])

        # Calculate optimal shrinkage coefficient

        X_m = stock_returns - stock_returns.mean(axis=0)

        p = 0
        r = 0
        g = 0

        for i in stock_returns.columns:
            for j in stock_returns.columns:
                p_ij = (1/len(stock_returns)) * np.sum(((X_m[i])*(X_m[j]) - S.loc[i, j])**2)
                p += p_ij

                f_ij = F.loc[i, j]
                
                t_ii_ij = (1/len(stock_returns))*np.sum((X_m[i]**2 - S.loc[i, i])*(X_m[i]*X_m[j] - S.loc[i, j]))
                t_jj_ij = (1/len(stock_returns))*np.sum((X_m[j]**2 - S.loc[j, j])*(X_m[i]*X_m[j] - S.loc[i, j]))

                if i == j:
                    r_ij = p_ij
                else:
                    r_ij = r_bar/2 * (np.sqrt((S.loc[j, j]/S.loc[i, i])*t_ii_ij + (S.loc[i, i]/S.loc[j, j])*t_jj_ij))

                r += r_ij

                g_ij = (f_ij - S.loc[i, j])**2
                g += g_ij
                
        k = (p - r)/g
        self.alpha = max(0, min(1, k/len(stock_returns)))

        return self.alpha*F + (1-self.alpha)*S

    def _james_stein_mean(self, data, target_mean=None, is_returns=True):
        if is_returns:
            stock_returns = data
        else:
            stock_returns = self._get_logreturns(data, period=1)
            
        mu = stock_returns.mean(axis=0)
        S = stock_returns.cov()
        
        l1 = sorted(np.linalg.eigvals(S))[-1]
        
        alpha = min(1, (2/len(stock_returns))*((np.trace(S)) - 2*l1**2)/((mu - target_mean) @ (mu - target_mean)))
        
        return alpha*target_mean + (1-alpha)*mu
        
    def shrink_cov(self, method, data, market_data=None, delta=None, period=1, is_returns=True):
        if is_returns:
            returns = data
            market_returns = market_data
        else:
            returns = self._get_logreturns(data, period=period)
            market_returns = self._get_logreturns(market_data, period=period)

        if method == 'identity_shrinkage':
            if delta == None:
                raise ValueError("Please give a value for delta")
            else:
                shrunk_cov = self._identity_shrinkage(data=returns, delta=delta, is_returns=True)
        elif method == "scaled_variance_shrinkage":
            if delta == None:
                raise ValueError("Please give a value for delta")
            else:
                shrunk_cov = self._scaled_variance_shrinkage(data=returns, delta=delta, is_returns=True)
        elif method == "ledoit_single_index":
            shurnk_cov = self._ledoit_single_index(data=returns, market_data=market_returns, is_returns=True)
        elif method == "ledoit_constant_corr":
            shurnk_cov = self._ledoit_constant_corr(data=returns, is_returns=True)
        else:
            raise NotImplementedError("Method not implmented...")

        return shurnk_cov

    def shrink_mean(self, method, data, target_mean, market_data=None, delta=None, is_returns=True):
        if method == 'james_stein':
            shrunk_mean = self._james_stein_mean(data=data, target_mean=target_mean, is_returns=is_returns)
        else:
            raise NotImplementedError("Method not implemented...")
        
        return shrunk_mean


class MLE:
    """
    Provide methods to calculate maximum likelihood estimators (MLE) of mean, covariance and higher moments. Currently
    implemented distributions:

    - Normal
    - Student-t

    Instance variables:

    - ``invariants`` (market invariants data)
    - ``dist`` (distribution choice)
    - ``n`` (number of assets)
    - ``mean`` (estimate of mean, initially None)
    - ``cov`` (estimate of covariance, initially None)
    - ``skew`` (estimate of skew, initially None)
    - ``kurt`` (estimate of kurtosis, initially None)

    Public methods:

    - ``norm_est`` (calculates the normally distributed maximum likelihood estimate of mean, covariance, skew and kurtosis)
    - ``st_est`` (calculates the student-t distributed maximum likelihood estimate of mean, covariance, skew and kurtosis)
    """
    def __init__(self, invariants, n, dist="normal"):
        """

        :param invariants: sample data of market invariants
        :type invariants: pd.Dataframe
        :param n: number of assets
        :type n: int
        :param dist: choice of distribution: "normal"
        :type dist: str
        """
        self.invariants = invariants
        self.dist = dist
        self.n = n
        self.mean = None
        self.cov = None
        self.skew = None
        self.kurt = None

    def norm_est(self):
        """
        Calculates MLE estimate of mean, covariance, skew and kurtosis, assuming normal distribution

        :return: dataframes of mean, covariance, skew and kurtosis
        :rtype: pd.Dataframe
        """
        if self.dist == "normal":
            self.mean = 1/self.n * np.sum(self.invariants)
            self.cov = 1/self.n * np.dot((self.invariants - self.mean), np.transpose(self.invariants - self.mean))
            self.skew = 0
            self.kurt = 0
        return self.mean, self.cov, self.skew, self.kurt

    def st_est(self):
        """
        Calculates MLE estimate of mean, covariance, skew and kurtosis, assuming student-t distribution

        :return: dataframe of mean, covariance, skew and kurtosis
        :rtype: pd.Dataframe
        """
        if self.dist == "student-t":
            self.mean, self.cov = expectation_max(self.invariants, max_iter=1000)
            self.skew = 0
            self.kurt = 6

