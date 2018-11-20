<p align="center">
    <img width=70% src="https://github.com/VivekPa/PortfolioAnalytics/blob/master/media/logo_v2.png">
</p>

<p align="center">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp;
</p>

**PortfolioAnalytics** is a robust open source library for portfolio optimisation and analytics. This library implements classical portfolio optimisation methods such as Efficient Frontier, Sharpe ratio and Mean Variance, along with modern developments in the field such as Shrinkage estimators, Maximum likelihood estimators and Kelly Criterion. It also implements a novel shrinkage estimator and optimisation with higher moments. For a full documentation on the usage, head over to [ReadTheDocs](https://portfolioanalytics.readthedocs.io/en/latest/)

Regardless of whether you are a fundamental investor, or an algorithmic trader, this library can aid you in allocating your capital in the most risk efficient way, allowing to optimise your utility. *For more details on the project design and similar content, please check out [Engineer Quant](https://medium.com/engineer-quant)*

*Disclaimer: This is not trading or investment advice. Trading involves significant risk and do so at your risk.*


## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Quickstart](#quickstart)
- [Market Invariants](#market-invariants)
- [Moment Estimation](#moment-estimation)
    - [Nonparametric Estimators](#nonparametric-estimators)
    - [Maximum Likelihood Estimators](#maximum-likelihood-estimators)
    - [Shrinkage Estimators](#shrinkage-estimators)
- [Efficient Frontier](#efficient-frontier)
    - [Sharpe Ratio](#sharpe-ratio)
    - [Efficient Return](#efficient-return)
- [Other Optimisations](#other-optimisations)
    - [Kelly Criterion](#kelly-criterion)
    - [Higher Moment Optimisation](#higher-moment-optimisation)
- [Utility Functions](#utility-functions)
- [Advances](#advances)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Overview
This library aims to make optimising portfolios accessible to every trader and investor. To install this library, download it and run
```python
run setup.py
```

## Quickstart
For those who want to see the library in action, run the following script:

```python
import pandas as pd
import portfolioopt.invariants as inv
import portfolioopt.moment_est as mest
import portfolioopt.eff_frontier as fron

# Load stock prices
df = pd.read_csv("stock_data.csv", parse_dates=True, index_col="date")

# Calculate invariants and estimate mean and covariance
invariants = inv.stock_invariants(df, 20)
mu = mest.sample_mean(invariants)
cov = mest.sample_cov(invariants)

# Optimise using the Sharpe Ratio
frontier = fron.EfficientFrontier(20, mu, cov, list(df.columns), gamma=0)
print(frontier.maximise_sharpe())
frontier.portfolio_performance(verbose=True)
```

This should output:

```
{'GOOG': 0.0,
'AAPL': 1.083816857952177e-18,
'FB': 0.0,
'BABA': 0.9999999999999993,
'AMZN': 0.0,
'GE': 0.0,
'AMD': 4.8090078788025475e-17,
'WMT': 6.713850949088525e-17,
'BAC': 2.727259355479373e-17,
'GM': 4.174867984747976e-17,
'T': 4.5848172405209455e-17,
'UAA': 0.0,
'SHLD': 4.0836164868648516e-17,
'XOM': 0.0,
'RRC': 0.0,
'BBY': 4.440892098500626e-16,
'MA': 0.0,
'PFE': 1.0225689161842779e-17,
'JPM': 3.961235752701918e-16,
'SBUX': 0.0}

Expected annual return: 2.1%
Annual volatility: 34.4%
Sharpe Ratio: -0.02
```

## Market Invariants
The first step to optimising any portfolio is calculating market invariants. Market invariants are defined as aspects of market prices that have some determinable statistical behaviour over time. For stock prices, the compounded returns are the market invariants. So when we calculate these invariants, we can statistically model them and gain useful insight into their behaviour. So far, calculating market invariants of stock prices and forex prices have been implemented. 

## Moment Estimation
Once the market invariants have been calculated, it is time to model the statistical properties of the invariants. This is a deeply researched and studied field and due to the nature of the complexity involved in modelling the statistical properties of large market data, I have tried my best to implement cutting edge procedures, but I welcome feedback and improvements.

### Nonparametric Estimators
The simplest method of estimating the mean and covariance of invariants are the sample mean and covariance. However, this can be extended by introducing weightage for the timestamps, i.e giving more weight to recent data than older data. One novel approach I have taken is introducing exponentially weighted mean and covariance, which intutively has backing. The following have been implemented:

- Sample Mean
- Sample Covariance
- Exponentially weighted mean
- Exponentially weighted covariance

### Maximum Likelihood Estimators
Maximum likelihood estimators (MLE) are intended to maximise the probability that the data points occur within a prescribed distribution. The procedure hence involves choosing a distribution or a class of distributions and then fitting the data to the distritbution such that the log probability of the data points are maximised by the parameters of the distributions. This will in turn give us the optimal estimators of the distribution for market invariants. MLE has been implemented for the following distributions:

- Multivariate Normal

### Shrinkage Estimators
Both nonparametric and MLE estimators require a large set of data and even then they might not produce the best estimators due to their inherent bias or lack there off. Akin to the bias-variance tradeoff in machine learning, too much bias and too much variance is not good in estimators. So, as a way to combine the two estimators, shrinkage was introduced. The idea is that you combine two weak estimators, one with high variance and the other with high bias, with some coefficient called the shrinkage coefficient, to produce a much better estimator. This is one of the cutting edge estimators and is still rigorously being researched. The following have been implemented:

- Manual Shrinkage
- Ledoit-Wolf Shrinkage
- Oracle Approximation
- Ledoit-Wolf Shrinkage for exponentially weighted moments
- Nonparametric and MLE manual shrinkage

## Efficient Frontier
Classical asset allocation is the efficient frontier allocation. This is also known as the mean-variance optimisation as it takes into account the estimators of the mean and variance. The procedure of optimisation involves choosing an utility function and optimising it for portfolio weights.

### Sharpe Ratio
One of the most commonly used metric for portfolios is the Sharpe ratio, which is essentially the risk adjusted excess return on a portfolio. We can use the sharpe ratio as the utility function to optimise for our portfolio weights. 

### Efficient Return
Another utility function is the efficient return, which is maximising return for a given risk tolerance, chosen by the investor. 

## Other Optimisations
Apart from the classical mean-variance optimisation, I have tried to implement some new research in the area. 

### Kelly Criterion
This optimisation takes insiration from Game Theory and proposes an optimal strategy for betting, given the estimation of the odds. 

### Higher Moment Optimisation
The main drawback of the mean-variance approach is the lack of consideration of higher moments such as skew and kurtosis, which can provide useful information. I have attempted to incorporate skew and kurtosis in the optimisation and I allow the investor to choose the wieghts he/she would like to place on each of the moments.

## Utility Functions
Here is a list of the utility functions available for usage:

- Sharpe Ratio
- Volatility
- Efficient Risk
- Efficient Return
- Kelly Criterion
- Higher Moment Utility

## Advances
The main advances implemented in this library are the nonparametric + MLE shirnkage, exponentially weighted moments and higher moment optimisation. 

## Roadmap
I have the following planned out and am working on implementing them:

- Market Invariants
  - Calculating invariants for bonds and derivatives

- Nonparametric Estimators
  - Different kernels for data
- Maximum Likelihood Estimators
  - Student-t Distribution
  - Stable Distributions
- Shrinkage Estimators
  - Optimal choosing of shrinkage for Nonparametric+MLE shrinkage
  - Shrinkage for higher moments

- Optimisations
  - CVaR optimisation
  - Monte Carlo simulations
 
## Contributing
I am always welcome to criticisms and suggestions regarding the library. 
  
