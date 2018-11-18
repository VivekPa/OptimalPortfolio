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

**PortfolioAnalytics** is a robust open source library for portfolio optimisation and analytics. This library implements classical portfolio optimisation methods such as Efficient Frontier, Sharpe ratio and Mean Variance, along with modern developments in the field such as Shrinkage estimators, Maximum likelihood estimators and Kelly Criterion. It also implements a novel shrinkage estimator and optimisation with higher moments. 

Regardless of whether you are a fundamental investor, or an algorithmic trader, this library can aid you in allocating your capital in the most risk efficient way, allowing to optimise your utility. 

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
df = pd.read_csv("stock_prices.csv", parse_dates=True, index_col="date")

# Calculate invariants and estimate mean and covariance
invariants = inv.stock_invariants(df, 20)
mu = mest.sample_mean(invariants)
cov = mest.sample_cov(invariants)

# Optimise using the Sharpe Ratio
frontier = fron.EfficientFrontier(20, mu, cov, list(df.columns), gamma=0)
print(frontier.max_sharpe())
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
The simplest method of estimating the mean and covariance of invariants are the sample mean and covariance. However, this can be extended by introducing weightage for the timestamps, i.e giving more weight to recent data than older data. One novel approach I have taken is introducing exponentially weighted mean and covariance, which intutively has backing. 

### Maximum Likelihood Estimators
Maximum likelihood estimators (MLE) are intended to maximise the probability that the data points occur within a prescribed distribution. 
