# Optimal Portfolio

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp;
</p>

**OptimalPortfolio** is an open source library for portfolio optimisation. This library extends classical portfolio optimisation methods for equities, options and bonds. Unlike modern portfolio theory (MPT), OptimalPortfolio takes into account the skew and kurtosis of the distribution of market invariants. Furthermore, novel methods of finding estimators of moments of the distribution is implemented.

Regardless of whether you are a fundamental investor, or an algorithmic trader, this library can aid you in allocating your capital in the most risk efficient way, allowing to optimise your utility. *For more details on the project design and similar content, please check out [Engineer Quant](https://medium.com/engineer-quant)*

*Disclaimer: This is not trading or investment advice. Trading involves significant risk and do so at your risk.*


## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Market Invariants](#market-invariants)
- [Moment Estimation](#moment-estimation)
    - [Nonparametric Estimators](#nonparametric-estimators)
    - [Maximum Likelihood Estimators](#maximum-likelihood-estimators)
    - [Shrinkage Estimators](#shrinkage-estimators)
- [Optimal Allocations](#optimal-allocations)
    - [Higher Moment Optimisation](#higher-moment-optimisation)
    - [Compared to Sharpe Ratio](#compared-to-sharpe-ratio)
- [Roadmap](#roadmap)

## Overview
This library aims to make optimising portfolios accessible to every trader and investor. To install this library, download it and run
```python
run setup.py
```


## Market Invariants
The first step to optimising any portfolio is calculating market invariants. Market invariants are defined as aspects of market prices that have some determinable statistical behaviour over time. For stock prices, the compounded returns are the market invariants. So when we calculate these invariants, we can statistically model them and gain useful insight into their behaviour. So far, calculating market invariants of stock prices and forex prices have been implemented. I plan to calculate invariants for options and bonds but data acquisition is difficult.

## Moment Estimation
Once the market invariants have been calculated, it is time to model the statistical properties of the invariants. This is an actively researched and studied field and due to the nature of the complexity involved in modelling the statistical properties of large market data, there are several limitations in estimating the moments of the distributions. I have tried implementing cutting edge research in shrinkage estimators.

### Nonparametric Estimators
The simplest method of estimating the mean and covariance of invariants are the sample mean and covariance. However, this can be extended by introducing weightage for the timestamps, i.e giving more weight to recent data than older data. One interesting approach I have taken is introducing exponentially weighted mean and covariance, which I read about [here](https://reasonabledeviations.science/2018/08/15/exponential-covariance/). I am now working on implementing exponential weightage for skew and kurtosis.

### Maximum Likelihood Estimators
Maximum likelihood estimators (MLE) are intended to maximise the probability that the data points occur within a prescribed distribution. The procedure hence involves choosing a distribution or a class of distributions and then fitting the data to the distribution such that the log probability of the data points are maximised by the parameters of the distributions. This will in turn give us the optimal estimators of the distribution for market invariants. MLE has been implemented for the following distributions:

- Multivariate Normal
- Multivariate Student t

The MLE estimate for Student-t distribution is computed using Expectation Maximisation (EM)
algorithm. 

### Shrinkage Estimators
Both nonparametric and MLE estimators require a large set of data and even then they might not produce the best estimators due to their inherent bias or lack there off. Akin to the bias-variance tradeoff in machine learning, too much bias and too much variance is not good in estimators. So, as a way to combine the two estimators, shrinkage was introduced. The idea is that you combine two weak estimators, one with high variance and the other with high bias, with some coefficient called the shrinkage coefficient, to produce a much better estimator. This is one of the cutting edge estimators and is still rigorously being researched. I have implemented shrinkage of nonparametric (exponential) estimates with MLE (student-t) estimates, with manual shrinkage. Working on finding the optimal shrinkage coefficient.

## Optimal Allocations
Classical asset allocation is the efficient frontier allocation. This is also known as the mean-variance optimisation as it takes into account the estimators of the mean and variance. The procedure of optimisation involves choosing an utility function and optimising it for portfolio weights. However, it struggles to capture the fat tail behaviour and skewness of the market prices.

### Higher Moment Optimisation
The core principle of optimisation with higher moments is identical to any other optimisation: given some utility function and constraints, find the weights of each of the portfolio entries such that the utility function is maximised. The only difference is that the utility function in this case would contain as arguments, higher moments. Furthermore, by adding coefficients to each moment, we are able to take into account investor risk aversion and preferences.
This version of the package includes higher moment optimization based on higher co-moments, which makes much more statistical sense than the column-wise higher order moments in the original package. 

### Compared to Sharpe Ratio
When doing backtests, higher moment optimisation works better than using Sharpe ratio to optimise allocations.

## Roadmap
I have the following planned out and am working on implementing them:

- Market Invariants
  - Calculating invariants for bonds and derivatives

- Nonparametric Estimators
  - Exponentially weighted skew and kurtosis
- Maximum Likelihood Estimators
  - Student-t Distribution
  - Stable Distributions
- Shrinkage Estimators
  - Optimal choosing of shrinkage for Nonparametric+MLE shrinkage
  - Shrinkage for higher moments

- Optimisations
  - Copula based CVaR optimisation
  - Monte Carlo simulations
