# Optimal Portfolio

<p align="left">
    <a href="https://www.python.org/">
        <img src="https://ForTheBadge.com/images/badges/made-with-python.svg"
            alt="python"></a> &nbsp;
    <a href="https://opensource.org/licenses/MIT">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square"
            alt="MIT license"></a> &nbsp;
</p>

**OptimalPortfolio** is an open source library for portfolio optimisation. This library implements classical portfolio optimisation techniques for equities, but is also extendable for non-equity products given the right adjustments in invariants. Furthermore, certain modern advances in portfolio optimisation, such as Hierarchical Risk Parity is also implemented. 

Regardless of whether you are a fundamental investor, or an algorithmic trader, this library can aid you in allocating your capital in the most risk efficient way, allowing to optimise your utility. *For more details on the project design and similar content, please check out [Engineer Quant](https://medium.com/engineer-quant)*

*Disclaimer: This is not trading or investment advice. Trading involves significant risk and do so at your risk.*


## Contents
- [Contents](#contents)
- [Overview](#overview)
- [Full Sequence](#full-sequence)
- [Functionality](#functionality)
  - [Expected Returns](#expected-returns)
  - [Risk Models](#risk-models)
  - [Objective Functions](#objective-functions)
  - [Constraints](#constraints)
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
```bash
bash install.sh
```

## Full Sequence
The pipeline for using this library is delibrately modular, so as to allow users to incorporate their own proprietary stacks and code into the optimisation. The full sequence from data to weights is in the ``examples/full_sequence.ipynb`` notebook, but for clarity, it is

- Acquire stock (or other asset class) data
- Calculate market invariants (for stocks is returns)
- Calculate moments of invariants (for simple mean-variance optimisation it is mean and covariance)
- Optimise weights according to a utility function, subject to constraints, with moments of invariants as inputs

```python
import pandas as pd
import numpy as np

import PortOpt.invariants as invs
import PortOpt.moment_est as moments
from PortOpt.opt_allocations import Optimiser
import PortOpt.utility_functions as utils

tickers = ['AAPL', 'MSFT', 'CVX', 'GE', 'GOOGL']
stock_data = utils.get_data(tickers, '1d', '2015-01-01', '2021-01-01')

# Compute market invariants

stock_returns = invs.stock_invariants(stock_data)

riskmodel = moments.RiskModel(tickers)
stock_cov = riskmodel.avg_hist_cov(stock_data)

# Optimise weights according to mean-variance

optimiser = Optimiser(tickers, exp_returns, stock_cov)
weights = optimiser.mean_variance(threshold=0.1, type='variance')
print(weights)
weight_tearsheet(weights)
```

This should have the following output:
```txt
AAPL     0.182566
MSFT     0.143655
CVX      0.190582
GE       0.117543
GOOGL    0.365654

Annual Return: 15.026
Annual Volatility: 23.203
Annual Sharpe: 0.561
```

## Functionality

Listed below is the current overall functionality of the library, split into the various parts of the pipeline.

### Expected Returns
- Nonparametric
  - Mean Historical Returns
  - Mean Exponentially Weighted Returns
  - Capital Asset Pricing Model (CAPM)
- Shrinkage
  - James-Stein Mean Shrinkage

### Risk Models
- Nonparametric
  - Historical Covariance
  - Exponentially Weighted Covariance
- Covariance Shrinkage
    - Identity Shrinkage
    - Scaled Variance Shrinkage
    - Ledoit-Wolf Single Index
    - Ledoit-Wolf Constant Correlation

### Objective Functions
- Mean-Variance Optimisation
  - Maximise Returns with a risk limit
  - Minimise risk with a returns limit
- Minimum Volatility
- Maximum Sharpe Ratio

### Constraints
- Long Short Neutral
- Maximum Position on an asset

### Adding Custom Constraints/Objectives
Since constraints are very case specific, I have added a functionality to be able to add your own constraints to the optimisation problem. We add a long/short neutral constraint using the custom method to demonstrate

```python
optimiser = Optimiser(tickers, exp_returns, stock_cov)

# Add LS neutral constraint
optimiser.add_constraints([lambda x: cp.sum(x) == 0])

# Optimise for mean-variance
weights = optimiser.mean_variance(threshold=0.1, type='variance')
print(weights)
weight_tearsheet(weights)
```

## Market Invariants
The first step to optimising any portfolio is calculating market invariants. Market invariants are defined as aspects of market prices that have some determinable statistical behaviour over time. For stock prices, the compounded returns are the market invariants. So when we calculate these invariants, we can statistically model them and gain useful insight into their behaviour. So far, calculating market invariants of stock prices. The same can be implemented for options and bonds, but data acquisition is an issue.

## Moment Estimation
Once the market invariants have been calculated, it is time to model the statistical properties of the invariants. This is an actively researched and studied field and due to the nature of the complexity involved in modelling the statistical properties of large market data, there are several limitations in estimating the moments of the distributions.

### Nonparametric Estimators
The simplest method of estimating the mean and covariance of invariants are the sample mean and covariance. However, this can be extended by introducing weightage for the timestamps, i.e giving more weight to recent data than older data. One interesting approach I have taken is introducing exponentially weighted mean and covariance, which I read about [here](https://reasonabledeviations.science/2018/08/15/exponential-covariance/).

<!-- ### Maximum Likelihood Estimators
Maximum likelihood estimators (MLE) are intended to maximise the probability that the data points occur within a prescribed distribution. The procedure hence involves choosing a distribution or a class of distributions and then fitting the data to the distribution such that the log probability of the data points are maximised by the parameters of the distributions. This will in turn give us the optimal estimators of the distribution for market invariants. MLE has been implemented for the following distributions:

- Multivariate Normal
- Multivariate Student t

The MLE estimate for Student-t distribution is computed using Expectation Maximisation (EM)
algorithm.  -->

### Shrinkage Estimators
Nonparametric estimators only converge to the population estimate as the number of independent, identically distribution (IID) data points tends to infinity. However, as anyone who has experimented with financial data will be aware, market data is everchaning, and in some cases the volume of data is minimal. In these circumstances, nonparametric estiamtors do not work as well as intended, and this will cause issues later on in the optimiser. 

One method to circumvent this is to impose some known structure in the market data. This is the essence of shrinkage estimation. We _shrink_ the sample moment towards a structure that we know _a-priori_. 

An example of shrinkage will be Ledoit-Wolf Single-Index Model Covariance Shrinkage. This approach assumes that stock returns are significantly influenced by market returns. So we model the stock returns as a linear regression of market returns

$$
\hat{r}_{i,t} = \alpha_{i} + \beta_{t} \hat{r}_{m, t} + \epsilon_{i,t} 
$$

We calculate $\beta$ and $\alpha$ from regression, and we assume the error $\epsilon$ is independent and normally distributed, i.e. $Cov(\epsilon_{i}, \epsilon_{j}) = 0$, $Cov(\hat{r}_m, \epsilon_{i}) = 0$, and $E[\epsilon] = 0$, $Var(\epsilon) = \sigma_{i}^2$. Given this, we let the shrinkage matrix be 

$$
F = \beta \beta^{T} \hat{\sigma}_{m} + \Sigma_{\epsilon}
$$
where $\Sigma_{\epsilon}$ is the diagonal matrix of error variances. We can now calculate the shrunk covariance matrix as 

$$
\Sigma = \alpha F + (1-\alpha) S
$$

We can go a step further by choosing $\alpha$ optimally, but for the sake of brevity, I will leave that to the interested reader to find. The source papers are in the \references directory for those interested. 

## Optimal Allocations
Classical asset allocation is the efficient frontier allocation. This is also known as the mean-variance optimisation as it takes into account the estimators of the mean and variance. The procedure of optimisation involves choosing an utility function and optimising it for portfolio weights. So far, I have implmented Mean-Variance, Minimum Volatility and Maximum Sharpe Optimisation. 

### Mean-Variance
Mean-variance optimisation can take on two forms: maximising returns with a variance limit, or minimising variance with a return limit. Both forms call the same ``mean_variance()`` method in the ``Optimiser`` class.

### Minimum Volatility
Minimum volatility optimisation minimises volatility whilst constraining the weights to fit a certain portfolio profile (Long only, Long/Short)

### Maximum Sharpe 
Maximum Sharpe optimisation requires a little more nuance as maximising sharpe ratio is not a convex optimisation problem. Hence, we have to do a variable transformation, with a few assumptions, to convert the problem into a convex optimisation one. 

<!-- ### Higher Moment Optimisation
The core principle of optimisation with higher moments is identical to any other optimisation: given some utility function and constraints, find the weights of each of the portfolio entries such that the utility function is maximised. The only difference is that the utility function in this case would contain as arguments, higher moments. Furthermore, by adding coefficients to each moment, we are able to take into account investor risk aversion and preferences.
This version of the package includes higher moment optimization based on higher co-moments, which makes much more statistical sense than the column-wise higher order moments in the original package.  -->

<!-- ### Compared to Sharpe Ratio
When doing backtests, higher moment optimisation works better than using Sharpe ratio to optimise allocations. -->

## Roadmap
I have the following planned out and am working on implementing them:

<!-- - Market Invariants
  - Calculating invariants for bonds and derivatives -->

- Nonparametric Estimators
  - Exponentially weighted skew and kurtosis
<!-- - Maximum Likelihood Estimators
  - Student-t Distribution
  - Stable Distributions -->
- Shrinkage Estimators
  - Shrinkage for higher moments
  - Optimal shrinkage coefficient for custom shrinkage matrices

- Optimisations
  - Extending Hierarchical Risk Parity
  - Higher Moment Optimisation
  - Backtesting optimisation
