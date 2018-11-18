.. _moment-estimation:

#################
Moment Estimation
#################

The core of optimising portfolios is determining the moments of the distribution
of market invariants. This is a well studied problem and there are several
methods to estimate the moments of the market invariants.

.. automodule:: portfolioopt.moment_est

  .. autofunction:: sample_mean

    This is the textbook method for estimating the mean of market invariants:

    .. math::
      \hat{\mu}:=\frac{1}{n}\sum_{i=1}^{n}x_{i}

    Although intuitive, this estimator has a number of issues. First, the
    number of data points required to produce an 'good' estimate is large.
    Second, the estimator is very sensitive to the data (has high variance).
    Therefore, this estimator is not the most effective.

    .. note::
      This should not be the estimator that you use. Opt for more robust
      estimators such as maximum likelihood or shrinkage.
  .. autofunction:: sample_cov

    This is the textbook method for estimating the covariance of market invariants:

      .. math::
        \hat{\Sigma}:=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\hat{\mu})(x_{i}-\hat{\mu})^{T}

    Although intuitive, this estimator has a number of issues. First, the
    number of data points required to produce an 'good' estimate is large.
    Second, the estimator is very sensitive to the data (has high variance).
    Therefore, this estimator is not the most effective.

    .. note::
      This should not be the estimator that you use. Opt for more robust
      estimators such as maximum likelihood or shrinkage.

  .. autofunction:: sample_skew

    Calculates the sample skew of the market invariants.

  .. autofunction:: sample_kurt

    Calculates the sample kurtosis of the market invariants.

  .. autofunction:: sample_moment

    Calculates the nth moment of the sample of market invariants.

  .. autofunction:: exp_mean

    Calculates the exponentially weighted sample mean of market invariants.

  .. autofunction:: exp_cov

    Calculates the exponentially weighted sample covariance of market invariants.

Maximum Likelihood Estimators
=============================

Maximum likelihood estimators aim to maximise the log probability of the occurrences
of the data points, and find a suitable estimator for a specified distribution.

.. autoclass:: MLE
  :members:

  .. automethod:: __init__

  .. automethod:: norm_est

    Uses MLE to estimate the mean, covariance, skew and kurtosis of market invariants, assuming
    their come from the normal distribution.

Shrinkage Estimators
====================

The concept of shrinkage is very simple. Combine two estimators, one with high
variance and one with high bias, to obtain an estimator that has the best qualities
of both the individual estimators. For those familiar with machine learning, it
akin to ensemble learning with the idea of combining weak learners to make one
strong learner.

.. autoclass:: Shrinkage
  :members:

  .. automethod:: __init__

  .. automethod:: shrunk_covariance

  .. automethod:: ledoit_wolf

  .. automethod:: oracle_approximation

  .. automethod:: exp_ledoit

  .. automethod:: param_mle
