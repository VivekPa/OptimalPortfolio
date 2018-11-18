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
