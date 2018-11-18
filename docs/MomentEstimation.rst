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
of the data points [1]_, and find a suitable estimator for a specified distribution.
Formally, we can write maximum likelihood estimators to be

.. math::
  (\hat{\mu}, \hat{\Sigma}) = \argmax_{\mu, \Sigma}\bigg[\sum_{i=1}^{n}\ln(f_{X}(x_{i}))\bigg]


.. autoclass:: MLE
  :members:

  .. automethod:: __init__




Shrinkage Estimators
====================

The concept of shrinkage is very simple. Combine two estimators, one with high
variance and one with high bias, to obtain an estimator that has the best qualities
of both the individual estimators. For those familiar with machine learning, it
akin to ensemble learning with the idea of combining weak learners to make one
strong learner. For those interested to learn more, do check out the references [2]_-[4]_.

.. autoclass:: Shrinkage
  :members:

  .. automethod:: __init__


References
==========

.. [1] XIAO-LI MENG, DONALD B. RUBIN; Maximum likelihood estimation via the ECM algorithm:
 A general framework, Biometrika, Volume 80, Issue 2, 1 June 1993, Pages 267–278,
 https://doi.org/10.1093/biomet/80.2.267
.. [2] Ledoit, O., & Wolf, M. (2003). `Honey, I Shrunk the Sample Covariance Matrix
 <http://www.ledoit.net/honey.pdf>`_ The Journal of Portfolio Management,
 30(4), 110–119. https://doi.org/10.3905/jpm.2004.110
.. [3] Ledoit, O., & Wolf, M. (2001). `Improved estimation of the covariance matrix
 of stock returns with an application to portfolio selection
  <http://www.ledoit.net/ole2.pdf>`_, 10, 603–621.
.. [4] Ledoit, O., & Wolf, M. (2004) `A Well-Conditioned Estimator for Large-Dimensional
 Covariance Matrices <http://perso.ens-lyon.fr/patrick.flandrin/LedoitWolf_JMA2004.pdf>`_,
 Journal of Multivariate Analysis, 88(2), 365-411
