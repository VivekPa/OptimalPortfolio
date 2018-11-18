.. _market-invariants

#################
Market Invariants
#################

The first step in optimising portfolios is determining market invariants. A
market invariant is some entity that does not change over time. In other words,
given historical data, we can estimate the future with some notion of certainty,
or at least some notion of probability. Every security has it's own market
invariant.

.. automodule:: portfolioopt.invariants

  .. autofunction:: stock_invariants

    This function, given the stock prices, calculates the invariants, which
    are the compound returns of stocks:

    .. math::
      I=\ln(P_{i})-\ln(P_{i-1})

    A lot of study has shown that with the right distribution and estimators,
    compounded returns are an effective market invariant for stocks.

  .. autofunction:: forex_invariants

    This function, given the forex prices, calculates the invariants, which are
    the compound returns.

      .. math::
        I=\ln(P_{i})-\ln(P_{i-1})

      A lot of study has shown that with the right distribution and estimators,
      compounded returns are an effective market invariant for stocks.

  .. note::
    This module is being extended to incorporate derivatives and bonds.
