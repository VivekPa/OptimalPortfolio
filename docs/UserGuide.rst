.. _user-guide

##########
User Guide
##########

Portfolio Optimisation has the following pipeline:

- Derive market invariants
- Statistically model invariants
- Optimise for certain utility function

This can be achieved by the following code template:


.. code-block:: python
  import pandas as pd
  import portfolioopt.invariants as inv
  import portfolioopt.moment_est as mest
  import portfolioopt.eff_frontier as fron

  # Load stock prices, from whichever file 
  df = pd.read_csv("stock_data.csv", parse_dates=True, index_col="date")

  # Calculate invariants and estimate mean and covariance, using the 
  # given methods in the class
  invariants = inv.stock_invariants(df, 20)
  mu = mest.sample_mean(invariants)
  cov = mest.sample_cov(invariants)

  # Optimise using preferred utility function
  frontier = fron.EfficientFrontier(20, mu, cov, list(df.columns), gamma=0)
  print(frontier.maximise_sharpe())
  frontier.portfolio_performance(verbose=True)


It is that simple. 
