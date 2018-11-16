"""
The invariants module calculates the market invariants for various securities and
returns an usage dataframe of those invariants. So far invariants of these securities have been implemented:
- Stocks
- Forex
Future plans include fixed-income and derivatives market invariants.
"""

import numpy as np
import pandas as pd
import warnings


def stock_invariants(prices, no_assets):
    """
    Calculates stock price invariants, which are the compounded returns
    :param prices: stock prices data of the various tickers
    :type prices: pd Dataframe
    :param no_assets: number of assets in data
    :type no_assets: int
    :return: pd Dataframe of stock invariants
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not a pd Dataframe", RuntimeWarning)

    asset_ret = pd.DataFrame()
    for j in range(no_assets):
        returns = []
        for i in range(1, len(prices)):
            log_ret = np.log(prices.iloc[i, j] / prices.iloc[i-1, j])
            returns.append(log_ret)
        asset_ret = pd.concat([pd.DataFrame(returns), asset_ret], axis=1, ignore_index=True)
    return asset_ret


def forex_invariants(prices, no_assets):
    """
    Calculates forex price invariants, which are the compounded returns
    :param prices: stock prices data of the various tickers
    :type prices: pd Dataframe
    :param no_assets: number of assets in data
    :type no_assets: int
    :return: pd Dataframe of stock invariants
    """
    if not isinstance(prices, pd.DataFrame):
        warnings.warn("prices are not a pd Dataframe", RuntimeWarning)

    asset_ret = pd.DataFrame()
    for j in range(no_assets):
        returns = []
        for i in range(1, len(prices)):
            log_ret = np.log(prices.iloc[i, j] / prices.iloc[i-1, j])
            returns.append(log_ret)
        asset_ret = pd.concat([pd.DataFrame(returns), asset_ret], axis=1, ignore_index=True)
    return asset_ret
