import pandas as pd 
import numpy as np 
import yfinance as yf 

start = '2010-01-10'
end = '2011-01-10'

tickers = ['AAPL', 'MSFT', 'CVX', 'XOM', 'AAL']

def get_data(tickers, interval, start, end):
    full_prices = pd.DataFrame()
    for i in tickers:
        df = yf.download(i, interval=interval, start=start, end=end)['Adj Close']
        print(f'{i} data downloaded...')
        
        full_prices = pd.concat([full_prices, df], axis=1, sort=False)

    full_prices.columns = tickers
    
    return full_prices

def get_asset_data():
    data = get_data(tickers, '1d', start, end)

    return data

def get_market_data():
    market_data = get_data(['SPY'], '1d', start, end)

    return market_data

