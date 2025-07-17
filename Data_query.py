import pandas as pd 
import numpy as np 
import yfinance as yf
from yahoofinancials import YahooFinancials
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import itertools
from pandas_datareader import data as pdr
from yahooquery import Ticker


class data_query:
    """Class to pull and process data from Yahoo Finance"""

    @staticmethod
    def pull_data(ticker, start, end, interval='1d'):
        data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):  # Drop level if exists
            data.columns = data.columns.droplevel('Ticker') 
        return data

    @staticmethod
    def remove_useless(data, ticker):
        data = data.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume'], errors='ignore')
        data = data.rename(columns={'Adj Close': ticker})
        return data

    @staticmethod
    def adj_close(ticker, start, end, interval='1d'):
        data = pd.DataFrame()

        if isinstance(ticker, list) or isinstance(ticker, pd.Series):
            for item in ticker:
                temp = data_query.pull_data(item, start, end, interval)
                temp = data_query.remove_useless(temp, item)
                data = pd.merge(data, temp, left_index=True, right_index=True, how='outer') if not data.empty else temp
        else:
            data = data_query.pull_data(ticker, start, end, interval)
            data = data_query.remove_useless(data, ticker)
        
        data.fillna(method='ffill', inplace=True)
    
        return data
