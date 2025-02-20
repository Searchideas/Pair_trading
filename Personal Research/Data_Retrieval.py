import pandas as pd 
import numpy as np 
import yfinance as yf
from yahoofinancials import YahooFinancials
import datetime as dt

class DataRetrieval:
    """
    A class used to retrieve financial data from Yahoo Finance.
    """

    def __init__(self, ticker):
        """
        Initialize the DataRetrieval class.

        Args:
            ticker (str): The ticker symbol of the stock.
        """

        self.ticker = ticker


    def pull_data(self, start_date, end_date):
        """
        Retrieve historical data for the specified ticker.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pandas.DataFrame: A DataFrame containing the historical data.
        """
        try:
            end_date = end_date
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=True,auto_adjust=False,multi_level_index=False)
            data = data['Adj Close']
            data = data.rename(f"{self.ticker}")
            data.fillna(method='ffill', inplace=True)
            return data
        except Exception as e:
            print(self.ticker+"ticker error")
        
    def merge_data(dataframes, suffixes=None, how="outer"):
        """
        Merge historical data from multiple tickers.

        Args:
            dataframes (list): A list of pandas DataFrames containing historical data.
            suffixes (list, optional): A list of suffixes to append to overlapping column names. Defaults to None.
            how (str, optional): The type of merge to perform. Defaults to "outer".

        Returns:
            pandas.DataFrame: A merged DataFrame containing historical data from all tickers.
        """
        merged_data = dataframes[0]
        for i, dataframe in enumerate(dataframes[1:]):
            suffix = f"_{i+1}" if suffixes is None else suffixes[i]
            merged_data = pd.merge(merged_data, dataframe, left_index=True, right_index=True, suffixes=(None, suffix), how='outer')
        merged_data.fillna(method='ffill', inplace=True)
        return merged_data
    
    def retrieve_and_merge_data(tickers, start_date, end_date):
        """
        Retrieve historical data for multiple tickers and merge the results.

        Args:
            tickers (list): A list of ticker symbols.
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            pandas.DataFrame: A merged DataFrame containing historical data for all tickers.
        """
        data_retrieval_instances = [DataRetrieval(ticker) for ticker in tickers if ticker is not None]
        dataframes = [instance.pull_data(start_date, end_date) for instance in data_retrieval_instances]
        dataframes = [df for df in dataframes if df is not None]  # Remove None values
        merged_df = DataRetrieval.merge_data(dataframes, suffixes=[f"_{ticker}" for ticker in tickers])
        merged_df = merged_df.drop(merged_df.filter(regex='_').columns, axis=1)
        return merged_df
