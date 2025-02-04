import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Retrieval import DataRetrieval
from Spread import Spread
import matplotlib.pyplot as plt
import scipy.optimize as optimize


class Backtest:
    def __init__(self, fpair, merged_df, hedge_table):
        self.pairs = fpair
        self.merged_df = merged_df
        self.hedge = hedge_table

    def signal(self, ticker1, ticker2):
        spread_calculator = Spread(self.merged_df)
        spread = spread_calculator.calculate_spread(ticker1, ticker2)
        vol, mean, upper, lower = spread_calculator.compute_volatility(spread)

        signals = []
        for date, value in spread.items():
            if value > upper:
                signals.append({'Date': date, 'Signal': 'Sell'})
            elif value < lower:
                signals.append({'Date': date, 'Signal': 'Buy'})

        signals_df = pd.DataFrame(signals)
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        signals_df.set_index('Date', inplace=True)
        return signals_df

    def position(self):
        for i, row in self.pairs.iterrows():
            ticker1 = row['Ticker 1']
            ticker2 = row['Ticker 2']
            signals_df = self.signal(ticker1, ticker2)
            ticker1_df = self.merged_df[ticker1].to_frame()
            merged_df = ticker1_df.join(signals_df, how='left')
            print(merged_df)
            return merged_df