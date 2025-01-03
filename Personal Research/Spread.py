import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Retrieval import DataRetrieval
import matplotlib.pyplot as plt
class Spread:
    def __init__(self, merged_df):
        self.merged_df = merged_df

    def calculate_spread(self, ticker1, ticker2):
        normalized_ticker1 = self.merged_df[ticker1] / self.merged_df[ticker1].iloc[0]
        normalized_ticker2 = self.merged_df[ticker2] / self.merged_df[ticker2].iloc[0]
        spread = normalized_ticker1 - normalized_ticker2
        return spread

    def compute_volatility(self, spread):
        vol = spread.std()
        mean = spread.mean()
        upper = mean+2*vol
        lower = mean-2*vol
        return vol, mean, upper,lower

    def plot_spread(self, spread, ticker1, ticker2, mean, upper,lower):
        plt.figure(figsize=(10, 6))
        plt.plot(spread)
        plt.axhline(y=mean, linestyle='-')
        plt.axhline(y=upper, linestyle='--')
        plt.axhline(y=lower, linestyle='--')
        plt.title('Spread between {} and {}'.format(ticker1, ticker2))
        plt.xlabel('Time')
        plt.ylabel('Spread')
        plt.grid(True)
        plt.show()

    def calculate_spread_and_volatility(self, pair):
        ticker1, ticker2 = pair
        if ticker1 in self.merged_df.columns and ticker2 in self.merged_df.columns:
            spread = self.calculate_spread(ticker1, ticker2)
            vol, mean, upper,lower = self.compute_volatility(spread)

            # Create a DataFrame to display the statistics
            stats_df = pd.DataFrame({
                'Statistic': ['Volatility', 'Mean', '90th Percentile', '10th Percentile'],
                'Value': [vol, mean, upper,lower]
            })

            print("Statistics for the spread between {} and {}: ".format(ticker1, ticker2))
            print(stats_df)

            self.plot_spread(spread, ticker1, ticker2, mean, upper,lower)
        else:
            print(f"Error: One or both tickers '{ticker1}' and '{ticker2}' not found in the merged DataFrame.")
