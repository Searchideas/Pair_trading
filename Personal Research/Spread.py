import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Retrieval import DataRetrieval
import matplotlib.pyplot as plt
import scipy.optimize as optimize

class Spread:
    def __init__(self, merged_df):
        self.merged_df = merged_df

    def calculate_spread(self, ticker1, ticker2):
        # Regression to find n 
        ticker1 = self.merged_df[ticker1] 
        ticker2 = self.merged_df[ticker2] 
        log_prices1 = np.log(ticker1)
        log_prices2 = np.log(ticker2)
        model = LinearRegression()
        model.fit(ticker1.values.reshape(-1, 1), ticker2.values.reshape(-1, 1),)
        n = (model.coef_[0])[0]
        # Calculate the spread
        spread = log_prices1 - n*log_prices2
        return spread

    def compute_volatility(self, spread):
        vol = spread.std()
        mean = spread.mean()
        upper = mean+1.75*vol
        lower = mean-1.75*vol
        return vol, mean, upper,lower

    def plot_spread(self, spread, ticker1, ticker2, mean, upper,lower):
        plt.figure(figsize=(10, 6))
        plt.plot(spread)
        plt.axhline(y=mean, linestyle='-')
        plt.axhline(y=upper, linestyle='--', c='red')
        plt.axhline(y=lower, linestyle='--', c='red')
        plt.title('Spread between {} and {}'.format(ticker1, ticker2))
        plt.xlabel('Time')
        plt.ylabel('Spread')
        plt.grid(True)
        plt.show()
    

    def revert_time(self, spread):
         # Calculate mean and standard deviation
        mean = np.mean(spread)
        sd = np.std(spread)

        # Initialize lists to store time indices
        neg_to_mean_indices = []
        pos_to_mean_indices = []

        # Iterate over spread values to find indices where spread crosses mean + SD and mean - SD
        for i in range(1, len(spread)):
            if spread[i-1] < mean - sd and spread[i] >= mean - sd:
                neg_to_mean_indices.append(i)
            elif spread[i-1] > mean + sd and spread[i] <= mean + sd:
                pos_to_mean_indices.append(i)

        # Calculate time differences
        neg_to_mean_times = [i - neg_to_mean_indices[j-1] if j > 0 else i for j, i in enumerate(neg_to_mean_indices)]
        pos_to_mean_times = [i - pos_to_mean_indices[j-1] if j > 0 else i for j, i in enumerate(pos_to_mean_indices)]

        # Combine times and calculate total average time
        total_times = neg_to_mean_times + pos_to_mean_times
        total_average_time = np.mean(total_times)

        return total_average_time


    def calculate_spread_and_volatility(self, pair):
        stats_df = pd.DataFrame(columns=['Ticker 1', 'Ticker 2', 'Volatility', 'Mean', '90th Percentile', '10th Percentile', 'Revert Time'])
        if len(pair)== 1:
            ticker1, ticker2 = pair[0]
            if ticker1 in self.merged_df.columns and ticker2 in self.merged_df.columns:
                spread = self.calculate_spread(ticker1, ticker2)
                vol, mean, upper,lower = self.compute_volatility(spread)
                revert_time = self.revert_time(spread) # Add half-life 
                # Create a DataFrame to display the statistics
                stats_df.loc[len(stats_df)] = [ticker1, ticker2, vol, mean, upper, lower, revert_time]

                print("Statistics for the spread between {} and {}: ".format(ticker1, ticker2))
                print(stats_df)

                self.plot_spread(spread, ticker1, ticker2, mean, upper,lower)
            else:
                print(f"Error: One or both tickers '{ticker1}' and '{ticker2}' not found in the merged DataFrame.")
        else:
            for i in pair:
                ticker1, ticker2 = i
                if ticker1 in self.merged_df.columns and ticker2 in self.merged_df.columns:
                    spread = self.calculate_spread(ticker1, ticker2)
                    vol, mean, upper,lower = self.compute_volatility(spread)
                    revert_time = self.revert_time(spread) # Add half-life 

                    # Create a DataFrame to display the statistics
                    stats_df.loc[len(stats_df)] = [ticker1, ticker2, vol, mean, upper, lower, revert_time]

                    print("Statistics for the spread between {} and {}: ".format(ticker1, ticker2))
                    print(stats_df)

                    self.plot_spread(spread, ticker1, ticker2, mean, upper,lower)
                else:
                    print(f"Error: One or both tickers '{ticker1}' and '{ticker2}' not found in the merged DataFrame.")
        return stats_df