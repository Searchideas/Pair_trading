import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Retrieval import DataRetrieval
from Spread import Spread
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from IPython.display import display
import pyfolio as pf
from scipy import stats


class Strategy:
    def __init__(self, fpair, merged_df, hedge_table):
        self.pairs = fpair
        self.merged_df = merged_df
        self.hedge = hedge_table
    
    def zscore(self, spread):
        spread= stats.zscore(spread)
        return spread
    
    def create_signal(self, ticker1, ticker2):
        spread_calculator = Spread(self.merged_df)
        spread = spread_calculator.calculate_spread(ticker1, ticker2)
        spread = self.zscore(spread)  # Use self to access instance methods
        # Logic is that when the value is above 2.0 then sell ticker1 and if below -2.0 buy ticker2
        signals = []
        for date, value in spread.items():
            if value > 2.0:
                signals.append({'Date': date, 'Signal': 'Sell'})
            elif value < -2.0:
                signals.append({'Date': date, 'Signal': 'Buy'}) 
                signals_df = pd.DataFrame(signals)
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
        signals_df.set_index('Date', inplace=True)
        spread =spread.to_frame()
        spread= spread.rename(columns={0:'Spread'})
        signals_df = spread.join(signals_df,how="outer")
        return signals_df

    def adaptive_multiplier(self, signal_df):
        min_z = signal_df['Spread'].min()
        max_z = signal_df['Spread'].max()

        def volatility_mutiplier(self,signal_df):
            # Calculate the mean volatility
            mean_volatility = signal_df['Spread'].std()
            signal_df['Volatility Mutiplier'] = mean_volatility/signal_df['Spread'].rolling(window=20).std()
            return signal_df

        # determine the scaling factor, my way of scaling is by Z-score [-2-Z]/[-2-min]
        def calculate_k(row):
            if row['Signal'] == 'Sell' and 2.0 < abs(row['Spread']) < 2.5:
                k = min(1.0, ((row['Spread'] - 2.0) / (max_z - 2.0))*row['Volatility Mutiplier'])
            elif row['Signal'] == 'Sell' and 2.5 < abs(row['Spread']) :
                k = min(1.2, ((row['Spread'] - 2.0) / (max_z - 2.0))*row['Volatility Mutiplier'])
            elif row['Signal'] == 'Buy' and -2.5 < (row['Spread']) < -2.0:
                k = min(1.0, (-(row['Spread'] - 2.0) / (-2.0 - min_z))*row['Volatility Mutiplier'])
            elif row['Signal'] == 'Buy' and  (row['Spread']) < -2.5:
                k = min(1.2, (-(row['Spread'] - 2.0) / (-2.0 - min_z))*row['Volatility Mutiplier'])
            else:
                k = 0
            return k

        # Apply conditions row-wise using the apply function
        signal_df = volatility_mutiplier(self,signal_df)
        signal_df['K']=signal_df.apply(calculate_k, axis=1)
        return signal_df


    def position(self,ticker1,ticker2):
        signal_df = self.create_signal(ticker1, ticker2)
        signal_df= self.adaptive_multiplier(signal_df)
        ticker1_df = self.merged_df[ticker1].to_frame()
        Price_df = ticker1_df.join(signal_df, how='outer')
        Price_df = Price_df.join(self.merged_df[ticker2].to_frame(), how='inner')
        hedge_ratio = self.hedge.loc[(self.hedge['ticker1'] == ticker1) | (self.hedge['ticker2'] == ticker2), 'hedge ratio'].values[0]

        Price_df[str(ticker1) + "'s trade"] = np.nan
        Price_df[str(ticker2) + "'s trade"] = np.nan
        Price_df[str(ticker1) + "'s position"] = np.nan
        Price_df[str(ticker2) + "'s position"] = np.nan
        Price_df[ticker1] = pd.to_numeric(Price_df[ticker1], errors='coerce')
        Price_df[ticker2] = pd.to_numeric(Price_df[ticker2], errors='coerce')
        Price_df["Cash"] = np.nan
        Price_df["Leverage"] = np.nan
        Price_df['Number of trades']=np.nan
        Price_df['Status']='Closed'

        cash = 5000000 #Set initial amount (NTD)
        leverage = 0
        gross_exposure, ticker1_position,ticker2_position,Num_trade= 0,0,0,0
        Price_df['Cash'].iloc[0] = cash
        Price_df['Number of trades'].iloc[0] = 0 
        for index, row in Price_df.iterrows():
            #Limit cash to 5% of leverage cash amount
            limit = min((cash*2)*0.05,(cash*2-gross_exposure)*0.05)*row['K']
            n = round(limit/(1* row[ticker1]+ hedge_ratio*row[ticker2]),0)
            if row['Signal'] == 'Buy':
                long_qty,short_qty = n, -n*hedge_ratio

                Price_df.at[index, str(ticker1) + "'s trade"] = long_qty
                Price_df.at[index, str(ticker2) + "'s trade"] = short_qty
        
                ticker1_position += long_qty
                ticker2_position += short_qty
                cash -= (long_qty* row[ticker1]+ short_qty*row[ticker2])
                Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                Price_df.at[index, "Cash"] = cash 
                gross_exposure = abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2]
                Price_df.at[index,"Leverage"] = (abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2]) / (cash + abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2])
                Price_df.at[index,"Portfolio's Value"] = cash + ticker1_position * row[ticker1]+ticker2_position * row[ticker2]
                Num_trade += 1
                Price_df.at[index,"Number of trades"] = Num_trade
                Price_df.at[index,"Status"] = 'Initiated'

            elif row['Signal'] == 'Sell':
                gross_exposure += (n*row[ticker1] + hedge_ratio*n*row[ticker2])
                short_qty, long_qty = -n, n*hedge_ratio

                
                Price_df.at[index, str(ticker1) + "'s trade"] = short_qty
                Price_df.at[index, str(ticker2) + "'s trade"] = long_qty
                cash -= (short_qty* row[ticker1]+ long_qty*row[ticker2])
                ticker1_position += short_qty
                ticker2_position += long_qty
                
                Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                Price_df.at[index, "Cash"] = cash 
                gross_exposure = abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2]
                Price_df.at[index,"Leverage"] =  (abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2]) / (cash + abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2])
                Price_df.at[index,"Portfolio's Value"] = cash + ticker1_position * row[ticker1]+ticker2_position * row[ticker2]
                Num_trade += 1
                Price_df.at[index,"Number of trades"] = Num_trade
                Price_df.at[index,"Status"] = 'Initiated'


            elif np.isnan(row['Signal']):
                prev_row = Price_df.iloc[Price_df.index.get_loc(index) - 1]
                #((float(prev_row['Spread']) > 0 and float(row['Spread']) < 0) or (float(prev_row['Spread']) < 0 and float(row['Spread']) > 0))
                if abs(row['Spread'])<0.5 and ticker1_position> 0 and index != 0:
                    Price_df.at[index, str(ticker1) + "'s trade"] = -prev_row[str(ticker1) + "'s position"]
                    Price_df.at[index, str(ticker2) + "'s trade"] = -prev_row[str(ticker2) + "'s position"]
                    cash-= Price_df.at[index, str(ticker1) + "'s trade"]*row[ticker1] + Price_df.at[index, str(ticker2) + "'s trade"]*row[ticker2]
                    ticker1_position = 0
                    ticker2_position = 0
                    Price_df.at[index, "Cash"] = cash 
                    
                    Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                    Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                    gross_exposure = abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2]
                    Price_df.at[index,"Leverage"] = max(1,gross_exposure/cash)
                    Price_df.at[index,"Portfolio's Value"] = cash + ticker1_position * row[ticker1]+ticker2_position * row[ticker2]
                    Num_trade += 1
                    Price_df.at[index,"Number of trades"] = Num_trade
                    Price_df.at[index,"Status"] = 'Closed'
                else:
                    Price_df.at[index, str(ticker1) + "'s trade"] = 0
                    Price_df.at[index, str(ticker2) + "'s trade"] = 0
                    Price_df.at[index, "Cash"] = cash 
                    Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                    Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                    gross_exposure = abs(ticker1_position) * row[ticker1] + abs(ticker2_position) * row[ticker2]
                    Price_df.at[index,"Leverage"] = max(1,gross_exposure/cash)
                    Price_df.at[index,"Portfolio's Value"] = cash + ticker1_position * row[ticker1]+ticker2_position * row[ticker2]
                    Price_df.at[index,"Number of trades"] = Num_trade
                    Price_df.at[index, "Status"] = prev_row['Status']
        return Price_df
        
    def signal_plot(self,Price_df, hedge_ratio):
        Columns = Price_df.columns
        ticker1 = Columns[0]
        ticker2 = Columns[5]

        hedge_ratio = self.hedge.loc[(self.hedge['ticker1'] == ticker1) | (self.hedge['ticker2'] == ticker2), 'hedge ratio'].values[0]

        # Calculate the log-transformed values
        log_ticker1 = np.log(Price_df[ticker1])
        log_ticker2 = hedge_ratio * np.log(Price_df[ticker2])

        # Create subplots (top and bottom)
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 10), sharex=True)

        # Plot log-transformed prices on the top graph (ax1)
        ax1.plot(Price_df.index, log_ticker1, label=f'{ticker1} Log Price', color='blue', linewidth=2)
        ax1.plot(Price_df.index, log_ticker2, label=f'{ticker2} Log Price (Hedge)', color='orange', linewidth=2)
        
        # Set dynamic y-limits for the top plot
        ymin = min(log_ticker1.min(), log_ticker2.min())
        ymax = max(log_ticker1.max(), log_ticker2.max())
        ax1.set_ylim([ymin - 0.05, ymax + 0.05])  # Add small buffer to the limits

        # Plot buy and sell signals for both tickers
        buy_signals = Price_df[(Price_df['Signal'] == 'Buy')]
        sell_signals = Price_df[(Price_df['Signal'] == 'Sell')]

        ax1.scatter(buy_signals.index, np.log(buy_signals[ticker1]), color='green', marker='^', s=100, label='Buy Ticker 1')
        ax1.scatter(buy_signals.index, hedge_ratio * np.log(buy_signals[ticker2]), color='cyan', marker='v', s=100, label='Sell Ticker 2')
        ax1.scatter(sell_signals.index, np.log(sell_signals[ticker1]), color='red', marker='v', s=100, label='Sell Ticker 1')
        ax1.scatter(sell_signals.index, hedge_ratio * np.log(sell_signals[ticker2]), color='brown', marker='^', s=100, label='Buy Ticker 2')

        ax1.set_ylabel('Log Price', fontsize=12)
        ax1.set_title(f'Price Comparison and Trading Signals for {ticker1} and {ticker2} (Log Prices)', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True)

        # Plot Spread on the bottom graph (ax2)
        ax2.plot(Price_df.index, Price_df['Spread'], label='Spread', color='black', linewidth=2)
        ax2.axhline(y=Price_df['Spread'].mean(), linestyle='--', color='red', label='Spread Mean')
        ax2.set_ylabel('Spread', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()


    def metric(self,Price_df):
        Price_df['returns'] = Price_df["Portfolio's Value"].pct_change()
        performance = pf.create_simple_tear_sheet(Price_df['returns'])
        return performance
