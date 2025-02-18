import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Retrieval import DataRetrieval
from Spread import Spread
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from IPython.display import display


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
        spread =spread.to_frame()
        spread= spread.rename(columns={0:'Spread'})
        signals_df = spread.join(signals_df,how="outer")
        #print(spread.index)
        return signals_df , mean

   
    def position(self,ticker1,ticker2):
        signals_df, mean = self.signal(ticker1, ticker2)
        ticker1_df = self.merged_df[ticker1].to_frame()

        Price_df = ticker1_df.join(signals_df, how='outer')
        Price_df = Price_df.join(self.merged_df[ticker2].to_frame(), how='inner')
        
        hedge_ratio = self.hedge.loc[(self.hedge['ticker1'] == ticker1) | (self.hedge['ticker2'] == ticker2), 'hedge ratio'].values[0]
        long_amount = 2500000
        
        Price_df[str(ticker1) + "'s trade"] = np.nan
        Price_df[str(ticker2) + "'s trade"] = np.nan
        Price_df[str(ticker1) + "'s position"] = np.nan
        Price_df[str(ticker2) + "'s position"] = np.nan
        Price_df["Cash"] = np.nan

        
        ticker1_position = 0
        ticker2_position = 0
        cash = int(5000000)
        Price_df['Cash'].iloc[0] = cash
        max_cash = 5000000

        for index, row in Price_df.iterrows():
            if row['Signal'] == 'Buy':
                long_qty = round(long_amount/row[ticker1])
                short_qty = - long_qty * hedge_ratio
                
                Price_df.at[index, str(ticker1) + "'s trade"] = long_qty
                Price_df.at[index, str(ticker2) + "'s trade"] = short_qty
                
                ticker1_position += long_qty
                ticker2_position += short_qty
                cash -= (long_qty* row[ticker1]+ short_qty*row[ticker2])
                Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                Price_df.at[index, "Cash"] = cash 
                
                
            elif row['Signal'] == 'Sell':
                long_qty = round(long_amount/row[ticker2])
                short_qty = - long_qty * hedge_ratio
                
                Price_df.at[index, str(ticker1) + "'s trade"] = short_qty
                Price_df.at[index, str(ticker2) + "'s trade"] = long_qty
                cash -= (short_qty* row[ticker1]+ long_qty*row[ticker2])
                ticker1_position += short_qty
                ticker2_position += long_qty
                
                Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                Price_df.at[index, "Cash"] = cash 

            elif row['Signal'] not in ['Sell', 'Buy'] and index != 0:
                prev_row = Price_df.iloc[Price_df.index.get_loc(index) - 1]
                if ((float(prev_row['Spread']) > mean and float(row['Spread']) < mean) or (float(prev_row['Spread']) < mean and float(row['Spread']) > mean)) and pd.notna(Price_df.at[index, str(ticker1) + "'s trade"]):
                    Price_df.at[index, str(ticker1) + "'s trade"] = -prev_row[str(ticker1) + "'s position"]
                    Price_df.at[index, str(ticker2) + "'s trade"] = -prev_row[str(ticker2) + "'s position"]
                    cash-= Price_df.at[index, str(ticker1) + "'s trade"]*row[ticker1] + Price_df.at[index, str(ticker2) + "'s trade"]*row[ticker2]
                    ticker1_position = 0
                    ticker2_position = 0
                    Price_df.at[index, "Cash"] = cash 
                    
                    Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                    Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
                    
                else:
                    Price_df.at[index, str(ticker1) + "'s trade"] = 0
                    Price_df.at[index, str(ticker2) + "'s trade"] = 0
                    Price_df.at[index, "Cash"] = cash 
                    Price_df.at[index, str(ticker1) + "'s position"] = ticker1_position
                    Price_df.at[index, str(ticker2) + "'s position"] = ticker2_position
        return Price_df

    def cashflow(self,Price_df):
        '''Define cash flow from trade + portfolio value'''
        Columns = Price_df.columns
        ticker1 = Columns[0]
        ticker2 =Columns[3]
        Price_df['Position MTM'] = Price_df[str(ticker1) + "'s position"] * Price_df[str(ticker1)] + Price_df[str(ticker2) + "'s position"] * Price_df[str(ticker2)] 
        Price_df['Portfolio Value'] = Price_df['Cash'] + Price_df['Position MTM']
        return Price_df

    def metric(self,Price_df):
        Price_df['returns'] = Price_df['Portfolio Value'].pct_change()
        Price_df['cum returns'] = (1 + Price_df['returns']).cumprod() - 1
        # Plot cumulative returns
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(Price_df.index, Price_df['cum returns'], label='Portfolio')
        ax2 = ax.twinx()
        ax2.plot(Price_df.index, Price_df['Portfolio Value'], label='Portfolio Value',c='red')
        ax2.set_ylim([Price_df['Portfolio Value'].min() * 0.9, Price_df['Portfolio Value'].max() * 1.1])
        ax.legend()
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax2.set_ylabel('Portfolio Value')
        ax2.legend(loc='upper right')
        plt.show()
        sharpe_ratio = (Price_df['returns'].mean()/ Price_df['returns'].std())*np.sqrt(252)
        print(f'Sharpe Ratio: {sharpe_ratio:.3f}')
        import pandas as pd

        # CAGR
        beginning_value = Price_df['Portfolio Value'].iloc[0]
        end_value = Price_df['Portfolio Value'].iloc[-1]
        number_of_years = (Price_df.index[-1] - Price_df.index[0]).days / 365

        cagr = (end_value / beginning_value)**(1 / number_of_years) - 1
        print(f'CAGR: {cagr*100:.2f}%')

        # Max drawdown
        # Max drawdown
        cumlative_max = Price_df['cum returns'].cummax()
        drawdown = (Price_df['cum returns'] - cumlative_max) / cumlative_max
        print(f'max drawdown: {min(drawdown[1:])*100:.2f}%')
        return Price_df
        
    def signal_plot(self,Price_df, hedge_ratio):
        Columns = Price_df.columns
        ticker1 = Columns[0]
        ticker2 = Columns[3]

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