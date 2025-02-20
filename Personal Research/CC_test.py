import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import itertools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import seaborn as sns
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

class CCstudy:
    def __init__(self, data):
        self.data = data


    def plot_correlation_matrix(self):
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.clustermap(self.data.corr(), cmap=cmap, center=0);
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.data.corr(), cmap='coolwarm', interpolation='nearest')
        plt.title('Correlation Matrix')
        plt.colorbar()
        plt.show()"""

    def return_tabulation(self):
        # Calculate daily returns
        returns_data = self.data.copy()
        '''
        for column in self.data.columns:
            returns_data[f"{column}_return"] = np.log(1+returns_data[column].pct_change())

        # Drop original columns and the first row
        returns_data.drop(self.data.columns, axis=1, inplace=True)'''
        returns_data.dropna(inplace=True)  # Drop rows with NaN values

        # Calculate correlation matrix
        correlation_matrix = returns_data.corr()

        return correlation_matrix

    def cointegration_test(self):
        """
        Perform a cointegration test for all combinations of columns.

        Returns:
            dict: A dictionary containing the cointegration test results and p-values for each pair of columns.
        """
        results = {}
        columns = self.data.columns
        self.data.fillna(method='ffill', inplace=True)
        # Generate all combinations of columns
        combinations = list(itertools.combinations(columns, 2))

        for combination in combinations:
            ticker1, ticker2 = combination
             # Calculate the log prices
            log_prices1 = np.log(self.data[ticker1])
            log_prices2 = np.log(self.data[ticker2])
            model = LinearRegression()
            model.fit(log_prices1.values.reshape(-1, 1), log_prices2.values.reshape(-1, 1),)
            n = (model.coef_[0])[0]
            # Calculate the R-squared value
            rsq = model.score(log_prices1.values.reshape(-1, 1), log_prices2.values.reshape(-1, 1))
            # Calculate the spread
            spread = log_prices1 - n*log_prices2
        
            # Perform the cointegration test
            try:
                ADF = adfuller(spread)
                result, p_value= ADF[0],ADF[1]
                results[(ticker1, ticker2)] = (result, p_value)
            except Exception as e:
                print(f"Error processing {ticker1} and {ticker2}: {str(e)}")
                results[(ticker1, ticker2)] = (None, None, None)

        return results

    
    def main(self):
        correlation_matrix = self.return_tabulation()
        self.plot_correlation_matrix()
        print("\nCointegration Test Results:")
        results = self.cointegration_test()


        # Create a DataFrame directly from the dictionary items
        results_df = pd.DataFrame(
            [(k[0], k[1], v[1]) for k, v in results.items()], 
            columns=['ticker1', 'ticker2', 'Cointegration result']
        )
        Correlation = pd.melt(correlation_matrix.reset_index(), id_vars='index', var_name='columns', value_name='correlation')
        Correlation.rename(columns={'index':'ticker1','columns':'ticker2'},inplace=True)
        Correlation = Correlation[Correlation['ticker1'] != Correlation['ticker2']]
        Correlation['ticker1'] = Correlation['ticker1'].str.removesuffix('_return')
        Correlation['ticker2'] = Correlation['ticker2'].str.removesuffix('_return')
        '''
        # Assuming that 'correlation_table' is your DataFrame
        Correlation = pd.melt(correlation_matrix.reset_index(), id_vars='index', var_name='column', value_name='correlation')

        # Rename the 'index' column to 'row'
        Correlation  = Correlation.rename(columns={'index': 'ticker1','column':'ticker2'})
'''     
        #Merge tables
        Cotable = results_df.merge(Correlation,left_on=['ticker1','ticker2'],right_on=['ticker1','ticker2'],how='left')

        return Cotable

