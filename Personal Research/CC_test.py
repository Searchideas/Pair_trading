import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
import itertools
from statsmodels.tsa.vector_ar.vecm import coint_johansen

class CCstudy:
    def __init__(self, data):
        """
        Initialize the CCstudy class.

        Args:
            data (pandas.DataFrame): A DataFrame containing the historical data.
        """
        self.data = data


    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix as a heatmap.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.data.corr(), cmap='coolwarm', interpolation='nearest')
        plt.title('Correlation Matrix')
        plt.colorbar()
        plt.show()

    def return_tabulation(self):
        """
        Calculate and return the correlation matrix.

        Returns:
            pandas.DataFrame: A DataFrame containing the correlation matrix.
        """
        # Calculate daily returns
        returns_data = self.data.copy()
        for column in self.data.columns:
            returns_data[f"{column}_return"] = np.log(1+returns_data[column].pct_change())

        # Drop original columns and the first row
        returns_data.drop(self.data.columns, axis=1, inplace=True)
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
            # Drop the first row to avoid NaN values
            result,p_value, _ = coint(self.data[ticker1].iloc[1:], self.data[ticker2].iloc[1:])
            results[(ticker1, ticker2)] = (result, p_value)

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

