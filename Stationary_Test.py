import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS
import seaborn as sns
from tqdm import tqdm

class stationary:
    def __init__(self, data):
        self.data = data

    def correlation_matrix(self):
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.clustermap(self.data.corr(), cmap=cmap, center=0)
        correlation_table = self.data.corr()
        return correlation_table

    def stationary_spread(self):
        stationarity_status = {}
        results = []

        tickers = list(self.data.columns)
        pairs = list(itertools.combinations(tickers, 2))

        # Step 1: Run DFGLS on individual time series
        for ticker in tickers:
            try:
                test = DFGLS(self.data[ticker].dropna())
                pval = test.pvalue
                status = 'stationary' if pval < 0.05 else 'non-stationary'
            except Exception as e:
                pval = None
                status = f"error: {e}"
            stationarity_status[ticker] = status

        # Step 2: Test each pair for cointegration
        for t1, t2 in tqdm(pairs, desc="Testing pairs for cointegration"):
            try:
                s1 = np.log(self.data[t1]).dropna()
                s2 = np.log(self.data[t2]).dropna()
                common_index = s1.index.intersection(s2.index)
                s1 = s1.loc[common_index]
                s2 = s2.loc[common_index]

                if len(s1) < 10 or len(s2) < 10:
                    raise ValueError("Insufficient overlapping data")

                ticker_1_status = stationarity_status[t1]
                ticker_2_status = stationarity_status[t2]

                if ticker_1_status == 'non-stationary' and ticker_2_status == 'non-stationary':
                    model = sm.OLS(s1, sm.add_constant(s2)).fit()
                    constants = model.params[0]
                    hedge_ratio = model.params[1]
                    residuals = s1 - model.predict(sm.add_constant(s2))

                    if residuals.nunique() <= 1:
                        adf_pval = 1.0
                        cointegration_status = 'Constant Residual - Not Applicable'
                    else:
                        adf_pval = adfuller(residuals)[1]
                        cointegration_status = 'Cointegrated' if adf_pval < 0.05 else 'Not Cointegrated'
                else:
                    constants = 0
                    hedge_ratio = 0
                    adf_pval = 1.0
                    cointegration_status = 'Not Applicable'

            except Exception as e:
                constants = np.nan
                hedge_ratio = np.nan
                adf_pval = np.nan
                cointegration_status = f'Error: {e}'

            results.append({
                'ticker_1': t1,
                'ticker_2': t2,
                'ticker_1_stationary': stationarity_status.get(t1, 'N/A'),
                'ticker_2_stationary': stationarity_status.get(t2, 'N/A'),
                'hedge_ratio': hedge_ratio,
                'intercept': constants,
                'adf_pval': adf_pval,
                'cointegration_status': cointegration_status
            })

        return pd.DataFrame(results)
