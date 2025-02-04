import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from Data_Retrieval import DataRetrieval
import matplotlib.pyplot as plt
class Hedge_ratio():
    def __init__(self, pairs,merged_df):
        self.pairs = pairs
        self.merged_df = merged_df
    
    def hedge(self,pairs,merged_df):
        hedge_table = pd.DataFrame(columns=['ticker1','ticker2','rsq','hedge ratio'])
        for i,row in pairs.iterrows():
            ticker1 = row['Ticker 1']
            ticker2 = row['Ticker 2']
            ticker1_data = np.log(self.merged_df[ticker1])
            ticker2_data = np.log(self.merged_df[ticker2])
            model = LinearRegression()
            model.fit(ticker1_data.values.reshape(-1, 1), ticker2_data.values.reshape(-1, 1),)
            r_sq = model.score(ticker1_data.values.reshape(-1, 1), ticker2_data.values.reshape(-1, 1))
            hedge_table.loc[i] = [ticker1,ticker2,r_sq,np.round(model.coef_[0])[0]]
        return hedge_table


            


    