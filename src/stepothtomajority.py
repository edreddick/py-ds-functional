import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import pdb

class StepOthToMajority(BaseEstimator, TransformerMixin):
    """ set other level to most occuring level when other level is sparse"""
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        
        def get_maj_lev(x):
            df = pd.DataFrame(x.value_counts()/len(x))
            df = df.rename(columns={ df.columns[0]: "x_name" })
            return list(df.query("x_name == x_name.max()").index)
        
        df = pd.DataFrame(X.copy())
        
        self.sup_levs = df.apply(get_maj_lev)

        return self
    
    def transform(self, X, y=None):
        def set_sup_lev(x, sup_lev, threshold):
            
            threshold_cond = sum(x=='other')/len(x) <= threshold and np.in1d(x.values, 'other').sum() > 0
            df = x.copy()
            if threshold_cond:
                df.loc[np.in1d(x.values, 'other')] = sup_lev[0]
            
            return df
        
        df = pd.DataFrame(X.copy())
        
        for i in range(df.shape[1]):
            df.iloc[:, i] = set_sup_lev(df.copy().iloc[:, i], self.sup_levs.iloc[:,i], self.threshold)
        X = df.to_numpy()
        return X
