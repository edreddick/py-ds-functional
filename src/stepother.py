import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class StepOther(BaseEstimator, TransformerMixin):
    """ pool infrequently occurring values into an "other" category """
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        
        def get_sup_lev(x, threshold):
            threshold = threshold[0]
            df = pd.DataFrame(x.value_counts()/len(x))
            df = df.rename(columns={ df.columns[0]: "x_name" })
            return list(df.query(f"x_name > {threshold}").index)
        df = pd.DataFrame(X.copy())
        self.sup_levs = df.apply(get_sup_lev, threshold = [self.threshold])
        return self

    def transform(self, X, y=None):
        def set_sup_lev(x, sup_lev):
            df = x.copy()
            df.loc[np.in1d(x.values, sup_lev.values, invert=True)] = 'other'
            return df
        
        df = pd.DataFrame(X.copy())
        
        for i in range(df.shape[1]):
            df.iloc[:, i] = set_sup_lev(df.copy().iloc[:, i], self.sup_levs.iloc[:,i])
        X = df.to_numpy()
        return X
