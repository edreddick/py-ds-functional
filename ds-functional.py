# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
Rewrite of:
https://towardsdatascience.com/functional-programing-in-data-science-projects-c909c11138bb
"""

########################## notes ##########################
## bad code for fitting models, but shows ex of pre-processing:
## https://www.youtube.com/watch?v=0Lt9w-BxKFQ
## import seaborn as sns
## import matplotlib.pyplot as plt
## from sklearn.ensemble import RandomForestClassifier 
## from sklearn.metrics import confusion_matrix, classification_report
## from sklearn.model_selection import train_test_split
## from sklearn.preprocessing import StandardScalar, LabelEncoder
## %matplotlib inline
## from sklearn.linear_model import LinearRegression
## model = LinearRegression(normalize=True)
## model.fit(X, y)

## sc = StandardScarler()
## X_train = sc.fit_transform(X_train)
## X_test = sc.transform(X_test)
## preprocessing.FunctionTransformer seems promising
## sklearn.pipeline also seems promising
## RepeatedStratifiedKFold seems promising
## grid search seems promising: https://scikit-learn.org/stable/modules/grid_search.html#grid-search
## nested cv seems promising: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-download-auto-examples-model-selection-plot-nested-cross-validation-iris-py
## Pipeline looks promising: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
## setting up custom transformers: https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

# import pdb 
# pdb.set_trace() is debugger, similar to browser() in R
# interact sometimes needed, followed by cntr + d to exit
import pandas as pd
import numpy as np
from toolz import thread_first

np.random.seed(42)

## function to either return a random value or a NA value
def cre_val(percent_na):
    if np.random.random() > percent_na:
        return np.random.random() 
    else: 
        return np.nan

## function that maps across cre_val function to create a random column
def cre_col(nb_rows=1000, percent_na = 0.01, name = 0):
    df = pd.DataFrame()
    df[f'val_{name}'] = list(map(cre_val, [percent_na]*nb_rows))
    return df

## function that maps across cre_col function to create a random dataset
def cre_df(nb_rows=1000, percent_na = [0.01, 0.1, 0.9], name = range(3)):
    df = pd.concat(list(map(cre_col, 3*[nb_rows], percent_na, name)), axis = 1)
    df.index = pd.date_range('2001/01/01', periods=nb_rows, freq='H')
    return df

## function that removes columns that may that have too many NA values
def nas_remover(df, na_percentage=0.2):
    na_df = df.isna().sum() / len(df)
    list_col_to_keep = na_df[na_df < na_percentage].index
    return df[list_col_to_keep]

## function that resamples data at user defined hour intervals    
def cre_resampler(df, resampling_str):
    return df.resample(resampling_str).mean()

## function to fill NA values
def fill_na(df):
    return df.interpolate().ffill().bfill()

## pipe data through all functions in list
if __name__ == '__main__':
    res = thread_first(cre_df(), 
                       nas_remover,
                       (cre_resampler, '2H'),
                       fill_na)