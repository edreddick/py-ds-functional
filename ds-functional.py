# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
Rewrite of:
https://towardsdatascience.com/functional-programing-in-data-science-projects-c909c11138bb
"""
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