# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
"""

## Step 0: Create fake small dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score, KFold

np.random.seed(1)
n = 10000
beta0 = -5
beta1 = 0.2
beta2 = .5
cleanup_nums = {"char_a": {"a": 0, "b": 1}}

df = pd.DataFrame({'num_a': np.random.rand(n)*1.5,
                   'char_a': np.random.choice(['a', 'b'], n),
                   'exposure': np.clip(np.random.rand(n), .01, 1)})

mu = np.exp((beta0 + beta1 * df.num_a + beta2 * (df.replace(cleanup_nums).char_a)) 
            * df.exposure)

df['count'] = np.random.poisson(lam = mu)
del(n, beta0, beta1, beta2, mu)

## Step 1: pre-process data
## naive pre-processing, this should occur within CV folds for any pre-processing
## that requires estimation to avoid leakage similar to recipes in R
def pre_pro(df):
    df = df.replace(cleanup_nums)
    return(df)

df_t = pre_pro(df)
y = df_t['count']
X = df_t.loc[:, df_t.columns != 'count']
del(df_t)

## messy to create 4 datasets, look into python alternatives similar to 
## rsample::initial_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

## Step 2: model specification
## need to specify case weight, currently treated as any other feature
glm = PoissonRegressor()

## Step 3: eval model protocol on training data using CV
## this should be nested CV if part of model protocol includes hyperparameter tuning
cv = KFold(n_splits=4, shuffle=True, random_state=1)

## according to the documentation this is the percentage of deviance explained
cross_val_score(glm, X=Xtrain, y=ytrain, cv=cv)

## Step 4: fit final model
glm.fit(Xtrain, ytrain)

## Step 5: eval final model on testing data
## according to the documentation this is the percentage of deviance explained
glm.score(Xtest, ytest)

## scrap code for hist
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, sharey=True, tight_layout=True)
axs.hist(df['count'], bins=20)
## end scrap code
