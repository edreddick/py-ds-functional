# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
"""

## Step 0: Create fake small dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pdb

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
y = df['count']
sw = df['exposure']
X = df.drop(['count', 'exposure'], axis=1)
del(df)

## split into training, test for features, target, and weight
Xtrain, Xtest, ytrain, ytest, swtrain, swtest = train_test_split(
    X, y, sw, test_size = 0.25, random_state = 5)
del(X, y, sw)

## introduce NaN and low freq values to apply column imputation
Xtrain.loc[Xtrain.index[0], 'num_a'] = np.NaN
Xtest.loc[Xtest.index[0], 'num_a'] = np.NaN
Xtrain.loc[Xtrain.index[0], 'char_a'] = np.NaN
Xtest.loc[Xtest.index[0], 'char_a'] = np.NaN
Xtrain.loc[Xtrain.index[1], 'char_a'] = 'c'
Xtest.loc[Xtest.index[1], 'char_a'] = 'c'

## pool infrequently occurring values into an "other" category
class StepOther(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns = ['x_char'])
        df = pd.DataFrame(df['x_char'].value_counts()/len(df))
        self.sup_lev = list(df.query('x_char > @self.threshold').index)
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns = ['x_char'])
        df = df.assign(x_char = [a if a in self.sup_lev else 'other' for a in df['x_char']])
        X = df.to_numpy()
        return X

## define transformer instances
si_n = SimpleImputer(missing_values=np.nan, strategy='mean')
si_c = SimpleImputer(strategy='constant', fill_value='other')
so_c = StepOther(.01)
ss = StandardScaler()
ohe = OneHotEncoder(drop='first')

## define column groups where same preprocessing steps will be carried out
cat_vars = ['char_a']
num_vars = ['num_a']

categorical_pipe = Pipeline([('si_c', si_c), ('so_c', so_c), ('ohe', ohe)])
numeric_pipe = Pipeline([('si_n', si_n), ('ss', ss)])

## set up columnTransformer
ct = ColumnTransformer(
                    transformers=[
                        ('nums', numeric_pipe, num_vars),
                        ('cats', categorical_pipe, cat_vars)
                    ],
                    remainder='drop',
                    n_jobs=-1
                    )

## check ColumnTransformer
#ct.fit_transform(Xtrain)
#ct.transform(Xtest)

## Step 2: model specification
glm = PoissonRegressor()

## specify pipeline
pipe = Pipeline([('ct', ct), ('glm', glm)])

## Step 3: eval model protocol on training data using CV
## this should be nested CV if part of model protocol includes hyperparameter tuning
## the folds should be set up with respect to the proportion of the response
cv = KFold(n_splits=4, shuffle=True, random_state=1)

## according to the documentation this is the percentage of deviance explained
cross_val_score(pipe, X=Xtrain, y=ytrain, cv=cv, 
                fit_params={'glm__sample_weight': swtrain})

## Step 4: fit final model
pipe.fit(Xtrain, ytrain, **{'glm__sample_weight': swtrain})

## Step 5: eval final model on testing data
## according to the documentation this is the percentage of deviance explained
pipe.score(Xtest, ytest, sample_weight=swtest)

## make predictions on training data and testing data
## neither take into account weights
pipe.predict(Xtrain)
pipe.predict(Xtest)

## scrap code for hist
#import matplotlib.pyplot as plt
#fig, axs = plt.subplots(1, sharey=True, tight_layout=True)
#axs.hist(df['count'], bins=20)
## end scrap code