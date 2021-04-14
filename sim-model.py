# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
"""

## Step 0: Create fake small dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.stepother import StepOther
from src.stepothtomajority import StepOthToMajority
from src.check_params_exist import check_params_exist
import pdb

np.random.seed(1)
n = 10000
beta0 = -5
beta1 = 0.2
beta2 = .5
beta3 = .3
cleanup_nums = {"char_a": {"a": 0, "b": 1}}
cleanup_nums_b = {"char_b": {"a": 0, "b": 1}}

df = pd.DataFrame({'num_a': np.random.rand(n)*1.5,
                   'char_a': np.random.choice(['a', 'b'], n),
                   'char_b': np.random.choice(['a', 'b'], n),
                   'exposure': np.clip(np.random.rand(n), .01, 1)})

mu = np.exp((beta0 + beta1 * df.num_a + beta2 * (df.replace(cleanup_nums).char_a) + 
             + beta3 * (df.replace(cleanup_nums_b).char_b)) * df.exposure)

df['count'] = np.random.poisson(lam = mu)
del(n, beta0, beta1, beta2, beta3, mu, cleanup_nums, cleanup_nums_b)

## Step 1: pre-process data, this will occur within CV folds so that 
## any pre-processing that requires estimation does not lead to leakage
y = df['count']
sw = df['exposure']
X = df.drop(['count', 'exposure'], axis=1)
del(df)

## split into training, test for features, target, and weight
## stratify based on target variable because data is highly imbalanced
Xtrain, Xtest, ytrain, ytest, swtrain, swtest = train_test_split(
    X, y, sw, test_size = 0.25, random_state = 5, stratify=y)
del(X, y, sw)

## introduce NaN and low freq values to test column imputation
Xtrain.loc[Xtrain.index[0], 'num_a'] = np.NaN
Xtest.loc[Xtest.index[0], 'num_a'] = np.NaN
Xtrain.loc[Xtrain.index[0], 'char_a'] = np.NaN
Xtest.loc[Xtest.index[0], 'char_a'] = np.NaN
Xtrain.loc[Xtrain.index[1], 'char_a'] = 'c'
Xtest.loc[Xtest.index[1], 'char_a'] = 'c'

## define transformer instances
## maybe re-write this so that instances are called inside the pipe
## because there's probably no need to pollute the namespace
si_n = SimpleImputer(missing_values=np.nan, strategy='mean')
si_c = SimpleImputer(strategy='constant', fill_value='other')
so_c = StepOther(.01)
sotm_c = StepOthToMajority(.01)
ss = StandardScaler()
ohe = OneHotEncoder(drop='first')

## define column groups where same preprocessing steps will be carried out
cat_vars = ['char_a', 'char_b']
num_vars = ['num_a']

categorical_pipe = Pipeline([('si_c', si_c), ('so_c', so_c), ('sotm_c', sotm_c), 
                             ('ohe', ohe)])
numeric_pipe = Pipeline([('si_n', si_n), ('ss', ss)])

## set up columnTransformer to combine all pre-processing
ct = ColumnTransformer(
                    transformers=[
                        ('nums', numeric_pipe, num_vars),
                        ('cats', categorical_pipe, cat_vars)
                    ],
                    remainder='drop'
                    )

## Step 2: model specification
glm = PoissonRegressor()

## specify pipeline by combining pre-processing with model specification
pipe = Pipeline([('ct', ct), ('glm', glm)])

## Step 3: eval model protocol on training data using CV
## this should be nested CV if part of model protocol includes hyperparameter tuning
## the folds are defined with respect to the proportion of the response
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

## specify potential hyper-parameters
## can also add choices from pre-processing: ex) 
## 'ct__nums__si_n__strategy' : ['mean', 'median']
param_grid = {'glm__alpha' : [0, 0.2, 0.4, 0.6, 0.8, 1.0]}

## Specify modelling protocol which is to choose best set of pre-processing and 
## hyper params using grid search
clf = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, n_jobs= 1, verbose=5)

## assess modelling protocol using nested CV
## according to the documentation this is the percentage of deviance explained
cross_val_score(clf, X=Xtrain, y=ytrain, cv=outer_cv, 
                fit_params={'glm__sample_weight': swtrain})

## Step 4: fit final model using modelling protocol
clf.fit(X=Xtrain, y=ytrain, **{'glm__sample_weight': swtrain})
print(clf.best_params_)

## Step 5: eval final model on testing data
## according to the documentation this is the percentage of deviance explained
clf.score(Xtest, ytest)

## make predictions on training data and testing data
clf.predict(Xtrain)
clf.predict(Xtest)