# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.stepother import StepOther
from src.stepothtomajority import StepOthToMajority
from src.check_params_exist import check_params_exist
import pdb

## Step 0: Simulate typical frequency data
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

## Step 1: Define problem criteria
## define column groups where same preprocessing steps will be carried out
num_vars = ['num_a']
cat_vars = ['char_a', 'char_b']

## Define features, target, and sample weight
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

## Step 2: pre-process data, this will occur within CV folds so that 
## any pre-processing that requires estimation does not lead to leakage
## specify transfomers pre-processing steps for numeric and categorical variables
numeric_pipe = Pipeline([('si_n', SimpleImputer(missing_values=np.nan)), 
                         ('ss_n', StandardScaler())])

categorical_pipe = Pipeline([('si_c', SimpleImputer(strategy='constant', fill_value='other')), 
                              ('so_c', StepOther(.01)), 
                              ('sotm_c', StepOthToMajority(.01)), 
                              ('oe_c', OrdinalEncoder(dtype=np.int32))])

## specify columnTransformer to combine all pre-processing steps
ct = ColumnTransformer(
                    transformers=[
                        ('nums', numeric_pipe, num_vars),
                        ('cats', categorical_pipe, cat_vars)
                    ],
                    remainder='drop'
                    )

## Step 3: model specification
## specify pipeline by combining pre-processing with model specification
modspec = lgb.LGBMRegressor()
pipe = Pipeline([('ct', ct), ('modspec', modspec)])

## Step 4: eval model protocol on training data using CV
## this should be nested CV if part of model protocol includes hyperparameter tuning
## the folds are defined with respect to the proportion of the response
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

## specify potential pre-processing choices and hyper-parameters
param_grid = {'ct__nums__si_n__strategy' : ['mean', 'median'],
              'modspec__boosting_type': ['gbdt'],
              'modspec__objective': ['poisson'],
              'modspec__n_estimators' : [10, 25, 50],
              'modspec__learning_rate' : [.1],
              'modspec__num_leaves': [15],
              'modspec__max_depth' : [5],
              'modspec__subsample': [.7],
              'modspec__colsample_bytree': [.8],
              'modspec__min_child_samples': [20],
              'modspec__reg_alpha': [8],
              'modspec__reg_lambda': [2]
              }

## Specify modelling protocol which is to choose best set of pre-processing and 
## hyper params using grid search
clf = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, n_jobs= 1, verbose=5)

## assess modelling protocol using nested CV
cross_val_score(clf, X=Xtrain, y=ytrain, cv=outer_cv, 
                fit_params={'modspec__sample_weight': swtrain})

## Step 5: fit final model using modelling protocol
clf.fit(X=Xtrain, y=ytrain, **{'modspec__sample_weight': swtrain})
print(clf.best_params_)

## Step 6: eval final model on testing data
clf.score(Xtest, ytest)

## Step 7: make predictions on training data and testing data
clf.predict(Xtrain)
clf.predict(Xtest)