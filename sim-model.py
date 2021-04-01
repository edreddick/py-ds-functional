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
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
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
#def pre_pro(df):
#    df = df.replace(cleanup_nums)
#    return(df)

#df_t = pre_pro(df)
df_t = df
y = df_t['count']
X = df_t.loc[:, df_t.columns != 'count']
del(df, df_t)

## messy to create 4 datasets, look into python alternatives similar to 
## rsample::initial_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
del(X, y)

## test out column imputation
## introduce NaN values
Xtrain.loc[Xtrain.index[0], 'num_a'] = np.NaN
Xtrain.loc[Xtrain.index[0], 'exposure'] = np.NaN
Xtest.loc[Xtest.index[0], 'num_a'] = np.NaN
Xtest.loc[Xtest.index[0], 'exposure'] = np.NaN

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

## test out sci-kit learn custom preprocessing
transformer_num = FunctionTransformer(func=imp
                                  , validate=False)

## for catagorical variables 
## pool infrequently occurring values into an "other" category
#Xtrain.loc[Xtrain.index[0], 'char_a'] = np.NaN

def get_support_levels(x_char, threshold):
    df = pd.DataFrame({'x_char' :x_char})
    df = pd.DataFrame(df['x_char'].value_counts()/len(x_char))
    return(list(df.query('x_char > @threshold').index))

sup_lev = get_support_levels(Xtrain['char_a'], .01)

def step_other(x_char, sup_lev):
    df = pd.DataFrame({'x_char' :x_char})
    df = df.assign(x_char = [a if a in sup_lev else 'other' for a in df['x_char']])
    return(df['x_char'])
    
step_other(Xtrain['char_a'], sup_lev)
    
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.means_ = None
        self.std_ = None

    def fit(self, X, y=None):
        X = X.to_numpy()
        self.means_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)

        return self

    def transform(self, X, y=None):
        X[:] = (X.to_numpy() - self.means_) / self.std_

        return X

my_CustomScaler = CustomScaler()


## this doesn't work probably because of first transformation being passed back
## as array with column names, or more likely because it's using the orignal num_a
## which has na values
#ct = ColumnTransformer(
#    [("num", imp, ['num_a', 'exposure']), 
#     ("num1", my_CustomScaler, ['num_a'])])

# define transformers
si_n = SimpleImputer(missing_values=np.nan, strategy='mean')
si_c = SimpleImputer(strategy='constant', fill_value='other')
ss = StandardScaler()
ohe = OneHotEncoder()
# define column groups with same processing
cat_vars = ['char_a']
num_vars = ['num_a', 'exposure']
# set up pipelines for each column group
categorical_pipe = Pipeline([('si_c', si_c), ('ohe', ohe)])
numeric_pipe = Pipeline([('si_n', si_n), ('ss', ss)])
# set up columnTransformer
ct = ColumnTransformer(
                    transformers=[
                        ('nums', numeric_pipe, num_vars),
                        ('cats', categorical_pipe, cat_vars)
                    ],
                    remainder='drop',
                    n_jobs=-1
                    )

#ct = ColumnTransformer(
#    [("num", imp, ['num_a', 'exposure']), 
#     ("char", transformer_char, ['char_a'])])

#transformer_num.transform(X)
## test values
ct.fit_transform(Xtrain)
ct.transform(Xtest)

## Step 2: model specification
## need to specify case weight, currently treated as any other feature
glm = PoissonRegressor()

## specify pipeline
pipe = Pipeline([('ct', ct), ('glm', glm)])

## Step 3: eval model protocol on training data using CV
## this should be nested CV if part of model protocol includes hyperparameter tuning
## the folds should be set up with respect to the proportion of the response
cv = KFold(n_splits=4, shuffle=True, random_state=1)

## according to the documentation this is the percentage of deviance explained
#cross_val_score(glm, X=Xtrain, y=ytrain, cv=cv)
cross_val_score(pipe, X=Xtrain, y=ytrain, cv=cv)

## Step 4: fit final model
#glm.fit(Xtrain, ytrain)
pipe.fit(Xtrain, ytrain)

## Step 5: eval final model on testing data
## according to the documentation this is the percentage of deviance explained
#glm.score(Xtest, ytest)
pipe.score(Xtest, ytest)

## scrap code for hist
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, sharey=True, tight_layout=True)
axs.hist(df['count'], bins=20)
## end scrap code
