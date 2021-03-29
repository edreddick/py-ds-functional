# -*- coding: utf-8 -*-
"""
Playing with python, typical data science tasks, and functional programming.
"""

############################### Create fake small dataset ###############################
import numpy as np
import pandas as pd

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
del(n, beta0, beta1, beta2, mu, cleanup_nums)

## scrap code for hist
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, sharey=True, tight_layout=True)
axs.hist(df['count'], bins=20)
## end scrap code
