
# coding: utf-8

# In[1]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn import linear_model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error

get_ipython().magic('pylab inline')
get_ipython().magic('matplotlib inline')

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (14, 7)

from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split


# In[2]:

df = pd.read_csv('Exmo_data0207.csv')
df.head()


# In[ ]:



