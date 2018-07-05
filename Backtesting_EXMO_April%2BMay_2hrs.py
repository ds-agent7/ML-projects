
# coding: utf-8

# In[1]:

import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import utils
from sklearn import linear_model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.metrics import accuracy_score, explained_variance_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold

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

df = pd.read_csv('EXMO_2hr.csv', sep=',')


# In[3]:

df.head()


# In[4]:

y = np.array(df['Close'])
X = np.array(df[['Open', 'High', 'Low']])


# In[46]:

criterion = 0.5


# In[47]:

def get_profit(y_test, y_pred):
    count = 0
    inc_array = []
    for i in range(len(y_pred)-1):
        delta_pred = ((y_pred[i+1] - y_pred[i])/y_pred[i]) *100
        if delta_pred > criterion: #BUY
            inc = (y_test[i]/y_test[i-1]) - 1
            inc_array.append(inc)
            count +=1
        if delta_pred < (-criterion): #SELL
            inc = 1 - (y_test[i]/y_test[i-1])
            inc_array.append(inc)
            count +=1
    total_inc = np.sum(inc_array)
    comission = 0.4*count
    final_profit = total_inc - comission
    return final_profit, count


# In[48]:

def rf_model(X_test, X_train, y_test, y_train):
    best_params = {'bootstrap': True, 'max_features': 'auto', 'max_leaf_nodes': 1000,
                   'min_samples_split': 4, 'n_estimators': 600}
    #Обучим модель, получим предсказание
    estimator = RandomForestRegressor(n_jobs=-1).set_params(**best_params)
    estimator.fit(X_train,y_train)
    y_pred = estimator.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    rmse = math.sqrt(mse)
    # Посчитаем точность определения класса (рост/падение цены)
    deltas_pos = []
    deltas_neg = []
    true_pos = []
    true_neg = []
    y_test = list(y_test)
    y_pred = list(y_pred)
    for i in range (len(y_test)-1):
        delta_true = ((y_test[i+1] - y_test[i])/y_test[i]) * 100
        delta_pred = ((y_pred[i+1] - y_pred[i])/y_pred[i]) * 100
        if (delta_true >= 0) & (delta_pred >= 0):
            deltas_pos.append(y_test[i])
        if (delta_true < 0) & (delta_pred < 0):
            deltas_neg.append(y_test[i])
        if (delta_true >= 0) & (delta_pred < 0):
            true_pos.append(y_test[i])
        if (delta_true < 0) & (delta_pred >= 0):
            true_neg.append(y_test[i])
    b = (len(deltas_pos)+len(deltas_neg)+len(true_pos)+len(true_neg))
    if b != 0:
        accuracy = (len(deltas_pos)+len(deltas_neg)) / (len(deltas_pos)+len(deltas_neg)+len(true_pos)+len(true_neg))
        accuracy = accuracy*100
        accuracy = round(accuracy, 2)
    else:
        accuracy = np.nan
    rmse = round(rmse, 2)
    # Считаем прибыльность и доход
    finpr, dealnum = get_profit(y_test, y_pred)
    return rmse, accuracy, finpr, dealnum


# In[49]:

def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]


# In[50]:

n = 2
zy = list(divide_chunks(y, n))
print('Количество недель: ', len(zy))


# In[51]:

nabor_arr = list(divide_chunks(df.index.values, n))
len(nabor_arr)


# In[52]:

all_x = list(divide_chunks(X, n))
all_y = list(divide_chunks(y, n))


# In[53]:

weeks = np.arange(1641, 1928)
len(weeks)


# In[54]:

accuracies_array = []
rmses_array = []
finpr_array = []
dealnum_array = []
for i in weeks:
   # while i<46:
        X_train = np.concatenate(all_x[:i], axis=0)
        X_test = all_x[i]
        y_train = np.concatenate(all_y[:i], axis=0)
        y_test = all_y[i]
        rmse_score, accuracy_score, finpr_score, dealnum_score = rf_model(X_test, X_train, y_test, y_train)
        accuracies_array.append(accuracy_score)
        rmses_array.append(rmse_score) 
        finpr_array.append(finpr_score)
        dealnum_array.append(dealnum_score)


# In[55]:

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.tight_layout()
fig.subplots_adjust(hspace = 0.7, top = 0.85, wspace = 0.7)
fig.suptitle('Backtesting EXMO, 2 hours, criterion = 1', fontweight='bold')
ax1.plot(weeks, rmses_array)
ax2.scatter(weeks, accuracies_array)
ax3.scatter(weeks, finpr_array)
ax4.scatter(weeks, dealnum_array)
ax1.set_title('RMSE')
ax2.set_title('Accuracy')
ax3.set_title('Total profit')
ax4.set_title('Number of deals')
ax1.set_xlabel('Week')
ax2.set_xlabel('Week')
ax3.set_xlabel('Week')
ax4.set_xlabel('Week')


# In[58]:

df = pd.read_csv('Exmo_BTCUSD_1h.csv', sep=';')


# In[61]:

df[::2].to_csv('new_df.csv')


# In[ ]:




# In[125]:

fin = pd.DataFrame({'Week':weeks, 'RMSE': rmses_array, 
                    'Accuracy':accuracies_array}, 
                  columns=['Week', 'RMSE', 'Accuracy'])


# In[126]:

fin.to_csv('EXMO_nopr.csv')


# In[130]:

XTr = X[:len(df)-84]
XTs = X[len(df)-84:]
YTr = y[:len(df)-84]
YTs = y[len(df)-84:]


# In[ ]:



#%%
from IPython.display import Latex
Latex('''The mass-energy equivalence is described by the famous equation

$$E=mc^2$$

discovered in 1905 by Albert Einstein.
In natural units ($c$ = 1), the formula expresses the identity

\\begin{equation}
E=m
\\end{equation}''')