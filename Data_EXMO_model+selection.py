
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 7)
from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

df_full = pd.read_csv('Exmo_data0207_sdvig.csv')
df_full.head()

X = np.array(df_full[['Open', 'High', 'Low']])
y = np.array(df_full['Close'])

plt.plot(y)
plt.plot(X)
#%%
t=0.999 
t = int(t*len(df_full)) 

X_train = X[:t] 
y_train = y[:t]  
X_test = X[t:] 
y_test = y[t:]

linear = LinearRegression().fit(X_train,y_train) 
predicted_price = linear.predict(X_test)  
#%%
plt.plot(predicted_price) 
plt.plot(y_test)  
plt.legend(['predicted_price','actual_price'])  
plt.show()
#%%
r2_score = linear.score(X[t:],y[t:])*100  
float("{0:.2f}".format(r2_score))
#%%
mses = mse(predicted_price, y_test)
mses

#%%
import math
math.sqrt(mses)

#%%
X_folds = np.array_split(X, 365)
y_folds = np.array_split(y, 365)
scores = list()
for k in range(365):
    # We use 'list' to copy, in order to 'pop' later on
    X_train = list(X_folds)
    X_test  = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test  = y_train.pop(k)
    y_train = np.concatenate(y_train)
    linear = LinearRegression().fit(X_train,y_train) 
    predicted_price = linear.predict(X_test) 
    mses = mse(X_test, y_test)
    rmse = math.sqrt(mses)
    scores.append(rmse)
print(scores)












#%%

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

%pylab inline
%matplotlib inline

plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (14, 7)

from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

#%%
criterion = 0.5
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
def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

n = 2
zy = list(divide_chunks(y, n))
print('Количество недель: ', len(zy))

#%%
nabor_arr = list(divide_chunks(df_full.index.values, n))
len(nabor_arr)
all_x = list(divide_chunks(X, n))
all_y = list(divide_chunks(y, n))
weeks = np.arange(4084, 4384)
len(weeks)
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
#%%
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