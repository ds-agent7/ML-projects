import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 7)
from pandas import read_csv, DataFrame

df_full = pd.read_csv('Exmo_data0207.csv')
df_full.head()

X = np.array(df_full[['Open', 'High', 'Low']])
Y = np.array(df_full['Close'])

df_full




