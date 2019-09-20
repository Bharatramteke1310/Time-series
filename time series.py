import os
import pandas as pd 
import numpy as np


dataset=pd.read_csv('D:\\cognitior\\Basics of data science\\AirPassengers.csv')
dataset

len(dataset)

#Using length(df.columns) function to check the total number of columns
len(dataset.columns)

#TAIL Function is used to get the values from the bottom.
dataset.tail(5)

#INFO fucntion is used to get information about the dataset what are the variable types etc.
dataset.info()

#DESCRIBE Function is used to know about how data is populated what are the basic statistics of the DATAFRAME
dataset.describe()

# This process will find out the number of missing values present in the dataset
dataset.isnull().any
dataset.isnull().sum()

#Convert it into time series
ts = dataset['#Passengers']
ts.head()

import matplotlib.pyplot as plt
plt.subplot(221)
plt.hist(ts)
plt.subplot(222)
ts.plot(kind = 'kde')
plt.tight_layout()

dataset['Month'] = pd.to_datetime(dataset['Month'])
dataset

dataset.plot()

dataset.set_index('Month', inplace=True)
dataset

dataset.plot()

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(dataset['#Passengers'], freq=12)
decomposition.plot()

from statsmodels.tsa.stattools import adfuller

adfuller(dataset['#Passengers'])
                 
 def adf_check(time_series):
    result = adfuller(time_series)
    print('Augmented Dickey Fuller Test')
    labels = ['ADF Test Statistic', 'P-value', '#Lags','No of Obs']
    for value, label in zip(result,labels):
        print(label + ":" + str(value))
        
    if result[1]<=0.05:
        print('Strong evidence against Null hypothesis and my time series is stationary')
    else: 
        print('Weak evidence against Null hypothesis and my time series is not stationary')                
                 
adf_check(dataset['#Passengers'])                 
 
dataset['#Passengers First Diff'] = dataset['#Passengers'] - dataset['#Passengers'].shift(1)
dataset                 

dataset['#Passengers Second Diff'] = dataset['#Passengers First Diff'] - dataset['#Passengers First Diff'].shift(1)
dataset

adf_check(dataset['#Passengers Second Diff'].dropna())                 
                  
dataset['Seasonal Difference'] = dataset['#Passengers']-dataset['#Passengers'].shift(12)
dataset  

adf_check(dataset['Seasonal Difference'].dropna())

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 

plot_acf(dataset['#Passengers Second Diff'].dropna(), lags=28)
#q=0

plot_pacf(dataset['#Passengers Second Diff'].dropna(),lags=14)
#p=0

plot_acf(dataset['Seasonal Difference'].dropna(), lags=12)
                  
plot_pacf(dataset['Seasonal Difference'].dropna(), lags=12)
#P=2 

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

model = sm.tsa.statespace.SARIMAX(dataset['#Passengers'], order=(1,2,1), seasonal_order=(2,2,0,12))
results = model.fit()
print(results.summary())

dataset['Forecast'] = results.predict(start=130, end=144, dynamic=True)
dataset[['#Passengers','Forecast']].plot()
         
from pandas.tseries.offsets import DateOffset
future_dates = [dataset.index[-1] + DateOffset(months=x) for x in range(0,6)]
future_dates 

future_dates_df = pd.DataFrame(index=future_dates[1:],columns=dataset.columns)
future_dates_df

future_df = pd.concat([dataset,future_dates_df])
future_df

future_df['Forecast'] = results.predict(start=145, end=151, dynamic=True)
future_df[['#Passengers','Forecast']].plot()

len(dataset)
future_df



                 
                  
                  
                  
                  
                  

                 



              
                  
                  
                  
                  
                  
                  
                 
                 
                 
                 


