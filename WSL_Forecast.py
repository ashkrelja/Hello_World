# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:06:09 2018

@author: ashkrelja
"""

#import packages

import pandas as pd
import numpy as np
from statsmodels.tsa.x13 import x13_arima_analysis

#import data

path = 'C:/Users/ashkrelja/Documents/Wall_Street_Lending/Technology/Analytics/Operations_Analytics/2019/Operations Analytics_03_2018.csv'

df = pd.read_csv(path, usecols = ['Status_ClosedDate', 'Loan_LoanWith'])

#manipulate data

df.dropna(inplace = True)
df['Status_ClosedDate'] = df['Status_ClosedDate'].apply(lambda x: pd.to_datetime(x))
df.set_index('Status_ClosedDate', inplace = True)
df2 = df.resample('MS').sum()
df2.plot()

#X13 seasonal decomposition

output = x13_arima_analysis(df2['Loan_LoanWith'])

df2['trend'] = output.trend
df2['seasadj'] = output.seasadj
df2['seasonal'] = df2['Loan_LoanWith'] - df2['seasadj']
df2['seasadj_log'] = df2['seasadj'].apply(lambda x: np.log(x)) #log-series


df2['seasonal'].plot(legend = 'seasonal')
df2['trend'].plot(legend = 'trend')
df2['seasadj'].plot(legend = 'seasadj')
df2['seasadj_log'].plot() # 1st difference model in order to eliminate trend

df2.head()

#ARIMA grid search

from pyramid.arima import auto_arima


stepwise_model = auto_arima(df2['seasadj_log'],
                            start_p = 1,
                            start_q = 1,
                            max_p = 5,
                            max_q = 5,
                            m = 12,
                            seasonal = False,
                            trace = True,
                            d = 1,
                            suppress_warnings = True,
                            stepwise = True)


stepwise_model.summary()

#ARIMA(1,1,1) best fit @ AIC = 161.994

#split dataset between train and test

train = df2.loc['2010-01-01T00:00:00.000000000':'2017-12-01T00:00:00.000000000']
test = df2.loc['2010-01-01T00:00:00.000000000':]

stepwise_model.fit(train['seasadj_log'])

#in-sample forecast and evaluate

future_forecast = stepwise_model.predict(n_periods=12)

print(future_forecast)













#visually examine trend-cycle component dataset and graph

print(output.trend)
output.trend.plot()

# manipulate trend-cycle data to yield 1st difference and test for non-stationarity
cyc_component = output.trend
log_cyc_component = pd.DataFrame(np.log(cyc_component))
lag_log_cyc_component = lagmat(log_cyc_component, maxlag=1, use_pandas=1)
dif1 = log_cyc_component['trend'] - lag_log_cyc_component['trend.L.1']


df01 = pd.DataFrame(cyc_component, column)
df02 = pd.DataFrame(log_cyc_component)

df3 = pd.concat([df2,df01,df02])

