# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:06:09 2018

@author: ashkrelja
"""


#import data

import pandas as pd
import numpy as np

path = 'SECRET'

df = pd.read_csv(path, usecols = ['Status_ClosedDate', 'Loan_LoanWith'])

#manipulate data

df.dropna(inplace = True)
df['Status_ClosedDate'] = df['Status_ClosedDate'].apply(lambda x: pd.to_datetime(x))
df.set_index('Status_ClosedDate', inplace = True)
df2 = df.resample('MS').sum()
df2.plot()

#X13 seasonal decomposition

from statsmodels.tsa.x13 import x13_arima_analysis

output = x13_arima_analysis(df2['Loan_LoanWith'])

df2['trend'] = output.trend
df2['seasadj'] = output.seasadj
df2['irregular'] = output.irregular
df2['seasonal'] = df2['Loan_LoanWith'] - df2['seasadj']
df2['seasadj_irr'] = df2['seasadj'] - df2['irregular']
df2['seasadj_log'] = df2['seasadj_irr'].apply(lambda x: np.log(x)) #log-series


df2['seasonal'].plot(legend = 'seasonal')
df2['trend'].plot(legend = 'trend')
df2['seasadj'].plot(legend = 'seasadj')
df2['irregular'].plot(legend = 'irregular')
df2['seasadj_irr'].plot(legend = 'fully adjusted')
df2['seasadj_log'].plot() # 1st difference model in order to eliminate trend

df2.head()

#stationarity

from statsmodels.tsa.statespace.tools import diff
from statsmodels.tsa.stattools import adfuller

df2['diff_1_seasadj'] = diff(diff(diff(diff(df2['seasadj_log']))))
df2['diff_1_seasadj'].plot()

df2['diff_1_seasadj'].replace(np.NaN,0,inplace=True)
adfuller(df2['diff_1_seasadj']) #reject Ho, conclude Ha: no unit root

#ACF(MA) - PACF(AR)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df2['diff_1_seasadj']) # MA(1,2,5)

plot_pacf(df2['diff_1_seasadj']) # AR(1)

#self-developed ARIMA

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df2['seasadj_log'], order=(1,4,(1,2,5)))


#ARIMA grid search

from pyramid.arima import auto_arima


stepwise_model = auto_arima(df2['seasadj_log'],
                            start_p = 1,
                            start_q = 1,
                            max_p = 10,
                            max_q = 10,
                            m = 12,
                            seasonal = False,
                            trace = True,
                            d = 1,
                            suppress_warnings = True,
                            stepwise = True,
                            with_intercept = False)


stepwise_model.summary()

#ARIMA(1,1,1) best fit @ AIC = 161.994

#diagnostic tests

residual_array = stepwise_model.resid()
residuals = pd.DataFrame(residual_array)
residuals.plot()

residuals.plot(kind='kde')
print(residuals.describe()) #residual mean 0.002732


#split dataset between train and test

train = df2.loc['2010-01-01T00:00:00.000000000':'2017-12-01T00:00:00.000000000']
test = df2.loc['2018-01-01T00:00:00.000000000':]

stepwise_model.fit(train['seasadj_log'])

#in-sample forecast

future_forecast = stepwise_model.predict(n_periods=15)

print(future_forecast)

#compare to actual data

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction']) #put predictions in dataframe
future_forecast['Prediction'] = future_forecast.apply(lambda x: np.exp(x)) #take predictions from log to level

new = pd.concat([future_forecast['Prediction'],test['seasonal'], test['Loan_LoanWith']],axis=1)
new['Predict_Lev_Seas'] = new['Prediction'] + new['seasonal']
pd.concat([new['Predict_Lev_Seas'], new['Loan_LoanWith']],axis=1).plot()






