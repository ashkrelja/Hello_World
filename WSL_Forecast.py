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
df2 = df2.loc['2013-01-01T00:00:00.000000000':]
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

df2['diff_1_seasadj'] = diff(diff(df2['seasadj_log']))
df2['diff_1_seasadj'].plot()

df2['diff_1_seasadj'].replace(np.NaN,0,inplace=True)
adfuller(df2['diff_1_seasadj']) #reject Ho, conclude Ha: no unit root

#ACF(MA) - PACF(AR)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df2['diff_1_seasadj']) # MA(4)

plot_pacf(df2['diff_1_seasadj']) # AR(0)

#self-developed ARIMA

from statsmodels.tsa.arima_model import ARIMA, ARIMAResults

model = ARIMA(df2['seasadj_log'], order=(1,2,4))

model_fit = model.fit(disp=0)

print(model_fit.summary())

#ARIMA(1,2,4) best fit @ AIC = -665.884

#diagnostics
residual_array_1 = pd.DataFrame(model_fit.resid)
residual_array_1.plot()
#residuals fluctuate around 0

residual_array_1.plot(kind='kde')
print(residual_array_1.describe())
#normal distribution of residuals with mean 0

#in-sample prediction vs actual

df2['insample_prediction'] = model_fit.predict(start = '2018-01-01T00:00:00.000000000',
                                               end = '2019-03-01T00:00:00.000000000')

df2['insample_prediction_level'] = model_fit.predict(start = '2018-01-01T00:00:00.000000000',
                                                     end = '2019-03-01T00:00:00.000000000',
                                                     typ='levels')

model_fit.plot_predict(start = '2018-01-01T00:00:00.000000000',
                       end = '2019-03-01T00:00:00.000000000',
                       alpha=0.05)

pd.concat([df2['insample_prediction'],df2['diff_1_seasadj']], axis=1).plot() #2nd differenced prediction vs actual

pd.concat([df2['insample_prediction_level'],df2['seasadj_log']], axis=1).plot() #level prediction vs actual

#performance

from statsmodels.tools.eval_measures import rmse

df2['insample_prediction_level_seas'] = df2['insample_prediction_level'].apply(lambda x: np.exp(x)) + df2['seasonal'] + df2['irregular']
pd.concat([df2['insample_prediction_level_seas'],df2['Loan_LoanWith']],axis = 1).plot()

pred_1= df2['insample_prediction'].loc['2018-01-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000']
obsv_1 = df2['diff_1_seasadj'].loc['2018-01-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000']
rmse(pred_1,obsv_1) #0.0014153190808289856

pred_2 = df2['insample_prediction_level'].loc['2018-01-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000']
obsv_2 = df2['seasadj_log'].loc['2018-01-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000']
rmse(pred_2,obsv_2) #0.0014153190808288993

pred_3 = df2['insample_prediction_level_seas'].loc['2018-01-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000']
obsv_3 = df2['Loan_LoanWith'].loc['2018-01-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000']
rmse(pred_3,obsv_3)

#out-sample forecast

model_fit.plot_predict(start = '2019-04-01T00:00:00.000000000',
                       end = '2020-03-01T00:00:00.000000000',
                       plot_insample=False)

os_prediction = model_fit.predict(start = '2019-04-01T00:00:00.000000000',
                                  end = '2020-03-01T00:00:00.000000000',
                                  typ = 'levels')

os_prediction_df = pd.DataFrame(os_prediction, columns=['outsample_prediction_level']).apply(lambda x: np.exp(x))

df3 = pd.concat([os_prediction_df, df2], axis=1)

df3['seasonal'].loc['2019-04-01T00:00:00.000000000':'2020-03-01T00:00:00.000000000'] = df2['seasonal'].loc['2018-04-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000'].values #repeat seasonal values

df3['irregular'].loc['2019-04-01T00:00:00.000000000':'2020-03-01T00:00:00.000000000'] = df2['irregular'].loc['2018-04-01T00:00:00.000000000':'2019-03-01T00:00:00.000000000'].values #repeat irregular values

df3['final_fcst'] = df3['outsample_prediction_level'] + df3['seasonal'] + df3['irregular']

pd.concat([df3['final_fcst'],df3['Loan_LoanWith']], axis =1).plot()


#Experiment
from pyramid.arima import auto_arima


stepwise_model = auto_arima(df2['seasadj_log'],
                            start_p = 1,
                            start_q = 1,
                            max_p = 10,
                            max_q = 10,
                            m = 12,
                            seasonal = False,
                            trace = True,
                            d = 2,
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






