# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:06:09 2018

@author: ashkrelja
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.x13 import x13_arima_analysis

path = 'C:/Users/ashkrelja/Documents/Wall_Street_Lending/Technology/Analytics/Operations_Analytics/Operations Analytics_03_2018_Copy.csv'

df = pd.read_csv(path, usecols = ['Status_ClosedDate', 'Loan_LoanWith'])


df.dropna(inplace = True)
df['Status_ClosedDate'] = df['Status_ClosedDate'].apply(lambda x: pd.to_datetime(x))
df.set_index('Status_ClosedDate', inplace = True)
df2 = df.resample('MS').sum()
df2.plot()

output = sm.x13_arima_analysis(df2['Loan_LoanWith'])

# examine results of X12/X13 procedure - Seasonality adjustment test failed, conclude series is not good candidate for
# seasonal adjustment

print(output.results)

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

