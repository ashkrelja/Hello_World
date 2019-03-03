# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:19:35 2018

@author: ashkrelja
"""
#ddd


import pandas as pd
import numpy as np
from tabula import read_pdf



df0 = []

for i in range(1,9):

    path = 'C:/Users/ashkrelja/Documents/Wall_Street_Lending/Technology/Analytics/Complimentary_Businesses/Real_Estate_Data/MI_Production_Report/Ranking Report_'+str(i)+'_2018.pdf'

    df = read_pdf(path, pages='all')
    
    df0.append(df)
    
df1 = pd.concat(df0)  

df2 = df1

df2.reset_index(inplace=True)
df2.fillna(' ',inplace=True)

df2['All'] = df2['Agent Name Office Name'] + df2['Listed'] + df2['Unnamed: 2'] + df2['Unnamed: 3']

df2.drop(['Agent Name Office Name', 'Listed','Unnamed: 2','Unnamed: 3'], axis=1, inplace = True)

df2.rename(columns={'Unnamed: 0': 'Rank'}, inplace=True)

df2['Agent Name']= df2['All'].str.split('(').str[0]
df2['Office Number'] = df2['All'].str.split('(').str[1]
df2['Office Number'] =df2['Office Number'].str.split(')').str[0]
df2.drop('All', axis=1,inplace=True)
df2.dropna(subset=['Office Number'], inplace=True)

df2.to_excel('C:/Users/ashkrelja/Documents/Wall_Street_Lending/Technology/Analytics/Complimentary_Businesses/Real_Estate_Data/MI_Production_Report/realtor_ranking.xlsx')




