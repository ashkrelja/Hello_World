import numpy as np
import pandas as pd
from validate_email import validate_email
import xlrd
import pyDNS

path = "C:/Users/ashkrelja/Documents/Wall_Street_Lending/Marketing/Email/Programmatic_Email/Realtor_Data/Realtor_MI_Emails_All.xlsx"

df = pd.read_excel(path)

#ad-hoc email client analysis

df1 = df.dropna(subset = ['Email'])

df1.to_excel("C:/Users/ashkrelja/Documents/Wall_Street_Lending/Marketing/Email/Programmatic_Email/Realtor_Data/Realtor_MI_Emails_All_Prepped.xlsx")

dfgmail = df1[df1['Email'].str.contains('gmail')].count() #1,794 or 35%

df1[df1['Email'].str.contains('hotmail')].count()

df1[df1['Email'].str.contains('yahoo')].count()

######

path2 = "C:/Users/ashkrelja/Documents/Wall_Street_Lending/Marketing/Email/Programmatic_Email/Realtor_Data/Realtor_MI_Emails_All_Prepped.xlsx"

workbook = xlrd.open_workbook(path2, "r")

worksheet = workbook.sheet_by_index(0)

valid_list = []

for index in range(1,worksheet.nrows):
    
    is_valid = validate_email(str(worksheet.cell(index,8).value), verify = True)
    
    valid_list.append(is_valid)
    
    print(index)
    
df0 = pd.DataFrame(valid_list)
df1=df0.rename(columns={0:"Valid"})

dataset = pd.read_excel(path2)
dataset2 = dataset.reset_index(drop=True)

dataset3 = pd.merge( df1 , dataset2 , left_index = True , right_index = True)
dataset4=dataset3[dataset3['Valid']==1]

path3="C:/Users/ashkrelja/Documents/Wall_Street_Lending/Marketing/Email/Programmatic_Email/Realtor_Data/Realtor_MI_Emails_Final.xlsx"

dataset4.to_excel(path3)




