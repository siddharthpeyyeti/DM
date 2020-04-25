# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:42:35 2020

@author: Siddharth
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
#import os
#x=os.getcwd()
rf_data=pd.read_csv(r'C:\Users\Siddharth\Desktop\BITS\3 2\DM\project\Sub_Division_IMD_2017.csv')

###Calculate Mean of a column
def cal_mean(readings):

    readings_total = sum(readings)
    number_of_readings = len(readings)
    mean = readings_total / float(number_of_readings)
    return mean
 
###Calculate variance of a column
def cal_variance(readings):
    # To calculate the variance we need the mean value
    # Calculating the mean value from the cal_mean function
    readings_mean = cal_mean(readings)
    # mean difference squared readings
    mean_difference_squared_readings = [pow((reading - readings_mean), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)
 
###Calculate Covariance of a column
def cal_covariance(readings_1, readings_2):
    readings_1_mean = cal_mean(readings_1)
    readings_2_mean = cal_mean(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in xrange(0, readings_size):
        covariance += (readings_1[i] - readings_1_mean) * (readings_2[i] - readings_2_mean)
    return covariance / float(readings_size - 1)
 

###Calculate the linear regression coefficients

def lin_reg(df):
    # Calculating the mean of the years and the annual rainfall
    years_mean = cal_mean(df['YEAR'])
    rainfall_mean = cal_mean(df['value'])
    
    # Calculating the variance of the years and the annual rainfall
    years_variance = cal_variance(df['YEAR'])
    
    # Calculating the regression    
    covariance_of_rainfall_and_years = df.cov()['YEAR']['value']

    w1 = covariance_of_rainfall_and_years / float(years_variance)
     
    w0 = rainfall_mean - (w1 * years_mean)
    
    return [w0, w1]

###Calculate rmse for given dataframe
def cal_rmse(df):
    soe = 0
    n = len(df['value'])
    for x in range (n):
        actual = test_vals['value'].iloc[x]
        predicted = test_vals['Predicted_Rainfall'].iloc[x]
        soe = soe + ((actual-predicted)**2)/n
    return sqrt(soe)
 
 
###Load the data

x = rf_data.SUBDIVISION
states=[]

for i in range(len(x)):
    if x[i] not in states:
        states.append(x[i])

###list of state dataframes
l_state_dfs = []

for i in range(len(states)):
    n=states[i]
    x=rf_data[rf_data.SUBDIVISION == n]
    l_state_dfs.append(x)

###create list of columns to drop
months = ['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JF', 'MAM', 'JJAS', 'OND']

###drop the columns for all dataframes and melt
for i in range(len(states)):
    l_state_dfs[i]=l_state_dfs[i].drop(months, axis=1)
    l_state_dfs[i] = l_state_dfs[i].melt('YEAR').reset_index()

###Remove nans (loop through states and replace with annual mean)
for i in range(len(states)):
    mean_of_state=l_state_dfs[i].value.mean()
    l_state_dfs[i].value = l_state_dfs[i].value.fillna(mean_of_state)

'''
###normalize annual rainfall
na=[]
for i in range(len(states)):
    a=min(l_state_dfs[i].value)
    b=max(l_state_dfs[i].value)
    
    for j in range(len(l_state_dfs[i].value)):
        u = (l_state_dfs[i]['value'][j]-a)/(b -a)
        na.append(u)
    
    l_state_dfs[i].drop(['value'],axis=1)
    l_state_dfs[i]['value'] = na
    na = []
'''
    
###Use functions defined to perform linear regression

errors = [] #list of RMSE for each state

for i in range(len(states)):
    #full contains original dataframe [i]
    full=l_state_dfs[i]
    #l_state_df[i] contains first 70% of dataframe
    l_state_dfs[i]=l_state_dfs[i].head(int(len(l_state_dfs[i])*(70/100)))
    #get coefficients using linear regression function
    coeffs = lin_reg(l_state_dfs[i])
    w0 = coeffs[0]
    w1 = coeffs[1]
    # Predictions
    full['Predicted_Rainfall'] = w0 + w1 * full['YEAR']
    
    #RMSE calculation
    test_vals = full.tail(int(len(l_state_dfs[i])*(30/100))+1)
    rmse = cal_rmse(test_vals)
    errors.append(rmse)
    
    ###plot
    plt.figure()
    plt.plot(full['YEAR'], full['value'])
    plt.plot(full['YEAR'], full['Predicted_Rainfall'])
    plt.title(states[i])
    plt.xlabel('YEAR')
    plt.ylabel('Annual Rainfall in mm')

    '''
    ###convert annual rainfall back to mm
    na=[]
    for j in range(len(l_state_dfs[i].value)):
        u=l_state_dfs[i]['Predicted_Rainfall'][j]*(b -a)+a 
        na.append(u)
    l_state_dfs[i].drop(['Predicted_Rainfall'],axis=1)
    l_state_dfs[i]['Predicted_Rainfall'] = na
    '''