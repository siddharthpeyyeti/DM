# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:42:35 2020

@author: Siddharth
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import os
x=os.getcwd()
rainfall=pd.read_csv(x+'/Sub_Division_IMD_2017.csv')


###Calculate Mean of a column
def cal_mean(readings):
    """
    Function to calculate the mean value of the input readings
    :param readings:
    :return:
    """
    readings_total = sum(readings)
    number_of_readings = len(readings)
    mean = readings_total / float(number_of_readings)
    return mean
 
###Calculate variance of a column
def cal_variance(readings):
    """
    Calculating the variance of the readings
    :param readings:
    :return:
    """
 
    # To calculate the variance we need the mean value
    # Calculating the mean value from the cal_mean function
    readings_mean = cal_mean(readings)
    # mean difference squared readings
    mean_difference_squared_readings = [pow((reading - readings_mean), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)
 
###Calculate Covariance of a column
def cal_covariance(readings_1, readings_2):
    """
    Calculate the covariance between two different list of readings
    :param readings_1:
    :param readings_2:
    :return:
    """
    readings_1_mean = cal_mean(readings_1)
    readings_2_mean = cal_mean(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in xrange(0, readings_size):
        covariance += (readings_1[i] - readings_1_mean) * (readings_2[i] - readings_2_mean)
    return covariance / float(readings_size - 1)
 
###Calculate the linear regression coefficients
def cal_simple_linear_regression_coefficients(x_readings, y_readings):
    """
    Calculating the simple linear regression coefficients (B0, B1)
    :param x_readings:
    :param y_readings:
    :return:
    """
    # Coefficient B1 = covariance of x_readings and y_readings divided by variance of x_readings
    # Directly calling the implemented covariance and the variance functions
    # To calculate the coefficient B1
    b1 = cal_covariance(x_readings, y_readings) / float(cal_variance(x_readings))
 
    # Coefficient B0 = mean of y_readings - ( B1 * the mean of the x_readings )
    b0 = cal_mean(y_readings) - (b1 * cal_mean(x_readings))
    return b0, b1
 
 
def predict_target_value(x, b0, b1):
    """
    Calculating the target (y) value using the input x and the coefficients b0, b1
    :param x:
    :param b0:
    :param b1:
    :return:
    """
    return b0 + b1 * x
 
 
def cal_rmse(actual_readings, predicted_readings):
    """
    Calculating the root mean square error
    :param actual_readings:
    :param predicted_readings:
    :return:
    """
    square_error_total = 0.0
    total_readings = len(actual_readings)
    for i in xrange(0, total_readings):
        error = predicted_readings[i] - actual_readings[i]
        square_error_total += pow(error, 2)
    rmse = square_error_total / float(total_readings)
    return rmse
 
 
###Load the data

x = rainfall.SUBDIVISION
states=[]

for i in range(len(x)):
    if x[i] not in states:
        states.append(x[i])

###list of state dataframes
l_state_dfs = []

for i in range(len(states)):
    n=g[i]
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

###Use functions defined to perform linear regression

for i in range(len(states)):
    mean = cal_mean(l_state_dfs[i]['value'])
    
    yrs = l_state_dfs[i]['YEAR']
    
    
    # Calculating the mean of the years and the annual rainfall
    years_mean = cal_mean(l_state_dfs[i]['YEAR'])
    rainfall_mean = cal_mean(l_state_dfs[i]['value'])
    
    
    years_variance = cal_variance(l_state_dfs[i]['YEAR'])
    rainfall_variance = cal_variance(l_state_dfs[i]['value'])
    
    # Calculating the regression
    covariance_of_rainfall_and_years = l_state_dfs[i].cov()['YEAR']['value']
    w1 = covariance_of_rainfall_and_years / float(years_variance)
     
    w0 = rainfall_mean - (w1 * years_mean)
     
    # Predictions
    l_state_dfs[i]['Predicted_Rainfall'] = w0 + w1 * l_state_dfs[i]['YEAR']
    
    ###plot
    plt.figure()
    plt.plot(l_state_dfs[i]['YEAR'], l_state_dfs[i]['value'])
    plt.plot(l_state_dfs[i]['YEAR'], l_state_dfs[i]['Predicted_Rainfall'])
    
    ###convert annual rainfall back to mm
    na=[]
    for j in range(len(l_state_dfs[i].value)):
        u=l_state_dfs[i]['Predicted_Rainfall'][j]*(b -a)+a 
        na.append(u)
    l_state_dfs[i].drop(['Predicted_Rainfall'],axis=1)
    l_state_dfs[i]['Predicted_Rainfall'] = na