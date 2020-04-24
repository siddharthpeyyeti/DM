# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:42:35 2020

@author: Siddharth
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
rainfall=pd.read_csv(r'C:\Users\Siddharth\Desktop\BITS\3 2\DM\project\Sub_Division_IMD_2017.csv')


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
tn_df = rainfall[rainfall['SUBDIVISION'] == 'Tamil Nadu']
months = ['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JF', 'MAM', 'JJAS', 'OND']
tn_df=tn_df.drop(months, axis=1)
melted = tn_df.melt('YEAR').reset_index()

###Arrange by years
df = melted[['YEAR','variable','value']].reset_index().sort_values(by=['YEAR','index'])

###Use functions defined to perform linear regression
mean = cal_mean(df['value'])

yrs = df['YEAR']


# Calculating the mean of the years and the annual rainfall
years_mean = cal_mean(df['YEAR'])
rainfall_mean = cal_mean(df['value'])


years_variance = cal_variance(df['YEAR'])
rainfall_variance = cal_variance(df['value'])

# Calculating the regression
covariance_of_rainfall_and_years = df.cov()['YEAR']['value']
w1 = covariance_of_rainfall_and_years / float(years_variance)
 
w0 = rainfall_mean - (w1 * years_mean)
 
# Predictions
df['Predicted_Rainfall'] = w0 + w1 * df['YEAR']


plt.plot(df['YEAR'], df['value'])
plt.plot(df['YEAR'], df['Predicted_Price'])