# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:48:12 2020

@author: Vijay
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from mpl_toolkits.axes_grid1 import make_axes_locatable

states = gpd.read_file("C:/Users/Siddharth/Desktop/BITS/3 2/DM/project/india_telengana.shp")

rf_data=pd.read_csv(r'C:\Users\Siddharth\Desktop\BITS\3 2\DM\project\Sub_Division_IMD_2017.csv')

x = rf_data.SUBDIVISION
g=[]

for i in range(len(x)):
    if x[i] not in g:
        g.append(x[i])
        
l=[]
for i in range(len(g)):
    n=g[i]
    state=rf_data[rf_data.SUBDIVISION == n]
    l.append(state)

#plotting raw data
#rainfall in mm from 1900 to 2015, statewise
x=rf_data.SUBDIVISION
for i in range(len(g)):
    y=rf_data[x==g[i]]
    fig = plt.figure()
    fig.suptitle(g[i], fontsize=10)
    plt.plot(y.YEAR,y.ANNUAL)
    plt.xlabel('YEAR')
    plt.ylabel('Annual Rainfall in mm')
    fig.savefig(g[i] + '.png')

#remove all the rows with more than 2 or more quarters with NaN
c=0
rowlist=[]
col = ['SUBDIVISION', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'JF', 'MAM', 'JJAS', 'OND']
rowlist1=pd.DataFrame(columns=['SUBDIVISION', 'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL', 'JF', 'MAM', 'JJAS', 'OND'])
for j in range(len(l)):
    for i in range(len(l[j])):
        for column in ['JF','MAM','JJAS','OND']:
            k=l[j].columns.get_loc(column)
            if np.isnan(l[j].iloc[i][k]):
                c+=1
        if c<2:
            rowlist.append(l[j].iloc[i])
        c=0
prelocalmean=rowlist1.append(rowlist, ignore_index=True)

#divide by states
p_states = [] #processed states, no Nan in quarters
for i in range(len(g)):
    n=g[i]
    s=prelocalmean[prelocalmean.SUBDIVISION==n]
    p_states.append(s)

#filling nans with mean values of the respective columns
lp=[]
for state in range(len(p_states)):
    h=p_states[state]    
    h=h.fillna(h.mean())
    lp.append(h)
    
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
lp[1].iloc[0]['JAN']
x=[]
sum=0

for i in range(len(lp)):
    for j in range(len(lp[i])):
        sum=0
        for k in months:
            sum+=lp[i].iloc[j][k]            
        x.append(sum)

df=pd.concat([lp[i] for i in range(len(lp))])
n=df.columns.get_loc('ANNUAL')
df=df.drop(['ANNUAL'],axis=1)
df['ANNUAL'] = x

col_values = []
x=[]
minmax=[]
na=[]

#Normalising Monthly rainfall
for i in range(len(lp)):
    df1=lp[i]
    for k in range(len(months)):
        a=min(df1[months[k]])
        b=max(df1[months[k]])
        n=lp[i].columns.get_loc(months[k])
        for j in range(len(df1[months[k]])):
            u = (lp[i].iloc[j][months[k]]-a)/(b -a)
            na.append(u)
        n=df1.columns.get_loc(months[k])
        df=df1.drop([months[k]],axis=1)
        df1[months[k]] = na
        lp[i]=df1
        na=[]


#Normalising Annual Rainfall
for i in range(len(lp)):
    df1=lp[i]
    a=min(lp[i].ANNUAL)
    b=max(lp[i].ANNUAL)
    n=lp[i].columns.get_loc('ANNUAL')
    for j in range(len(lp[i].ANNUAL)):
        u = (lp[i].iloc[j]['ANNUAL']-a)/(b -a)
        na.append(u)
    n=df1.columns.get_loc('ANNUAL')
    df=df1.drop(['ANNUAL'],axis=1)
    df1['ANNUAL'] = na
    lp[i]=df1
    na=[]

# Rescale dataset columns to the range 0-1         

#plot annual rainfall per year for each state
for i in range(len(lp)):
    plt = lp[i].plot('YEAR','ANNUAL', title=g[i])
    plt.figure.savefig(g[i] + '.png')


# Getting Mean of Monthly and Annual Rainfall 
means_months = pd.DataFrame(columns=['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
means_months['SUBDIVISION']=g
for k in range(len(months)):
    mon=[]
    n=means_months.columns.get_loc(months[k])
    for i in range(len(lp)):
        df1=lp[i]
        mean=df1[months[k]].mean()
        mon.append(mean)
    means_months[months[k]]=mon
# Estimate Mean and standard deviation
means_annual = []
for i in range(len(lp)):
    df1=lp[i]
    mean=lp[i].ANNUAL.mean()
    means_annual.append(mean)
    

# Calculate Annual Standard Deviations

stdevs_annual = []
s=0;
for i in range(len(lp)):
    df1=lp[i]
    n=df1.columns.get_loc('ANNUAL')
    for j in range(len(lp[i].ANNUAL)):
        k=df1.iloc[j][n]
        variance=pow(k-means_annual[i],2) 
        s+=variance
    stdevs_annual.append(s)
    
stdevs_annual[i]
for i in range(len(stdevs_annual)):
    stdevs_annual[i]= [sqrt(stdevs_annual[i]/(float(len(lp[i])-1)))]
    
#Calculating Standard Deviation of all the months for each state over the years 


###VISUALIZATION	

#create data frame of states and means of annual rainfall
annual_means = pd.DataFrame(columns=['SUBDIVISION', 'MEAN'])
annual_means['SUBDIVISION']=g #states
annual_means['MEAN']=means_annual #mean of annual rainfall 


#dictionary to map to states_on_geojson : meteorological regions
mydict = {"Andaman and Nicobar": ["Andaman & Nicobar Islands"],
"Telangana": ["Telangana"],
"Andhra Pradesh": ["Rayalseema", "Coastal Andhra Pradesh"],
"Arunachal Pradesh": ["Arunachal Pradesh"],
"Assam": ["Assam & Meghalaya"],
"Bihar": ["Bihar"],
"Chandigarh": ["Haryana Delhi & Chandigarh"],
"Chhattisgarh": ["Chhattisgarh"],
"Dadra and Nagar Haveli": [],
"Daman and Diu": [],
"Delhi": ["Haryana Delhi & Chandigarh"],
"Goa": ["Konkan & Goa"],
"Gujarat": ["Gujarat Region", "Saurashtra & Kutch"],
"Haryana": ["Haryana Delhi & Chandigarh"],
"Himachal Pradesh": ["Himachal Pradesh"],
"Jammu and Kashmir": ["Jammu & Kashmir"],
"Jharkhand": ["Jharkhand"],
"Karnataka": ["Coastal Karnataka", "North Interior Karnataka", "South Interior Karnataka"],
"Kerala": ["Kerala"],
"Lakshadweep": ["Lakshadweep"],
"Madhya Pradesh": ["East Madhya Pradesh", "West Madhya Pradesh"],
"Maharashtra": ["Madhya Maharashtra", "Matathwada", "Vidarbha"],
"Manipur": ["Naga Mani Mizo Tripura"],
"Meghalaya": ["Assam & Meghalaya"],
"Mizoram": ["Naga Mani Mizo Tripura"],
"Nagaland": ["Naga Mani Mizo Tripura"],
"Orissa" : ["Orissa"],
"Puducherry": [],"Punjab": ["Punjab"],
"Rajasthan": ["East Rajasthan", "West Rajasthan"],
"Sikkim": ["Sub Himalayan West Bengal & Sikkim"],
"Tamil Nadu": ["Tamil Nadu"],
"Tripura": ["Naga Mani Mizo Tripura"],
"Uttar Pradesh": ["East Uttar Pradesh", "West Uttar Pradesh"],
"Uttaranchal": ["Uttarakhand"],
"West Bengal": ["Gangetic West Bengal", "Sub Himalayan West Bengal & Sikkim"]
}


#create a list of means for the geojson states
plot_mean = []

#iterate through the dictionary, take average of multiple states in value array
for key, val in mydict.items():
    sumof = 0
    count=0
    for i in range(len(val)):
        reg = val[i]
        print (val[i])
        si=annual_means[annual_means.SUBDIVISION==reg]
        sumof+=si.iloc[0][1]
    if len(val)==0:
        plot_mean.append(0) #append the average values
    else:
        plot_mean.append(sumof/len(val)) #append the average values
        
states['MEANS']=plot_mean #add the average annual rainfall of all the years as a column

#plot the column on the map
fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad=0.1)
states.plot(cmap='jet',column='MEANS',figsize=(10,10), ax=ax, legend=True, cax=cax)
make_axes_locatable(ax)

#Aggregation

