# -*- coding: utf-8 -*-
"""
Created on May 22, 2019

@author: Junjie Zhu
"""


import numpy as np
from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return MAPE

def adjusted_R_squared(X,R_squared):
    a = (1-R_squared)
    b = (len(X)-1)
    c = (len(X)-X.shape[1]-1)
    Adj_R2 = 1- ((a*b)/c)
    return Adj_R2   


data_train = pd.read_excel('Your training dataset.xlsx')
data_valid = pd.read_excel('Your validation dataset.xlsx')
data_test = pd.read_excel('Your testing dataset.xlsx')

#data preprocessing, changing them based on your datasets
data_tr = data_train.iloc[:,:].values
obs_tr = len(data_tr.T[0])
num_var = len(data_tr.T)-1

data_vd = data_valid.iloc[:,:].values
obs_vd = len(data_vd.T[0])

data_te = data_test.iloc[:,:].values
obs_te = len(data_te.T[0])


#split x and y
X_tr = data_tr[:,:-1]
y_tr = data_tr[:,-1]
y_tr = y_tr.reshape(len(y_tr), 1)

X_vd = data_vd[:,:-1]
y_vd = data_vd[:,-1]
y_vd = y_vd.reshape(len(y_vd), 1)

X_te = data_te[:,:-1]
y_te = data_te[:,-1]
y_te = y_te.reshape(len(y_te), 1)


# scale data
scaler = preprocessing.MinMaxScaler()
scaler.fit(data_tr)
scaled_data_tr = scaler.transform(data_tr)
scaled_data_vd = scaler.transform(data_vd)
scaled_data_te = scaler.transform(data_te)

scaler_rv = preprocessing.MinMaxScaler()
scaler_rv.fit(pd.DataFrame(data_tr.iloc[:,-1]))

# split scaled data into training, validation, and testing #
scaled_X_tr = scaled_data_tr[:,:-1]
scaled_y_tr = scaled_data_tr[:, -1]

scaled_X_vd = scaled_data_vd[:,:-1]
scaled_y_vd = scaled_data_vd[:, -1]

scaled_X_te  = scaled_data_te[:,:-1]
scaled_y_te = scaled_data_te[:,-1]

  
acfun = ['sigmoid', 'relu']   

def func(x):
    np.random.seed(300)  # for reproducibility
    x0 = int(round(x[0]))
    x1 = int(round(x[1]))
    x2 = int(round(x[2]))    
    x3 = int(round(x[3]))
    x7 = int(round(x[7]))  
    x8 = int(round(x[8]))    
    x9 = int(round(x[9]))
    
    if x7 == 0:
        opfun = optimizers.SGD(lr=x[4], decay=x[5], momentum=x[6], nesterov=True)
    else:
        opfun = optimizers.Adam(lr=x[4], beta_1=0.9, beta_2=0.999, amsgrad=False)
        
# create model
    model1 = Sequential()
    model1.add(Dense(x0, input_dim=num_var, kernel_initializer='normal', activation=acfun[x1]))
    model1.add(Dense(x2, activation=acfun[x3]))
    model1.add(Dense(1))
# Compile model

    model1.compile(loss='mean_squared_error', optimizer= opfun)

    model1.fit(scaled_X_tr, scaled_y_tr, epochs=x8, batch_size=x9, verbose=0, shuffle=False)

    y_pred_vd_scaled = model1.predict(scaled_X_vd)
    y_pred_vd  = scaler_rv.inverse_transform(y_pred_vd_scaled)
    
    MAPE_vd = mean_absolute_percentage_error(y_vd, y_pred_vd) 
 
    return MAPE_vd


def f(x):
    n_particles = x.shape[0]
    j = [func(x[i]) for i in range(n_particles)]
    return np.array(j)
    
from pyswarms.single.global_best import GlobalBestPSO

bounds = (np.array([5,   0,   5, 0,  1E-4,   0,   0.5,  0,  5, 5]), 
          np.array([200, 1, 200, 1,  0.5, 0.1, 0.999, 1, 60, 20]))

options = {'c1': 1, 'c2': 1, 'w': 0.5, 'k': 2, 'p': 1}

optimizer = GlobalBestPSO(n_particles=50, dimensions=10, options=options, bounds=bounds, ftol=0.0001)

cost, pos = optimizer.optimize(f, iters=100, n_processes = 20)


