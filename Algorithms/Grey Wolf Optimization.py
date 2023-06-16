# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 18:04:57 2019

@author: Junjie Zhu
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0
        

import random
import numpy
import time
from multiprocessing import Pool 
from functools import partial
from colorama import Fore, Back, Style  

class GWOfun():
    
    def compute_objective_function(self,Positions):
        results = self.p.map(
            partial(self.objf), Positions)
        return np.array(results) 
                        
    def GWO(self, objf, lb, ub, dim, SearchAgents_no, Max_iter):       
        # initialize alpha, beta, and delta_pos
        self.Alpha_pos=numpy.zeros(dim)
        self.Alpha_score=float("inf")
        
        self.Beta_pos=numpy.zeros(dim)
        self.Beta_score=float("inf")
        
        self.Delta_pos=numpy.zeros(dim)
        self.Delta_score=float("inf")
        
        self.objf=objf
        self.Max_iter=Max_iter
        self.SearchAgents_no=SearchAgents_no
        self.dim=dim
        
        if not isinstance(lb, list):
            lb = [lb] * dim
        if not isinstance(ub, list):
            ub = [ub] * dim
        
        #Initialize the positions of search agents
        self.Positions = numpy.zeros((SearchAgents_no, dim))
        for i in range(dim):
            self.Positions[:, i] = numpy.random.uniform(0,1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        
        Convergence_curve=numpy.zeros(Max_iter)
        s=solution()
    
         # Loop counter
        print("GWO is optimizing  \""+objf.__name__+"\"")    
        
        timerStart=time.time() 
        s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        
        timeriter = numpy.zeros((Max_iter,1))
        timerdelta = numpy.zeros((Max_iter,1))
        # Main loop

        #j = range(dim)
        self.p=Pool(20) #the number of parallel processes           
        #Distribute the parameter sets evenly across the cores
        termination_index = 0
        for l in range(0, Max_iter):
                      
            #Generate values for each parameter
            i = range(SearchAgents_no)
            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(self.dim):
                self.Positions[i,j]=numpy.clip(self.Positions[i,j], lb[j], ub[j])
                #fitness0=[]
            if __name__=='__main__':
                fitness=f.compute_objective_function(self.Positions)
                
            for i in range(SearchAgents_no):
                
                if fitness[i]<self.Alpha_score:
                    self.Delta_score=self.Beta_score
                    self.Beta_score=self.Alpha_score
                    self.Alpha_score=fitness[i]; # Update alpha
                    self.Alpha_pos=self.Positions[i].copy()
                    
                if (fitness[i]>self.Alpha_score and fitness[i]<self.Beta_score):
                    self.Delta_score=self.Beta_score
                    self.Beta_score=fitness[i]  # Update beta
                    self.Beta_pos=self.Positions[i].copy()
                
                if (fitness[i]>self.Alpha_score and fitness[i]>self.Beta_score and fitness[i]<self.Delta_score): 
                    self.Delta_score=fitness[i] # Update delta
                    self.Delta_pos=self.Positions[i].copy()
            
            
            a=2-l*((2)/Max_iter); # a decreases linearly fron 2 to 0
            

            # Update the Position of search agents including omegas
            for i in range(0,self.SearchAgents_no):
                for j in range (0,self.dim):     
                                                 
                    r1=random.random() # r1 is a random number in [0,1]
                    r2=random.random() # r2 is a random number in [0,1]
                    
                    A1=2*a*r1-a; # Equation (3.3)
                    C1=2*r2; # Equation (3.4)
                    
                    D_alpha=abs(C1*self.Alpha_pos[j]-self.Positions[i,j]); # Equation (3.5)-part 1
                    X1=self.Alpha_pos[j]-A1*D_alpha; # Equation (3.6)-part 1
                               
                    r1=random.random()
                    r2=random.random()
                    
                    A2=2*a*r1-a; # Equation (3.3)
                    C2=2*r2; # Equation (3.4)
                    
                    D_beta=abs(C2*self.Beta_pos[j]-self.Positions[i,j]); # Equation (3.5)-part 2
                    X2=self.Beta_pos[j]-A2*D_beta; # Equation (3.6)-part 2       
                    
                    r1=random.random()
                    r2=random.random() 
                    
                    A3=2*a*r1-a; # Equation (3.3)
                    C3=2*r2; # Equation (3.4)
                    
                    D_delta=abs(C3*self.Delta_pos[j]-self.Positions[i,j]); # Equation (3.5)-part 3
                    X3=self.Delta_pos[j]-A3*D_delta; # Equation (3.5)-part 3             
                    
                    self.Positions[i,j]=(X1+X2+X3)/3  # Equation (3.7)
            
            Convergence_curve[l]=self.Alpha_score
  
          
            if l == 0:
                timeriter[0,0] = time.time()
                timerdelta[0,0] = timeriter[0,0] - timerStart
            if l > 0:
                timeriter[l,0] = time.time()
                timerdelta[l,0] = (timeriter[l,0] - timeriter[l-1,0]) 
            
            if (l%1==0):
                   print('At iteration '+ str(l)+ ' the best fitness is '+ Fore.RED + str(round(self.Alpha_score,5))
                    + Style.RESET_ALL +' and computing time is '+str(round(timerdelta[l,0],3))
                    + ' seconds;\nBest positions are ' + str([round(float(i),4) for i in self.Alpha_pos]))
        
        
            if l == 0:
                global_best = self.Alpha_score
            else:
                if (global_best == self.Alpha_score) or ((self.Positions[-2] == self.Positions[-1]).all()):
                    termination_index = termination_index +1
                elif (global_best != self.Alpha_score) and (global_best - self.Alpha_score < 0.0001):
                    print('Cost change is less than the threshold '+str(0.0001))
                    break
                else:
                    global_best = self.Alpha_score
                    termination_index = 1
            if termination_index == 10: #if the cost is staying the same for 10 iterations
                print('Same solution repeats '+str(termination_index)+' times')
                break      
        
        timerEnd=time.time()  
        s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
        s.executionTime=timerEnd-timerStart
        
        print('All iterations are completed, '+ ' the final best fitness is '+ str(self.Alpha_score)+
              '\nThe final best positions are '+ str([round(float(i),5) for i in self.Alpha_pos])+
              ' \nThe total computing time is '+str(s.executionTime)+ ' seconds') 
        
        s.convergence=Convergence_curve
        s.optimizer="GWO"
        s.objfname=objf.__name__
        s.positions = self.Alpha_pos
        s.iter = l
        return s






###------------------------------------------------------------------------------------------
##model initialization        

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

def MLPfun(x):
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


#lower and upper bounds
lb = [5,   0,   5, 0,  1E-4,   0,   0.5,  0,  5, 5]
ub = [200, 1, 200, 1,  0.5, 0.1, 0.999, 1, 60, 20]


f = GWOfun()
SImodel = f.GWO(objf=MLPfun, lb=lb, ub=ub, dim=10, SearchAgents_no=50, Max_iter=100)



###-----------------------------------------------------------------------------------------------
##obtain model info

comptime = SImodel.executionTime
costs = SImodel.convergence
postions = SImodel.positions
numiter = SImodel.iter
bestcost = SImodel.convergence[numiter]
x = SImodel.positions


