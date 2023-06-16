# -*- coding: utf-8 -*-
"""
Created on Oct, 2019

@author: Junjie Zhu
"""


import numpy as np
from sklearn import preprocessing
import pandas as pd
import random
from deap import base
from deap import creator
from deap import tools
import time
import multiprocessing
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

def evalfunc(individual):
    np.random.seed(300)  # for reproducibility
    x0 = int(5 + individual[0]//(1/196))
    x1 = int(round(individual[1]))
    x2 = int(5 + individual[2]//(1/196))   
    x3 = int(round(individual[3]))
    x4 = (0.5 - 0.0001) * individual[4] + 0.0001
    x5 = individual[5]/10
    x6 = (0.999 - 0.5) * individual[6] + 0.5
    x7 = int(round(individual[7]))
    x8 = int(5 + individual[8]//(1/56)) 
    x9 = int(5 + individual[9]//(1/16))
    if x7 == 0:
        opfun = optimizers.SGD(lr=x4, decay=x5, momentum=x6, nesterov=True)
    else:
        opfun = optimizers.Adam(lr=x4, beta_1=0.9, beta_2=0.999, amsgrad=False)
        
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
   
    return MAPE_vd,


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
 
toolbox = base.Toolbox()
pool = multiprocessing.Pool(20) #the number of parallel processes
toolbox.register("map", pool.map)

toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, 10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# register the goal / fitness function
toolbox.register("evaluate", evalfunc)
 
# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)
 
# register a mutation operator with a probability to
# flip each attribute/gene of 0.01
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
 
# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=8)

#create an initial population of 50 individuals
ppl = 50
pop = toolbox.population(n = ppl)
 
#CXPB is the probability with which two individuals are crossed
#MUTPB is the probability for mutating an individual
#NGEN  is the maximum number of generations for which the evolution runs
CXPB, MUTPB, NGEN = 0.6, 0.2, 100
genbest_ind = []
genbest_fit = []    



print("Start of evolution")
first_start = time.time()    
# Evaluate the entire population
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit
print("  Evaluated %i individuals" % len(pop))
total_ind = len(pop)
print("  Evaluation time for initial generation is %s sec" % (time.time() - first_start))    
# Begin the evolution
termination_index = 1
for g in range(NGEN):
    print("-- Generation %i --" % g)
    start_time = time.time()    
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
 
        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
 
            # fitness values of the children must be recalculated later
            del child1.fitness.values
            del child2.fitness.values
 
    for mutant in offspring:
 
        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    total_ind = total_ind + len(invalid_ind)    
    print("  Evaluated %i individuals" % len(invalid_ind))
        
    # The population is entirely replaced by the offspring
    pop[:] = offspring
        
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]
        
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    end_time = time.time()    
    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)
    
    candidate_ind = tools.selBest(pop, ppl)
    for j in range(0, ppl):
        if candidate_ind[j].fitness.values[0] == min(fits):        
            best_ind = candidate_ind[j]
            break 

    genbest_ind.append(best_ind)
    genbest_fit.append(best_ind.fitness.values)
    print("Best individual for generation %s is %s, %s" % (g, best_ind, best_ind.fitness.values))
    print("Evaluation time for generation %s is %s sec" % (g, (end_time - start_time)))
    if g == 0:
        global_best = best_ind.fitness.values
    else:
        if (global_best == best_ind.fitness.values) or ((genbest_ind[-2] == genbest_ind[-1]).all()):
            termination_index = termination_index +1
        else:
            global_best = best_ind.fitness.values
            termination_index = 1
    if termination_index == 10: #if the cost is staying the same for 10 iterations
        break
    if (global_best != best_ind.fitness.values) and (global_best[0] - best_ind.fitness.values[0] < 0.0001):
        break
print("-- End of (successful) evolution --")
    
print("Totally evaluated %i individuals" % total_ind)
print("Best individual for whole generation is %s, %s" % (best_ind, best_ind.fitness.values))
print("Total evaluation time for genetic algorithm is %s sec" %(time.time() - first_start)) 





