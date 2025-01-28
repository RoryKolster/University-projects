# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:40:38 2024

@author: kolst
"""

import numpy as np

import datetime
import matplotlib.pyplot as plt
import math
import time
from sklearn.metrics import balanced_accuracy_score
# np.random.seed(9)


# Obtaining the data
X = np.loadtxt("Data_X.csv", delimiter=",", dtype=float)
Y = np.loadtxt("Data_y.csv", delimiter=",", dtype=str)
Y = np.where(Y=="P",1, 0)

N = np.shape(X)[0]
J = np.shape(X)[1] # number of features
halfJ = int(np.floor(J/2))
halfJ2 = J - halfJ

# Training, validation and test split of the data
train_size = int(np.ceil(0.7 * N))
validation_size = int(np.ceil(0.2 * N))
test_size = N - train_size - validation_size
inds = np.arange(N)
train_inds = np.random.choice(inds, size=train_size, replace=False)
train_inds.sort()
rem_inds = np.setdiff1d(inds, train_inds)
validation_inds = np.random.choice(rem_inds, size=validation_size, replace=False)
test_inds = np.setdiff1d(rem_inds, validation_inds)
X_train = X[train_inds]
X_test = X[test_inds]
X_validation = X[validation_inds]
y_validation = Y[validation_inds]
y_train = Y[train_inds]
y_test = Y[test_inds]

def Objective_function(X,Y,s_j, tau_j):
    # Extract variables
    N = np.shape(X)[0] # number of patients
    J = np.shape(X)[1] # number of features

    # Select upper and lower Xjs and taujs
    upper_inds = np.argwhere(s_j==1)[:,1] 
    lower_inds = np.argwhere(s_j==0)[:,1] 
    V_j = X[:,upper_inds]
    W_j = X[:,lower_inds]
    upper_tau = tau_j[0,upper_inds] 
    lower_tau = tau_j[0,lower_inds] 
    y_hat = np.zeros(N)
    
    balanced_error = 1 - balanced_accuracy_score(Y, y_hat)

    
    for i in range(N):
        
        if np.all(V_j[i,:] < upper_tau):
            # uppers[i] = 1
            upper = 1
            if np.all(W_j[i,:] > lower_tau):
                y_hat[i] = 1
                pass

    obj_val = (1-Y).T @ y_hat + Y.T @ (1-y_hat)
    
    return obj_val, y_hat, balanced_error



def Neighbourhood_s(s_j, tau_j, X, patients): #, tau_steps, tau_flag, tau_counter):
    N = np.shape(X)[0] # number of patients
    J = np.shape(X)[1] # number of features

    var_neighbourhood_s = np.tile(s_j, (J,1))
    var_neighbourhood_s[np.arange(J), np.arange(J)] ^= 1 # XOR operation

    max_patients_X = np.max(X[patients, :], axis=0)
    min_patients_X = np.min(X[patients, :], axis=0)
    
    var_neighbourhood_tau = np.zeros((J,J))
    
    
    for s in range(J):
        
        new_tau = np.zeros((1,J))
        n_up = np.sum(var_neighbourhood_s[s,:])
        n_low = J - n_up
        
        
        tau1s = max_patients_X[var_neighbourhood_s[s,:].flatten() == 1] + 1
        tau0s = min_patients_X[var_neighbourhood_s[s,:].flatten() == 0] - 1
        new_tau[0,var_neighbourhood_s[s,:].flatten() == 1] = tau1s.reshape((1,n_up))
        new_tau[0,var_neighbourhood_s[s,:].flatten() == 0] = tau0s.reshape((1,n_low))
        
        var_neighbourhood_tau[s,:] = new_tau
    

    return var_neighbourhood_s, var_neighbourhood_tau

def Neighbourhood_tau(s_j, tau_j, tau_steps, phase2_depth):
    var_neighbourhood_tau = np.zeros((J,J))
    for s in range(J):
        s_condition = s_j[s]
        if s_condition == 1:
            new_tau = tau_j
            new_tau[s] = tau_steps[1+phase2_depth,s]
            var_neighbourhood_tau[s,:] = new_tau
        else:
            new_tau = tau_j
            new_tau[s] = tau_steps[-2-phase2_depth,s]
            var_neighbourhood_tau[s,:] = new_tau
            
    return var_neighbourhood_tau
    
def Local_Search(X_train, y_train, X_validation, y_validation, init_s, init_tau, Objective_function, Neighbourhood_s, Neighbourhood_tau, tau_size=50, tau_len=10, phase2_depth=2):
    N = np.shape(X_train)[0] # number of datapoints in training set
    J = np.shape(X_train)[1] # number of features
    M = np.shape(X_validation)[0] # number of datapoints in validation set
    patients = np.argwhere(y_train == 1) .flatten()
    max_patients_X = np.max(X_train[patients, :], axis=0)
    min_patients_X = np.min(X_train[patients, :], axis=0)
    
    # initializing parameters
    phase1_tau = init_tau
    phase1_s = init_s
    iterations = 0
    tau_flag = 0
    tau_iter = 0
    save_flag = 0
    
    
    # linspace of tau steps
    tau_steps = np.zeros((tau_size, J))
    
    
    for j in range(J):
        steps = np.linspace(min_patients_X[j], max_patients_X[j], num=tau_size)
        tau_steps[:,j] = steps
    
    # Initialise path
    path_obj = []
    second_search_s = np.zeros((tau_len, J))
    second_search_tau = np.zeros((tau_len, J))
    
    while True:
    
        
        if tau_flag == 0:
            var_neighbourhood_s, var_neighbourhood_tau = Neighbourhood_s(phase1_s, phase1_tau, X_train, patients) #, tau_steps, tau_flag, tau_counter)
            
            obj_vals = np.zeros((J, 1))
            balanced_errors = np.zeros((J,1))
            
            for k in range(J):
                obj_vals[k], _, balanced_errors[k] = Objective_function(X_train,y_train, var_neighbourhood_s[k,:].reshape((1,J)), var_neighbourhood_tau[k,:].reshape((1,J)))
                
                

            phase1_ind = np.argmin(obj_vals)
            phase1_obj = np.min(obj_vals) 
            phase1_tau = var_neighbourhood_tau[phase1_ind, :].reshape(1,J)
            phase1_s = var_neighbourhood_s[phase1_ind, :].reshape(1,J)
            
            
        # Save path
        path_obj.append(phase1_obj)
        
        if len(path_obj) >= 10 and np.all(path_obj[-6:]==path_obj[-1]):
            save_flag = 1
            
        if save_flag == 1 and tau_iter < tau_len:
            
            second_search_s[tau_iter, :] = phase1_s
            second_search_tau[tau_iter, :] = phase1_tau
            tau_iter += 1
            if tau_iter == tau_len:
                tau_flag = 1
                print("Entering phase 2")
            
        if tau_flag == 1:
            best_second_search_s = np.zeros((tau_len, J))
            best_second_search_tau = np.zeros((tau_len, J))
            obj_second_search = np.zeros((tau_len, 1))
            validation_objs = np.zeros((tau_len, 1))
            y_hats_phase2 = np.zeros((tau_len, M))
            validation_errors = np.zeros((tau_len,1))
            var2_neighbourhood_tau = np.zeros((J, J))
            phase2_iter = 1
            phase2_best_obj = 1000
            phase2_best_s = np.zeros((phase2_depth,J))
            phase2_best_tau = np.zeros((phase2_depth,J))
            while phase2_iter < phase2_depth:
                for t in range(tau_len):
                    best_second_search_s[t, :] = second_search_s[t,:]
                    var2_neighbourhood_tau = Neighbourhood_tau(second_search_s[t,:], second_search_tau[t, :], tau_steps, phase2_depth)
                    obj_vals2 = np.zeros((J, 1))
                    balanced_errors2 = np.zeros((J, 1))
                    for u in range(J):
                        obj_vals2[u,0], _, balanced_errors2[u] = Objective_function(X_train, y_train, best_second_search_s[t,:].reshape((1,J)), var2_neighbourhood_tau[u,:].reshape((1,J)))
                        
                    # chosen_ind2 = np.argmin(obj_vals2)
                    chosen_ind2 = np.argmin(obj_vals2)
                    chosen_obj = obj_vals2[chosen_ind2] # Not needed?
                    obj_second_search[t] = chosen_obj # Not needed? Also issues here
                    best_second_search_tau[t,:] = var_neighbourhood_tau[chosen_ind2, :]
                    
                    print("Phase 2 iteration (change #/sol #): "+ str(t+1)+ "/" +str(phase2_iter))
                    
                    
                    
                    
                # Testing phase2 searchs on validation set
                for v in range(tau_len):
                    validation_objs[v], y_hats_phase2[v,:], validation_errors[v] = Objective_function(X_validation, y_validation, best_second_search_s[v,:].reshape((1,J)), best_second_search_tau[v,:].reshape((1,J)))
                    
            
                # final_ind = np.argmin(validation_objs)
                phase2_ind = np.argmin(validation_errors)
                phase2_obj = validation_objs[phase2_ind]
                phase2_accuracy = 1 - phase2_obj/M
                phase2_balanced_accuracy = balanced_accuracy_score(y_validation, y_hats_phase2[phase2_ind, :])
                phase2_s = best_second_search_s[phase2_ind, :].reshape((1,J))
                phase2_tau = best_second_search_tau[phase2_ind, :].reshape((1,J))
                
                if phase2_obj < phase2_best_obj or phase2_iter == 1:
                    phase2_best_obj = phase2_obj
                    phase2_best_s = phase2_s
                    phase2_best_tau = phase2_tau
                
                phase2_iter += 1
                
            # Saving the final solution
            obj_phase1, y_hat_phase1, _ = Objective_function(X_validation, y_validation, phase1_s, phase1_tau)
            phase1_accuracy = 1 - obj_phase1/M
            phase1_balanced_accuracy = balanced_accuracy_score(y_validation, y_hat_phase1)
            phase2_best_obj, y_hat_phase2, _ = Objective_function(X_validation, y_validation, phase2_best_s, phase2_best_tau)
            phase2_accuracy = 1 - phase2_best_obj/M
            phase2_balanced_accuracy = balanced_accuracy_score(y_validation, y_hat_phase2)

            if phase2_accuracy < phase1_accuracy:
                final_obj = phase1_obj
                final_s = phase1_s
                final_tau = phase1_tau
                final_balanced_accuracy = phase1_balanced_accuracy
                final_accuracy = phase1_accuracy
            else:
                final_obj = phase2_best_obj
                final_s = phase2_best_s
                final_tau = phase2_best_tau
                final_balanced_accuracy = phase2_balanced_accuracy
                final_accuracy = phase2_accuracy
            break
        
        
        iterations += 1
        print("Phase 1 iteration " + str(iterations))

    
    return phase1_obj, phase1_s, phase1_tau, phase1_accuracy, phase1_balanced_accuracy, phase2_best_obj, phase2_best_s, phase2_best_tau, phase2_accuracy, phase2_balanced_accuracy, final_obj, final_s, final_tau, final_accuracy, final_balanced_accuracy, path_obj, iterations+1

# Initialization
patients = np.argwhere(y_train == 1) .flatten()
max_X = np.max(X_train, axis=0)
min_X = np.min(X_train, axis=0)
init_s = np.zeros((1, J), dtype=int)
init_s_inds = np.sort(np.random.choice(J, halfJ, replace=False))
init_s_inds_complement = np.setdiff1d(np.arange(J), init_s_inds)
init_s[0,init_s_inds] = 1
init_tau = np.zeros((J,1))
max_patients_X = np.max(X_train[patients, :], axis=0)
min_patients_X = np.min(X_train[patients, :], axis=0)
s1s = max_patients_X[init_s.flatten() == 1] + 1
s0s = min_patients_X[init_s.flatten() == 0] - 1
init_tau[init_s.flatten() == 1] = s1s.reshape((halfJ,1))
init_tau[init_s.flatten() == 0] = s0s.reshape((halfJ2,1))
init_tau = init_tau.reshape((1,J))

init_obj_val, init_y_hat, init_balanced_error = Objective_function(X_train,y_train,init_s, init_tau)

# Hyperparameters
tau_size = 50
tau_len = 10
phase2_depth = 5

# Running the algorithm
phase1_obj, phase1_s, phase1_tau, phase1_accuracy, phase1_balanced_accuracy, phase2_obj, phase2_s, phase2_tau, phase2_accuracy, phase2_balanced_accuracy, final_obj, final_s, final_tau, final_accuracy, final_balanced_accuracy, path_obj, iterations = Local_Search(X_train, y_train, X_validation, y_validation, init_s, init_tau, Objective_function, Neighbourhood_s, Neighbourhood_tau, tau_size=tau_size, tau_len=tau_len, phase2_depth=phase2_depth)

# Plotting phase 1 path of objective function
plt.figure()
plt.plot(np.arange(iterations), path_obj)
plt.ylabel("Objective Value")
plt.xlabel("Iterations")
plt.show()

# Testing the classification of phase 1
print("Accuracy (phase 1): "  +str(phase1_accuracy))
print("Balanced accuracy (phase1): " + str(phase1_balanced_accuracy))

# Testing the classification of phase 2 
print("Accuracy (phase 2): "  +str(phase2_accuracy))
print("Balanced accuracy (phase2): " + str(phase2_balanced_accuracy))

# Testing the classification of final
print("Accuracy (final): "  +str(final_accuracy))
print("Balanced accuracy (final): " + str(final_balanced_accuracy))