# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:56:27 2024

@author: kolst
"""

# importing packages
import numpy as np
import gurobipy as gp
import math
import random
import scipy.stats as stats
import time
import matplotlib.pyplot as plt

def instance_generator(n,c,N):
    # Initialize arrays to store the instance
    w_est = np.zeros((N,n))
    w = np.zeros((N,n))
    p = np.zeros((N,n))
    
    for i in range(N):# Loop through how many instances to be created (N)
        for j in range(n): # Loop through the number of items to be offered in an instance (n)
            w_est[i,j] = random.randint(1,np.floor(0.8*c).astype(int)) # generate expected weight
            w[i,j] = random.randint(w_est[i,j]-1,w_est[i,j]+1) # Determine actual weight based on expected weight
            if w[i,j] == 0: # Handle egde case
                w[i,j] = 1
            p[i,j] = random.randint(1,np.floor(0.8*c).astype(int)) # Generate profit based on capacity
        
    return n,c,w_est,w,p


def GreedyHeuristic(n,c,w_est,w,p):
    # Initialise list to store items and current weight/profit/capacity remaining
    KP = []
    weight = 0
    profit = 0
    current_c = c
    for i in range(n): # Loop through offered items
        if w_est[i] < current_c: # If there is enough space add the item
            # Add this item
            KP.append(i)
            weight += w[i]
            profit += p[i]
            current_c -= w[i]
            
    if weight > c: # At the end if the actual total weight exceeds the capacity set profit to 0
        profit = 0
    return KP, weight, profit

def EstimateDistHeuristic(n,c,w_est,w,p):
    # Initialize array to store accepted items and rejected items
    KP = []
    weight = 0
    profit = 0
    current_c = c
    observed_items_profits = []
    observed_items_weights = []
    greedy_flag = 0 # Code flag to revert back to greedy
    
    for i in range(n): # Loop through offered items
        if greedy_flag == 1:
            if w_est[i] < current_c:
                # Add this item
                KP.append(i)
                weight += w[i]
                profit += p[i]
                current_c -= w[i]
        
        else:
            if i < n*(0.15): # If in the first 15% of items, only observe
                observed_items_profits.append(p[i])
                observed_items_weights.append(w[i])
                mean_value = np.mean((np.array(observed_items_profits))/(np.array(observed_items_weights)))
                
            else:
                if i > n*(3/4) and current_c/c > 0.7: # If in last 25% of items and very low current weight of KP
                    greedy_flag = 1 # Switch to greedy
                else:
                    if p[i]/w_est[i] > 0.8*mean_value and w_est[i] < current_c: # If value density is larger than observed mean and there is still space
                        # Add this item
                        KP.append(i)
                        weight += w[i]
                        profit += p[i]
                        current_c -= w[i]
                        # Update observations
                        observed_items_profits.append(p[i])
                        observed_items_weights.append(w[i])
                        mean_value = np.mean((np.array(observed_items_profits))/(np.array(observed_items_weights)))
                        
    if weight > c: # At end check if weight exceeds capacity
        profit = 0
            
    return KP, weight, profit

def ClairvoyantAgent(n,c,w_est,w,p):
    
    # initiate gurobi model
    m = gp.Model("KP")
    # add binary selection variable
    x = m.addVars(n, vtype=gp.GRB.BINARY, name="x")
    # add capacity constraint
    m.addConstr(gp.quicksum(w[i]*x[i] for i in range(n)) <= c)
    # set objective function, maximising profit
    m.setObjective(gp.quicksum(p[i]*x[i] for i in range(n)), gp.GRB.MAXIMIZE)
    # suppress output messages
    m.setParam('OutputFlag',0)
    # optimize
    m.optimize()
    profit = m.ObjVal
    all_vars = m.getVars()
    values = m.getAttr("X", all_vars)
    KP = np.argwhere(np.array(values).astype(int)==1).reshape(np.sum(values).astype(int)).tolist()
    # KP = KP_array.tolist

    return KP, profit


N = 100
n = 100
c = 300
n,c,w_est,w,p = instance_generator(n,c,N)
KP_greedy = []
profit_greedy = []
KP_est = []
profit_est = []
KP_opt = []
profit_opt = []
for k in range(N):
    KP_greedy_k, weight_greedy_k, profit_greedy_k = GreedyHeuristic(n,c,w_est[k,:],w[k,:],p[k,:])
    KP_est_k, weight_est_k, profit_est_k = EstimateDistHeuristic(n,c,w_est[k,:],w[k,:],p[k,:])
    KP_opt_k, profit_opt_k = ClairvoyantAgent(n,c,w_est[k,:],w[k,:],p[k,:])
    KP_greedy.append(KP_greedy_k)
    profit_greedy.append(profit_greedy_k)
    KP_est.append(KP_est_k)
    profit_est.append(profit_est_k)
    KP_opt.append(KP_opt_k)
    profit_opt.append(profit_opt_k)


instances = np.arange(N)
plt.figure()
plt.scatter(instances, profit_greedy, label="Greedy Profit", color="blue")
plt.scatter(instances, profit_est, label="Estimated Profit", color="orange")
plt.scatter(instances, profit_opt, label="Optimal Profit", color="green")
plt.title("Simple Heuristic Profits per Instance")
plt.xlabel("Instances")
plt.ylabel("Profit")
plt.legend()  # Add the legend
plt.show()