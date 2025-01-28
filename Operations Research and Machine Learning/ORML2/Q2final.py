# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 21:34:33 2024

@author: kolst
"""
import pickle
import numpy as np
import scipy.stats as stats
import time
import gurobipy as gp
import random


# np.random.seed(1)

# =============================================================================
# Greedy Heuristic
# =============================================================================

def GreedyHeuristic(n,l,w,W):
    K = int(np.sum(l)) # Upper bound on optimal solution
    K_i = K -l + 1 # last bin where item i can be packed
    f = np.zeros(K)
    a = np.zeros(n)
    
    for k in range(K):
        for i in range(n):
            if a[i] == 0 and w[i] + f[k] <= W:
                a[i] = k + 1
                for k_prime in range(k, k + l[i]):
                    if k_prime <= K - l[i]: # Upper bound on k_prime
                        f[k_prime] = f[k_prime] + w[i]
                    
    return f, a

# =============================================================================
# Gurobi solver
# =============================================================================

def ILP(n,W,w,l):
    m = gp.Model("1CBP_ILP")

    K = int(np.sum(l)) # Upper bound on optimal solution
    K_i = K -l + 1 # last bin where item i can be packed

    # add binary decision variables
    z_k = m.addVars(K, vtype=gp.GRB.BINARY, name="z_k")
    x_ik = m.addVars(n, K, vtype=gp.GRB.BINARY, name="x_ik")

    # constraint 1
    for i in range(n):
        m.addConstr(gp.quicksum(x_ik[i,k] for k in range(K_i[i])) == 1)
    # constraint 2 
    for k in range(K):
        m.addConstr(
            gp.quicksum(
                    w[i] * x_ik[i,k_prime]
                    for i in range(n)
                    for k_prime in range(max(0,k - l[i] + 1), k + 1)
                    if k_prime <= K - l[i]
            ) <= W * z_k[k]
        )


    # constraint 3
    for k in range(K-1):
        m.addConstr(z_k[k] >= z_k[k+1])
        
    # Objective function
    m.setObjective(gp.quicksum(z_k[k] for k in range(K)), gp.GRB.MINIMIZE)
    # suppress gurobi outputs
    m.setParam('OutputFlag',0)
    # Set time limit to 5 seconds
    timelimit = 5
    m.setParam("TimeLimit", timelimit)
    m.optimize() # Solve ILP
    status = m.status # retrieve problem status
    if status == 2:
        print("Optimal solution found")
    elif status == 3:
        print("Problem infeasible")
    elif status == 9: # status for time limit
        print(f"Time limit of {timelimit} seconds reached")

    # extracting solution
    z_vals = m.getAttr("X", z_k)
    x_vals = m.getAttr("X", x_ik)

    # Count active bins
    active_bins = sum(1 for k in range(K) if z_vals[k] == 1)

    # Initialise solution lists
    item_assignment = []
    bin_weights = [0] * K  

    # Extract and save solution
    for i in range(n):
        for k in range(K):
            if x_vals[i, k] == 1: 
                item_assignment.append(k+1)
                for b in range(k, k + l[i]):
                    if b < K:  
                        bin_weights[b] += w[i]
                        
    return item_assignment, bin_weights, active_bins

# =============================================================================
# Combined greedy and gurobi heuristic
# =============================================================================

def GreedyHeuristicN(n,l,w,W,N):
    # GreedyHeuristic algorithm that continues until N items have been fixed
    K = int(np.sum(l)) # Upper bound on optimal solution
    K_i = K -l + 1 # last bin where item i can be packed
    f = np.zeros(K)
    a = np.zeros(n)
    
    for k in range(K):
        for i in range(n):
            if sum(np.where(a > 0, 1,0)) >= N:
                break
            elif a[i] == 0 and w[i] + f[k] <= W:
                a[i] = k + 1
                for k_prime in range(k, k + l[i]):
                    if k_prime <= K - l[i]:
                        f[k_prime] = f[k_prime] + w[i]

    return f, a

def Mixed_Heuristic(n,N,W,w,l):
    # Greedy N algorithm
    f, a = GreedyHeuristicN(n,l,w,W,N)
    active_bins = np.sum(np.where(f!=0, 1, 0))
    
    # Regular greedy algorithm
    fG, aG = GreedyHeuristic(n,l,w,W)
    active_binsG = np.sum(np.where(fG!=0, 1, 0))

    # Obtain the remaining items that need to be packed and necessary indices
    inds = np.argwhere(a == 0)
    new_w = w[inds].reshape((np.size(inds),))
    new_l = l[inds].reshape((np.size(inds),))
    new_n = np.size(inds)
    new_K = int(np.sum(new_l)) # Upper bound on optimal solution
    new_K_i = new_K - new_l + 1 # last bin where item i can be packed
    new_K_i.reshape((np.size(inds),))
    
    m = gp.Model("1CBP_ILP")

    # add binary decision variables
    z_k = m.addVars(new_K, vtype=gp.GRB.BINARY, name="z_k")
    x_ik = m.addVars(new_n, new_K, vtype=gp.GRB.BINARY, name="x_ik")

    # constraint 1
    for i in range(new_n):
        m.addConstr(gp.quicksum(x_ik[i,k] for k in range(new_K_i[i])) == 1)
    # constraint 2 
    for k in range(new_K):
        m.addConstr(
            gp.quicksum(
                    new_w[i] * x_ik[i,k_prime]
                    for i in range(new_n)
                    for k_prime in range(max(0,k - new_l[i] + 1), k + 1)
                    if k_prime <= new_K - new_l[i]
            ) <= W * z_k[k]
        )


    # constraint 3
    for k in range(new_K-1):
        m.addConstr(z_k[k] >= z_k[k+1])
        
    # Objective function
    m.setObjective(gp.quicksum(z_k[k] for k in range(new_K)), gp.GRB.MINIMIZE)
    # Suppress outputs from gurobi
    m.setParam('OutputFlag',0)
    # Set time limit to 5 seconds
    timelimit = 5
    m.setParam("TimeLimit", timelimit)
    m.optimize() # Solve problem
    status = m.status # Retrieve problem status
    if status == 2:
        print("Optimal solution found")
    elif status == 3:
        print("Problem infeasible")
    elif status == 9: # status for time limit
        print(f"Time limit of {timelimit} seconds reached")

    # extracting solution
    z_vals = m.getAttr("X", z_k)
    x_vals = m.getAttr("X", x_ik)

    # Count active bins
    active_bins2 = sum(1 for k in range(new_K) if z_vals[k] == 1)

    # Initialize solution lists
    item_assignment2 = []
    bin_weights2 = [0] * new_K  
    
    # Extract solution and add to lists
    for i in range(new_n):
        for k in range(new_K):
            if x_vals[i, k] == 1: 
                item_assignment2.append(k+1)
                for b in range(k, k + new_l[i]):
                    if b < new_K:  
                        bin_weights2[b] += new_w[i]
                        
    # Total active bins from both heuristics
    total_active_bins = active_bins + active_bins2
    # From the ILP, items are assigned to bin 0 onwards, but 
    for i in range(len(item_assignment2)):
        item_assignment2[i] += np.min(np.where(f==0))

    # concatenate both solutions to obtain new solution
    total_item_assignment = a
    for i in range(len(inds)):
        total_item_assignment[inds[i]] = item_assignment2[i]
        
    total_bin_weights = f
    total_bin_weights[active_bins:active_bins+active_bins2] = bin_weights2[:active_bins2]
    
    return total_active_bins, active_binsG, total_item_assignment, total_bin_weights


# =============================================================================
# 1CBP instance generator
# =============================================================================

def Instance_1CBP(n_instances, n, w_l, w_u, l_u_upper):
    # Generate instances for the 1CBP
    # Initialize arrays to store instances
    instances_W = np.zeros((n_instances, 1), dtype=int)
    instances_w = np.zeros((n_instances, n), dtype=int)
    instances_l = np.zeros((n_instances, n), dtype=int)
    l_us = stats.randint(2,l_u_upper).rvs(n_instances) # Upper bounds on upper bound of contiguity constraints
    for i in range(n_instances): # Loop through each instance generating n contiguity constraints
        instances_l[i,:] = stats.randint(1, l_us[i]).rvs(n)
    instances_w = stats.randint(w_l, w_u).rvs((n_instances,n)) # weights  
    instances_W = np.floor(1.5 * np.max(instances_w, axis=1)).astype(int) # Capacities
    return n, instances_W, instances_w, instances_l

# =============================================================================
# Train Q-learning
# =============================================================================


def trainQlearn(Q,V,n,n_sizes,instances_W,instances_w,instances_l,N_len,epsilon,alpha,iteration):

    # Initialize game
    W = instances_W[iteration]
    w = instances_w[iteration,:]
    l = instances_l[iteration,:]

    # Compute mean and set up states and buckets
    mean_l = np.mean(l)
    l_values = [[0,3], [3,6], [6,9], [9,14], [14,20]]
    N_values = np.linspace(0,int(np.floor(max(n_sizes) * 2/3)),num=N_len, dtype=int)
    M_len = len(l_values)
    
    # Determine which bucket the instance falls into
    for i in range(M_len):
        if l_values[i][0] <= mean_l < l_values[i][1]:
            state = i
   
    p = random.random() # random number between 0,1
    if (p < epsilon or all(q == 0 for q in Q[state])): # epsilon random policy or all are zeros
        decision = random.randint(0,N_len-1) # Random action
    else:
        decision = Q[state].index(max(Q[state])) # Maximum action
        
    # Update visit table     
    V[state][decision] += 1

    
    
    # Compute reward by running mixed heuristic
    total_active_bins, active_binsG, total_item_assignment, total_bin_weights = Mixed_Heuristic(n,N_values[decision],W,w,l)
    reward = - (total_active_bins - active_binsG)
    Q[state][decision] = (1 - alpha) * Q[state][decision] + alpha * reward            
        
    return Q,V

def PlayTrainedAgent(Q,n,n_sizes,N_len,instances_W,instances_w,instances_l,iteration):

    # Initialize game
    W = instances_W[iteration]
    w = instances_w[iteration,:]
    l = instances_l[iteration,:]

    # Solve with ILP for 5 seconds
    item_assignmentILP, bin_weightsILP, active_binsILP = ILP(n,W,w,l)

    # Compute mean and set up state space
    mean_l = np.mean(l)
    l_values = [[0,3], [3,6], [6,9], [9,14], [14,20]]
    N_values = np.linspace(0,int(np.floor(max(n_sizes) * 2/3)),num=N_len, dtype=int)
    M_len = len(l_values)

    # Determine which bucket the instance falls into
    for i in range(M_len):
        if l_values[i][0] < mean_l <= l_values[i][1]:
            state = i
    if state == -1:
        print("mean_l not in state space")
        state = 4

    if all(q == 0 for q in Q[state]):
        decision = random.randint(0,N_len-1)
    else:
        decision = Q[state].index(max(Q[state]))
    

    # Do Mixed heuristic based on Q value decision
    total_active_bins, active_binsG, total_item_assignment, total_bin_weights = Mixed_Heuristic(n,N_values[decision],W,w,l)
    # Calculate the difference between greedy heuristic and found solution
    objective_gainG = active_binsG - total_active_bins
    objective_gainILP = active_binsILP - total_active_bins
        
    return objective_gainG, objective_gainILP

# =============================================================================
# Main 
# =============================================================================
# Code flags
question_flag = 1 # For which question you want to run
run_training_flag = 1 # If you want to run training, otherwise you will load a Q_table
saveQ_flag = 0 # If you want to save the Q_table
# Number of episodes and tests
n_game_trained = 2000
n_game_tested = 200


# Obtain instances
if question_flag == 1:
    # Example instance
    n = 3
    W = 10
    w = np.array([4,6,1])
    l = np.array([1,2,3])
    
    f,a = GreedyHeuristic(n,l,w,W)
    active_bins = np.sum(np.where(f!=0, 1, 0))
    item_assignment, bin_weights, active_binsILP = ILP(n,W,w,l)

    print("w: " + str(w))
    print("l: " + str(l))
    print("Greedy Heuristic result:")
    print("Active bins: " + str(active_bins))
    print("Item assignment: " + str(a))
    print("Bin weights: " + str(f))
    print("---------------------------------------------------")
    print("Gurobi result: ")
    print("Active bins: " + str(active_binsILP))
    print("Item assignment: " + str(item_assignment))
    print("Bin weights: " + str(bin_weights))
    
elif question_flag == 2:
    # Obtain random instance from instance generator
    n = 60
    l_u = 10
    n, instances_W, instances_w, instances_l = Instance_1CBP(1,n,20,50,l_u)
    W = instances_W[0]
    w = instances_w[0,:]
    l = instances_l[0,:]
    
    # Greedy heuristic solution
    f,a = GreedyHeuristic(n,l,w,W)
    active_bins = np.sum(np.where(f!=0, 1, 0))
    # ILP solution
    item_assignment, bin_weights, active_binsILP = ILP(n,W,w,l)
    
    print("w: " + str(w))
    print("l: " + str(l))
    print("Greedy Heuristic result:")
    print("Active bins: " + str(active_bins))
    print("Item assignment: " + str(a))
    print("Bin weights: " + str(f))
    print("---------------------------------------------------")
    print("Gurobi result: ")
    print("Active bins: " + str(active_binsILP))
    print("Item assignment: " + str(item_assignment))
    print("Bin weights: " + str(bin_weights))
    
else:
    if run_training_flag == 1:
        # Q-learning parameters
        alpha = 0.1
        epsilon = 0.4
        
        # Obtain training instances
        n_sizes = [40,60,80]
        l_u_upper = 20
        w_l = 1
        _, instances_W1, instances_w1, instances_l1 = Instance_1CBP(np.floor(n_game_trained/3).astype(int),n_sizes[0],w_l,int(np.ceil(n_sizes[0]*0.5)),l_u_upper)
        _, instances_W2, instances_w2, instances_l2 = Instance_1CBP(np.floor(n_game_trained/3).astype(int),n_sizes[1],w_l,int(np.ceil(n_sizes[1]*0.5)),l_u_upper)
        _, instances_W3, instances_w3, instances_l3 = Instance_1CBP(int(n_game_trained - 2 * np.floor(n_game_trained/3)),n_sizes[2],w_l,int(np.ceil(n_sizes[2]*0.5)),l_u_upper)
        # _, instances_W, instances_w, instances_l = Instance_1CBP(n_game_trained,n_sizes[1],w_l,int(np.ceil(n_sizes[1]*0.5)),l_u_upper)
    
    
        # Set up buckets and state space to make Q and V tables
        l_values = [[0,3], [3,6], [6,9], [9,14], [14,20]]
        N_len = 10
        N_values = np.linspace(0,int(np.floor(max(n_sizes) * 2/3)),num=N_len, dtype=int)
        M_len = len(l_values)
        
        # Initialize Q-table and visit table
        Q = [[0] * N_len for i in range(M_len)]
        V = [[0] * N_len for i in range(M_len)]
        
        # Train the agent: First 1/3 instances are n = 40, next are n = 60 and the rest of n = 80
        for iteration in range(n_game_trained):
            # Q,V = trainQlearn(Q,V,n_sizes[1],[60],instances_W,instances_w,instances_l,N_len, epsilon, alpha,iteration)
            if iteration < np.floor(n_game_trained/3).astype(int):
                Q,V = trainQlearn(Q,V,n_sizes[0],n_sizes,instances_W1,instances_w1,instances_l1,N_len, epsilon, alpha,iteration)
            elif iteration < 2 * np.floor(n_game_trained/3).astype(int):
                Q,V = trainQlearn(Q,V,n_sizes[1],n_sizes,instances_W2,instances_w2,instances_l2,N_len, epsilon, alpha,iteration-np.floor(n_game_trained/3).astype(int))
            else:
                Q,V = trainQlearn(Q,V,n_sizes[2],n_sizes,instances_W3,instances_w3,instances_l3,N_len, epsilon, alpha,iteration-2 * np.floor(n_game_trained/3).astype(int))
            if iteration % 10 == 0 and iteration != 0:
                print("Trained instance: " + str(iteration))
              
        if saveQ_flag == 1:
            # Saving Q table
            with open("Q_table_2_new", "wb") as fp:   #Pickling
                pickle.dump(Q, fp)
        print("Completed training")
        print("-----------------------------------------------------------------------")
        # Q_60 = Q
    else:
        # Retrieving Q table
        with open('Q_table_2', 'rb') as f:
            Q = pickle.load(f)
        with open('Q_table_2_60', 'rb') as f:
            Q_60 = pickle.load(f)   
        
            
# =============================================================================
#     Test agent
# =============================================================================
    # Obtain instances
    n_sizes = [40,60,80]
    N_len = 10
    l_u_upper = 20
    w_l = 1
    _, instances_W1, instances_w1, instances_l1 = Instance_1CBP(np.floor(n_game_tested/3).astype(int),n_sizes[0],w_l,int(np.ceil(n_sizes[0]*0.5)),l_u_upper)
    _, instances_W2, instances_w2, instances_l2 = Instance_1CBP(np.floor(n_game_tested/3).astype(int),n_sizes[1],w_l,int(np.ceil(n_sizes[1]*0.5)),l_u_upper)
    _, instances_W3, instances_w3, instances_l3 = Instance_1CBP(int(n_game_tested - 2 * np.floor(n_game_tested/3)),n_sizes[2],w_l,int(np.ceil(n_sizes[2]*0.5)),l_u_upper)
    # _, instances_W, instances_w, instances_l = Instance_1CBP(n_game_tested,60,w_l,int(np.ceil(60*0.5)),l_u_upper)
    
    # Initialize save lists
    objectives_G = np.zeros(n_game_tested)
    objectives_ILP = np.zeros(n_game_tested)
    # objectives_G60 = np.zeros(n_game_tested)
    # objectives_ILP60 = np.zeros(n_game_tested)
    for iteration in range(n_game_tested):
        # objective_G60, objective_ILP60 = PlayTrainedAgent(Q_60,60,[60],N_len,instances_W,instances_w,instances_l,iteration)
        if iteration < np.floor(n_game_tested/3).astype(int):
            objective_G, objective_ILP = PlayTrainedAgent(Q,n_sizes[0],n_sizes,N_len,instances_W1,instances_w1,instances_l1,iteration)
        if iteration < 2 * np.floor(n_game_tested/3).astype(int):
            objective_G, objective_ILP = PlayTrainedAgent(Q,n_sizes[1],n_sizes,N_len,instances_W2,instances_w2,instances_l2,iteration-np.floor(n_game_tested/3).astype(int))
        else:
            objective_G, objective_ILP = PlayTrainedAgent(Q,n_sizes[2],n_sizes,N_len,instances_W3,instances_w3,instances_l3,iteration-2*np.floor(n_game_tested/3).astype(int))
        objectives_G[iteration] = objective_G
        objectives_ILP[iteration] = objective_ILP
        # objectives_G60[iteration] = objective_G60
        # objectives_ILP60[iteration] = objective_ILP60
        if iteration % 10 == 0 and iteration != 0:
            print(f"Tested instance {iteration}")
            
    print("------------------------------------------")
    avg_improvement_greedy = np.mean(objectives_G)
    avg_improvement_greedymix40 = np.mean(objectives_G[:np.floor(n_game_tested/3).astype(int)])
    avg_improvement_greedymix60 = np.mean(objectives_G[np.floor(n_game_tested/3).astype(int):2*np.floor(n_game_tested/3).astype(int)])
    avg_improvement_greedymix80 = np.mean(objectives_G[2*np.floor(n_game_tested/3).astype(int):])
    n_worse_greedy = np.sum(np.where(objectives_G < 0, 1, 0))
    avg_improvement_ILP = np.mean(objectives_ILP)
    avg_improvement_ILPmix40 = np.mean(objectives_ILP[:np.floor(n_game_tested/3).astype(int)])
    avg_improvement_ILPmix60 = np.mean(objectives_ILP[np.floor(n_game_tested/3).astype(int):2*np.floor(n_game_tested/3).astype(int)])
    avg_improvement_ILPmix80 = np.mean(objectives_ILP[2*np.floor(n_game_tested/3).astype(int):])
    n_worse_ILP = np.sum(np.where(objectives_ILP < 0, 1, 0))
    print("Average improvement on Greedy Heuristic: " + str(avg_improvement_greedy))
    print("Number of instances performing worse than Greedy: " + str(n_worse_greedy))
    print("Average improvement on ILP: " + str(avg_improvement_ILP))
    print("Number of instances performing worse than ILP: " + str(n_worse_ILP))
    
    # print("------------------------------------------")
    # avg_improvement_greedy60 = np.mean(objectives_G60)
    # n_worse_greedy60 = np.sum(np.where(objectives_G60 < 0, 1, 0))
    # avg_improvement_ILP60 = np.mean(objectives_ILP60)
    # n_worse_ILP60 = np.sum(np.where(objectives_ILP60 < 0, 1, 0))
    # print("Average improvement on Greedy Heuristic: " + str(avg_improvement_greedy60))
    # print("Number of instances performing worse than Greedy: " + str(n_worse_greedy60))
    # print("Average improvement on ILP: " + str(avg_improvement_ILP60))
    # print("Number of instances performing worse than ILP: " + str(n_worse_ILP60))
    
    
    