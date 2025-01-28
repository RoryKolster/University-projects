# =============================================================================
# Import packages
# =============================================================================

import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import *
import math
import random
import numpy as np
import scipy.spatial as spatial
import scipy.stats as stats
import scipy.optimize as optimize
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpy import loadtxt
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import pickle

# =============================================================================
# Functions
# =============================================================================
# Return Euclidian Distance
def ED(xi,xj,yi,yj):
    return ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

def ccw(xA,yA,xB,yB,xC,yC):
    return (yC-yA) * (xB-xA) > (yB-yA) * (xC-xA)

# Return true if line segments AB and CD intersect
def intersect(xA,yA,xB,yB,xC,yC,xD,yD):
    return ccw(xA,yA,xC,yC,xD,yD) != ccw(xB,yB,xC,yC,xD,yD) and ccw(xA,yA,xB,yB,xC,yC) != ccw(xA,yA,xB,yB,xD,yD)

# def regression_model(n, avg_edge_length, a, b, c):
#     return a + b * (1 / np.sqrt(n)) + c * avg_edge_length

def GreedyHeuristic(dist, start=0):
    # Function that performs greedy heuristic starting at given node and 
    # returning the tour, the cost and the cost minus last edge, the average 
    # and standard deviation of lengths of edges in the tour
    distance_matrix = np.array(dist)
    num_cities = distance_matrix.shape[0]
    visited = [False] * num_cities
    tour = [start]
    visited[start] = True
    total_cost = 0
    lengths = []

    current_city = start
    for _ in range(num_cities - 1):
        # Find the nearest unvisited city
        nearest_city = None
        min_distance = float('inf')
        for next_city in range(num_cities):
            if not visited[next_city] and distance_matrix[current_city, next_city] < min_distance:
                nearest_city = next_city
                min_distance = distance_matrix[current_city, next_city]

        # Visit the nearest city
        tour.append(nearest_city)
        visited[nearest_city] = True
        total_cost += min_distance
        current_city = nearest_city
        lengths.append(min_distance)

    # Return to the starting city
    total_cost += distance_matrix[current_city, start]
    tour.append(start)
    total_cost_minus_last = total_cost - distance_matrix[tour[-2],start]
    lengths.append(distance_matrix[current_city, start])
    avg_len = np.mean(np.array(lengths))
    std_len = np.std(np.array(lengths))

    return tour, total_cost, total_cost_minus_last, avg_len, std_len

def Gurobi(dist):
    # Function that takes a distance matrix in list form and solves the TSP to
    # optimality using gurobi, set to a time limit of 60 seconds, 
    # returning objective value
    nbCity = len(dist)
    
    m = gp.Model("ModelTSP") 

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[m.addVar(vtype=GRB.BINARY) for j in range(nbCity)] for i in range(nbCity)]

    # continuous variable to prevent subtours: each city will have a different sequential id in the planned route except the first one
    y = [m.addVar(vtype=GRB.CONTINUOUS)  for i in range(nbCity)]

    # set objective
    m.setObjective(quicksum(dist[i][j] * x[i][j] for i in range(nbCity) for j in range(nbCity)), GRB.MINIMIZE)

    # constraint : leave each city only once
    m.addConstrs(quicksum(x[i][j] for j in range(nbCity) if j != i) == 1 for i in range(nbCity))
      
    # constraint : enter each city only once
    m.addConstrs(quicksum(x[i][j] for i in range(nbCity) if i != j) == 1 for j in range(nbCity))

    # subtour elimination
    m.addConstrs(y[i] - (nbCity+1)*x[i][j] >= y[j]-nbCity for i in range(1,nbCity) for j in range(1,nbCity) if i != j)

    # optimizing
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 60)
    m.optimize()
    

    return m.ObjVal   

def GurobiBounds(dist):
    # Function that takes distance matrix in list form and solves the TSP,
    # given a time limit of 1 second, returning the best found lower bound
    # and the best found objective value.
    nbCity = len(dist)
    
    m = gp.Model("ModelTSP") 

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[m.addVar(vtype=GRB.BINARY) for j in range(nbCity)] for i in range(nbCity)]

    # continuous variable to prevent subtours: each city will have a different sequential id in the planned route except the first one
    y = [m.addVar(vtype=GRB.CONTINUOUS)  for i in range(nbCity)]

    # set objective
    m.setObjective(quicksum(dist[i][j] * x[i][j] for i in range(nbCity) for j in range(nbCity)), GRB.MINIMIZE)

    # constraint : leave each city only once
    m.addConstrs(quicksum(x[i][j] for j in range(nbCity) if j != i) == 1 for i in range(nbCity))
      
    # constraint : enter each city only once
    m.addConstrs(quicksum(x[i][j] for i in range(nbCity) if i != j) == 1 for j in range(nbCity))

    # subtour elimination
    m.addConstrs(y[i] - (nbCity+1)*x[i][j] >= y[j]-nbCity for i in range(1,nbCity) for j in range(1,nbCity) if i != j)

    # optimizing
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 1)
    m.optimize()
    
    upper_bound = m.ObjVal
    lower_bound = m.ObjBound

    return upper_bound, lower_bound 

def ConvexHullPath(nbCity,Coords):
    # Function that takes the number of cities, and the coordinates of the cities
    # in Euclidean space and determines the convex hull that contains all of the cities.
    # Returns the path, the length (perimeter) and the area.
    
    points = np.array(Coords)
    hull = spatial.ConvexHull(points)
    hull_path = hull.vertices.tolist()
    hull_path.append(hull_path[0])
    hull_path_length = hull.area
    hull_area = hull.volume

    return hull_path, hull_path_length, hull_area

def instance_generator(nbCity=50,plot=0,output=0):
    # Function that takes the number of cities and generates an instance
    # for Euclidean TSP. Possible to plot the graph and have output,
    # default set to zero. Returns distance matrix list and coordinates.
    
    maxDis = 1000
    xCoord = [random.randint(1,maxDis) for i in range(nbCity)]
    yCoord = [random.randint(1,maxDis) for i in range(nbCity)]
    Coords = [(x,y) for x,y in zip(xCoord,yCoord)]
    dist = [[int(ED(xCoord[i],xCoord[j],yCoord[i],yCoord[j])) for j in range(nbCity)] for i in range(nbCity)]
    distR = [dist[i].copy() for i in range(nbCity)]
    distRT = []
    for i in range(nbCity):
        distR[i].sort()
        distR[i].pop(0)

    distRT = [dist[i][j] for i in range(nbCity) for j in range(i+1,nbCity)]
    distRT.sort()
    q1 = [np.quantile(dist[i], 0.25) for i in range(nbCity)]
    q3 = [np.quantile(dist[i], 0.75) for i in range(nbCity)]
    infos = []
    minD = 99999
    maxD = 0
    # Draw instance
    if plot==1:
        for i in range(nbCity):
            plt.scatter(xCoord[i], yCoord[i])
            plt.annotate(i,(xCoord[i], yCoord[i]))
        plt.show()
    if output == 1:
        print(q1)
        print(xCoord)
        print(yCoord)
        for i in range(nbCity):
            minD = min(minD, min(x for x in dist[i] if x != 0))
            maxD = max(maxD,max(dist[i]))
        print(minD,maxD)
    
    return dist, Coords


def iteration(nbCity, save_load_flag):
    # Function that takes number of cities and save_load_flag, and generates an instance,
    # saves the distance matrix, solves to optimality, within 60 seconds, 
    # extracts features from the instance and saves these (if required).
    
    # Generate instance
    maxDis = 1000
    # nbCity = 50
    xCoord = [random.randint(1,maxDis) for i in range(nbCity)]
    yCoord = [random.randint(1,maxDis) for i in range(nbCity)]
    Coords = [(x,y) for x,y in zip(xCoord,yCoord)]
    dist = [[int(ED(xCoord[i],xCoord[j],yCoord[i],yCoord[j])) for j in range(nbCity)] for i in range(nbCity)]
    distR = [dist[i].copy() for i in range(nbCity)]
    distRT = []
    for i in range(nbCity):
        distR[i].sort()
        distR[i].pop(0)
        
    dist_a = np.array(dist)
    upper_dist_ind = np.triu_indices(dist_a.shape[0],k=1)
    distRT = [dist[i][j] for i in range(nbCity) for j in range(i+1,nbCity)]
    distRT.sort()
    # q1 = [np.quantile(dist[i], 0.25) for i in range(nbCity)]
    q1 = np.quantile(dist_a[upper_dist_ind], 0.25)
    # q3 = [np.quantile(dist[i], 0.75) for i in range(nbCity)]
    q3 = np.quantile(dist_a[upper_dist_ind], 0.75)
    avg_edge_len = np.mean(dist_a[upper_dist_ind])
    std_edge_len = np.std(dist_a[upper_dist_ind])
    infos = []
    minD = 99999
    maxD = 0
    

    # Save instance
    if save_load_flag == 0:
        if nbCity == 20:
            file = open("Training_instances_20.txt", "a")
            for row in range(nbCity):
                for col in range(nbCity-1):
                    file.write("%s," % str(dist[row][col]))
                file.write("%s" % str(dist[row][col]))
                file.write("\n")
            file.close()
        elif nbCity == 40:
            file = open("Training_instances_40.txt", "a")
            for row in range(nbCity):
                for col in range(nbCity-1):
                    file.write("%s," % str(dist[row][col]))
                file.write("%s" % str(dist[row][col]))
                file.write("\n")
            file.close()
        else:
            file = open("Training_instances_60.txt", "a")
            for row in range(nbCity):
                for col in range(nbCity-1):
                    file.write("%s," % str(dist[row][col]))
                file.write("%s" % str(dist[row][col]))
                file.write("\n")
            file.close()
        
    # Draw instance
    # for i in range(nbCity):
    #     plt.scatter(xCoord[i], yCoord[i])
    #     plt.annotate(i,(xCoord[i], yCoord[i]))
    # plt.show()
    # print(q1)
    # print(xCoord)
    # print(yCoord)
    # for i in range(nbCity):
    #     minD = min(minD, min(x for x in dist[i] if x != 0))
    #     maxD = max(maxD,max(dist[i]))
    # print(minD,maxD)

    # Solve to opimality
    m = gp.Model("ModelTSP") 

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[m.addVar(vtype=GRB.BINARY) for j in range(nbCity)] for i in range(nbCity)]

    # continuous variable to prevent subtours: each city will have a different sequential id in the planned route except the first one
    y = [m.addVar(vtype=GRB.CONTINUOUS)  for i in range(nbCity)]
    
    # set objective
    m.setObjective(quicksum(dist[i][j] * x[i][j] for i in range(nbCity) for j in range(nbCity)), GRB.MINIMIZE)

    # constraint : leave each city only once
    m.addConstrs(quicksum(x[i][j] for j in range(nbCity) if j != i) == 1 for i in range(nbCity))
      
    # constraint : enter each city only once
    m.addConstrs(quicksum(x[i][j] for i in range(nbCity) if i != j) == 1 for j in range(nbCity))

    # subtour elimination
    m.addConstrs(y[i] - (nbCity+1)*x[i][j] >= y[j]-nbCity for i in range(1,nbCity) for j in range(1,nbCity) if i != j)

    # optimizing
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 60)
    m.optimize()
    
    if save_load_flag == 0:
        file = open("Optimal_values.txt", "a")
        file.write(f"{m.ObjVal:.2f}\n")
        file.close()
    
    
    # Solve greedy from all starting nodes
    greedy_objs = []
    greedy2_objs = []
    for a in range(0,nbCity):
        path, cost, cost2, _, _ = GreedyHeuristic(dist,a)
        # greedy_paths[a] = path
        greedy_objs.append(cost)
        greedy2_objs.append(cost2)
        
    best_greedy = min(greedy_objs)
    best_greedy2 = min(greedy2_objs)
    
        
    # Convex hull tour
    hull_path, hull_path_length, hull_area = ConvexHullPath(nbCity, Coords)
    
    # Features
    info = []
    # Size
    info.append(nbCity)
    # Average edge length
    info.append(avg_edge_len)
    # Variance edge length
    info.append(std_edge_len)
    # Number of nodes with mean edge length in lower/upper quartile
    node_means = np.mean(dist_a, axis=0)
    countl = 0
    countu = 0
    for i in range(nbCity):
        if node_means[i] < q1:
            countl += 1
        elif node_means[i] > q3:
            countu += 1
    info.append(countl)
    info.append(countu)
    # Best Greedy solution
    info.append(best_greedy)
    # Best Greedy minus last solution
    info.append(best_greedy2)
    # Area and length and number of nodes of convex hull
    info.append(hull_area)
    info.append(hull_path_length)
    info.append(len(hull_path))
    # LP relaxation
    for i in range(nbCity):
        for j in range(nbCity):
            x[i][j].VType = GRB.CONTINUOUS 
    m.optimize()
    info.append(m.ObjVal)
    
    if save_load_flag == 0:
        file = open("Features.txt", "a")
        for k in range(len(info)-1):
            file.write("%s," % str(round(info[k],3)))
        file.write("%s" % str(round(info[k],3)))
        file.write("\n")
        file.close()
    

def data_features(dist):
    # Function that takes a distance matrix in list form and extracts the 
    # information needed to compute greedy heuristic approximations.
    # Returns the extracted features
    n = len(dist)
    dist_a = np.array(dist)
    upper_dist_ind = np.triu_indices(dist_a.shape[0],k=1)
    avg_edge_len = np.mean(dist_a[upper_dist_ind])
    Greedy_tour, Greedy_cost, Greedy_cost2, avg_len, std_len = GreedyHeuristic(dist)
    # opt = Gurobi(dist)
    feature = [n, avg_edge_len, Greedy_cost]
    feature_2 = [n, avg_edge_len, Greedy_cost2]
    return feature, feature_2


# =============================================================================
# Greedy Regressions
# =============================================================================

def Greedy_regressions(dists_20, dists_40, dists_60, y):
    # Training function for the greedy heuristic regressions,
    # using the training set instances. Returns the two regression models.
    
    
    features = []
    features2 = []
    print("---------------------------------------------------------------")
    print("Greedy Approximation training")
    for i in range(1000):
        dist_list = dists_20[i*20:i*20 + 20,:]
        feature, feature_2 = data_features(dist_list)
        features.append(feature)
        features2.append(feature_2)
        print("Iteration: " + str(i))
        
    for i in range(500):
        dist_list = dists_40[i*40:i*40 + 40,:]
        feature, feature_2 = data_features(dist_list)
        features.append(feature)
        features2.append(feature_2)
        print("Iteration: " + str(i + 1000))
        
    for i in range(500):
        dist_list = dists_60[i*60:i*60 + 60,:]
        feature, feature_2 = data_features(dist_list)
        features.append(feature)
        features2.append(feature_2)
        print("Iteration: " + str(i + 1500))
    
    model = LinearRegression()
    model.fit(features, y)
    model2 = LinearRegression()
    model2.fit(features2, y)
        
    return model, model2

# =============================================================================
# Gurobi approximations
# =============================================================================

def Gurobi_approximations(dists_20, dists_40, dists_60):
    # Training function for the gurobi approximations used in the neural network,
    # (DIFFERENT THAN THE METHOD TESTED IN FINAL SECTION).
    
    gurobi_approx = []
    print("---------------------------------------------------------------")
    print("Gurobi Approximations of the training set")
    for i in range(1000):
        dist = dists_20[i*20:i*20+20, :]
        dist_list = dist.tolist()
        _, Greedy_cost, Greedy_cost2,_,_ = GreedyHeuristic(dist_list)
        upper_bound, lower_bound = GurobiBounds(dist_list)
        if upper_bound == lower_bound:
            gurobi_approx.append(upper_bound)
        else:
            gurobi_approx.append(0.2*lower_bound + 0.8*Greedy_cost2)
        print("Iteration: " + str(i))
        
    for i in range(500):
        dist = dists_40[i*40:i*40+40, :]
        dist_list = dist.tolist()
        _, Greedy_cost, Greedy_cost2,_,_ = GreedyHeuristic(dist_list)
        upper_bound, lower_bound = GurobiBounds(dist_list)
        if upper_bound == lower_bound:
            gurobi_approx.append(upper_bound)
        else:
            gurobi_approx.append(0.2*lower_bound + 0.8*Greedy_cost2)
        print("Iteration: " + str(i + 1000))
        
    for i in range(500):
        dist = dists_60[i*60:60*i+60, :]
        dist_list = dist.tolist()
        _, Greedy_cost, Greedy_cost2,_,_ = GreedyHeuristic(dist_list)
        upper_bound, lower_bound = GurobiBounds(dist_list)
        if upper_bound == lower_bound:
            gurobi_approx.append(upper_bound)
        else:
            gurobi_approx.append(0.2*lower_bound + 0.8*Greedy_cost2)
        print("Iteration: " + str(i+1500))
        
    file = open("Gurobi_approximations.txt", "a")
    for a in gurobi_approx:
        file.write(f"{a}\n")
    file.close()
        
    return gurobi_approx

def extract_features(dist, Coords, time=60):
    # Function that takes the distance matrix in list form and coordinates,
    # and returns the features extracted from the instance.
    
    nbCity = len(dist)
    # Determine optimal
    m = gp.Model("ModelTSP") 

    # binary variables indicating if arc (i,j) is used on the route or not
    x = [[m.addVar(vtype=GRB.BINARY) for j in range(nbCity)] for i in range(nbCity)]

    # continuous variable to prevent subtours: each city will have a different sequential id in the planned route except the first one
    y = [m.addVar(vtype=GRB.CONTINUOUS)  for i in range(nbCity)]
    
    # set objective
    m.setObjective(quicksum(dist[i][j] * x[i][j] for i in range(nbCity) for j in range(nbCity)), GRB.MINIMIZE)

    # constraint : leave each city only once
    m.addConstrs(quicksum(x[i][j] for j in range(nbCity) if j != i) == 1 for i in range(nbCity))
      
    # constraint : enter each city only once
    m.addConstrs(quicksum(x[i][j] for i in range(nbCity) if i != j) == 1 for j in range(nbCity))

    # subtour elimination
    m.addConstrs(y[i] - (nbCity+1)*x[i][j] >= y[j]-nbCity for i in range(1,nbCity) for j in range(1,nbCity) if i != j)

    # optimizing
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time)
    m.optimize()
    optimal = m.ObjVal
    lower_bound = m.ObjBound
    near_optimal = 0.5*optimal + 0.5*lower_bound
    # if m.status == 2:
    #     print("Optimality reached")
    # elif m.status == 9:
    #     print("Time limit reached")
    
    # Extract rest of the features
    info = []
    info.append(nbCity)
    dist_a = np.array(dist)
    upper_dist_ind = np.triu_indices(dist_a.shape[0],k=1)
    q1 = np.quantile(dist_a[upper_dist_ind], 0.25)
    q3 = np.quantile(dist_a[upper_dist_ind], 0.75)
    avg_edge_len = np.mean(dist_a[upper_dist_ind])
    std_edge_len = np.std(dist_a[upper_dist_ind])
    
    greedy_objs = []
    greedy2_objs = []
    for a in range(0,nbCity):
        path, cost, cost2, _, _ = GreedyHeuristic(dist,a)
        # greedy_paths[a] = path
        greedy_objs.append(cost)
        greedy2_objs.append(cost2)
        
    best_greedy = min(greedy_objs)
    best_greedy2 = min(greedy2_objs)
    
        
    # Convex hull tour
    hull_path, hull_path_length, hull_area = ConvexHullPath(nbCity, Coords)
    
    # Info
    info = []
    # Size
    info.append(nbCity)
    # Average edge length
    info.append(avg_edge_len)
    # Variance edge length
    info.append(std_edge_len)
    # Number of nodes with mean edge length in lower/upper quartile
    node_means = np.mean(dist_a, axis=0)
    countl = 0
    countu = 0
    for i in range(nbCity):
        if node_means[i] < q1:
            countl += 1
        elif node_means[i] > q3:
            countu += 1
    info.append(countl)
    info.append(countu)
    # Best Greedy solution
    info.append(best_greedy)
    # Best Greedy minus last solution
    info.append(best_greedy2)
    # Area and length and number of nodes of convex hull
    info.append(hull_area)
    info.append(hull_path_length)
    info.append(len(hull_path))
    # LP relaxation
    for i in range(nbCity):
        for j in range(nbCity):
            x[i][j].VType = GRB.CONTINUOUS 
    m.optimize()
    info.append(m.ObjVal)
    return info, optimal, near_optimal
# =============================================================================
# __main__
# =============================================================================
# Section flags:
section_flag = 3 # Flag to run each section 1 for training instances and gathering features, 2 for NN, 3 for testing methods
save_load_flag = 1 # Flag: 0 to save outputs in to text files, 1 to load text files
hundred_flag = 0 # Flag to run the larger instance size


# =============================================================================
# Gathering Features
# =============================================================================
if section_flag == 1:
    n_instances = [1000,500,500] # Number of instances per size of instance
    n_city = [20,40,60] # Instance sizes
    for it in range(len(n_city)):
        for b in range(n_instances[it]):
            iterations = b + sum(n_instances[:it])
            print("start of iteration", iterations)
            iteration(n_city[it], save_load_flag)
        
        
# =============================================================================
# Extracting features
# =============================================================================
elif section_flag == 2:
    dists_20 = np.loadtxt("Training_instances_20.txt", delimiter=",")
    dists_40 = np.loadtxt("Training_instances_40.txt", delimiter=",")
    dists_60 = np.loadtxt("Training_instances_60.txt", delimiter=",")
    y = loadtxt("Optimal_values.txt", delimiter=",")
    
    dataset_20 = loadtxt('training_set_20.txt', delimiter=",")
    dataset_40 = loadtxt('training_set_40.txt', delimiter=",")
    dataset_60 = loadtxt('training_set_60.txt', delimiter=",")
    
    
    # Features = np.concatenate((dataset_20, dataset_40, dataset_60), axis=0)
    X = np.concatenate((dataset_20, dataset_40, dataset_60), axis=0)
    X[:,-2] = X[:,-1]
    Features = X[:,:-1]
    
    # Load Gurobi Approximations of the training set
    if save_load_flag == 0:
        gurobi_approx = Gurobi_approximations(dists_20, dists_40, dists_60)
    else:
        gurobi_approx = loadtxt("Gurobi_approximations.txt", delimiter=",")
    # Appending gurobi approximations to the features used in the NN
    Features = np.concatenate((Features,gurobi_approx.reshape((2000,1))), axis=1)
    
    # Features = loadtxt("Features_test.txt", delimiter=",")
    
    # Greedy approximation training
    model_greedy, model_greedy2 = Greedy_regressions(dists_20, dists_40, dists_60, y)
    # Save the model
    if save_load_flag == 0:
        with open('model_greedy.pkl','wb') as f:
            pickle.dump(model_greedy,f)
        with open('model_greedy2.pkl','wb') as f:
            pickle.dump(model_greedy2,f)

    
    # Prune Features
    # Selection = [0,1,2,5,6,8,9,10,11]
    # Features = Features[:, selection]
    
    # Train Neural Network
    X_train, X_test, y_train, y_test = train_test_split(Features, y, test_size=0.2, random_state=3)
    
    
    # define the keras model
    model = Sequential()
    model.add(Dense(10, input_dim=np.shape(X_train)[1], activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    
    # model.add(Dense(10, activation='relu'))
    # # model.add(BatchNormalization())
    # # # model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='linear'))
    
    # compile the keras model
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(loss='mae', optimizer=opt, metrics=['mse', 'mae'])
    
    log_name = "logs_TSPNN/fit/bestsofar_2hidden_0_01_300_all"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_name, histogram_freq=1, write_graph=True)
    
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=300, batch_size=256, validation_split=0.1, callbacks=[tensorboard_callback])

    if save_load_flag == 0:
        model.save("model_TSP.keras")
    
    # predict with the keras model
    prediction = model.predict(X_test)
    mse_test = mean_squared_error(y_test,prediction)
    mae_test = mean_absolute_error(y_test, prediction)
    prediction_error = prediction + mae_test
    print(mae_test)

    
# =============================================================================
# Testing against new data
# =============================================================================
elif section_flag == 3:
    n_tested_instances = 500
    
    if save_load_flag == 0:
        # Load the models/approximations
        modelNN = keras.models.load_model('model_TSP.keras')
        with open('model_greedy.pkl', 'rb') as f:
            model_greedy = pickle.load(f)
        with open('model_greedy2.pkl', 'rb') as f:
            model_greedy2 = pickle.load(f)
            
        
        # Sample of number cities
        n_sample = stats.randint(20,80).rvs(n_tested_instances)
        # Lists to store approximations
        Features_list = []
        features = []
        features2 = []
        gurobi_uppers = []
        gurobi_lowers = []
        gurobi_predictions = []
        # NN_predictions = []
        optimal_values = []
        # Generate the new instances
        for i in range(n_tested_instances):
            # Generate number of cities 
            n = n_sample[i]
            dist, Coords = instance_generator(n)
            # Extract features and determine optimal
            Features, optimal,_ = extract_features(dist, Coords)
            Features_list.append(Features)
            optimal_values.append(optimal)
            feature, feature2 = data_features(dist)
            features.append(feature)
            features2.append(feature2)
            gurobi_upper_bound, gurobi_lower_bound = GurobiBounds(dist)
            gurobi_uppers.append(gurobi_upper_bound)
            gurobi_lowers.append(gurobi_lower_bound)
            print("Instance: " + str(i))
            
        # Predict using the methods
        greedy_predictions = model_greedy.predict(features)
        greedy2_predictions = model_greedy2.predict(features2)
        
        
        for i in range(len(gurobi_uppers)):
            if gurobi_uppers[i] == gurobi_lowers[i]:
                gurobi_predictions.append(gurobi_lowers[i])
            else:
                gurobi_predictions.append(0.2*gurobi_lowers[i] + 0.8*greedy_predictions[i])
        
        Features = np.concatenate((np.array(Features_list), np.array(gurobi_predictions).reshape(n_tested_instances,1)), axis=1)
        # selection = []
        # Features = Features[:, selection]
        NN_predictions = modelNN.predict(Features)
        # NN_predictions.append(NN_prediction)
            
        lists = [greedy_predictions, greedy2_predictions, gurobi_predictions, NN_predictions]
        names = ["greedy_predictions.txt", "greedy2_predictions.txt", "gurobi_predictions.txt", "NN_predictions.txt"]
        for i in range(len(lists)):    
            with open(names[i], 'w') as f:
                for s in lists[i]:
                    f.write(str(s) + '\n')
                
    if save_load_flag == 1:
        greedy_predictions = loadtxt("greedy_predictions.txt", delimiter=",")
        greedy2_predictions = loadtxt("greedy2_predictions.txt", delimiter=",")
        gurobi_predictions = loadtxt("gurobi_predictions.txt", delimiter=",")
        # I saved the NN_predictions incorrectly as individual arrays so I have to use this to extract it properly
        NN_predictions = []
        with open("NN_predictions.txt", 'r') as file:
            for line in file:
                # Strip whitespace and brackets, then convert to float
                stripped_line = line.strip().strip('[]')  # Remove whitespace and brackets
                NN_predictions.append(float(stripped_line))  
        # NN_predictions = loadtxt("NN_predictions.txt", delimiter=",")
        optimal_values = loadtxt("Optimal_test_values.txt", delimiter=",")
    
    
    # with open("file.txt", 'r') as f:
    #     score = [line.rstrip('\n') for line in f]
        
    mae_greedy = mean_absolute_error(optimal_values, greedy_predictions)
    errors_greedy = np.array(optimal_values) - np.array(greedy_predictions)
    mae_greedy2 = mean_absolute_error(optimal_values, greedy2_predictions)
    errors_greedy2 = np.array(optimal_values) - np.array(greedy2_predictions)
    mae_gurobi = mean_absolute_error(optimal_values, gurobi_predictions)
    errors_gurobi = np.array(optimal_values) - np.array(gurobi_predictions)
    mae_NN = mean_absolute_error(optimal_values, NN_predictions)
    errors_NN = np.array(optimal_values) - np.array(NN_predictions)
    print("Mean absolute difference (greedy): " + str(mae_greedy))
    print("Mean absolute difference (greedy minus last): " + str(mae_greedy2))
    print("Mean absolute difference (gurobi approximation): " + str(mae_gurobi))
    print("Mean absolute difference (NN): " + str(mae_NN))
    
    
    
    plt.figure()
    plt.hist(errors_greedy)
    plt.title("Errors for greedy approximations")
    plt.ylabel("Density")
    plt.xlabel("Error")
    plt.show()
    
    
    plt.figure()
    plt.hist(errors_greedy2)
    plt.title("Errors for greedy minus last approximations")
    plt.ylabel("Density")
    plt.xlabel("Error")
    plt.show()
    
    
    plt.figure()
    plt.hist(errors_gurobi)
    plt.title("Errors for gurobi approximations")
    plt.ylabel("Density")
    plt.xlabel("Error")
    plt.show()
    
    plt.figure()
    plt.hist(errors_NN)
    plt.title("Errors for NN approximations")
    plt.ylabel("Density")
    plt.xlabel("Error")
    plt.show()
    
    # Higher size instance:
        
    if hundred_flag == 1:
        modelNN = keras.models.load_model('model_TSP.keras')
        with open('model_greedy.pkl', 'rb') as f:
            model_greedy = pickle.load(f)
        with open('model_greedy2.pkl', 'rb') as f:
            model_greedy2 = pickle.load(f)
    
        n = 100
        dist, Coords = instance_generator(n)
        # Extract features and determine optimal
        time = 480
        Features,_, near_optimal = extract_features(dist, Coords, time)
        feature, feature2 = data_features(dist)
        gurobi_upper_bound, gurobi_lower_bound = GurobiBounds(dist)
        
        greedy_prediction = model_greedy.predict([feature])
        greedy2_prediction = model_greedy2.predict([feature2])
        
        if gurobi_upper_bound == gurobi_lower_bound:
            gurobi_prediction = gurobi_lower_bound
        else:
            gurobi_prediction = 0.2*gurobi_lower_bound + 0.8*greedy_prediction
        
        Features = np.concatenate((np.array(Features), np.array(gurobi_prediction).reshape(1,)), axis=0)
        # selection = []
        # Features = Features[:, selection]
        NN_prediction = modelNN.predict(Features.reshape(1,12))
        # NN_predictions.append(NN_prediction)
        
        error_gurobi = near_optimal - gurobi_prediction
        error_NN = near_optimal - NN_prediction
        print("Gurobi approximation error: " + str(error_gurobi))
        print("Neural Network approximation error: " + str(error_NN))