# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:08:06 2024

@author: kolst
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import itertools as it
import scipy


def mapping(X):
    # Mapping from R^nxn to R^{n choose 2}, taking the entries above diagonal 
    # of symmetric matrix and converting it into a vector.
    n = np.shape(X)[0]
    tri_ind = np.triu_indices(n, 1)
    x = X[tri_ind]

    return x

def mapping_inverse(x):
    # Inverse mapping of the above mapping
    # m = np.shape(x)[0]
    # n = int((1 + np.sqrt(1 + 4*2*m))/2) # Obtaining dimension of symmetric output
    # X = np.eye(n)
    # X1 = np.zeros((n,n))
    # utri_ind = np.triu_indices(n, 1)
    # X[utri_ind] = x
    # X1[utri_ind] = x
    # X = X + X1.T
    
    # Inverse mapping of the above mapping
    m = np.shape(x)[0]
    n = int((1 + np.sqrt(1 + 4 * 2 * m)) / 2)  # Dimension of symmetric output matrix
    X = np.zeros((n, n))  # Initialize an nxn matrix with zeros

    # Upper triangle indices excluding the diagonal
    utri_ind = np.triu_indices(n, 1)
    
    # Populate the upper triangle and symmetrize
    X[utri_ind] = x
    X = X + X.T + np.identity(n)  # Mirror the upper triangle to the lower triangle
    
    return X
                    

# 2. Ellipsoid method generic framework
def ellipsoid_method(n, m, oracle, z_init, A_init, W, mapping, mapping_inverse, iter_max=20000, tolerance=1e-7):
    """
    Generalized Ellipsoid method for convex optimization, relying only on a separation oracle.

    Inputs:
    - oracle: A function that takes the current point (z_old) and returns:
              * a separating vector 'a' for the current point if z_old is infeasible, or
              * a direction 'a' in case of feasibility
    - z_init: Initial guess for the ellipsoid center, shape (n,).
    - A_init: Initial ellipsoid matrix (positive definite), shape (n, n).
    - iter_max: Maximum number of iterations allowed.
    - tolerance: Convergence tolerance for stopping criteria.

    Outputs:
    - z: Final point after optimization (near-optimal or feasible solution).
    - A: Final ellipsoid matrix.
    - z_list: List of all ellipsoid centers throughout the iterations.
    - A_list: List of all ellipsoid matrices throughout the iterations.
    """
    # m = z_init.shape[0]  # Dimension of the problem
    z_old = z_init  # Initialize ellipsoid center
    A_old = A_init  # Initialize ellipsoid matrix
    objective_value_best = 0 # Initialize best saved objective value

    z_list = [z_old]  # Store ellipsoid centers for tracking
    A_list = [A_old]  # Store ellipsoid matrices for tracking

    iter_nr = 0  # Initialize iteration counter

    while True:
        # Call the oracle to get the separating vector (always returns a vector 'a')
        a = oracle(z_old, c)

        # Update the ellipsoid
        a_tilde = (1 / np.sqrt(a.T @ A_old @ a)) * (A_old @ a)  # Normalize the separating vector
        A = (m**2 / (m**2 - 1)) * (A_old - (2 / (m + 1)) * np.outer(a_tilde, a_tilde))  # Update ellipsoid matrix
        z = z_old - (1 / (n + 1)) * a_tilde  # Update the ellipsoid center

        # Store the current ellipsoid data for tracking
        z_list.append(z)  # Append the new ellipsoid center
        A_list.append(A)  # Append the new ellipsoid matrix

        # Update z_old and A_old for the next iteration
        z_old = z
        A_old = A

        # Stopping criteria: based on ellipsoid size and maximum iterations
        ellipsoid_size = np.sqrt(np.max(a.T @ A @ a, 0))  # Measure ellipsoid size
        if iter_nr >= iter_max:
            print('Maximum number of iterations reached')
            break
        elif ellipsoid_size < tolerance:
            print('Optimal or near-optimal solution found')
            break

        iter_nr += 1  # Increment iteration counter

    return z, A, z_list, A_list, iter_nr # Return the final solution, ellipsoid matrix, and the lists

# Call the generalized ellipsoid method with the LP oracle
def oracle(z_old, c):
    Z = mapping_inverse(z_old)
    # n = np.shape(Z)[0]
    # Eigendecomposition
    Z_eigenvalues, Z_eigenvectors = np.linalg.eigh(Z)
    eig = Z_eigenvalues[0]
    v = Z_eigenvectors[:,0]
    
    if eig >= 0:
        return c
    else:
        return -mapping(np.outer(v,v))


def plot_objective(z_list, c, title):
    obj_list = []
    for z in z_list:
        obj_list.append(c.T @ z)
        
    iterations = np.arange(len(obj_list))
    plt.figure()
    plt.plot(iterations, obj_list)
    plt.title(title)
    plt.ylabel("Objective value")
    plt.xlabel("Iterations")
    plt.show()

# Generate data for LP
# min c^Tx
# s.t. x \in P

# Generating W
W_dict = {}
for n in range(3,8):
    W = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if j == np.mod(i+1,n):
                W[i,j] = 1
            elif i == np.mod(j+1,n):
                W[i,j] = 1
    W_dict["W_" + str(n)] = W
    

for i in range(3,8):   
    W = W_dict["W_" + str(i)]
    n = W.shape[0]
    m = int(scipy.special.comb(n,2))
    c = 2 * mapping(W)
    
    print("For n = " + str(i) + ":")
    
    # Radius of ball containing feasible region
    R = np.sqrt(m)
    
    # Construct initial ellipsoid
    z_init = np.zeros(m)  # center
    A_init = R**2 * np.identity(m)  # initial ellipsoid matrix
    
    
        
    
    # Run the ellipsoid method
    z_opt, A_opt, z_list, A_list, iterations = ellipsoid_method(n, m, oracle, z_init, A_init, W, mapping, mapping_inverse, iter_max = 20000)
    
    # Output the solution
    optimal_value = c.T @ z_opt
    # print("Optimal x:", z_opt)
    # print("Optimal value:", optimal_value)
    # print("Number of iterations: " + str(iterations))
    print(mapping_inverse(z_opt))
    
    # visualize path
    title = "Objective path for n = "+ str(n)
    plot_objective(z_list, c, title)
