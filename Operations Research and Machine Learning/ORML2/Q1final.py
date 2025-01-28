# Import packages
import pickle
from random import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# =============================================================================
# Heuristic functions
# =============================================================================

def SimpleHeuristic1(T):
    # Initialize game
    v = randint(1,12)
    t = 0
    state = 0
    profit = 0
    if v > 9:
        project_profit = [5, 0, 0]
    elif 6 < v <= 9:
        project_profit = [5, 50, 0]
    else:
        project_profit = [5, 50, 500]
    
    
    # Creating sample of project durations
    t_As = []
    t_Bs = []
    t_Cs = []
    for i in range(T): # 48 is an arbitrary large number so that we won't run out of projects
        t_As.append(randint(1,v))
        t_Bs.append(randint(v,2*v) + 6)
        t_Cs.append(randint(2*v,4*v) + 12)
        
        
    while t < 48:
        
        if t + t_Cs[0] < T: # Check if there is enough time left to do project C
            # Update profit and time
            profit += project_profit[2]
            t += t_Cs[0]
            t_Cs.pop(0) # Delete project duration, to access new one on next iteration
        else:
            break
        
    return profit

def SimpleHeuristic2(T):
    
    
    # Initialize game
    v = randint(1,12)
    t = 0
    state = 0
    profit = 0
    if v > 9:
        project_profit = [5, 0, 0]
    elif 6 < v <= 9:
        project_profit = [5, 50, 0]
    else:
        project_profit = [5, 50, 500]
    
    
    # Creating sample of project durations
    t_As = []
    t_Bs = []
    t_Cs = []
    for i in range(T): # 48 is an arbitrary large number so that we won't run out of projects
        t_As.append(randint(1,v))
        t_Bs.append(randint(v,2*v) + 6)
        t_Cs.append(randint(2*v,4*v) + 12)
        
        
    while t < T:
        
        if t == 0: # Do project A to estimate v
            if t_As[0] <= 5: # If estimated low v
                C_flag = 1
            else:
                C_flag = 0
            # Update profit and time
            profit += project_profit[0]
            t += project_profit[0]
            t_As.pop(0)
        else:
            if C_flag == 0 and t < 11: # Fire
                profit += 2* (T-t) # Profit from firing
                break
            else:
                if t > T - 14: # Too late to finish project C
                    if t > T - 7: # Too late to finish project B
                        if t + t_As[0] < T: # Enough time to finish project A
                            # Update profit and time
                            profit += project_profit[0]
                            t += t_As[0]
                            t_As.pop(0)
                        else:
                            break
                    else:
                        # Update profit and time
                        profit += project_profit[1]
                        t += t_Bs[0]
                        t_Bs.pop(0)
                        
                elif t + t_Cs[0] < T: # Enough time to do project C
                    # Update profit and time
                    profit += project_profit[2]
                    t += t_Cs[0]
                    t_Cs.pop(0)
                else:
                    break
                
        
    return profit

def SimpleHeuristic3(T):
    
    
    # Initialize game
    v = randint(1,12)
    t = 0
    state = 0
    profit = 0
    if v > 9:
        project_profit = [5, 0, 0]
    elif 6 < v <= 9:
        project_profit = [5, 50, 0]
    else:
        project_profit = [5, 50, 500]
    
    
    # Creating sample of project durations
    t_As = []
    t_Bs = []
    t_Cs = []
    for i in range(T): # 48 is an arbitrary large number so that we won't run out of projects
        t_As.append(randint(1,v))
        t_Bs.append(randint(v,2*v) + 6)
        t_Cs.append(randint(2*v,4*v) + 12)
        
        
    while t < T:
        
        
        if t == 0:             
            # Always do project C 
            if t + t_Cs[0] < T: # Enough time to do project C
                # Update profit and time
                profit += project_profit[2]
                t += t_Cs[0]
                t_Cs.pop(0)
                if project_profit[2] == 0: # If profit from C was 0 don't do C
                    C_flag = 0
                else:
                    C_flag = 1
            else:
                break
        else:
            if C_flag == 0:
                if t + t_Bs[0] < T: # Enough time for project B
                    # Update profit and time
                    profit += project_profit[1]
                    t += t_Bs[0]
                    t_Bs.pop(0)
                else:
                    break
            else:
                if t > T - 14: # Too late to finish project C
                    if t > T - 7: # Too late to finish project B
                        if t + t_As[0] < T: # Enough time for project A
                            # Update profit and time
                            profit += project_profit[0]
                            t += t_As[0]
                            t_As.pop(0)
                        else:
                            break
                    else:
                        # Update profit and time
                        profit += project_profit[1]
                        t += t_Bs[0]
                        t_Bs.pop(0)
                        
                elif t + t_Cs[0] < T: # Enough time for project C
                    # Update profit and time
                    profit += project_profit[2]
                    t += t_Cs[0]
                    t_Cs.pop(0)
                else:
                    break
                
    return profit

# =============================================================================
# Q-learning functions
# =============================================================================

def ManagerGameOnce(Q,N,T,alpha,epsilon,gamma, iteration):
    # Initialize game
    v = randint(1,12)
    t = 0
    state = 0
    profit = 0
    if v > 9:
        project_profit = [5, 0, 0]
    elif 6 < v <= 9:
        project_profit = [5, 50, 0]
    else:
        project_profit = [5, 50, 500]
    
    
    # Creating sample of project durations
    t_As = [randint(1, v) for _ in range(T)]
    t_Bs = [randint(v, 2 * v) + 6 for _ in range(T)]
    t_Cs = [randint(2 * v, 4 * v) + 12 for _ in range(T)]
    durations = [t_As, t_Bs, t_Cs]
    
    
    while t < T: # Check with timing of finishing jobs
        
        p = random() # random number between 0,1
        if t < 11:
            if (p < epsilon or Q[t][state][0] == Q[t][state][1] == Q[t][state][2] == Q[t][state][3]): # all same q-values or epsilon random policy
                a = randint(0,3)
            else: # Otherwise go for max action
                a = Q[t][state].index(max(Q[t][state])) 
        else: # Beyond first year
            if (p < epsilon or Q[t][state][1] == Q[t][state][2] == Q[t][state][3]):
                a = randint(1,3) # No fire action
            else:
                a = Q[t][state].index(max(Q[t][state][1:])) # No fire action
                 
        # Update vist count and decaying alpha if beyond 80% of iterations
        N[t][state][a] += 1 
        if iteration > 0.8 * n_game_trained:
            # Compute alpha        
            alpha = 1 / N[t][state][a]
                    
                    
        if a == 0: # If fired
            reward = 2 * (T - t) #profit from firing
            Q[t][state][a] = (1 - 0.05) * Q[t][state][a] + 0.05 * reward # Update Q table (terminal state)
            break
        else:
            if t + durations[a-1][0] < T: # If there is enough time to do project, a-1 because there are three profits and 4 total actions
                reward = project_profit[a-1] # Reward is profit of project
                Q[t][state][a] = (1 - alpha) * Q[t][state][a] + alpha * (reward + gamma * max(Q[t + durations[a-1][0]][state])) #Update Q table
                # Update time
                t += durations[a-1][0]
                durations[a-1].pop(0)
                state = a
            else:
                break
        
    return Q,N

def ManagerGameOnce_DoubleQ(Q1, Q2, N, T, alpha, epsilon, gamma, iteration):
    # Initialize game
    v = randint(1, 12)
    t = 0
    state = 0
    profit = 0
    if v > 9:
        project_profit = [5, 0, 0]
    elif 6 < v <= 9:
        project_profit = [5, 50, 0]
    else:
        project_profit = [5, 50, 500]

    # Creating sample of project durations
    t_As = [randint(1, v) for _ in range(T)]
    t_Bs = [randint(v, 2 * v) + 6 for _ in range(T)]
    t_Cs = [randint(2 * v, 4 * v) + 12 for _ in range(T)]
    durations = [t_As, t_Bs, t_Cs]

    while t < T: 
    
        p = random()  # random number between 0, 1
        # Epsilon-greedy action selection
        if t < 11: # While in first year of employment
            if p < epsilon or all(Q1[t][state][i] + Q2[t][state][i] == 0 for i in range(4)): # If all Q_values are zero or epsilon random
                a = randint(0, 3)
            else: # Otherwise max q_value action
                q_values = [Q1[t][state][i] + Q2[t][state][i] for i in range(4)] # Obtain additive q_value
                a = q_values.index(max(q_values))
        else: # Past first year
            if p < epsilon or all(Q1[t][state][i] + Q2[t][state][i] == 0 for i in range(1, 4)): # All zero/epsilon random policy
                a = randint(1, 3) # No fire action
            else: # Max
                q_values = [Q1[t][state][i] + Q2[t][state][i] for i in range(1, 4)] # Obtain additive q_value
                a = q_values.index(max(q_values[1:]))  # No fire action
                
        # Update the visitation count and alpha
        N[t][state][a] += 1
        alpha = 1 / N[t][state][a]

        if a == 0: # If fire
            reward = 2 * (T - t) # Profit from firing
            if random() > 0.5:  # Randomly update Q1 or Q2
                Q1[t][state][a] = (1 - alpha) * Q1[t][state][a] + alpha * reward
            else:
                Q2[t][state][a] = (1 - alpha) * Q2[t][state][a] + alpha * reward
            break
        else:
            if t + durations[a - 1][0] < T: # Enough time to do project, a - 1 since three project durations but four total actions
                # compute reward and end date of project
                reward = project_profit[a - 1]
                t_next = t + durations[a - 1][0]

                if random() > 0.5:  # Update Q1
                    max_action = Q1[t_next][state].index(max(Q1[t_next][state])) # Take max action from Q1
                    Q1[t][state][a] = (1 - alpha) * Q1[t][state][a] + alpha * (reward + gamma * Q2[t_next][state][max_action]) #Update Q1 table, based on best action from Q2 in next state
                else:  # Update Q2
                    max_action = Q2[t_next][state].index(max(Q2[t_next][state])) # Take max action from Q2
                    Q2[t][state][a] = (1 - alpha) * Q2[t][state][a] + alpha * (reward + gamma * Q1[t_next][state][max_action]) #Update Q2 table, based on best action from Q1 in next state
                # Update time and state
                t += durations[a - 1][0]
                durations[a - 1].pop(0)
                state = a
            else:
                break

    return Q1, Q2, N

def play_trained(Q, T, rand=0):
    # Initialize game
    v = randint(1,12)
    t = 0
    state = 0
    profit = 0
    if v > 9:
        project_profit = [5, 0, 0]
    elif 6 < v <= 9:
        project_profit = [5, 50, 0]
    else:
        project_profit = [5, 50, 500]
    
    
    # Creating sample of project durations
    t_As = []
    t_Bs = []
    t_Cs = []
    for i in range(T): # 48 is an arbitrary large number so that we won't run out of projects
        t_As.append(randint(1,v))
        t_Bs.append(randint(v,2*v) + 6)
        t_Cs.append(randint(2*v,4*v) + 12)
        
    durations = [t_As, t_Bs, t_Cs]
    
    # Normalize Q for random test
    # Takes the sum of each Q value in each state and divides each entry by sum, such that each state adds up to one
    Q_rand = Q
    if rand == 1:
        for t_rand in range(T):
            for state_rand in range(4):
                state_sum = sum(Q[t_rand][state_rand])
                if state_sum == 0:
                    continue
                for a_rand in range(4):
                    Q_rand[t_rand][state_rand][a_rand] = Q_rand[t_rand][state_rand][a_rand] / state_sum
                    
                    
    while t < T: # Check with timing of finishing jobs

        if rand == 1:
            if Q_rand[t][state][0] == Q_rand[t][state][1] == Q_rand[t][state][2] == Q_rand[t][state][3]:
                a = randint(0,3) # If all equal do discrete uniform choice
            else:
                a = np.random.choice(np.arange(0,4), p=np.array(Q_rand[t][state])) # Choose action based on discrete distribution given by normalized q table
        else:
            a = Q[t][state].index(max(Q[t][state])) # Otherwise always do maximum action
        
    
        if a == 0: # If fire
            profit += 2 * (T - t) # Profit from firing
            break
        else:
            if t + durations[a-1][0] < T: # Enough time for project
                # Update profit, state and time
                profit += project_profit[a-1] 
                t += durations[a-1][0]
                durations[a-1].pop(0)
                state = a
            else:
                break        
            
        
    return profit, Q_rand


# Universal parameters and flags
T = 48
n_game_tested = 1000
rand_flag = 0
heuristic_flag = 1
run_training_flag = 0
saveQ_flag = 0


# =============================================================================
# Test simple heuristics
# =============================================================================
if heuristic_flag == 1:
    profits1 = []
    profits2 = []
    profits3 = []
    for i in range(n_game_tested):
        profit1 = SimpleHeuristic1(T)
        profits1.append(profit1)
        profit2 = SimpleHeuristic2(T)
        profits2.append(profit2)
        profit3 = SimpleHeuristic3(T)
        profits3.append(profit3)
        if i % 10000 == 0 and i != 0:
            print(f"Tested heuristic iteration {i}")
        
    mean_profit1 = np.mean(profits1)
    sd_profit1 = np.std(profits1)
    mean_profit2 = np.mean(profits2)
    sd_profit2 = np.std(profits2)
    mean_profit3 = np.mean(profits3)
    sd_profit3 = np.std(profits3)
    print("Simple Heuristic tests complete")
    print("------------------------------------------------------------------")

# =============================================================================
# Train Q-learning
# =============================================================================

# Define the RL parameters
n_game_trained = 100000
epsilon = 0.25
gamma = 1
alpha = 0.01

if run_training_flag == 1:
    # Create state-space list and visit list
    # Q[time][project/start/fired][a]
    # action in state 0 is to fire, and state 1,2,3 to go to A or B or C
    # action in state 1,2,3 is to fire (index 0, if t <= 11) or to a different project (1,2,3)
    Q = [[[0] * 4 for s in range(4)] for t in range(T)]
    N = [[[0] * 4 for s in range(4)] for t in range(T)]
    Q1 = [[[0] * 4 for s in range(4)] for t in range(T)]
    Q2 = [[[0] * 4 for s in range(4)] for t in range(T)]
    Q_double = [[[0] * 4 for s in range(4)] for t in range(T)]
    
    
    for i in range(n_game_trained):
        Q,N = ManagerGameOnce(Q, N, T, alpha, epsilon, gamma, i)
        Q1, Q2, Nd = ManagerGameOnce_DoubleQ(Q1, Q2, N, T, alpha, epsilon, gamma, i)
        if i % 10000 == 0 and i != 0:
            print("Trained Q-learning instance: " + str(i))
    
    # Convert Q1 and Q2 into one table
    for i in range(len(Q1)):
        for j in range(len(Q1[0])):
            for k in range(len(Q2[0][0])):
                           Q_double[i][j][k] = Q1[i][j][k] + Q2[i][j][k]
    if saveQ_flag == 1:                       
        # Saving Q table
        with open("Q_table_1", "wb") as fp:   #Pickling
            pickle.dump(Q, fp) 
    
        with open("Q_double_table_1", "wb") as fp:   #Pickling
            pickle.dump(Q_double, fp) 
    
    print("Completed Q-learning training")
    print("-----------------------------------------------------")
else:
    with open('Q_table_1', 'rb') as f:
        Q = pickle.load(f)
    with open('Q_double_table_1', 'rb') as f:
        Q_double = pickle.load(f)
    


profit_test = []
profit_double = []
profit_test_rand = []
profit_double_rand = []
for i in range(n_game_tested):
    if i % 10000 == 0 and i != 0:
        print("Tested Q-learning instance " + str(i))
    profit,_ = play_trained(Q, T)
    profit_test.append(profit)
    profitd,_ = play_trained(Q_double, T)
    profit_double.append(profitd)
    if rand_flag == 1:
        profit_rand, _ = play_trained(Q,T,1)
        profit_test_rand.append(profit_rand)
        profit_d_rand, _ = play_trained(Q_double,T, 1)
        profit_double_rand.append(profit_d_rand)

print("Completed Q-learning testing")


print("--------------------------------------------------")
if heuristic_flag == 1:
    print("Simple Heuristic 1 average profit: " + str(mean_profit1))
    print("Simple Heuristic 1 standard deviation: " + str(sd_profit1))
    print("Simple Heuristic 2 average profit: " + str(mean_profit2))
    print("Simple Heuristic 2 standard deviation: " + str(sd_profit2))
    print("Simple Heuristic 3 average profit: " + str(mean_profit3))
    print("Simple Heuristic 3 standard deviation: " + str(sd_profit3))
    print("--------------------------------------------------")
avg_test_profit = np.mean(profit_test)
sd_test_profit = np.std(profit_test)
print("Q-learning average profit: " + str(avg_test_profit))
print("Q-learning profit standard deviation: " + str(sd_test_profit))
avg_double_profit = np.mean(profit_double)
sd_double_profit = np.std(profit_double)
print("Double-Q average profit of double Q: " + str(avg_double_profit))
print("Double-Q Profit standard deviation of double Q: " + str(sd_double_profit))
if rand_flag == 1:
    avg_test_profit_rand = np.mean(profit_test_rand)
    sd_test_profit_rand = np.std(profit_test_rand)
    print("Q-learning average random profit: " + str(avg_test_profit_rand))
    print("Q-learning profit standard deviation: " + str(sd_test_profit_rand))
    avg_double_profit_rand = np.mean(profit_double_rand)
    sd_double_profit_rand = np.std(profit_double_rand)
    print("Double-Q average random profit: " + str(avg_double_profit_rand))
    print("Double-Q profit standard deviation: " + str(sd_double_profit_rand))
    
    