# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:18:43 2024

@author: kolst
"""

import numpy as np
from tensorflow import keras
import datetime
import matplotlib as plt
import math
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# np.random.seed(seed=1)




X = np.loadtxt("Data_X.csv", delimiter=",", dtype=float)
y = np.loadtxt("Data_y.csv", delimiter=",", dtype=str)
y = np.where(y=="P",1, 0)

N = np.shape(X)[0] # number of patients
# K = 10 # number of clauses
J = np.shape(X)[1] # number of features
input_shape = (J, )


# Training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)





#%% Step 2: Building and training the model
def build_categorical_model(xtrain, ytrain, xtest, ytest, input_shape, learning_rate, dropout_rate, opt, activation_hidden, activation_output, batch_size, log_name, epochs, nrlayers, nrnodes):
    # This function builds and trains a neural network.
    # This network has "nrlayes" layers and "nrnodes" nodes in each layer,
    #   where "nrlayers" is a number and "nrnodes" is a vector specifying how many nodes each layer should contain
    # "xtrain" and "ytrain" contain the training data, while "xtest" and "ytest" contain the test data

    # Initialize the model
    model = keras.Sequential()

    # Add the input layer and dropout
    model.add(keras.layers.Dropout(dropout_rate, input_shape=input_shape))
    model.add(keras.Input(shape=input_shape))

    # Add the hidden layers, using the sigmoid activation function
    for i in range(nrlayers):
        # model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(nrnodes[i], activation=activation_hidden)) 
        
    # Add the final layer, using the softmax activation function
    model.add(keras.layers.Dense(1, activation=activation_output))

    # Give a summary of the model
    model.summary()

    model.compile(loss="binary_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])
    
    print(log_name)


    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_name, histogram_freq=1, write_graph=True)
    # Fit the model, note that keras takes care of the train set/validation set
    model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard_callback]) #TENSORBOARD Add the callback to the model training such that Tensorboard will create logfiles

    # Evaluate the trained model on the test data
    score = model.evaluate(xtest, ytest, verbose=1)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Return the trained model
    return model

# Build and train the model

# rates = [0.0001,0.001, 0.01, 0.1]
# dropout_rates = [0.05, 0.1, 0.15,0.2,0.25]
# rhos = [0.5,0.7,1]
d_rate = 0.05
learning_rate = 0.001
opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
activation_hidden = "sigmoid"
activation_output = "sigmoid"
activation_hiddens = ["relu", "leaky_relu", "tanh, sigmoid"]
activation_outputs = ["sigmoid", "softmax"]
batch_size = 32
log_name = "logs_assignment1/fit/configfinal"
epochs = 100
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_name, histogram_freq=1, write_graph=True)
optimizers = {"sgd": keras.optimizers.SGD(learning_rate=learning_rate),
              "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
              "adam": keras.optimizers.Adam(learning_rate=learning_rate),
              "adagrad": keras.optimizers.Adagrad(learning_rate=learning_rate)}
# k = 0
# for activation_hidden in activation_hiddens:
#     for activation_output in activation_outputs:
#         k +=1
#         log_name = "logs_assignment1/fit/config_again"+ str(58+k)+ "_" + activation_hidden +"_"+  activation_output
#         model = build_categorical_model(X_train, y_train, X_test, y_test, input_shape, learning_rate, d_rate, opt, activation_hidden, activation_output, batch_size, log_name, epochs, nrlayers=4, nrnodes = [200, 100, 50, 10])


# for learning_rate in rates:
#     opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
#     log_name = "logs_rate_lab2/fit/" + "learning_rate" + "-" + str(learning_rate)
#     model = build_categorical_model(X_train, y_train, X_test, y_test, learning_rate, opt, activation_hidden, activation_output, batch_size, log_name, nrlayers=2, nrnodes=[512,512])
    
# for opt_name, opt in optimizers.items():
#     for learning_rate in rates:
#         log_name = "logs_assignment1/fit/" + opt_name + "_" + str(learning_rate)
#         model = build_categorical_model(X_train, y_train, X_test, y_test, input_shape, learning_rate, d_rate, opt, activation_hidden, activation_output, batch_size, log_name, epochs, nrlayers=4, nrnodes = [200, 100, 50, 10])

model = build_categorical_model(X_train, y_train, X_test, y_test, input_shape, learning_rate, d_rate, opt, activation_hidden, activation_output, batch_size, log_name, epochs, nrlayers=4, nrnodes = [200, 100, 50, 10])

