#A simple Neural Network from scratch(One input,one output with no hidden layers)

import numpy as np

#Steps to follow:
###1. Define independent variables and dependent variable
###2. Define Hyperparameters
###3. Define Activation Function and its derivative
###4. Train the model
        #- Forward pass
        #- Backward propagation
        #- Update Weight
        
###5. Make predictions

###1. Define independent variables and dependent variable
#Independent variables
input_set = np.array([[0,1,0],
                      [0,0,1],
                      [1,0,0],
                      [1,1,0],
                      [1,1,1],
                      [0,1,1],
                      [0,1,0]])#Dependent variable
labels = np.array([[1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1]])
labels = labels.reshape(7,1) #to convert labels to vector


###2. Define Hyperparameters
np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05 #learning rate

###3. Define Activation Function and its derivative
#We make use of the sigmoid function

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

###4. Train the model
for epoch in range(25000):
    
    inputs = input_set
    
    #feedforward
    XW = np.dot(inputs, weights)+ bias
    
    z = sigmoid(XW)
    
    #Backpropagtion
    error = z - labels
    print(error.sum())
    
    dcost = error
    dpred = sigmoid_derivative(z)
    z_del = dcost * dpred
    inputs = input_set.T
    
    #update weights
    weights = weights - lr*np.dot(inputs, z_del)
    
    for num in z_del:
        bias = bias - lr*num
        
   
###5. Make predictions for a single point
single_pt = np.array([1,1,0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)   