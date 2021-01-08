import numpy as np
import math

# Create random input and output data
x = 1.5
y = 0.5


# Randomly initialize weights
w = 0.8


learning_rate = 1e-6
for t in range(2000):
    # Forward pass: compute predicted y
    # y = a + b x + c x^2 + d x^3
    y_pred = w * x 

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (y_pred - y)**2
    
    

    # Update weights
    w -= learning_rate * grad_a
    
    

print(f'Result: y = {w} * {x} ')