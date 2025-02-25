import copy
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

#X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
#y_train = np.array([0, 0, 0, 1, 1, 1])

X_train = np.array([[1.149, 0.959], [1.194, 1.457], [0.930, 0.930], [1.474, 1.230], [0.859, 1.163],
 [0.861, 0.860], [1.073, 0.426], [0.483, 0.831], [0.696, 1.094], [0.728, 0.576],
 [1.440, 0.932], [1.020, 0.573], [0.837, 1.033], [0.655, 1.113], [0.820, 0.912],
 [2.319, 3.056], [2.496, 2.183], [2.747, 2.134], [2.563, 1.912], [2.102, 2.559],
 [2.722, 2.551], [2.465, 2.410], [2.056, 2.284], [2.362, 2.817], [2.603, 1.971],
 [2.597, 2.384], [2.297, 2.684], [2.809, 2.779], [2.248, 2.407], [2.599, 2.793]])

y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
)


## function 1/ (1 + exp(-x))

def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    return g

## cost fuction J

def cost_function(X_train, y_train, w, b):
    
    m = len(y_train)

    z = np.dot(X_train, w) + b
    f_wb = sigmoid(z)
    cost = - np.dot(y_train, np.log(f_wb)) - np.dot((1 - y_train),np.log(1 - f_wb))
    return cost / m
    
print(cost_function(X_train, y_train, np.array([1,1]), -3))

## derivative of cost fuction

def derivative(x, y, w, b):
    m = len(y)
    d_dw = 1/ m * np.dot(sigmoid(np.dot(x, w) + b) - y, x)
    d_db = 1/ m * np.sum(sigmoid(np.dot(x, w) + b) - y)

    return d_dw, d_db

# test
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_dw_tmp, dj_db_tmp = derivative(X_train, y_train, w_tmp, b_tmp)
print(dj_dw_tmp, dj_db_tmp)
# true : dj_db: 0.49861806546328574
#        dj_dw: [0.498333393278696, 0.49883942983996693]


## gradient descent of logistic regression

def gradient_descent(x, y, w, b, iter_max, alpha):
    

    for i in range(0, iter_max):
        d_dw, d_db = derivative(x, y, w, b)
        w = w - alpha * d_dw
        b = b - alpha * d_db
        if(i % 10000 == 0):
            print(f"iter : {i} cost : {cost_function(x, y, w, b)}")
    return w, b


an_w, an_b = gradient_descent(X_train, y_train, np.array([0,0]), 0, 10000, 10e-2)
print(an_w, an_b)

# plotting
x = range(0, 4)
#print(X_train[:,0])
plt.scatter(X_train[:,0], X_train[:,1])
plt.plot(x,  - an_w[0]/an_w[1] * x - an_b/an_w[1])
plt.show()



