import copy
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


x_train = np.array([0.5275850943761389, 1.259210705775408, -1.3997210858463067, -0.14545194148894652, -0.23315632348299936, 
 2.1999302867483994, 2.5476636693983554, -2.2310687558043543, 0.4933533888500445, -0.5634904487431606, 
 0.899505843602414, 2.119506982039696, 1.011733592100994, 2.4084739956458625, -0.3542800076240181, 
 0.14906821627683425, 1.7779857914293544, -0.35872463616964695, 1.2490800710190126, 0.10971305928887527, 
 -0.9697406124407666, -1.156431957655343, 0.06173908885714956, -0.41062401042778185, -0.8463793500354725, 
 -0.8599868241915396, 0.33132809824229253, -0.08101704351970175, 0.49375703270979177, 0.4021201809140967]
)
y_train = np.array([2.1824739823144164, 0.6493057227386678, -0.3539293470103265, -0.1643743346320112, 1.0196856026090764, 
 0.9316211154950603, 1.8551333089955293, -0.8185847561972408, 0.1221200453749413, 0.1364418076166868, 
 0.6646946233850652, 0.4174703823758589, 0.7476386419038326, 2.6759954383806552, 1.1837255691180597, 
 -0.2625394614296533, 1.7186296254696445, 0.3474560616227083, 0.4372160708010098, -1.0145928429770463, 
 0.9268024732118861, 0.24808932604515688, -0.12940537888706796, -2.0373845333386913, 0.13778238179358707, 
 -1.2948562430284052, -0.5198790773730163, 0.5276681604798161, -0.029703913813753254, -1.3884553171333034]
)

def cost_function(x, y, w1, w2, b):
    m = len(y)
    cost = 1/(2*m) * np.sum(  (np.dot(x**2, w1)+ np.dot(x, w2) + b - y)**2  )
    return cost


def derivative(x, y, w1, w2, b):
    m = len(y)
    d_dw1 = 1 / m * np.dot((np.dot(x**2, w1) + np.dot(x, w2) + b - y), x**2)
    d_dw2 = 1 / m * np.dot((np.dot(x**2, w1) + np.dot(x, w2) + b - y), x)
    d_db = 1 / m * np.sum((np.dot(x**2, w1) + np.dot(x, w2) + b - y))
    return d_dw1, d_dw2, d_db


def gradient_descent(x, y, w1, w2, b, alpha):   
    i = 0
    #while(sum((np.dot(x, w) + b - y))> 10**-7):
    while(i <= 300000):
        d_dw1, d_dw2, d_db = derivative(x, y, w1, w2, b)
        w1 = w1 - alpha * d_dw1
        w2 = w2 - alpha * d_dw2
        b = b - alpha * d_db
        i += 1
        if (i %10000== 0):
            print(cost_function(x, y, w1, w2, b))
    print(w1, w2, b)
    return w1, w2, b


# an_w1, an_w2, an_b = gradient_descent(x_train, y_train, w1 = 0, w2 = 0, b = 0, alpha= 0.001)

# x = np.arange(-3, 5)
# plt.scatter(x_train, y_train)
# plt.plot(x , float(an_w1) * x**2 + float(an_w2) * x +  float(an_b))
# plt.show()


x = np.arange(-10, 20, 1)
y = 18 + -6.82 * x**2 + 0 * x + x**4

an_w1, an_w2, an_b = gradient_descent(x, y, w1 = 0, w2 = 0, b = 0, alpha= 10**-4)

plt.scatter(x, y)
plt.plot(x , float(an_w1) * x**2 + float(an_w2) * x +  float(an_b))
plt.show()

