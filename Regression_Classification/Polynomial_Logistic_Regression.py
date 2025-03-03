import copy
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    return g


X_train = np.array([[ 1.04967142,  0.0324084 ],
        [ 1.07530134,  0.10191925],
        [ 1.19749876,  0.22934456],
        [ 1.26450785,  0.51144947],
        [ 1.00240548,  0.68083147],
        [ 0.86543975,  0.75649658],
        [ 0.88547559,  0.62067531],
        [ 0.64612886,  0.68306545],
        [ 0.38641089,  0.74799623],
        [ 0.37845677,  0.82992969],
        [ 0.18615069,  0.73542683],
        [ 0.09393524,  0.85052712],
        [ 0.05542807,  0.86329043],
        [-0.29455791,  0.95046504],
        [-0.42980749,  1.20862636],
        [-0.46978143,  1.25938021],
        [-0.65063692,  1.04580738],
        [-0.61486438,  1.02685733],
        [-0.78880652,  0.8047437 ],
        [-0.85468875,  0.542864  ],
        [-0.56743959,  0.50090496],
        [-0.74787056,  0.50308632],
        [-0.76033708,  0.25110879],
        [-0.98746326,  0.32102585],
        [-1.00175698, -0.20115463],
        [-1.03747982,  0.01486969],
        [-1.23322923, -0.20905107],
        [-1.09345179, -0.40542578],
        [-1.13670868, -0.50930874],
        [-0.99134112, -0.82506129],
        [-0.8696042 , -0.711048  ],
        [-0.46136985, -0.67626463],
        [-0.49929393, -0.56605106],
        [-0.48149387, -0.77201724],
        [-0.19492295, -0.83403   ],
        [-0.30981052, -0.87265775],
        [-0.06792981, -0.82913231],
        [-0.1630957 , -0.9921734 ],
        [ 0.04587297, -1.15824657],
        [ 0.35624883, -1.08266471],
        [ 0.55921258, -1.08674231],
        [ 0.62056364, -0.89855114],
        [ 0.66602939, -0.91988159],
        [ 0.67903402, -0.71953426],
        [ 0.5661216 , -0.57206331],
        [ 0.64483217, -0.55005508],
        [ 0.69556478, -0.27064014],
        [ 0.90757205, -0.18416688],
        [ 0.92881409, -0.1148155 ],
        [ 0.82369598, -0.02345871],
        [ 1.85846293,  0.02504929],
        [ 2.05317132,  0.30479557],
        [ 2.10219732,  0.49224562],
        [ 2.0278811 ,  0.87670405],
        [ 1.98777015,  1.15788198],
        [ 1.86886274,  1.29317827],
        [ 1.78485076,  1.7324421 ],
        [ 1.34559401,  1.71281345],
        [ 1.07250423,  1.60759193],
        [ 0.76385693,  1.80803893],
        [ 0.32184639,  1.63343682],
        [ 0.27409806,  1.79050162],
        [ 0.06055513,  1.81637168],
        [ 0.08189442,  1.62241861],
        [-0.4120858 ,  1.81752464],
        [-0.60973237,  1.78005806],
        [-0.91080333,  1.82139189],
        [-1.30463713,  1.89243594],
        [-1.35095257,  1.5888476 ],
        [-1.64304211,  1.38738152],
        [-1.84527661,  1.16368099],
        [-2.15639039,  0.91308806],
        [-1.99134882,  0.70003917],
        [-2.26394585,  0.44771757],
        [-1.99443484,  0.15948379],
        [-1.71961546, -0.04174784],
        [-1.90152951, -0.34973196],
        [-1.72122451, -0.40733012],
        [-1.52845869, -0.76733155],
        [-1.47832018, -0.65749025],
        [-1.48261476, -1.06759171],
        [-1.21711241, -1.43344133],
        [-1.20692769, -1.68502162],
        [-0.89546176, -1.75896411],
        [-0.83351554, -2.03744   ],
        [-0.34224059, -2.10712458],
        [-0.29798764, -2.22970585],
        [ 0.04146804, -2.30471563],
        [ 0.44300186, -2.32161334],
        [ 0.5013004 , -2.25525115],
        [ 0.87057785, -1.95991842],
        [ 1.15753535, -1.60820889],
        [ 1.00507465, -1.4404864 ],
        [ 1.29562862, -1.36143943],
        [ 1.40318511, -1.01051061],
        [ 1.55955839, -0.79576374],
        [ 1.47586313, -0.73597509],
        [ 1.60066502, -0.43899774],
        [ 1.92411836, -0.23553703],
        [ 2.02969847, -0.11429703]])

y_train =  np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])



##  z = w1 * x1 + w2 * x2 + w3 * x1*x2 + w4 * x1**2 + w5 * x2**2

##  w = [w1,w2,w3,w4,w5]
##  b = b

## here we need to do feature engineering first

"""
x1    = X_train[:,0]
x2    = X_train[:,1]
x1x2  = X_train[:,0] * X_train[:,1]
x12   = X_train[:,0]**2
x22   = X_train[:,1]**2

X_train_eng = np.c_[x1, x2, x1x2, x12, x22]

"""
x1    = X_train[:,0]
x2    = X_train[:,1]
x1x2  = X_train[:,0] * X_train[:,1]
x12   = X_train[:,0]**2
x22   = X_train[:,1]**2

X_train_eng = np.c_[x1, x2, x1x2, x12, x22]



## cost fuction J 

def cost_function(X_train, y_train, w, b):
    
    m = len(y_train)

    z = np.dot(X_train, w) + b
    f_wb = sigmoid(z)
    cost = - np.dot(y_train, np.log(f_wb)) - np.dot((1 - y_train),np.log(1 - f_wb))
    return cost / m
    
print(cost_function(X_train_eng, y_train, np.array([1,1,1,1,1]), -3))

## derivative of cost fuction

def derivative(x, y, w, b):
    m = len(y)
    d_dw = 1/ m * np.dot(sigmoid(np.dot(x, w) + b) - y, x)
    d_db = 1/ m * np.sum(sigmoid(np.dot(x, w) + b) - y)

    return d_dw, d_db

# test
w_tmp = np.array([2.,3.,2., 3., 5.])
b_tmp = 1.
dj_dw_tmp, dj_db_tmp = derivative(X_train_eng, y_train, w_tmp, b_tmp)
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
            pass
    return w, b



an_w, an_b = gradient_descent(X_train_eng, y_train, np.array([0,0,0,0,0]), 0, 200000, 10e-2)
print(an_w, an_b)


# plotting

# 타원 방정식의 계수 정의
a, b, c, d, e, f = an_w[0], an_w[1], an_w[2], an_w[3], an_w[4], an_b  # x_1^2 + x_2^2 - 1 = 0 형태

# x1과 x2의 범위 생성
x1 = np.linspace(-2, 2, 500)
x2 = np.linspace(-2, 2, 500)
x1, x2 = np.meshgrid(x1, x2)

# 타원 방정식
z = a * x1 + b * x2 + c * x1 * x2 + d * x1**2 + e * x2**2 +  f

#print(X_train[:,0])
plt.scatter(X_train[:,0], X_train[:,1], c = y_train)
plt.contour(x1, x2, z, levels=[0], colors='blue')  # 타원 경계를 그림
plt.show()