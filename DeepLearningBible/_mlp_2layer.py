# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

class MLP2:
  def __init__(self, lr = 1e-2, input_size = 30, hidden_size = 10, output_size = 1):

    self.lr = lr
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.initialize_weights()

  def initialize_weights(self):
    """initalize weights and bias"""

    self.W1 = np.random.randn(self.input_size, self.hidden_size)
    self.b1 = np.zeros((1, self.hidden_size))
    self.W2 = np.random.randn(self.hidden_size, self.output_size)
    self.b2 = np.zeros((1, self.output_size))

  def relu(self, x, derivative = False):

    if derivative == True:
      return np.where(x >= 0, 1, 0)
    return np.maximum(0, x)

  def sigmoid(self, x, derivative = False):

    g = 1 / (1 + np.exp(-x))
    if derivative == True:
      return g * (1 - g)
    return g

  def forward(self, X):

    self.Z1 = np.dot(X, self.W1) + self.b1
    self.A1 = self.sigmoid(self.Z1)
    self.Z2 = np.dot(self.A1, self.W2) + self.b2
    self.yhat = self.sigmoid(self.Z2)

    return self.yhat

  def backward(self, X, y):
    """
    backpropagation of MLP
    """
    m = y.shape[0]

    dZ2 = self.yhat - y

    dW2 = np.dot(self.A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims= True) / m

    dA1 = np.dot(dZ2, self.W2.T)
    dZ1 = np.multiply(dA1, self.sigmoid(self.Z1, derivative= True))

    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims= True) / m

    return dW2, db2, dW1, db1

  def BCEloss(self, y, yhat):
    # binary cross entropy
      bce_loss = -(np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))) / len(y)
      return bce_loss


  def update_param(self, dW2, db2, dW1, db1):
    """
    optimizer
    : Batch Gradient Descent (BGD)

    """

    self.W2 -= self.lr * dW2
    self.b2 -= self.lr * db2
    self.W1 -= self.lr * dW1
    self.b1 -= self.lr * db1

  def fit(self, X,y, epochs = 100):
    """
    Train NN
    """

    start_time = time.time()

    for i in range(epochs):

      self.forward(X)

      train_loss = self.BCEloss(y, self.yhat)

      dW2, db2, dW1, db1 = self.backward(X, y)
      self.update_param(dW2, db2, dW1, db1)

      print(f"\rEpochs = {i}, Time = {time.time()-start_time}, Train Loss = {train_loss}", end = "")

    print("\nTraing Complete")
    print("------------------------------------------------------------")

X_train = np.array([[1, 0], [0, 1], [0, 0], [1, 1]]) # input features (4 x 2 design matrix)
y_train = np.array([[1], [1], [0], [0]])             # ground truth y labels (4x1)

model = MLP2(input_size=2, hidden_size=30, output_size=1)

model.fit(X_train, y_train, epochs=100000)

model.forward(np.array([[1,1], [0,1], [0,0], [1,0]]))

