# -*- coding: utf-8 -*-

# Multi-Layer Perceptron


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

## optimizer

pass

## layer

class layer:
  def __init__(self, units = 1, activation = None, name = None):

    self.units = units
    self.activation = activation
    self.layer_name = name

    self.W = None

  def initialize_param(self, n_in, units):
    """Intializeing Parameter"""

    self.W = np.random.randn(n_in, units)
    self.b = np.zeros((1, units))

  def forward(self, X):
    """Forward Propagation"""
    if self.W is None:
      self.initialize_param(X.shape[-1], self.units)

    self.Z = np.dot(X, self.W) + self.b

    self.A = self.activation_func(self.Z)

    return self.Z ,self.A

  def activation_func(self, x, derivative = False):
    """Fucntion to decision activation function"""

    if self.activation == "linear":
      if derivative:
        return self.linear(x, derivative = True)
      return self.linear(x, derivative = False)

    if self.activation == "relu":
      if derivative:
        return self.relu(x, derivative = True)
      return self.relu(x, derivative = False)

    if self.activation == "sigmoid":
      if derivative:
        return self.sigmoid(x, derivative = True)
      return self.sigmoid(x, derivative = False)

    if self.activation == "softmax":
      if derivative:
        return self.softmax(x, derivative = True)
      return self.softmax(x, derivative = False)

  def linear(self, x, derivative = False):

    return np.where(derivative == True, 1, x)

  def relu(self, x, derivative = False):

    if derivative:
      return np.where(x >= 0, 1, 0)
    return np.maximum(0, x)

  def sigmoid(self, x, derivative = False):

    g = 1 / (1 + np.exp(-x))
    if derivative:
      return g * (1 - g)
    return g

  def softmax(self, x, derivative = False):

    pass

## sequential

class sequential:
  def __init__(self, layers = None, input_size = None):

    self.layers = {}
    if layers is not None:
      for layer in layers:
        self.layers[len(self.layers) + 1] = layer

    self.input_size = input_size

    self.A = {}
    self.Z = {}
    self.grads = {}

    self.loss = []

  def Input_size(self, input_size):

    self.input_size = input_size

  def add(self, layer):
    """Add Layer"""

    self.layers[len(self.layers) + 1] = layer

  def forward(self, X):
    """Forward Propagation"""

    self.A[0] = X

    for i in range(len(self.layers)):
      self.Z[i+1] ,self.A[i+1] = self.layers[i+1].forward(self.A[i]) ## forward porp of one layer

    return self.A[len(self.layers)]

  def backward(self, y):
    """
    back propagation algorithm only apply to classification

    """
    m = y.shape[0]

    length_layer = len(self.layers)

    for idx in range(length_layer, 0, -1):
      if idx == length_layer:
        self.grads[f"dZ{idx}"] = self.A[idx] - y # (dL/dA @ dA/dZ) only for cross entropy(softmax, sigmoid)
      else:
        self.grads[f"dA{idx}"] = self.grads[f"dZ{idx+1}"] @ self.layers[idx+1].W.T # (dL/dZ_n @ dZ_n/dA_n-1)
        self.grads[f"dZ{idx}"] = self.grads[f"dA{idx}"] * self.layers[idx].activation_func(self.Z[idx], derivative = True) # (dL/dA_n-1 @ dA_n-1/dZ_n-1)

      self.grads[f"dW{idx}"] = self.A[idx - 1].T @ self.grads[f"dZ{idx}"] / m
      self.grads[f"db{idx}"] = np.sum(self.grads[f"dZ{idx}"], axis= 0, keepdims=True) / m

  def complie(self, optimizer = None, loss = None, learning_rate = 1e-2):

    self.loss = loss
    self.optimizer = optimizer
    self.learning_rate = learning_rate

  def update_param(self):

    """
    batch gradient descent
    """

    for idx in range(1, len(self.layers)+1):
      self.layers[idx].W -= self.learning_rate * self.grads[f"dW{idx}"]
      self.layers[idx].b -= self.learning_rate * self.grads[f"db{idx}"]

  def compute_loss(self, y):

    l_sum = np.sum(y * np.log(self.A[len(self.layers)]))
    m = y.shape[0]
    loss = -(1./m)* l_sum
    return loss


  def fit(self, X, y, epochs = 1000):
    """Train model"""

    for i in range(1, epochs+1):

      self.forward(X)

      self.backward(y)

      self.update_param()

      print(f"\repochs = {i}, loss = {self.compute_loss(y)}", end = "")

    print("\n Training end\n------------------------------------------")

np.random.seed(42)

# 데이터 크기 및 차원 설정
n_samples = 1000
n_features = 4

# 클래스 0의 데이터 생성 (다차원 정규분포)
X0 = np.random.randn(n_samples // 2, n_features) + np.array([-1, -1, -1, -1])
y0 = np.zeros((n_samples // 2, 1))

# 클래스 1의 데이터 생성 (다차원 정규분포)
X1 = np.random.randn(n_samples // 2, n_features) + np.array([1, 1, 1, 1])
y1 = np.ones((n_samples // 2, 1))

# 데이터셋 합치기
X = np.vstack([X0, X1])
y = np.vstack([y0, y1])

# 데이터 섞기
indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

# 훈련/테스트 분할 (80% 훈련, 20% 테스트)
split = int(n_samples * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 데이터 시각화 (확인용)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='coolwarm', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Binary Classification Dataset')
plt.show()

model = sequential(layers = [
    layer(50, activation="relu"),
    layer(30, activation="relu"),
    layer(1, activation="sigmoid")
    ])

from sklearn.metrics import accuracy_score

model.complie(learning_rate=1e-2)

model.fit(X_train, y_train, epochs= 3000)

train_pred = np.where(model.forward(X_train) >= 0.5, 1, 0)
test_pred = np.where(model.forward(X_test) >= 0.5, 1, 0)

print(accuracy_score(y_train, train_pred))
print(accuracy_score(y_test, test_pred))

