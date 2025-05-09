# -*- coding: utf-8 -*-
"""MLP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1L1WVi7X3S4T4edspRrZ6hOW8jnAy3J5o

# Multi-Layer Perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

## optimizer

class Optimizer:
    def __init__(self, learning_rate=None, name=None):
        self.learning_rate = learning_rate
        self.name = name

    def config(self, layers):

        pass

    def optimize(self, idx, layers: list, grads: dict, *args):
        '''# Args: Takes in idx of the layer, list of the layers and the gradients as a dictionary
            Performs updates in the list of layers passed into it'''
        pass


## SGDM

class SGDM(Optimizer):
  """
  mu_init   : inital momentum coefficient
  max_mu    : max mu

  demon     : decaying momentum
  beta_init : inital beta value of demon, controlling the rate of decaying
  """
  def __init__(self, learning_rate = 1e-2, mu_init = 0.5, max_mu = 0.99, beta_init = 0.9, demon = False, **kwargs):
    super().__init__(**kwargs)
    self.mu_init = mu_init
    self.max_mu = max_mu
    self.demon = demon
    if self.demon:
      self.beta = beta_init
    self.m = dict()

  def config(self, layers):
    for i in range(1, len(layers)+1):
      self.m[f"W{i}"] = 0
      self.m[f"b{i}"] = 0

  def optimize(self, idx, layers, grads, epoch_num, steps):

    mu = min(self.mu_init * 1.2**(epoch_num - 1), self.max_mu)

    if self.demon:
      pass

    """

    self.m[f"W{idx}"] = self.m[f"W{idx}"] * mu + grads[f"dW{idx}"]
    self.m[f"b{idx}"] = self.m[f"b{idx}"] * mu + grads[f"db{idx}"]

    layers[idx].W -= self.learning_rate * self.m[f"W{idx}"]
    layers[idx].b -= self.learning_rate * self.m[f"b{idx}"]


    """
    # Another expression of SGDM

    self.m[f"W{idx}"] = self.m[f"W{idx}"] * mu - self.learning_rate * grads[f"dW{idx}"]
    self.m[f"b{idx}"] = self.m[f"b{idx}"] * mu - self.learning_rate * grads[f"db{idx}"]

    layers[idx].W += self.m[f"W{idx}"]
    layers[idx].b += self.m[f"b{idx}"]

## SGDM Nesterov

class Nesterov(SGDM):
  def __init__(self, learning_rate = 1e-2, **kwargs):

    self.learning_rate = learning_rate
    super().__init__(**kwargs)

  def optimize(self, idx, layers, grads, epoch_num, steps):

    mu = min(self.mu_init * 1.2**(epoch_num - 1), self.max_mu)

    if self.demon:
      pass

    self.m[f"W{idx}"] = self.m[f"W{idx}"] * mu - self.learning_rate * grads[f"dW{idx}"]
    self.m[f"b{idx}"] = self.m[f"b{idx}"] * mu - self.learning_rate * grads[f"db{idx}"]

    layers[idx].W += mu * self.m[f"W{idx}"] - self.learning_rate * grads[f"dW{idx}"]
    layers[idx].b += mu * self.m[f"b{idx}"] - self.learning_rate * grads[f"db{idx}"]


## AdaGrid

class AdaGrad(Optimizer):
  def __init__(self, learning_rate = 1e-2, epsilon = 1e-8, **kwargs):

    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.V = dict()
    super().__init__(**kwargs)

    def config(self, layers):
      for i in range(1, len(layers) + 1):
        self.V[f"W{i}"] = 0
        self.V[f"b{i}"] = 0

    def optimize(self, idx, layers, grads, epochs_num, steps):

      self.V[f"W{idx}"] += grads[f"dW{idx}"]**2
      self.V[f"b{idx}"] += grads[f"db{idx}"]**2

      layers[f"W{idx}"] -=  grads[f"dW{idx}"] * (self.learning_rate) / np.sqrt(self.V[f"W{idx}"] + self.epsilon)
      layers[f"b{idx}"] -=  grads[f"db{idx}"] * (self.learning_rate) / np.sqrt(self.V[f"b{idx}"] + self.epsilon)


## RMSProp

class RMSProp(Optimizer):
  def __init__(self, learning_rate = 1e-2, decay_rate = 0.9, epsilon = 1e-8, **kwargs):

    self.learning_rate = learning_rate
    self.decay_rate = decay_rate
    self.epsilon = epsilon
    super().__init__(**kwargs)
    self.V = dict()

  def config(self, layers):
    for i in range(1, len(layers) + 1):
      self.V[f"W{i}"] = 0
      self.V[f"b{i}"] = 0

  def optimize(self, idx, layers, grads, epochs_num, steps):

    self.V[f"W{idx}"] = self.decay_rate * self.V[f"W{idx}"] + (1 - self.decay_rate) * grads[f"dW{idx}"]**2
    self.V[f"b{idx}"] = self.decay_rate * self.V[f"b{idx}"] + (1 - self.decay_rate) * grads[f"db{idx}"]**2

    layers[idx].W -=  grads[f"dW{idx}"] * (self.learning_rate) / np.sqrt(self.V[f"W{idx}"] + self.epsilon)
    layers[idx].b -=  grads[f"db{idx}"] * (self.learning_rate) / np.sqrt(self.V[f"b{idx}"] + self.epsilon)



## Adam

class Adam(Optimizer):
  def __init__(self, learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, **kwargs): ## default recommended from paper

    self.learning_rate = learning_rate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    super().__init__(**kwargs)

    self.t = 0      # initalize step
    self.M = dict()
    self.V = dict()

  def config(self, layers):

    self.t = 0
    for i in range(1, len(layers) + 1):
      self.M[f"W{i}"] = 0
      self.M[f"b{i}"] = 0
      self.V[f"W{i}"] = 0
      self.V[f"b{i}"] = 0

  def optimize(self, idx, layers, grads, epochs_num, steps):

    self.t += 1 # update step

    # Update first moment estimate
    self.M[f"W{idx}"] = self.beta1 * self.M[f"W{idx}"] + (1 - self.beta1) * grads[f"dW{idx}"]
    self.M[f"b{idx}"] = self.beta1 * self.M[f"b{idx}"] + (1 - self.beta1) * grads[f"db{idx}"]

    # Update second moment estimate
    self.V[f"W{idx}"] = self.beta2 * self.V[f"W{idx}"] + (1 - self.beta2) * grads[f"dW{idx}"]**2
    self.V[f"b{idx}"] = self.beta2 * self.V[f"b{idx}"] + (1 - self.beta2) * grads[f"db{idx}"]**2


    # bias correction
    M_hat_W = self.M[f"W{idx}"] / (1 - self.beta1**self.t)
    M_hat_b = self.M[f"b{idx}"] / (1 - self.beta1**self.t)

    V_hat_W = self.V[f"W{idx}"] / (1 - self.beta2**self.t)
    V_hat_b = self.V[f"b{idx}"] / (1 - self.beta2**self.t)

    # Update parameters
    layers[idx].W -= self.learning_rate * M_hat_W / (np.sqrt(V_hat_W) + self.epsilon)
    layers[idx].b -= self.learning_rate * M_hat_b / (np.sqrt(V_hat_b) + self.epsilon)

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

    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    if derivative:
      exp = np.exp(x - np.max(x, axis=1, keepdims=True))
      return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=1, keepdims=True)

from math import perm
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


    self.optimizer.config(self.layers)
    self.optimizer.learning_rate = learning_rate

  def update_param_by_optimizer(self, epoch_num, steps):
    for idx in self.layers.keys():
      self.optimizer.optimize(idx, self.layers, self.grads, epoch_num, steps)

  def update_param(self):

    """
    batch gradient descent
    """

    for idx in range(1, len(self.layers)+1):
      self.layers[idx].W -= self.learning_rate * self.grads[f"dW{idx}"]
      self.layers[idx].b -= self.learning_rate * self.grads[f"db{idx}"]

  def compute_loss(self, y):

    l_sum = np.sum(np.log(self.A[len(self.layers)] + 1e-8))
    m = y.shape[0]
    loss = -(1./m)* l_sum

    if(self.loss == "CAT"):
      l_sum = np.sum(y * np.log(self.A[len(self.layers)] + 1e-8))
      m = y.shape[0]
      loss = -(1./m)* l_sum
      return loss
    return loss


  def fit(self, X, y, epochs = 1000, batch_size = 32):
    """Training cycle of model"""

    self.train_losses = []
    self.batch_size = batch_size
    self.epochs = epochs
    num_batches = (X.shape[0] + self.batch_size - 1) // self.batch_size

    for epoch in range(epochs):

      epoch_loss = []
      step = 0

      permutation = np.random.permutation(X.shape[0])
      X_suffle = X[permutation]
      y_suffle = y[permutation]

      for batch_idx in range(num_batches):
        start = batch_idx * self.batch_size
        end = np.minimum((batch_idx + 1) * self.batch_size, X.shape[0]-1)
        X_batch = X[start: end]
        y_batch = y[start: end]

        # Forward

        step += 1
        yhat = self.forward(X_batch)
        loss = self.compute_loss(y_batch)
        epoch_loss.append(loss)

        # Backprop

        self.backward(y_batch)

        # update_param
        self.update_param_by_optimizer(epoch, step)


      # loss
      train_loss = np.mean(epoch_loss) / len(epoch_loss)
      self.train_losses.append(train_loss)
      print(f"epoch = {epoch + 1}/{epochs}, loss = {train_loss}")
    print("------------------------------------------------------------")

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

import tensorflow.keras.datasets.mnist as mnist

# Load data from tensorflow
data = mnist.load_data()

(X_train, y_train), (X_test, y_test) = data
plt.imshow(X_train[0])
plt.show()


# Reduce the sample size
from sklearn.model_selection import train_test_split
train_size = 10000
test_size  = 5000
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=train_size,
                                                  test_size=test_size, shuffle=True)

print("Raw Data Shape")
print("X_train.shape :", X_train.shape)
print("X_test.shape  :", X_test.shape)
print("y_train.shape :", y_train.shape)
print("y_test.shape  :", y_test.shape)

# Preprocess data
# Reshape (flatten)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize data within {0,1} + dtype conversion
X_train = np.array(X_train/255., dtype=np.float32)
X_test  = np.array(X_test/255., dtype=np.float32)

def one_hot(y):
  lst = []
  for i in y:
    a = np.zeros(10)
    a[i] = 1.0
    lst.append(a)
  return np.array(lst)
hot_y_train = one_hot(y_train)

# Commented out IPython magic to ensure Python compatibility.
# visualizing the first 10 images in the dataset and their labels
# %matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 1))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.show()
print('label for each of the above image: %s' % (y_train[0:10]))

model = sequential(layers = [
    layer(64, activation="relu"),
    layer(32, activation="relu"),
    layer(1, activation="sigmoid")
    ])

sgdm = SGDM()
nes = Nesterov()
rmsprop = RMSProp()
adagrad = AdaGrad()
adam = Adam()

opt = [sgdm, nes, rmsprop, adam]
name = ["sgdm", "nes", "rmsprop", "adam"]

def run_model(opt = None):
  np.random.seed(100)
  model = sequential(layers = [
    layer(64, activation="relu"),
    layer(32, activation="relu"),
    layer(1, activation="sigmoid")
    ])
  model.complie(optimizer=opt, learning_rate=1e-3)
  model.fit(X_train, y_train, epochs= 30)
  return model.train_losses

plt.figure(figsize = (25, 12))
for col, (i, l) in enumerate(zip(opt, name)):
  loss_red = run_model(i)
  plt.plot(np.arange(len(loss_red)), loss_red, alpha = 0.8, color=f"C{col}", label=l)
plt.legend()
plt.show()

model = sequential([
    layer(64, activation="relu"),
    layer(32, activation="relu"),
    layer(10, activation="softmax")
])

model.complie(optimizer=adam, loss="CAT", learning_rate=1e-2)

model.fit(X_train, hot_y_train, epochs = 30)

from sklearn.metrics import accuracy_score
an = []
for i in model.forward(X_train):
  answer = i.argmax()
  an.append(answer)
y_pred = np.array(an)

print(f"Train accuracy socre : {accuracy_score(y_train, y_pred)}")

an = []
for i in model.forward(X_test):
  answer = i.argmax()
  an.append(answer)
y_train_pred = np.array(an)

print(f"Test accuracy socre : {accuracy_score(y_test, y_train_pred)}")

## Opimizer compare

def run_model(opt = None):
  np.random.seed(100)
  model = sequential(layers = [
    layer(16, activation="relu"), ## make more simple model
    layer(10, activation="softmax")
    ])
  model.complie(optimizer=opt, loss = "CAT",learning_rate=1e-3)
  model.fit(X_train, hot_y_train, epochs= 30)
  return model.train_losses

plt.figure(figsize=(25, 12))
for col, (i, l) in enumerate(zip(opt, name)):
  loss_red = run_model(i)
  plt.plot(np.arange(len(loss_red)), loss_red, alpha = 0.8, color=f"C{col}", label=l)
plt.legend()
plt.show()

