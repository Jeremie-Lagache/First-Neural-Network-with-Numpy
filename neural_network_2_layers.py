import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from tqdm import tqdm
"import warnings"

"warnings.filterwarnings('ignore')"

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
X = X.T
y = y.reshape((1, y.shape[0]))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

'''plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')'''

def initialisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1)
    b2 = np.zeros((n2, 1))

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }
    return parametres

def forward_propagation(X, parametres):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2,
    }
    return activations

def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def back_propagation(X, y, activations, parametres):

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dZ2': dZ2,
        'dW2': dW2,
        'db2': db2,
        'dZ1': dZ1,
        'dW1': dW1,
        'db1': db1,
    }

    return gradients

def update(gradients, parametres, learning_rate):
    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW2 = gradients['dW2']
    db2 = gradients['db2']
    dW1 = gradients['dW1']
    db1 = gradients['db1']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

    return parametres

def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    A2 = activations['A2']

    return A2 >= 0.5

def neural_network(X, y, n1=2, learning_rate = 0.1, n_iter=1000):
    n0 = X.shape[0]
    n2 = y.shape[0]
    parametres = initialisation(n0, n1, n2)

    train_loss = []
    train_acc =[]

    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, parametres)
        A2 = activations['A2']

        if i %10 == 0:
            # Train
            train_loss.append(log_loss(y, A2))
            y_pred = predict(X, parametres)
            current_accuracy = accuracy_score(y.flatten(), y_pred.flatten())
            train_acc.append(current_accuracy)

        gradients = back_propagation(X, y, activations, parametres)
        parametres = update(gradients, parametres, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="train accuracy")
    plt.legend()
    plt.show()

    return parametres

from utilities import *

X_train, y_train, X_test, y_test = load_data()

y_train = y_train.T
y_test = y_test.T

X_train = X_train.T
X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()

X_test = X_test.T
X_test_reshape = X_test.reshape(-1, X_test.shape[-1]) / X_train.max()

m_train = 300
m_test = 80
X_test_reshape = X_test_reshape[:, :m_test]
X_train_reshape = X_train_reshape[:, :m_train]
y_train = y_train[:, :m_train]
y_test = y_test[:, :m_test]

parametres = neural_network(X_train_reshape, y_train, n1=64, n_iter=10000, learning_rate=0.01)

y_pred = predict(X_test_reshape, parametres)
current_accuracy = accuracy_score(y_test.flatten(), y_pred.flatten())
print('current acc', current_accuracy)



