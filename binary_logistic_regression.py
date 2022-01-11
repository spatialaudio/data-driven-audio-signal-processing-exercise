# Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
# Institute of Communications Engineering (INT), Faculty of Computer Science
# and Electrical Engineering (IEF), University of Rostock, Germany
#
# Data Driven Audio Signal Processing - A Tutorial with Computational Examples
# Feel free to contact lecturer frank.schultz@uni-rostock.de
#
# Exercise 10: Binary logistic regression model with just one layer
# Training using gradient descent and forward/backward propagation
# following the derivations and coding conventions from the brilliant
# course https://www.coursera.org/learn/neural-networks-deep-learning
# cf. especially week 2

import numpy as np
# rng = np.random.RandomState(1)  # for debug
rng = np.random.RandomState()


def my_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost(y, a):
    # vectorized loss
    L = - (y*np.log(a) + (1-y)*np.log(1-a))
    # cost function as avg of all L entries
    J = np.mean(L)
    return(J)


def create_data(nx, m, mean_offset):
    # std=1,  all features have same +mean
    X1 = rng.randn(nx, m//2) + mean_offset
    # vary the variance [0...1] of the features
    for i in range(nx):
        X1[i, :] *= rng.rand(1)
    # class 1 code
    Y1 = np.ones((1, m//2))

    X2 = rng.randn(nx, m//2) - mean_offset
    for i in range(nx):
        X2[i, :] *= rng.rand(1)
        Y2 = np.zeros((1, m//2))

    X = np.concatenate((X1, X2), axis=1)
    Y = np.concatenate((Y1, Y2), axis=1)
    return X, Y


# gradient descent hyper parameters
# set up to see how things evolve
# in practice do automated hyper parameter search
step_size = 0.5
steps = 500

# we create some training data
mean_offset = 0.75  # apply +-mean_offset to randn data to simulate two classes
m = 100000  # training data examples, even int!
m_train = m//10  # test data examples, even int!
nx = 5  # number of features, increasing -> higher F1 score
X, Y = create_data(nx, m, mean_offset)

# we init weights and bias
w = (rng.rand(nx, 1) - 1/2) * 2
b = (rng.rand(1, 1) - 1/2) * 2

# we do gradient descent
for step in range(steps):

    # forward propagation = calc actual prediction, i.e. the model output
    # using the current weights and bias:
    Z = np.dot(w.T, X) + b  # forward step 1 = inner product + bias
    A = my_sigmoid(Z)  # forward step 2 = activation function
    print('cost function on training data', cost(Y, A))

    # backward propagation, start at the output from model and subsequently
    # move back to the model input
    # vectorized
    da = -Y / A + (1-Y) / (1-A)  # step1 -> dL/da
    dz = da * A*(1-A)  # step 2 -> (dL/da) * da/dz
    dw = np.dot(X, dz.T) / m  # step 3 -> dL/dw = (dL/da * da/dz) * dz/dw
    db = np.mean(dz)  # dL/dw = dL/da * da/dz * dz/db

    # gradient descent update rule
    w = w - step_size * dw
    b = b - step_size * db

# we trained the model and hope for useful weights and bias
print('w', w, '\nb', b)

# we check model performance on unseen! test data
# therefore we create some test data with same
# PDF characteristics as training data
X, Y = create_data(nx, m_train, mean_offset)
# do model prediction == forward propagation using test data
A = my_sigmoid(np.dot(w.T, X) + b)  # Yhat
print('cost function on test data:', cost(Y, A))
print('       Y   ', Y)
# print('Yhat', A)
A[A > 0.5] = 1
A[A <= 0.5] = 0
print('binary Yhat', A)

TP = np.sum(np.logical_and(Y, A))
TN = np.sum(np.logical_and(np.logical_not(Y), np.logical_not(A)))
FN = np.sum(np.logical_xor(Y[Y == 1], A[Y == 1]))
FP = np.sum(np.logical_xor(Y[Y == 0], A[Y == 0]))
confusion_matrix = np.array([[TP, FP], [FN, TN]])
n_confusion_matrix = confusion_matrix / np.sum(np.sum(confusion_matrix))
precision = TP / (TP + FP)  # TP normalized by sum row1
recall = TP / (TP + FN)  # TP normalized by sum column 1
F1_score = 2 / (1/precision + 1/recall)  # harmonic mean

print('normalized confusion matrix in % \n[TP FP]\n[FN TN] =\n',
      n_confusion_matrix*100)
print('precision', precision)
print('recall', recall)
print('F1_score in %', F1_score*100)
