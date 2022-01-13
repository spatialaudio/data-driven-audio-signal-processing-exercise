# Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
# Institute of Communications Engineering (INT), Faculty of Computer Science
# and Electrical Engineering (IEF), University of Rostock, Germany
#
# Data Driven Audio Signal Processing - A Tutorial with Computational Examples
# Feel free to contact lecturer frank.schultz@uni-rostock.de
#
# Exercise 11: Binary logistic regression model with just one layer
# Training using gradient descent and forward/backward propagation
# following the derivations and coding conventions from the brilliant
# course https://www.coursera.org/learn/neural-networks-deep-learning
# cf. especially week 2
# compare results against a TensorFlow model

import numpy as np
import tensorflow as tf


print(tf.__version__)

# rng = np.random.RandomState(1)  # for debug
rng = np.random.RandomState()


def my_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(y_true, y_pred):
    # vectorized loss
    L = - (y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    # cost function as average of all entries in L
    J = np.mean(L)
    return(J)


def evaluate(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1  # actually not nice,
    y_pred[y_pred <= 0.5] = 0  # since we generally overwrite y_pred
    # which might get dangerous if we need original data outside the function
    # therefore we call it evaluate(np.copy(), np.copy())
    TP = np.sum(np.logical_and(y_true, y_pred))
    TN = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
    FN = np.sum(np.logical_xor(y_true[y_true == 1], y_pred[y_true == 1]))
    FP = np.sum(np.logical_xor(y_true[y_true == 0], y_pred[y_true == 0]))
    cm = np.array([[TP, FP], [FN, TN]])
    n_cm = cm / np.sum(np.sum(cm))
    precision = TP / (TP + FP)  # TP normalized by sum row1
    recall = TP / (TP + FN)  # TP normalized by sum column 1
    F1_score = 2 / (1/precision + 1/recall)  # harmonic mean
    return cm, n_cm, precision, recall, F1_score


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
step_size = 0.3
steps = 5000

# we create some training data
mean_offset = 0.75  # apply +-mean_offset to randn data to simulate two classes
m = 100000  # training data examples, even int!
m_train = m//10  # test data examples, even int!
nx = 5  # number of features, increasing -> higher F1 score
X, Y = create_data(nx, m, mean_offset)

# TRAIN
#     OUR MODEL
# we init weights and bias
w = (rng.rand(nx, 1) - 1/2) * 2
b = (rng.rand(1, 1) - 1/2) * 2

# we do gradient descent
for step in range(steps):

    # forward propagation = calc actual prediction, i.e. the model output
    # using the current weights and bias:
    Z = np.dot(w.T, X) + b  # forward step 1 = inner product + bias
    A = my_sigmoid(Z)  # forward step 2 = activation function
    print('epoch', step, '/', steps, ', our cost on training data', cost(Y, A))

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

J_train = cost(Y, A)
_, n_cm_train, precision_train, recall_train, F1_score_train =\
    evaluate(np.copy(Y), np.copy(A))

#     TensorFlow MODEL
initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
optimizer = tf.optimizers.SGD(learning_rate=step_size, momentum=0)
# use some other gradient descent methods:
# optimizer = tf.optimizers.Adam()
# optimizer = tf.optimizers.SGD()
loss = tf.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
metrics = [tf.metrics.BinaryCrossentropy(),
           tf.metrics.BinaryAccuracy(),
           tf.metrics.Precision(),
           tf.metrics.Recall()]
input = tf.keras.Input(shape=(nx,))
output = tf.keras.layers.Dense(1, kernel_initializer=initializer,
                               activation=tf.math.sigmoid)(input)
# use default kernel_initializer:
# output = tf.keras.layers.Dense(1, activation=tf.math.sigmoid)(input)
model = tf.keras.Model(inputs=input, outputs=output)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(X.T, Y.T, batch_size=m, epochs=steps, verbose=1)
# explicit usage of epochs, batch_size (here hard coded):
# model.fit(X.T, Y.T, epochs=20, batch_size=100, verbose=1)
results_train_tf = model.evaluate(X.T, Y.T, batch_size=m, verbose=0)

# print results of our model / TF model
print(model.summary())
print('\n\nmetrics on training data:')
print('our cost', J_train)
print('TF cost ', results_train_tf[0])
print('our precision', precision_train)
print('TF precision ', results_train_tf[3])
print('our recall', recall_train)
print('TF recall ', results_train_tf[4])
print('our confusion matrix \n[TP FP]\n[FN TN] =\n',
      n_cm_train*100, '%')
print('our F1_score', F1_score_train*100, '%')
print('\nmodel weights:')
print('ours\nw', w.T,
      '\nb', b)
print('TF\nw', model.get_weights()[0].T,
      '\nb', model.get_weights()[1])

# TEST
# we check model performance on !! unseen !! test data
# therefore we create some test data with same
# PDF characteristics as training data
X, Y = create_data(nx, m_train, mean_offset)

#     OUR MODEL
# do model prediction == forward propagation using test data
A = my_sigmoid(np.dot(w.T, X) + b)  # Yhat
J_test = cost(Y, A)
_, n_cm_test, precision_test, recall_test, F1_score_test =\
    evaluate(np.copy(Y), np.copy(A))

#     TensorFlow MODEL
results_test_tf = model.evaluate(X.T, Y.T, batch_size=m_train, verbose=0)

# print results of our model / TF model
print('\n\nmetrics on test data:')
print('our cost', J_test)
print('TF cost ', results_test_tf[0])
print('our precision', precision_test)
print('TF precision ', results_test_tf[3])
print('our recall', recall_test)
print('TF recall ', results_test_tf[4])
print('our confusion matrix\n[TP FP]\n[FN TN] =\n',
      n_cm_test*100, '%')
print('our F1_score', F1_score_test*100, '%')
