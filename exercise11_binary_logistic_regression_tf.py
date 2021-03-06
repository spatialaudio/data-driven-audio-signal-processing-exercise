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
# compare against model that is trained by TensorFlow

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras


print('TF version', tf.__version__,  # we used 2.4.3
      '\nKeras version', keras.__version__)  # we used 2.4.0

# rng = np.random.RandomState(1)  # for debug
rng = np.random.RandomState()

verbose = 1  # plot training status


def my_sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(y_true, y_pred):
    # vectorized loss
    L = - (y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    # cost function as average of all entries in L
    J = np.mean(L)
    return(J)


def evaluate(y_true, y_pred):
    # actually not nice, since we generally overwrite y_pred
    y_pred[y_pred < 0.5], y_pred[y_pred >= 0.5] = 0, 1
    # which might get dangerous if we need original data outside the function
    # therefore we should call evaluate(np.copy(), np.copy())

    # inverted logic to be consistent with the TF confusion matrix
    # of labels starting with 0:
    # label positive == 0
    # label negative == 1
    # confusion matrix (row = actual label, column = predicted label):
    # [TP    FN] = [0,0    0,1<-this 1 is negative label and hene false]
    # [FP    TN] = [1,0    1,1]
    TP = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred)))
    TN = np.sum(np.logical_and(y_true, y_pred))
    FN = np.sum(np.logical_xor(y_true[y_true == 0], y_pred[y_true == 0]))
    FP = np.sum(np.logical_xor(y_true[y_true == 1], y_pred[y_true == 1]))
    cm = np.array([[TP, FN], [FP, TN]])
    n_cm = cm / np.sum(np.sum(cm))

    # sensitivity, recall, hit rate, or true positive rate (TPR)
    recall = TP / (TP + FN)
    # specificity, selectivity or true negative rate (TNR)
    TN / (TN + FP)
    # precision or positive predictive value (PPV)
    precision = TP / (TP + FP)
    # negative predictive value (NPV)
    TN / (TN + FN)
    # accuracy (ACC)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # balanced F-score, F1 score
    F1_score = 2 / (1/precision + 1/recall)  # harmonic mean
    return cm, n_cm, precision, recall, F1_score, accuracy


# gradient descent hyper parameters
# set up to see how things evolve
# in practice do hyper parameter tuning, see exercise 12
step_size = 0.25
steps = 500

# we create some data
m = 100000  # data examples
nx = 2  # number of features
train_size = 0.8  # 80% are used for training

X, Y = make_classification(n_samples=m,
                           n_features=nx, n_informative=nx,
                           n_redundant=0,
                           n_classes=2, n_clusters_per_class=1,
                           class_sep=1,
                           flip_y=1e-2,
                           random_state=8)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=train_size,
                                                    random_state=None)
m_train = X_train.shape[0]
print('\nm_train', m_train)

X = X_train.T  # our implementation needs transposed data
Y = np.expand_dims(Y_train, axis=0)
# X.shape = (nx, m_train)
# Y.shape = (1, m_train)
print('X train dim', X.shape, 'Y train dim', Y.shape)


# TRAINING PHASE
#     OUR MODEL
# we init weights and bias with uniform PDF noise
w = (rng.rand(nx, 1) - 1/2) * 2
b = (rng.rand(1, 1) - 1/2) * 2

# we do the batch gradient descent, i.e. take all data at once per epoch
# to calc new weights
for step in range(steps):

    # forward propagation = calc actual prediction, i.e. the model output
    # using the current weights and bias:
    Z = np.dot(w.T, X) + b  # forward step 1 = inner product + bias
    A = my_sigmoid(Z)  # forward step 2 = activation function
    if verbose:
        print('epoch', step, '/', steps, ', our cost on training data',
              cost(Y, A))

    # backward propagation, start at the output from model and subsequently
    # move back to the model input
    # vectorized
    da = -Y / A + (1-Y) / (1-A)  # step1 -> dL/da
    dz = da * A*(1-A)  # step 2 -> (dL/da) * da/dz
    dw = np.dot(X, dz.T) / m_train  # step 3 -> dL/dw = (dL/da * da/dz) * dz/dw
    db = np.mean(dz)  # dL/dw = dL/da * da/dz * dz/db

    # gradient descent update rule
    w = w - step_size * dw
    b = b - step_size * db

J_train = cost(Y, A)
cm_train, n_cm_train, precision_train, recall_train,\
    F1_score_train, accuracy_train = evaluate(np.copy(Y), np.copy(A))

#     TensorFlow MODEL
initializer = keras.initializers.RandomUniform(minval=0., maxval=1.)
optimizer = keras.optimizers.SGD(learning_rate=step_size, momentum=0)
# we can also use some other gradient descent methods:
# optimizer = keras.optimizers.Adam()
# optimizer = keras.optimizers.SGD()
loss = keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
metrics = [keras.metrics.BinaryCrossentropy(),
           keras.metrics.BinaryAccuracy(),
           keras.metrics.Precision(),
           keras.metrics.Recall()]
input = keras.Input(shape=(nx,))
output = keras.layers.Dense(1, kernel_initializer=initializer,
                               activation='sigmoid')(input)
# we can also use default kernel_initializer:
# output = keras.layers.Dense(1, activation='sigmoid')(input)
model = keras.Model(inputs=input, outputs=output)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(X.T, Y.T, batch_size=m_train, epochs=steps, verbose=verbose)
# explicit usage of epochs, batch_size hard coded:
# model.fit(X.T, Y.T, epochs=20, batch_size=100, verbose=verbose)
results_train_tf = model.evaluate(X.T, Y.T, batch_size=m_train,
                                  verbose=verbose)

# print results of our model vs. TF model
print(model.summary())
print('\n\nmetrics on training data:')
print('our cost', J_train)
print('TF cost ', results_train_tf[0])
print('our accuray', accuracy_train)
print('TF accuracy', results_train_tf[2])
print('our precision', precision_train)
print('TF precision ', results_train_tf[3])
print('our recall', recall_train)
print('TF recall ', results_train_tf[4])

print('our F1_score', F1_score_train*100, '%')
print('our confusion matrix\n[TP FN]\n[FP TN] =\n',
      n_cm_train*100, '%')
Y_pred = model.predict(X_train)
Y_pred[Y_pred < 0.5], Y_pred[Y_pred >= 0.5] = 0, 1
cm = tf.math.confusion_matrix(labels=Y_train,
                              predictions=Y_pred,
                              num_classes=2)
print('TF confusion matrix')
print(cm / m_train * 100)

print('\nmodel weights:')
print('ours\nw', w.T,
      '\nb', b)
print('TF\nw', model.get_weights()[0].T,
      '\nb', model.get_weights()[1])

# TESTING PHASE
# we check model performance on !! unseen !! test data
m_test = X_test.shape[0]
print('\nm_test', m_test)
X = X_test.T  # our implementation needs transposed data
Y = np.expand_dims(Y_test, axis=0)
# X.shape = (nx, m_test)
# Y.shape = (1, m_test)
print('X test dim', X.shape, 'Y test dim', Y.shape)

#     OUR MODEL
# do model prediction == forward propagation using test data
A = my_sigmoid(np.dot(w.T, X) + b)  # Yhat
J_test = cost(Y, A)
cm_test, n_cm_test, precision_test, recall_test,\
    F1_score_test, accuracy_test = evaluate(np.copy(Y), np.copy(A))

#     TensorFlow MODEL
results_test_tf = model.evaluate(X.T, Y.T, batch_size=m_test,
                                 verbose=verbose)

# print results of our model vs. TF model
print('\n\nmetrics on test data:')
print('our cost', J_test)
print('TF cost ', results_test_tf[0])
print('our accuray', accuracy_test)
print('TF accuracy', results_test_tf[2])
print('our precision', precision_test)
print('TF precision ', results_test_tf[3])
print('our recall', recall_test)
print('TF recall ', results_test_tf[4])
print('our F1_score', F1_score_test*100, '%')
print('our confusion matrix\n[TP FN]\n[FP TN] =\n',
      n_cm_test*100, '%')
Y_pred = model.predict(X_test)
Y_pred[Y_pred < 0.5], Y_pred[Y_pred >= 0.5] = 0, 1
cm = tf.math.confusion_matrix(labels=Y_test,
                              predictions=Y_pred,
                              num_classes=2)
print('TF confusion matrix')
print(cm / m_test * 100)

# plot
if nx == 2:  # 2D plot of data and classification line
    f1 = np.arange(-6, 6, 0.1)
    f2 = np.arange(-6, 6, 0.1)
    xv, yv = np.meshgrid(f1, f2)
    tmp = my_sigmoid(w[0]*xv + w[1]*yv + b)
    tmp[tmp < 0.5], tmp[tmp >= 0.5] = 0, 1
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 'C0o', ms=1)
    plt.plot(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 'C1o', ms=1)
    plt.contourf(f1, f2, tmp, cmap='RdBu_r')
    plt.axis('equal')
    plt.colorbar()
    plt.title(X_train.shape)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.subplot(2, 1, 2)
    plt.plot(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1], 'C0o', ms=1)
    plt.plot(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], 'C1o', ms=1)
    plt.contourf(f1, f2, tmp, cmap='RdBu_r')
    plt.axis('equal')
    plt.colorbar()
    plt.title(X_test.shape)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.savefig('exercise11_binary_logistic_regression_tf.png')


if False:  # check our confusion matrix handling
    # inverted logic!
    # label positive == 0
    # label negative == 1
    # TP
    y_true = np.array([0])
    y_pred = np.array([0])
    cm, n_cm, precision, recall, F1_score, accuracy = evaluate(y_true, y_pred)
    print('TP', cm)

    # FN
    y_true = np.array([0])
    y_pred = np.array([1])  # <- 1...neg label, which is false against y_true=0
    cm, n_cm, precision, recall, F1_score, accuracy = evaluate(y_true, y_pred)
    print('FN', cm)

    # FP
    y_true = np.array([1])
    y_pred = np.array([0])  # <- 0...pos label which is false against y_true=1
    cm, n_cm, precision, recall, F1_score, accuracy = evaluate(y_true, y_pred)
    print('FP', cm)

    # TN
    y_true = np.array([1])
    y_pred = np.array([1])
    cm, n_cm, precision, recall, F1_score, accuracy = evaluate(y_true, y_pred)
    print('TN', cm)
