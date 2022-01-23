# Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
# Institute of Communications Engineering (INT), Faculty of Computer Science
# and Electrical Engineering (IEF), University of Rostock, Germany
#
# Data Driven Audio Signal Processing - A Tutorial with Computational Examples
# Feel free to contact lecturer frank.schultz@uni-rostock.de
#
# Exercise 11: Binary logistic regression with TensorFlow & Keras API
# using convenient stuff from scikit learn

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.backend as K


print(tf.__version__)

# rng = np.random.RandomState(1)  # for debug
rng = np.random.RandomState()

verbose = 1  # plot training status


# DATA
m = int(5/4*80000)  # data examples
nx = 2  # number of features

# train_size = 1/2  # 50% are used for training
train_size = 4/5  # 80% are used for training
# train_size = 95/100  # 95% are used for training

# these seeds produce 'nice' two classes each with
# two clusters for chosen m, nx and train_size
random_state_idx = 1
random_state = np.array([7, 21, 24, 25, 29, 33, 38])
X, Y = make_classification(n_samples=m,
                           n_features=nx, n_informative=nx,
                           n_redundant=0,
                           n_classes=2, n_clusters_per_class=2,
                           class_sep=1.5,
                           flip_y=1e-2,
                           random_state=random_state[random_state_idx])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    train_size=train_size,
                                                    random_state=None)
m_train = X_train.shape[0]
m_test = X_test.shape[0]
print('\nm_train', m_train)
print('X train dim', X_train.shape, 'Y train dim', Y_train.shape)
print('\nm_test', m_test)
print('X test dim', X_test.shape, 'Y test dim', Y_test.shape, '\n')


# SETUP of TensorFlow MODEL
# hyper parameters (should be learned as well)_
epochs = 5
batch_size = 32

# no_perceptron_in_hl = np.array([64, 64])  # trainable params 4417
# no_perceptron_in_hl = np.array([64, 32, 16, 8, 4, 2])  # trainable params 2985
# no_perceptron_in_hl = np.array([64, 16, 4, 16, 64])  # trainable params 2533
# no_perceptron_in_hl = np.array([64, 4, 2, 4, 64])  # trainable params 859
# no_perceptron_in_hl = np.array([32, 16, 8, 4, 2])  # trainable params 809
# no_perceptron_in_hl = np.array([16, 16, 4, 2])  # trainable params 401
# no_perceptron_in_hl = np.array([16, 8, 4, 2])  # trainable params 233
# no_perceptron_in_hl = np.array([8, 8, 4, 2])  # trainable params 145
# no_perceptron_in_hl = np.array([5, 5, 5])  # trainable params 81
# no_perceptron_in_hl = np.array([5, 3, 2])  # trainable params 44
# no_perceptron_in_hl = np.array([5, 4])  # trainable params 44
# no_perceptron_in_hl = np.array([5, 3])  # trainable params 37
# no_perceptron_in_hl = np.array([8])  # trainable params 33
no_perceptron_in_hl = np.array([5, 2])  # trainable params 30
# no_perceptron_in_hl = np.array([5])  # trainable params 21


optimizer = tf.optimizers.SGD()
loss = tf.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0)
metrics = [tf.metrics.BinaryCrossentropy(),
           tf.metrics.BinaryAccuracy(),
           tf.metrics.Precision(),
           tf.metrics.Recall()]

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(nx,)))
# hidden layers:
for n in no_perceptron_in_hl:
    model.add(tf.keras.layers.Dense(n, activation=tf.nn.relu))
# output layer with sigmoid for binary classificaton
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
tw = np.sum([K.count_params(w) for w in model.trainable_weights])
print('\ntrainable_weights', tw, '\n')

# TRAINING PHASE
if True:
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose)
print(model.summary())
# print(model.get_weights())

print('\n\nmetrics on train data:')
results = model.evaluate(X_train, Y_train,
                         batch_size=m_train,
                         verbose=verbose)
Y_pred = model.predict(X_train)
Y_pred[Y_pred < 0.5], Y_pred[Y_pred >= 0.5] = 0, 1
cost = results[0]
accuracy = results[2]
precision = results[3]
recall = results[4]
F1_score = 2 / (1/precision + 1/recall)  # harmonic mean
print('binary_crossentropy cost ', cost)
print('precision ', precision)
print('recall ', recall)
print('accuracy', accuracy)
print('F1', F1_score)
print('confusion matrix [[TP, FN], [FP, TN]] in % on train data:\n')
cm = tf.math.confusion_matrix(labels=Y_train,
                              predictions=Y_pred,
                              num_classes=2)
print(cm / m_train * 100)
print('accuracy manually:')
print((cm.numpy()[0, 0] + cm.numpy()[1, 1]) / m_train * 100,
      '% are correct predictions')


# TESTING PHASE
print('\n\nmetrics on test data:')
# we check model performance on !! unseen !! test data
results = model.evaluate(X_test, Y_test,
                         batch_size=m_test,
                         verbose=verbose)
Y_pred = model.predict(X_test)
Y_pred[Y_pred < 0.5], Y_pred[Y_pred >= 0.5] = 0, 1
cost = results[0]
accuracy = results[2]
precision = results[3]
recall = results[4]
F1_score = 2 / (1/precision + 1/recall)  # harmonic mean
print('binary_crossentropy cost ', cost)
print('precision ', precision)
print('recall ', recall)
print('accuracy', accuracy)
print('F1', F1_score)
print('confusion matrix [[TP, FN], [FP, TN]] in % on test data:\n')
cm = tf.math.confusion_matrix(labels=Y_test,
                              predictions=Y_pred,
                              num_classes=2)
print(cm / m_test * 100)
print('accuracy manually:')
print((cm.numpy()[0, 0] + cm.numpy()[1, 1]) / m_test * 100,
      '% are correct predictions')

if nx == 2:  # 2D plot of data and classification (curved) line
    f1 = np.arange(-6, 6, 0.05)
    f2 = np.arange(-6, 6, 0.05)
    xv, yv = np.meshgrid(f1, f2)
    Xgrid = np.concatenate((np.reshape(xv, (1, -1)),
                            np.reshape(yv, (1, -1))), axis=0).T
    # print('Xgrid.shape', Xgrid.shape)
    ygrid = model.predict(Xgrid)
    # print('ygrid.shape vec', ygrid.shape)
    ygrid[ygrid < 0.5], ygrid[ygrid >= 0.5] = 0, 1
    ygrid = np.reshape(ygrid, (xv.shape[0], xv.shape[1]))
    # print('ygrid.shape grid', ygrid.shape)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)  # train data
    plt.plot(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], 'C0o', ms=1)
    plt.plot(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], 'C1o', ms=1)
    plt.contourf(f1, f2, ygrid, cmap='RdBu_r')
    plt.colorbar()
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title(X_train.shape)
    plt.xlabel('feature 1 for train data')
    plt.ylabel('feature 2 for train data')
    plt.subplot(1, 2, 2)  # test data
    plt.plot(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1], 'C0o', ms=1)
    plt.plot(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], 'C1o', ms=1)
    plt.contourf(f1, f2, ygrid, cmap='RdBu_r')
    plt.colorbar()
    plt.axis('equal')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.title(X_test.shape)
    plt.xlabel('feature 1 for test data')
    plt.ylabel('feature 2 for test data')
    tmp = str(no_perceptron_in_hl)
    tmp = tmp.replace('[', '')
    tmp = tmp.replace(']', '')
    tmp = tmp.replace(' ', '_')
    fstr = ('exercise11_binary_logistic_regression_tf_' +
            'seed_' + str(random_state[random_state_idx]) +
            '_epochs_' + str(epochs) +
            '_batchsize_' + str(batch_size) +
            '_hiddenlayer_' + tmp +
            '.png')
    print(fstr)
    plt.savefig(fstr)
