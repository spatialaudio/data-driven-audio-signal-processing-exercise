# Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
# Institute of Communications Engineering (INT), Faculty of Computer Science
# and Electrical Engineering (IEF), University of Rostock, Germany
#
# Data Driven Audio Signal Processing - A Tutorial with Computational Examples
# Feel free to contact lecturer frank.schultz@uni-rostock.de
#
# Exercise 13: Basics of Convolutional Neural Networks


import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


print("element-wise multiplication of matrix slices")
x1 = np.random.randint(low=-10, high=10, size=(1, 3, 7, 1))
x2 = np.random.randint(low=-10, high=10, size=(1, 3, 7, 1))
y = x1 * x2  # element-wise multiplication since x1 and x2 have same shape
# we check this manually
tmp = 0
for n1 in range(3):
    for n2 in range(7):
        tmp += x1[0, n1, n2, 0] * x2[0, n1, n2, 0]
print(y.shape, np.sum(y), tmp, "\n")


print("conv2d")
x = tf.constant(
    np.random.randint(low=-10, high=10, size=(1, 28, 28, 3)), dtype=tf.int32
)
f = tf.constant(
    np.random.randint(low=-10, high=10, size=(5, 5, 3, 16)), dtype=tf.int32
)
y = tf.nn.conv2d(x, f, strides=1, padding="VALID")
print(x.shape, f.shape, y.shape)
# some hard coding stuff to get same results as TF
# we check only some nw entries here:
for nw in range(10):
    tmp = 0
    for ch_i in range(3):
        tmp += x[0, 0 + nw : 5 + nw, 0:5, ch_i] * f[:, :, ch_i, 0]
    print(tmp.shape, np.allclose(np.sum(tmp), y[0, nw, 0, 0].numpy()))
print("\n")


print("1x1 Convolution case for conv2d")
# comparably small width/height but very deep feature map
x = tf.constant(
    np.random.randint(low=-100, high=100, size=(1, 9, 9, 1024)), dtype=tf.int32
)
# 16 instances of 1x1 deep filters
f = tf.constant(
    np.random.randint(low=-100, high=100, size=(1, 1, 1024, 16)), dtype=tf.int32
)
# leads to 1x1 Convolution
y = tf.nn.conv2d(x, f, strides=1, padding="VALID")
print(x.shape, f.shape, y.shape)
# we get this manually
for nw in range(9):
    for nh in range(9):
        for nf in range(16):
            # take dot product of the two deep 'tubes'
            tmp_we = np.dot(x[0, nw, nh, :], f[0, 0, :, nf])
            tmp_tf = y[0, nw, nh, nf].numpy()
            if not np.allclose(tmp_we, tmp_tf):
                print("1D conv calc went wrong at nw, nh, nf:", nw, nh, nf)
print("\n")


print("Transpose conv2d_transpose")
# example/numbers taken from
# Deep Learning Specialization -> Convolutional Neural Networks, week 3
# https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning
# 1st dimension follows TF convention used for data sample index
# (1, 2, 2, 1) -> (data samples, height, width, channels)
x = tf.constant(np.array([[[[2], [1]], [[3], [2]]]]), dtype=tf.int32)
# (3, 3, 1, 1) -> (height, width, in channels, out channels)
f = tf.constant(
    np.array(
        [[[[1]], [[2]], [[1]]], [[[2]], [[0]], [[1]]], [[[0]], [[2]], [[1]]]]
    ),
    dtype=tf.int32,
)
# apply transposed conv
conv = tf.nn.conv2d_transpose(
    x,
    f,
    output_shape=(1, 4, 4, 1),
    strides=[1, 2, 2, 1],
    padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
)
# get it hard coded
y = np.zeros((6, 6), dtype=np.intc)
y[0:3, 0:3] += f[:, :, 0, 0] * x[0, 0, 0, 0]  # left upper 3x3
y[0:3, 2:5] += f[:, :, 0, 0] * x[0, 0, 1, 0]  # right upper 3x3
y[2:5, 0:3] += f[:, :, 0, 0] * x[0, 1, 0, 0]  # left lower 3x3
y[2:5, 2:5] += f[:, :, 0, 0] * x[0, 1, 1, 0]  # right lower 3x3
print("y shape", conv.shape)
print("TF\n", np.squeeze(conv.numpy()))
print("we\n", y[1:5, 1:5], "\n")


print("check how dimensions in a CNN/DNN evolve:")


def get_dim(n, f, p, s):
    """
    n...number of samples in considered data axis
    f...number of filter coeff according to considered data axis
    p...number of padding samples according to considered data axis
    s...number of stride samples according to considered data axis
    returns number of samples in output data according to considered data axis
    cf. Gilbert Strang, Linear Algebra and Learning from Data,
    Wellesley, 2019, p. 392
    cf. Sergios Theodoridis, Machine Learning,
    Academic Press, 2nd ed, 2020, p. 962, eq. (18.65)
    """
    return int((n + 2 * p - f) / s + 1)


# mix of hard and soft coded parameters allows us to recognize things
# conveniently getting this code worked as soft-coded version would be
# a good exercise to think about the dimensions and calculations involved
# the actual network is a toy example and probably not to be used in an
# actual application
n0 = 64  # we start with 64x64x3 image
model = keras.Sequential()
model.add(keras.Input(shape=(n0, n0, 3)))
print("Input (None,", n0, n0, ",3)")

kernel_size = 7
model.add(
    keras.layers.Conv2D(
        filters=10,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        strides=(1, 1),
    )
)
n1 = get_dim(n0, kernel_size, 3, 1)  # 3 is for padding='same'
print("Layer 1 (Conv2D): (None, ", n1, ", ", n1, ", 10), ", sep="", end="")
print("Param #:", (kernel_size**2 * 3 + 1) * 10)


model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
n2 = get_dim(n1, 2, 0, 2)
# no change of number of feature maps
print("Layer 2 (MaxPool2D): (None, ", n2, ", ", n2, ", 10), ", sep="", end="")
print("Param #: 0")  # pool op has parameter to learn


kernel_size = 5
model.add(
    keras.layers.Conv2D(
        filters=20,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        strides=(1, 1),
    )
)
n3 = get_dim(n2, kernel_size, 2, 1)  # p=2 for padding='same' if n=32, f=5, s=1
print("Layer 3 (Conv2D): (None, ", n3, ", ", n3, ", 20), ", sep="", end="")
print("Param #:", (kernel_size**2 * 10 + 1) * 20)


model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
n4 = get_dim(n3, 3, 0, 2)
# no change of number of feature maps
print("Layer 4 (MaxPool2D): (None, ", n4, ", ", n4, ", 20), ", sep="", end="")
print("Param #: 0")  # pool op has parameter to learn


kernel_size = 3
model.add(
    keras.layers.Conv2D(
        filters=40,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        strides=(1, 1),
    )
)
n5 = get_dim(n4, kernel_size, 1, 1)  # p=1 for padding='same' if n=15, f=3, s=1
print("Layer 5 (Conv2D): (None, ", n5, ", ", n5, ", 40), ", sep="", end="")
print("Param #:", (kernel_size**2 * 20 + 1) * 40)


kernel_size = 3
model.add(
    keras.layers.Conv2D(
        filters=80,
        kernel_size=kernel_size,
        activation="relu",
        padding="valid",
        strides=(2, 2),
    )
)
n6 = get_dim(n5, kernel_size, 0, 2)
print("Layer 6 (Conv2D): (None, ", n6, ", ", n6, ", 80), ", sep="", end="")
print("Param #:", (kernel_size**2 * 40 + 1) * 80)


kernel_size = 1  # 1x1 conv
model.add(
    keras.layers.Conv2D(
        filters=160,
        kernel_size=kernel_size,
        activation="relu",
        padding="valid",
        strides=(1, 1),
    )
)
n7 = get_dim(n6, kernel_size, 0, 1)
print("Layer 7 (Conv2D 1x1): (None, ", n6, ", ", n6, ", 160), ", sep="", end="")
print("Param #:", (kernel_size**2 * 80 + 1) * 160)


model.add(keras.layers.Flatten())
print("Layer 8 (Flatten): (None, ", n7 * n7 * 160, "), ", sep="", end="")
print("Param #: 0")  # no learning param


model.add(keras.layers.Dense(100, activation="relu"))
print("Layer 9 (Dense): (None, 100), ", sep="", end="")
print("Param #:", n7 * n7 * 160 * 100 + 100)


model.add(keras.layers.Dense(110, activation="relu"))
print("Layer 10 (Dense): (None, 110), ", sep="", end="")
print("Param #:", 100 * 110 + 110)


# 9 classes to predict
model.add(keras.layers.Dense(9, activation="softmax"))
print("Layer 11 (Dense): (None, 9), ", sep="", end="")
print("Param #:", 110 * 9 + 9)


optimizer = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy(
    from_logits=False, label_smoothing=0
)
metrics = [
    keras.metrics.CategoricalCrossentropy(),
    keras.metrics.CategoricalAccuracy(),
]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
print(model.summary())
