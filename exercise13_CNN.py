# Sascha Spors, Professorship Signal Theory and Digital Signal Processing,
# Institute of Communications Engineering (INT), Faculty of Computer Science
# and Electrical Engineering (IEF), University of Rostock, Germany
#
# Data Driven Audio Signal Processing - A Tutorial with Computational Examples
# Feel free to contact lecturer frank.schultz@uni-rostock.de
#
# Exercise 13:

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


print('Convolution')
x = tf.constant(np.random.randn(1, 28, 28, 3), dtype=tf.float32)
f = tf.constant(np.random.randn(5, 5, 3, 16), dtype=tf.float32)
y = tf.nn.conv2d(x, f, strides=1, padding='VALID')
print(x.shape, f.shape, y.shape)
# some hard coding stuff is going on to get same result as TF:
for nw in range(10):
    tmp = 0
    for ch_i in range(3):
        tmp+=x[0,0+nw:5+nw,0:5,ch_i] * f[:,:,ch_i,0]
    print(np.allclose(np.sum(np.sum(tmp)), y[0,nw,0,0].numpy()))

print('Transpose Convolution')
# numbers from Andrew Ng's example in Ng_Coursera_For_DeepLearningAI, Course4 CNN, week 3

# (1, 2, 2, 1) -> (data samples, height, width, channels)
x = tf.constant(np.array([[
    [[2], [1]], 
    [[3], [2]]
]]), dtype=tf.float32)

# (3, 3, 1, 1) -> (height, width, in channels, out channels)
f = tf.constant(np.array([
    [[[1]], [[2]], [[1]]], 
    [[[2]], [[0]], [[1]]], 
    [[[0]], [[2]], [[1]]]
]), dtype=tf.float32)

conv = tf.nn.conv2d_transpose(x, f,
                              output_shape=(1, 4, 4, 1),
                              strides=[1, 2, 2, 1],
                              padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
print('y shape', conv.shape)
print('TF', np.squeeze(conv.numpy()))

# hard coded!!!
y = np.zeros((6, 6))
y[0:3, 0:3] += f[:, :, 0, 0] * x[0, 0, 0, 0]  # left upper 3x3
y[0:3, 2:5] += f[:, :, 0, 0] * x[0, 0, 1, 0]  # right upper 3x3
y[2:5, 0:3] += f[:, :, 0, 0] * x[0, 1, 0, 0]  # left lower 3x3
y[2:5, 2:5] += f[:, :, 0, 0] * x[0, 1, 1, 0]  # right lower 3x3
print('we', y[1:5, 1:5])


print('check how dimensions in a CNN/DNN evolve:')


def get_dim(n, f, p, s):
    return int((n+2*p-f)/s+1)


# mix of hard and soft coded parameters forces us to think about but also to recognize things conveniently:
n0 = 64
model = keras.Sequential()
model.add(keras.Input(shape=(n0, n0, 3)))
print('Input (None,', n0, n0, ',3)')

kernel_size = 7
model.add(keras.layers.Conv2D(filters=10, kernel_size=kernel_size, activation='relu', padding='same', strides=(1,1)))
n1 = get_dim(n0, kernel_size, 3, 1)  # 3 is for padding='same'
print('L1 (conv2D): (None,', n1, ',', n1, ',  10)')
print('Param #:', (kernel_size**2 * 3 + 1) * 10)

kernel_size = 5
model.add(keras.layers.Conv2D(filters=20, kernel_size=kernel_size, activation='relu', padding='valid', strides=(2,2)))
n2 = get_dim(n0, kernel_size, 0, 2)  # 0 is for padding='valid'
print('L2 (conv2D): (None,', n2, ',', n2, ',  20)')
print('Param #:', (kernel_size**2 * 10 + 1) * 20)

model.add(keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1)))
n3 = get_dim(n2, 3, 0, 1)
print('L3 (maxpool): (None,', n3, ',', n3, ',  20)')  # no change of number of feature maps, since pool acts on all these equally
print('Param #: 0')  # pool has nothing to learn

model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
n4 = get_dim(n3, 2, 0, 2)
print('L4 (maxpool): (None,', n4, ',', n4, ',  20)')  # no change of number of feature maps, since pool acts on all these equally
print('Param #: 0')  # pool has nothing to learn

kernel_size = 3
model.add(keras.layers.Conv2D(filters=30, kernel_size=kernel_size, activation='relu', padding='valid', strides=(1,1)))
n5 = get_dim(n4, kernel_size, 0, 1)
print('L5 (conv2D): (None,', n5, ',', n5, ',  30)')
print('Param #:', (kernel_size**2 * 20 + 1) * 30)

kernel_size = 3
model.add(keras.layers.Conv2D(filters=40, kernel_size=kernel_size, activation='relu', padding='same', strides=(2,2)))
n6 = get_dim(n5, kernel_size, 1, 2)
print('L6 (conv2D): (None,', n6, ',', n6, ',  40)')
print('Param #:', (kernel_size**2 * 30 + 1) * 40)

model.add(keras.layers.Flatten())
print('L7 (Flatten): (None,', n6*n6*40, ')')
print('Param #: 0')  # no learning param

model.add(keras.layers.Dense(140, activation='relu'))
print('L8 (Dense): (None, 140)')
print('Param #:', n6*n6*40*140 + 140)

model.add(keras.layers.Dense(70, activation='relu'))
print('L9 (Dense): (None, 70)')
print('Param #:', 140*70 + 70)

model.add(keras.layers.Dense(10, activation='softmax'))
print('L10 (Dense): (None, 10)')
print('Param #:', 70*10 + 10)

optimizer = keras.optimizers.Adam()
loss = keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0)
metrics = [keras.metrics.CategoricalCrossentropy(),
           keras.metrics.CategoricalAccuracy()]
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
print(model.summary())

