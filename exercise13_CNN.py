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
model = keras.Sequential()
model.add(keras.Input(shape=(32,32,3)))
model.add(keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
model.add(keras.layers.Conv2D(filters=16, kernel_size=5, strides=(1,1), activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(120, activation='relu'))
model.add(keras.layers.Dense(84, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
model.compile()
print(model.summary())

