#!/bin/env/python3
# a simple feed-forward neural network

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle

from tensorflow.keras import models
from tensorflow.keras import layers

#X = np.load("x.npy") # (3740, 306)
#Y = np.load("y.npy") # (3740,)

X = np.load("x3.npy") # (3559, 24)
Y = np.load("y3.npy") # (3559,)


transformer = Normalizer().fit(X)
X = transformer.transform(X) # normalizes data according to columns

X, Y = shuffle(X, Y, random_state=0) # shuffle the samples

#import IPython ; IPython.embed() ; exit(1)

X_train = X[:3000]
Y_train = Y[:3000]


X_test = X[3000:]
Y_test = Y[3000:]

# 3000 training samples
print(X_train.shape)
print(Y_train.shape)

# 740 test samples
print(X_test.shape)
print(Y_test.shape)

nn = models.Sequential()
nn.add(layers.Dense(400, activation='relu', input_shape=(30,)))
nn.add(layers.Dense(200, activation='relu'))
nn.add(layers.Dense(100, activation='relu'))
nn.add(layers.Dropout(0.5))
nn.add(layers.Dense(1, activation='sigmoid'))

nn.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

history = nn.fit(
                X_train,
                Y_train,
                epochs=20,
                batch_size=30,
                validation_data=(X_test,Y_test)
                )

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='training loss')
plt.plot(epochs, val_loss_values, 'b', label='validation loss')
plt.title('losses')
plt.xlabel('epocs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.clf()
