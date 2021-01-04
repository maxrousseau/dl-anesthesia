#!/bin/env/python3
# a basic recurrent neural network

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle

from tensorflow.keras import models
from tensorflow.keras import layers

X = np.load("x3.npy") # (3740, 30)
Y = np.load("y3.npy") # (3740,)

transformer = Normalizer().fit(X)
X = transformer.transform(X) # normalizes data according to columns

X, Y = shuffle(X, Y, random_state=0) # shuffle the samples

X = np.reshape(X, (3559, 6, 5), order='F')
# np.save("x_time.npy", x_time)

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

# import IPython ; IPython.embed() ; exit(1)

rnn = models.Sequential()
# rnn.add(layers.SimpleRNN(51, return_sequences=True))
#rnn.add(layers.SimpleRNN(51, return_sequences=True))
#rnn.add(layers.SimpleRNN(51, return_sequences=True))
rnn.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
rnn.add(layers.Bidirectional(layers.LSTM(64)))
rnn.add(layers.Dense(1, activation='sigmoid'))

rnn.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

history = rnn.fit(
                X_train,
                Y_train,
                epochs=200,
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
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

values = history_dict['accuracy']
val_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, values, 'bo', label='training accuracy')
plt.plot(epochs, val_values, 'b', label='validation accuracy')
plt.title('losses')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
