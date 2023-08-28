import tensorflow as tf
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('data.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:,11]

# define the keras model
model = Sequential()
model.add(Dense(10, input_dim=8, activation='sigmoid'))
model.add(Dense(1))

# compile the keras model
model.compile(loss='mse',
              optimizer='sgd', 
              metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('MSE: ', accuracy)


