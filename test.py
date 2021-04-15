from sklearn.datasets import make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
import numpy as np
# generate 2d classification dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# split into train and test
n_train = 500
trainX, testX = X[:n_train, :], X[n_train:, :]
trainy, testy = y[:n_train], y[n_train:]
# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(trainX, trainy, verbose=0)
_, test_acc = model.evaluate(testX, testy, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training

# for i in range(2):
#     samples_ix = np.where(y == i)
#     pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
# pyplot.legend()
# pyplot.show()

predY = model.predict(testX)


samples_tx = np.where(predY > 0.5)
pyplot.scatter(testX[samples_tx, 0], testX[samples_tx, 1], label=str(1))
samples_fx = np.where(predY <= 0.5)
pyplot.scatter(testX[samples_fx, 0], testX[samples_fx, 1], label=str(0))
pyplot.legend()
pyplot.show()