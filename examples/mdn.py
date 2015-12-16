#!/usr/bin/env python

from __future__ import print_function

"""This examples demonstrates the use of a Mixture Density output layer"""

import climate
import logging
import matplotlib.pylab as pl
import numpy as np
import theanets
import sys

climate.enable_default_logging()

# simple test
N = 1000
numhidden = 30
models = []
Xs = []
Ys = []
tXs = []
tYs = []
sigs = np.arange(1.0, 0.6, -0.1)

# for sig in sigs:
#     print("sig = %f\n" % sig)
#     models.append(theanets.Regressor([1, (numhidden, "tanh"), 1], loss="mae"))
#     # Xs.append(np.random.randn(1000, 1).astype('f') * sig)
#     Xs.append(np.random.uniform(-1., 1., (1000, 1)).astype('f') * sig)
#     Ys.append(np.cos(Xs[-1]) * 1.) # np.random.randn(1000, 1).astype('f')
#     tXs.append(np.random.randn(100, 1).astype('f'))

# models.append(theanets.Regressor([1, (30, "tanh"), 1]))
# Xs.append(np.random.normal(0, 0.5, (N, 1)).astype('f'))
# Ys.append(np.cos(Xs[-1]).astype('f'))
# tXs.append(np.random.randn(100, 1).astype('f'))

# models.append(theanets.Regressor([1, (30, "tanh"), 1]))
# Xs.append(np.random.normal(0, 1, (N, 1)).astype('f'))
# Ys.append((np.cos(Xs[-1]) + np.random.uniform(-0.1, 0.1, (N, 1))).astype('f'))
# tXs.append(np.random.randn(100, 1).astype('f'))

models.append(theanets.Regressor([1, (numhidden, "tanh"), 1]))
# Xs.append(np.random.normal(0, 1, (N, 1)).astype('f'))
Xs.append(np.random.uniform(-1., 1., (1000, 1)).astype('f'))
Ys.append((Xs[-1] + 0.3 * np.sin(2*3.1415926*Xs[-1]) + np.random.uniform(-0.1, 0.1, (N, 1))).astype('f'))
# Ys.append((np.cos(Xs[-1]) + np.random.uniform(-0.1, 0.1, (N, 1))).astype('f'))
tXs.append(np.random.randn(100, 1).astype('f'))

# model = theanets.Regressor([1, dict(size=30, activation="tanh"), 1])
# model = theanets.Regressor([1, 20, 1])

# pl.subplot(211)
# pl.plot(inputs)
# pl.plot(outputs)
# pl.subplot(212)
# pl.plot(inputs1)
# pl.plot(outputs1)
# pl.show()

# pl.plot(inputs, outputs, "bo")
# pl.show()

for i, _ in enumerate(models):
    models[i].train([Xs[i], Ys[i]], algo="rmsprop", regularizers=()) # min_improvement=0.1) # algo="rmsprop", learning_rate=1e-3, max_updates=5000, , min_improvement=0.1
    tYs.append(models[i].predict(tXs[i]))

# print(testprd)

for i, _ in enumerate(models):
    pl.subplot(len(models), 1, i+1)
    try:
        pl.title("sig(X) = %f" % sigs[i])
    except Exception:
        print("fail")
    pl.plot(Xs[i],  Ys[i], "bo")
    pl.plot(tXs[i], tYs[i], "ko")
    pl.gca().set_xlim((-4, 4))
pl.show()

from keras.models import Sequential
from keras.layers.core import Dense, Activation

model = Sequential()
model.add(Dense(input_dim = 1, output_dim = numhidden))
model.add(Activation("tanh"))
model.add(Dense(output_dim = 1))
model.add(Activation("linear"))

model.compile(loss='mse', optimizer='rmsprop')

model.fit(Xs[0], Ys[0], nb_epoch=100, batch_size=32)

prd = model.predict(tXs[0])

pl.plot(tXs[0], prd, "ko")
pl.show()

sys.exit()

# generate some 1D regression data (reproducing Bishop book data, page 273). 
# Note that the P(y|x) is not a nice distribution. E.g. it has three modes for x ~= 0.5
N = 1000
# X = np.linspace(0,1,N).astype('f')
X = np.random.uniform(-1, 1, (N, 1)).astype('f')
# Y = np.cos(X)
Y = (X + 0.3 * np.sin(2*3.1415926*X) + np.random.uniform(-0.1, 0.1, (N, 1))).astype('f')
# X,Y = Y,X

# print(X.shape, Y.shape)
X = X.reshape((1, N, 1))
Y = Y.reshape((1, N, 1))
# X = X.reshape((N, 1))
# Y = Y.reshape((N, 1))



pl.subplot(111)
pl.scatter(X,Y,color='g')
pl.show()

model = theanets.recurrent.Regressor([1, (30, "tanh"), 1])
model.train([X, Y], min_improvement=0.01)
test = np.random.randn(1, 100, 1).astype('f')
testprd = model.predict(test)
print(testprd)

pl.plot(X[0], Y[0], "bo")
pl.plot(test[0], testprd[0], "ko")
pl.show()



BATCH_SIZE = 1
numix = 3
numhidden = 100

# standard fit
# net = theanets.Regressor([1, dict(size=numhidden, activation="tanh"), 1])
net = theanets.recurrent.Regressor([1, dict(size=numhidden, activation="tanh", form="rnn"), 1])
# net.set_loss("mse")

net.train([X, Y], min_improvement=0.01)

# model initialization
# networks = [
#     dict()
#     ]

# kw = dict(inputs={"hid1:out": numhidden}, size=numix)
# # net = theanets.recurrent.MixtureDensity([1, ("linear", numhidden),
# net = theanets.feedforward.MixtureDensity([1, dict(size=numhidden, activation="tanh"),
#                                         dict(name="mu", activation="linear", **kw),
#                                         dict(name="sig", activation="exp", **kw),
#                                         dict(name="pi", activation="softmax", **kw),
#                                          ])


# net.set_loss("nll", mu_name="mu", sig_name="sig", pi_name="pi", numcomp=numix)
    
losses = []

# for tm, _ in net.itertrain([X, Y],
#     monitor_gradients=True,
#     batch_size=BATCH_SIZE,
#     algo="rmsprop",
#     # algo = "adagrad",
#     # algo = "adadelta",
#     # algo = "adam",
#     # algo = "rprop",
#     # algo = "esgd",
#     # algo = "sgd",
#     # algo = "nag", # nope
#     learning_rate=0.0001,
#     momentum=0.1,
#     # nesterov=True,
#     min_improvement=0.01): #, save_progress="recurrent_waves_{}", save_every=100):
#     losses.append(tm['loss'])

# X_ = np.random.uniform(-0.2, 1.2, (1, 1000, 1)).astype(np.float32)
X_ = np.random.uniform(-0.2, 1.2, (100, 1)).astype(np.float32)
prd = net.predict(X)
print("prd.shape", prd.shape)

# pl.scatter(X[0],Y[0],color='g')
pl.subplot(211)
pl.scatter(X, Y, color='g')
pl.scatter(X, prd, color="k")

pl.subplot(212)
# pl.plot(net.find("hid1", "w"))
pl.plot(losses)
pl.show()
