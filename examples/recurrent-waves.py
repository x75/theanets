#!/usr/bin/env ipython

from __future__ import print_function

'''This example compares recurrent layer performance on a sine-generation task.

The task is to generate a complex sine wave that is constructed as a
superposition of a small set of pure frequencies. All networks are constructed
with one input (which receives all zero values), one recurrent hidden layer, and
one output (which is tasked with matching the target sine wave). Each model is
trained and then its predicted output is plotted for easy visual comparison of
the behavior of the different layer models.

For this task, the clockwork RNN layer tends to perform the best of the layer
models, even though the clockwork layer uses the simplest activation (linear)
and has the fewest parameters (~2000 for a 64-node hidden layer, versus ~4000
for a vanilla RNN and ~17000 for an LSTM). The vanilla RNN layer tends to do the
worst, or at the least is the most sensitive to the initialization of the
parameters. The other layer models fall somewhere in the middle but tend only to
match the dominant frequency in the target wave.
'''

import climate
import logging
import matplotlib.pyplot as pl
import numpy as np
import theanets
import sys
# from 
from smp.datasets import wavdataset

climate.enable_default_logging()

COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
          '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#17becf']

BATCH_SIZE = 2

# NUMDATAPTS = 8192
# NUMDATAPTS = 1000
NUMDATAPTS = 256

# Construct a complex sine wave as a sum of pure-frequency waves.
TAU = 2 * np.pi
T = np.linspace(0, TAU, NUMDATAPTS)
SIN = sum(c * np.sin(TAU * f * T) for c, f in ((2, 1.5), (3, 1.8), (4, 1.1)))


# Create an input dataset consisting of all zeros, and an output dataset
# containing the target sine wave. We have to stack the target sine wave here
# because recurrent models expect a tensor with three dimensions, and the batch
# size for recurrent networks must be greater than 1.
ZERO = np.zeros((BATCH_SIZE, len(T), 1), 'f')
WAVES_SIN = np.concatenate([SIN[None, :, None]] * BATCH_SIZE, axis=0).astype('f')

print("T.shape, SIN.shape, ZERO.shape, WAVES_SIN.shape", T.shape, SIN.shape, ZERO.shape, WAVES_SIN.shape)

################################################################################
# # load wave data #1
# ds = wavdataset(
#     sample_len = len(T), # 7*441,
#     n_samples = 1,
#     filename = "../../smp/playground/sequence/drinksonus44.wav")
# print("len(ds)", len(ds))
# print("len(ds[0][0])", len(ds[0][0]))

# print("ds[0][0].shape, SIN.shape", ds[0][0].shape, SIN.shape)

# WAVES_WAV = np.concatenate([ds[0][0][None, :]] * BATCH_SIZE, axis=0).astype('f')
# print("WAVES_WAV.shape", WAVES_WAV.shape)

################################################################################
# load wave data #2
ds = wavdataset(
    sample_len = len(T), # 7*441,
    n_samples = 10,
    filename = "../../smp/playground/sequence/drinksonus44.wav")
print("len(ds)", len(ds))
print("len(ds[0][0])", len(ds[0][0]))

print("ds[0][0].shape, SIN.shape", ds[1][0].shape, SIN.shape)

# WAVES_WAV = np.concatenate([ds[0][0][None, :]] * BATCH_SIZE, axis=0).astype('f')
WAVES_WAV = np.array(ds[:])[:,0,:,:].astype(np.float32)
print("WAVES_WAV.shape", WAVES_WAV.shape)

# sys.exit()
# massage wave data

# WAVES = WAVES_SIN.copy()
WAVES = WAVES_WAV.copy()

INPUT = np.roll(WAVES, 1, axis=1)# * 0.1

print("INPUT.shape", INPUT.shape)

pl.subplot(311)
pl.title("SIN")
pl.plot(SIN)
pl.subplot(312)
pl.title("WAVES / INPUT")
pl.plot(WAVES[:,:,0].T)# + np.array((0.1, 0.2)))
pl.plot(INPUT[:,:,0].T)# + np.array((0.1, 0.2)))
pl.subplot(313)
pl.title("INPUT")
pl.plot(INPUT[:,:,0].T) # + np.array((0.1, 0.2)))
pl.show()

# sys.exit()



# Set up plotting axes to show the output result and learning curves.
_, (wave_ax, learn_ax, freerun_ax) = pl.subplots(3, 1)

print("T.shape, WAVES.shape", T.shape, WAVES.shape)

# Plot the target wave.
# wave_ax.plot(T, SIN, ':', label='Target', alpha=0.7, color='#111111')
wave_ax.plot(T, ds[0][0], ':', label='Target', alpha=0.7, color='#111111')
# wave_ax.plot(T, WAVES, ':', label='Target', alpha=0.7, color='#111111')


# For each layer type, train a model containing that layer, and plot its
# predicted output.

networks = [
    dict(form='rnn', activation='relu', diagonal=0.5),
        dict(form='rrnn', activation='relu', rate='vector', diagonal=0.5),
        dict(form='gru', activation='relu'),
        dict(form='lstm', activation='tanh'),
    dict(form='clockwork', activation='linear', periods=(1, 2, 4, 8, 16, 64))
]

# networks = [
#         dict(form='clockwork', activation='linear', periods=(1, 4, 16, 64))
# ]

for i, layer in enumerate(networks):
    print("layer",layer)
    name = layer['form']
    layer['size'] = 126 # 64
    logging.info('training %s model', name)
    
    net = theanets.recurrent.Regressor([1, layer, 1])
    losses = []
    # """
    # for tm, _ in net.itertrain([ZERO, WAVES],
    print("INPUT.shape, WAVES.shape", INPUT.shape, WAVES.shape)
    for tm, _ in net.itertrain([INPUT, WAVES],
                               monitor_gradients=True,
                               batch_size=BATCH_SIZE,
                               algorithm='rmsprop',
                               learning_rate=0.0001,
                               momentum=0.9,
                               min_improvement=0.01): #, save_progress="recurrent_waves_{}", save_every=100):
        losses.append(tm['loss'])
    # """
    # net.load("recurrent_waves_net_rnn")
    # prd = net.predict(ZERO)
    prd = net.predict(INPUT)
    print("prd.shape", prd.shape)
    wave_ax.plot(T, prd[0, :, 0].flatten(), label=name, alpha=0.7, color=COLORS[i])
    learn_ax.plot(losses, label=name, alpha=0.7, color=COLORS[i])

    # net.save("recurrent_waves_net_%s" % name)
    
    # freerunning
    freerun_steps = 200
    prd2 = np.zeros((freerun_steps-1, 1), dtype=np.float32)
    inp = np.concatenate((INPUT[0], np.zeros((freerun_steps, 1), dtype=np.float32))) # prd[0,-1,:].reshape((1,1,1))
    print("inp.dtype", inp.dtype)
    print("inp.shape", inp.shape)
    # outp = net.predict(inp)
    # print("inp.shape, outp.shape", inp.shape, outp.shape, outp[0].shape)
    for j in range(freerun_steps-1):
        bi = freerun_steps-j
        print("bi = %d" % bi)
        rinp = inp[np.newaxis,0:-bi]
        outp = net.predict(rinp)
        # print("outp.dtype", outp.dtype)
        print("%d, inp.shape, rinp.shape, outp.shape" % j, inp.shape, rinp.shape, outp.shape, outp[0].shape)
        # prd2[j,:] = outp[0,-1]
        inp[-(freerun_steps-j)] = outp[0,-1] # prd2[j,:].reshape((1,1,1))
        # print("inp.dtype", inp.dtype)
    prd2 = outp[0].copy()
    print("prd2.shape", prd2.shape)
    freerun_ax.plot(prd2, label="%s" % name, alpha=0.7, color=COLORS[i])
    freerun_ax.plot(inp, "--", label="%s" % name, alpha=0.2, color=COLORS[i])
    # freerun_ax.plot(outp.flatten(), "ko", label="%s" % name, alpha=0.7, color=COLORS[i])
    
# Make the plots look nice.
for ax in [wave_ax, learn_ax]:
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['left'].set_position(('outward', 6))

wave_ax.set_ylabel('Amplitude')
wave_ax.set_xlabel('Time')

learn_ax.set_ylabel('Loss')
learn_ax.set_xlabel('Training Epoch')
learn_ax.grid(True)

pl.legend()
pl.show()
