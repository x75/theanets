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

import os, time, argparse
import climate
import logging
import matplotlib.pyplot as pl
import numpy as np
import theanets
import sys

from smp.datasets import wavdataset

# timestamp
timestamp = time.strftime("%Y%m%d-%H%M%S")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--activation", type=str, default="tanh", help="Activation function: tanh|linear|relu|sigmoid|... [tanh]")
parser.add_argument("-bs", "--batch_size", type=int, default=2)
parser.add_argument("-m", "--mode", type=str, default="mse", help="mse, mae, or mdn [mse]")
parser.add_argument("-ms", "--modelsize", type=int, default=100)
parser.add_argument("-o", "--optimizer", type=str, default="rmsprop", help="rmsprop, adagrad, adadelta, adam, rprop, esgd, sgd, nag [rmsprop]")
parser.add_argument("-t", "--target", type=str, default="sin", help="Target type: sin|wav [sin]")
parser.add_argument("-w", "--weights", type=str, default=None, help="modelfile to load [recurrent_waves_net_lstm]")

args = parser.parse_args()


climate.enable_default_logging()

COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
          '#e377c2', '#8c564b', '#bcbd22', '#7f7f7f', '#17becf']

BATCH_SIZE = args.batch_size


extendo = 1
# NUMDATAPTS = 8192
NUMDATAPTS = 1024
# NUMDATAPTS = 256 * extendo

# Construct a complex sine wave as a sum of pure-frequency waves.
TWOPI = 2 * np.pi
TAU = TWOPI * extendo
T = np.linspace(0, TAU*4, NUMDATAPTS)
SIN = sum(c * np.sin(TWOPI * f * T) for c, f in ((2, 1.5), (3, 1.8), (4, 1.1)))
# SIN = sum(c * np.sin(TWOPI * f * T) for c, f in ((2, 1.5), (3, 2.8), (4, 3.1)))
SIN /= np.max(SIN)

# Create an input dataset consisting of all zeros, and an output dataset
# containing the target sine wave. We have to stack the target sine wave here
# because recurrent models expect a tensor with three dimensions, and the batch
# size for recurrent networks must be greater than 1.
ZERO = np.zeros((BATCH_SIZE, len(T), 1), 'f')
WAVES_SIN = np.concatenate([SIN[None, :, None]] * BATCH_SIZE, axis=0).astype('f')

# SIN = np.vstack((SIN, SIN, SIN)).T
# WAVES_SIN = np.concatenate([SIN[None, :]] * BATCH_SIZE, axis=0).astype('f')

print("T.shape, SIN.shape, ZERO.shape, WAVES_SIN.shape", T.shape, SIN.shape, ZERO.shape, WAVES_SIN.shape)

# ################################################################################
# # load wave data #2
ds = wavdataset(
    sample_len = len(T), # 7*441,
    n_samples = 1,
    filename = "../../smp/playground/sequence/data/music.wav")
# print("len(ds)", len(ds))
# print("len(ds[0][0])", len(ds[0][0]))

# # print("ds[0][0].shape, SIN.shape", ds[1][0].shape, SIN.shape)

# # WAVES_WAV = np.concatenate([ds[0][0][None, :]] * BATCH_SIZE, axis=0).astype('f')
WAVES_WAV = np.array(ds[:])[:,0,:,:].astype(np.float32)
# print("WAVES_WAV.shape", WAVES_WAV.shape)

# select target
if args.target == "sin":
    WAVES = WAVES_SIN.copy()
elif args.target == "wav":
    WAVES = WAVES_WAV.copy()
else:
    print("unknown target %s" % args.target)
    sys.exit()

INPUT = np.roll(WAVES, 1, axis=1)# * 0.1

print("INPUT.shape", INPUT.shape)

# pl.subplot(311)
# pl.title("SIN")
# pl.plot(SIN)
# # pl.plot(T)
# pl.subplot(312)
# pl.title("WAVES / INPUT")
# pl.plot(WAVES[:,:,0].T)# + np.array((0.1, 0.2)))
# pl.plot(INPUT[:,:,0].T)# + np.array((0.1, 0.2)))
# pl.subplot(313)
# pl.title("INPUT")
# pl.plot(INPUT[:,:,0].T) # + np.array((0.1, 0.2)))
# pl.show()

# sys.exit()



# Set up plotting axes to show the output result and learning curves.
if args.weights != None:
    _, (wave_ax, freerun_ax) = pl.subplots(2, 1)
else:
    _, (wave_ax, learn_ax, freerun_ax) = pl.subplots(3, 1)
# _, (wave_ax, learn_ax) = pl.subplots(2, 1)

print("T.shape, WAVES.shape", T.shape, WAVES.shape)

# Plot the target wave.
# wave_ax.plot(T, SIN, ':', label='Target', alpha=0.7, color='#111111')
# wave_ax.plot(T, ds[0][0], ':', label='Target', alpha=0.7, color='#111111')
# wave_ax.plot(T, WAVES[0], ':', label='Target', alpha=0.7, color='#111111')
wave_ax.plot(WAVES[0], ':', label='Target', alpha=0.7, color='#111111')


# For each layer type, train a model containing that layer, and plot its
# predicted output.

activation = args.activation

networks = [
    dict(form='rnn', activation=activation, diagonal=0.5),
    dict(form='rrnn', activation=activation, rate='vector', diagonal=0.5),
    dict(form='scrn', activation=activation),
    dict(form='gru', activation=activation),
    dict(form='lstm', activation=activation),
    dict(form='clockwork', activation=activation, periods=(1, 2, 4, 8, 16, 64)),
    # dict(form='clockwork', activation=activation, periods=(1, 2, 4, 8, 16, 64)),
]

# networks = [
#         dict(form='clockwork', activation='linear', periods=(1, 4, 16, 64))
# ]

for i, layer in enumerate(networks):
    print("layer",layer)
    name = layer['form']
    layer['size'] = args.modelsize # 300 # 64
    # check size for clockwork partitioning
    if layer["form"] == "clockwork":
        layer["size"] = (layer["size"]//len(layer["periods"]))*len(layer["periods"])
        print("cw adjust", layer["size"])
    # layer['size'] = 64
    logging.info('training %s model', name)
    
    # net = theanets.recurrent.Regressor([1, layer, 1])
    
    # kw = dict(inputs={"%s:out" % name: 64}, size=numix)
    if args.mode == "mse":
        print("using MSE mode")
        net = theanets.recurrent.Regressor([1, layer, 1], loss="mse")
    elif args.mode == "mae":
        print("using MAE mode")
        net = theanets.recurrent.Regressor([1, layer, 1], loss="mae")
    elif args.mode == "mdn":
        print("using MDN output")
        numix = 3
        kw = dict(inputs={"hid1:out": layer["size"]}, size=numix)
        outkw = dict(inputs={"mu:out": numix, "sig:out": numix, "pi:out": numix}, size=numix*3)
        net = theanets.recurrent.MixtureDensity([1, layer,
                                       dict(name="mu", activation="linear", **kw),
                                       dict(name="sig", activation="exp", **kw),
                                       dict(name="pi", activation="softmax", **kw),
                                       # dict(name="out", activation="linear", **outkw)
                                       ])
                                       # dict(size=3 * numix, inputs={"mu:out": numix, "sig:out": numix, "pi:out": numix})])# , loss="nll")
        net.set_loss("nll", mu_name="mu", sig_name="sig", pi_name="pi", numcomp=numix)
    
    losses = []
    #"""
    # for tm, _ in net.itertrain([ZERO, WAVES],
    algo = args.optimizer
    # algo="rmsprop",
    # algo = "adagrad",
    # algo = "adadelta",
    # algo = "adam",
    # algo = "rprop",
    # algo = "esgd",
    # algo = "sgd",
    # algo = "nag", # nope
    
    print("INPUT.shape, WAVES.shape", INPUT.shape, WAVES.shape)

    
    if args.weights != None and os.path.exists(args.weights + "_%s" % name):
        # thats kinda defunct in the loop over different models, thus all predictions are the same
        print("loading model %s" % (args.weights + "_%s" % name))
        net = theanets.graph.Network.load(args.weights + "_%s" % name)
    else:
        # net.train([INPUT, WAVES], learning_rate=0.0001, algo="rmsprop")    
        for tm, _ in net.itertrain([INPUT, WAVES],
                               monitor_gradients=True,
                               batch_size=BATCH_SIZE,
                               algo=algo,
                               learning_rate=0.0001,
                               momentum=0.9,
                               nesterov=True,
                               patience = 10,
                               min_improvement=0.01): #, save_progress="recurrent_waves_{}", save_every=100):
            losses.append(tm['loss'])
        # save network
        net.save("recurrent-waves/recurrent_waves_net_%s_%s" % (timestamp, name))
        np.save("recurrent-waves/recurrent_waves_losses_%s_%s" % (timestamp, name), np.asarray(losses))
    # """
    # net = net.load("recurrent_waves_net_rnn")
    # prd = net.predict(ZERO)
    # standard predict
    prd = net.predict(INPUT)
    # print("net state", net.layers[1].cell)
    print("prd.shape", prd.shape)
    # wave_ax.plot(T, prd[0, :, 0].flatten(), label=name, alpha=0.7, color=COLORS[i])
    wave_ax.plot(prd[0, :, 0].flatten(), label=name, alpha=0.7, color=COLORS[i])
    if args.weights == None:
        learn_ax.plot(losses, label=name, alpha=0.7, color=COLORS[i])

    # freerunning, maintaining state? nope
    # print("net state", net.cell)
    freerun_steps = 200 * 1 # extendo
    prd2 = np.zeros((freerun_steps-1, 1), dtype=np.float32)
    # inp = np.concatenate((INPUT[0], np.zeros((freerun_steps, 1), dtype=np.float32))) # prd[0,-1,:].reshape((1,1,1))
    inp = np.concatenate((INPUT[0,:-freerun_steps], np.zeros((freerun_steps, 1), dtype=np.float32))) # prd[0,-1,:].reshape((1,1,1))
    inpc = INPUT[0].copy()
    print("inp.dtype", inp.dtype)
    print("inp.shape", inp.shape)
    # outp = net.predict(inp)
    # print("inp.shape, outp.shape", inp.shape, outp.shape, outp[0].shape)
    for j in range(freerun_steps-1):
        bi = freerun_steps-j
        # print("bi = %d" % bi)
        rinp = inp[np.newaxis,0:-bi]
        now = time.time()
        # single timestep predict
        outp = net.predict(rinp)
        took = time.time() - now
        # print("outp.dtype", outp.dtype)
        print("pred time = %f, %d/%d, inp.shape = %s, rinp.shape = %s, outp.shape = %s, outp[0].shape = %s" % (took, bi, j, inp.shape, rinp.shape, outp.shape, outp[0].shape))
        # prd2[j,:] = outp[0,-1]
        inp[-(freerun_steps-j)] = outp[0,-1] # prd2[j,:].reshape((1,1,1))
        # print("inp.dtype", inp.dtype)
    prd2 = outp[0].copy()
    print("prd2.shape", prd2.shape)
    freerun_ax.plot(prd2, label="%s prd fr" % name, alpha=0.7, color=COLORS[i])
    if i == 0: # add target only once
        freerun_ax.plot(inpc, "--", label="target", alpha=0.2, color=COLORS[i])
    # freerun_ax.plot(outp.flatten(), "ko", label="%s" % name, alpha=0.7, color=COLORS[i])

np.save("input.npy", INPUT[0])
np.save("prd.npy", prd[0, :, 0].flatten())
np.save("prd2.npy", prd2)
        
# Make the plots look nice.
if args.weights == None:
    niceaxes = [wave_ax, learn_ax]
else:
    niceaxes = [wave_ax]
for ax in niceaxes:
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('outward', 6))
    ax.spines['left'].set_position(('outward', 6))

wave_ax.set_ylabel('Amplitude')
wave_ax.set_xlabel('Time')

if args.weights == None:
    if args.mode != "mdn":
        learn_ax.set_yscale('log')
    learn_ax.set_ylabel('Loss')
    learn_ax.set_xlabel('Training Epoch')
    learn_ax.grid(True)

pl.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=8)

pl.gcf().set_size_inches(18, 12)
pl.gcf().savefig("recurrent-waves-H%d_%s_%s_%s_%s_%s.pdf" % (args.modelsize, args.mode, args.activation, args.target, args.optimizer, timestamp), dpi=300, bbox_inches="tight")

pl.show()
