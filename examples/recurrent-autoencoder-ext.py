#!/usr/bin/env python

import argparse, os
import climate
import logging
import numpy as np
import numpy.random as rng
import pylab as pl
import theanets

climate.enable_default_logging()

TIME = 10
BATCH_SIZE = 32

def generate():
    return [rng.randn(TIME, BATCH_SIZE, 3).astype('f')]


def main_ae_1(args):
    e = theanets.Experiment(
        theanets.recurrent.Autoencoder,
        layers=(3, ('rnn', 10), 3),
        )

    # batch_size=BATCH_SIZE

    batch = generate()
    logging.info('data batches: %s', batch[0].shape)

    print len(batch)
    print batch[0].shape

    losses = []
    for tm, _ in e.itertrain(batch,
        monitor_gradients=True,
        batch_size=BATCH_SIZE,
        algo="rmsprop",
        learning_rate=0.0001,
        momentum=0.9,
        nesterov=True,
        min_improvement=0.01): #, save_progress="recurrent_waves_{}", save_every=100):
        losses.append(tm['loss'])

    # t, v = e.train(generate)

    e.save("recurrent_autoencoder")

    pl.plot(losses)
    pl.show()


def main_ae_2(args):
    model = theanets.recurrent.Autoencoder([10, (20, 'rnn'), 10])
    inputs = np.random.randn(1000, 100, 10).astype('f')
    
    # t, v = model.train([inputs], learning_rate=0.0001, momentum=0.1)
    t, v = model.train([inputs], learning_rate=0.0001, algo="adadelta")
    losses = []
    # for tm, _ in model.itertrain(inputs,
    #     monitor_gradients=True,
    #     batch_size=BATCH_SIZE,
    #     algo="rmsprop",
    #     learning_rate=0.0001,
    #     momentum=0.9,
    #     nesterov=True,
    #     min_improvement=0.01): #, save_progress="recurrent_waves_{}", save_every=100):
    #     losses.append(tm['loss'])

    # print t
    
    test = np.random.randn(3, 200, 10).astype('f')
    pred = model.predict(test)

    enc = model.encode(test)
    print "enc.shape", enc.shape
    
    pl.subplot(311)
    # pl.plot(t["loss"])
    # pl.plot(losses)
    pl.subplot(312)
    pl.plot(pred[0])
    pl.subplot(313)
    pl.plot(enc[0])
    pl.show()

def batches_from_wav(batch_size, seqlen, dim=1):
    from scipy.io import wavfile
    rate, data = wavfile.read("../../smp/playground/sequence/drinksonus44.wav")
    # pl.plot(data[0:44100])
    # pl.show()
    idx = np.random.randint(10000, 100000)
    data_ = data[idx:idx+(batch_size * seqlen)].copy().reshape((batch_size, seqlen, dim)).astype(np.float32)
    # data_ = data_.transpose([1, 0, 2])
    data_ /= np.max(np.abs(data_))
    return data_
            
def main_ae_3(args):
    dim = 1
    form = "clockwork"
    activation = "tanh"
    seqlen = 4410
    inputs = batches_from_wav(BATCH_SIZE, seqlen)
    losses = []
    
    print "inputs.shape", inputs.shape
    for i in range(BATCH_SIZE):
        pl.plot(inputs[i,:,:])
    pl.show()

    if os.path.exists("recurrent_autoencoder_ae3_net"):
        print "loading network"
        model = theanets.graph.Network.load("recurrent_autoencoder_ae3_net")
    else:
        print "creating and training network"
        # model = theanets.recurrent.Autoencoder([dim, (40, 'lstm'), dim])
        # model = theanets.recurrent.Autoencoder([dim, (40, 'rnn'), (20, 'rnn'), (40, 'rnn'), dim])
        # layer_h_1 = dict(form=form, activation=activation, size=100)
        # layer_h_2 = dict(form=form, activation=activation, size=40)
        # layer_h_3 = dict(form=form, activation=activation, size=20)
        layer_h_1 = dict(form=form, activation=activation, size=100, periods=[1, 2, 4, 8])
        layer_h_2 = dict(form=form, activation=activation, size=40, periods=[8, 16, 32, 64])
        layer_h_3 = dict(form=form, activation=activation, size=20, periods=[64, 128])
        model = theanets.recurrent.Autoencoder([
            1, #         theanets.layers.base.Input(size=1, name="In"),
            layer_h_1,
            layer_h_2,
            layer_h_3,
            layer_h_2,
            layer_h_1,
            1
            ])
        # t, v = model.train([inputs], learning_rate=0.0001, momentum=0.1)
        # t, v = model.train(inputs, learning_rate=0.00005, algo="adadelta")
        for tm, _ in model.itertrain(inputs,
            monitor_gradients=True,
            batch_size=BATCH_SIZE,
            algo="rmsprop",
            learning_rate=0.00001,
            momentum=0.9,
            nesterov=True,
            max_updates=1000,
            min_improvement=0.01): #, save_progress="recurrent_waves_{}", save_every=100):
            losses.append(tm['loss'])

        # print t

        model.save("recurrent_autoencoder_ae3_net")
    
    # test = np.random.randn(3, 200, 10).astype('f')
    test = batches_from_wav(3, seqlen * 2)
    pred = model.predict(test)

    enc = model.encode(test)
    print "enc.shape", enc.shape

    enc2 = enc.copy()
    enc2[:,:,:-2] = 0.
    dec = model.decode(enc2)

    enc3 = np.zeros_like(enc2)
    enc3[:,:,np.random.randint(0,20)] = (np.random.uniform(-1., 1, (enc3.shape[0], enc3.shape[1])) > 0.8) * 1.
    dec3 = model.decode(enc3)
                
    pl.subplot(411)
    # pl.plot(t["loss"])
    pl.plot(losses)
    pl.subplot(412)
    pl.plot(pred[0])
    pl.plot(test[0])
    pl.subplot(413)
    pl.plot(enc[0])
    pl.subplot(414)
    pl.plot(dec[0])
    pl.plot(enc3[0])
    pl.plot(dec3[0])
    pl.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="ae2", help="select experiment: ae1, ae2, ...")
    args = parser.parse_args()

    if args.mode == "ae1":
        main_ae_1(args)
    elif args.mode == "ae2":
        main_ae_2(args)
    elif args.mode == "ae3":
        main_ae_3(args)
    else:
        print "unknown mode, bye"
