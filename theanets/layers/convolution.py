# -*- coding: utf-8 -*-

'''Convolutional layers "scan" over input data.'''

from __future__ import division

import numpy as np
import theano
import theano.tensor as TT

from . import base
from .. import util

__all__ = [
    'Conv1',
]


class Convolution(base.Layer):
    '''Convolution layers convolve filters over the input arrays.

    Parameters
    ----------
    filter_shape : (int, int)
        Shape of the convolution filters for this layer.
    stride : (int, int), optional
        Apply convolutions with this stride; i.e., skip this many samples
        between convolutions. Defaults to (1, 1)---that is, no skipping.
    border_mode : str, optional
        Compute convolutions with this border mode. Defaults to 'valid'.
    '''

    def __init__(self, filter_shape, stride=(1, 1), border_mode='valid', **kwargs):
        self.filter_shape = filter_shape
        self.stride = stride
        self.border_mode = border_mode
        super(Convolution, self).__init__(**kwargs)

    def add_conv_weights(self, name, mean=0, std=None, sparsity=0):
        '''Add a convolutional weight array to this layer's parameters.

        Parameters
        ----------
        name : str
            Name of the parameter to add.
        mean : float, optional
            Mean value for randomly-initialized weights. Defaults to 0.
        std : float, optional
            Standard deviation of initial matrix values. Defaults to
            :math:`1 / sqrt(n_i + n_o)`.
        sparsity : float, optional
            Fraction of weights to set to zero. Defaults to 0.
        '''
        nin = self.input_size
        nout = self.size
        mean = self.kwargs.get(
            'mean_{}'.format(name),
            self.kwargs.get('mean', mean))
        std = self.kwargs.get(
            'std_{}'.format(name),
            self.kwargs.get('std', std or 1 / np.sqrt(nin + nout)))
        sparsity = self.kwargs.get(
            'sparsity_{}'.format(name),
            self.kwargs.get('sparsity', sparsity))
        arr = np.zeros((nout, nin) + self.filter_shape, util.FLOAT)
        for r in range(self.filter_shape[0]):
            for c in range(self.filter_shape[1]):
                arr[:, :, r, c] = util.random_matrix(
                    nout, nin, mean, std, sparsity=sparsity, rng=self.rng)
        self._params.append(theano.shared(arr, name=self._fmt(name)))


class Conv1(Convolution):
    '''1-dimensional convolutions run over one data axis.

    One-dimensional convolution layers can only be included in ``theanets``
    models that use recurrent inputs and outputs, i.e.,
    :class:`theanets.recurrent.Autoencoder`,
    :class:`theanets.recurrent.Predictor`,
    :class:`theanets.recurrent.Classifier`, or
    :class:`theanets.recurrent.Regressor`. The convolution will always be
    applied over the "time" dimension (axis 1).

    Parameters
    ----------
    filter_size : int
        Length of the convolution filters for this layer.
    stride : int, optional
        Apply convolutions with this stride; i.e., skip this many samples
        between convolutions. Defaults to 1, i.e., no skipping.
    border_mode : str, optional
        Compute convolutions with this border mode. Defaults to 'valid'.
    '''

    def __init__(self, filter_size, stride=1, border_mode='valid', **kwargs):
        super(Conv1, self).__init__(
            filter_shape=(1, filter_size),
            stride=(1, stride),
            border_mode=border_mode,
            **kwargs)

    def setup(self):
        '''Set up the parameters and initial values for this layer.'''
        self.add_conv_weights('w')
        self.add_bias('b', self.size)

    def transform(self, inputs):
        '''Transform the inputs for this layer into an output for the layer.

        Parameters
        ----------
        inputs : dict of Theano expressions
            Symbolic inputs to this layer, given as a dictionary mapping string
            names to Theano expressions. See :func:`Layer.connect`.

        Returns
        -------
        outputs : dict of Theano expressions
            A map from string output names to Theano expressions for the outputs
            from this layer. This layer type generates a "pre" output that gives
            the unit activity before applying the layer's activation function,
            and an "out" output that gives the post-activation output.
        updates : list of update pairs
            A sequence of updates to apply inside a Theano function.
        '''
        # input is:     (batch, time, input)
        # conv2d wants: (batch, input, 1, time)
        x = self._only_input(inputs).dimshuffle(0, 2, 'x', 1)

        pre = TT.nnet.conv.conv2d(
            x,
            self.find('w'),
            image_shape=(None, self.input_size, 1, None),
            filter_shape=(self.size, self.input_size) + self.filter_shape,
            border_mode=self.border_mode,
            subsample=self.stride,
        ).dimshuffle(0, 3, 1, 2)[:, :, :, 0] + self.find('b')
        # conv2d output is: (batch, output, 1, time)
        # we want:          (batch, time, output)
        # (have to do [:, :, :, 0] to remove unused trailing dimension)

        return dict(pre=pre, out=self.activate(pre)), []
