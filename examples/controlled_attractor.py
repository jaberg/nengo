
"""
1-D "pose cell" network

"""

import numpy as np
#from ..nonlinear import LIF

class PoseCellNetwork1(object):
    def __init__(self, model, heading,
                 N=10,
                 neurons_per_grid=50,
                ):
        """
        One-dimensional proof of concept

        dP = conv(P, e) - c
        F dP = F G (F P .* Fe) - Fc
        F dP = FP .* Fe

        """

        # -- Fourier of the Pose Cell Network (PCN)
        fourier_pcn = model.signal(N)

        # -- Fourier heading is the Fourier transform
        #    of the filterbank that would shift us by the
        #    amount described by the `heading`
        fourier_heading = model.signal(N)

        PCN = model.nonlinearity()



import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt

from nengo.nonlinear import LIF
from nengo.simulator_objects import SimModel
from nengo.simulator_objects import Signal
from nengo.simulator_objects import Encoder
#from nengo.simulator_objects import Decoder
#from nengo.simulator_objects import Filter
#from nengo.simulator_objects import Transform
from nengo.simulator import Simulator
from nengo.simulator import register_handler
from nengo.old_api import filter_coefs


def simple_filter_decoded_signal(model, signal, pstc):
    a, b = filter_coefs(pstc, model.dt)
    model.filter(a, signal, signal)
    model.transform(b, signal, signal)


class Clamp(object):
    """
    Output
    """
    def __init__(self, data, dt):
        self.data = data
        self.dt = dt
        L, N = data.shape
        self.output_signal = Signal(N)


@register_handler(Clamp)
class SimClamp(object):
    def __init__(self, nl):
        self.nl = nl
        self.t = 0.0

    def step(self, dt, J, output):
        idx = int(self.t / self.nl.dt) % len(self.nl.data)
        output[:] = self.nl.data[idx]
        self.t += dt


if __name__ == '__main__':


    N_NEURONS = 100 
    N_POSITIONS = 20  # -- position is grid of this many points
    N_BASIS = 8       # -- represent position in terms of this many Fourier components

    # 1: create signals
    basis_signal = Signal(N_BASIS * 2, dtype='float')
    position_signal = Signal(N_POSITIONS, dtype='float')
    direction_signal = Signal(1, dtype='float')

    # 2: create neurons
    basis_neurons = LIF(N_NEURONS)

    rng = np.random.RandomState(1234)
    max_rates = rng.uniform(size=N_NEURONS, low=200, high=300)
    threshold = rng.uniform(size=N_NEURONS, low=-1.0, high=1.0)
    basis_neurons.set_gain_bias(max_rates, threshold)
    basis_neurons.bias_signal.value[:] = basis_neurons.bias

    # 3: create encoder/decoder circuit
    enc_basis = Encoder(basis_signal, basis_neurons)
    enc_direc = Encoder(direction_signal, basis_neurons)

    enc_basis.weights *= basis_neurons.gain[:, None]
    enc_direc.weights *= basis_neurons.gain[:, None]


    # 4: train encoder/decoder circuit

    # 4.1: Sample all possible positions
    data_positions = np.eye(N_POSITIONS)

    # 4.2: Sample all possible directions
    data_directions = np.asarray([-1, 0, 1])

    # 4.3: Convert data positions to Fourier domain
    blobs = scipy.signal.lfilter([.15, .7, .15], [1], data_positions)
    basis_blobs = scipy.fftpack.fft(blobs)
    basis_blobs[:, N_BASIS:] = 0
    if 0:
        d2 = scipy.fftpack.ifft(basis_blobs)
        plt.imshow(np.real(d2))
        plt.show()
    data_basis = np.empty((len(basis_blobs), 2 * N_BASIS))
    data_basis[:, ::2] = np.real(basis_blobs[:, :N_BASIS])
    data_basis[:, 1::2] = np.imag(basis_blobs[:, :N_BASIS])

    # 4.4: Build a training set
    Xbasis = list()
    Xdirec = list()
    Ybasis = list()
    Yposit = list()
    # loop forward over the positions
    for ii in range(N_POSITIONS):
        Xbasis.append(data_basis[ii])
        Xdirec.append(0)
        Ybasis.append(data_basis[ii])
        Yposit.append(data_positions[ii])
        Xbasis.append(data_basis[ii])
        Xdirec.append(1)
        Ybasis.append(data_basis[(ii + 1) % N_POSITIONS])
        Yposit.append(data_positions[(ii + 1) % N_POSITIONS])
    # loop backward over the positions
    for ii in range(N_POSITIONS):
        Xbasis.append(data_basis[-ii])
        Xdirec.append(0)
        Ybasis.append(data_basis[-ii])
        Yposit.append(data_positions[-ii])
        Xbasis.append(data_basis[-ii])
        Xdirec.append(-1)
        Ybasis.append(data_basis[(-ii - 1) % N_POSITIONS])
        Yposit.append(data_positions[(-ii - 1) % N_POSITIONS])

    clamp_basis = Clamp(np.asarray(Xbasis), dt=.1)
    clamp_direc = Clamp(np.asarray(Xdirec)[:, None], dt=.1)

    # 4.4: Run simulator
    training_model = SimModel()
    #nl_readout = training_model.signal(N_NEURONS)
    training_model.signals.update(
        [basis_signal, position_signal, direction_signal,
         basis_neurons.input_signal, basis_neurons.bias_signal,
         basis_neurons.output_signal,
         clamp_basis.output_signal,
         clamp_direc.output_signal,
        ])
    training_model.nonlinearities.update([basis_neurons, clamp_basis, clamp_direc])
    training_model.encoders.update([enc_basis, enc_direc])
    training_model.decoder(clamp_basis, basis_signal, 1.0)
    training_model.decoder(clamp_direc, direction_signal, 1.0)
    #training_model.decoder(basis_neurons, nl_readout, 1.0)
    #training_model.transform(1.0, nl_readout, nl_readout)

    simple_filter_decoded_signal(training_model, basis_signal, pstc=.01)
    simple_filter_decoded_signal(training_model, position_signal, pstc=.01)

    training_model.transform(1.0, clamp_direc.output_signal, clamp_direc.output_signal)
    p1 = training_model.probe(clamp_direc.output_signal, dt=0)

    training_model.transform(1.0, clamp_basis.output_signal, clamp_basis.output_signal)
    p2 = training_model.probe(clamp_basis.output_signal, dt=0)

    training_model.transform(1.0, basis_neurons.output_signal, basis_neurons.output_signal)
    p3 = training_model.probe(basis_neurons.output_signal, dt=0)

    p4 = training_model.probe(basis_signal, dt=0)

    sim = Simulator(training_model)
    sim.run_steps(1000)

    #print sim.probe_outputs[p]

    if 1:
        plt.subplot(1, 4, 1)
        plt.imshow(sim.probe_data(p1))
        plt.subplot(1, 4, 2)
        plt.imshow(sim.probe_data(p2))
        plt.subplot(1, 4, 3)
        plt.imshow(sim.probe_data(p3))
        plt.subplot(1, 4, 4)
        plt.imshow(sim.probe_data(p4))
        print sim.probe_data(p4)
        plt.show()


