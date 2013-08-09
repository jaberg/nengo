
"""
1-D "pose cell" network

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.fftpack

from nengo.nonlinear import LIF
from nengo.simulator_objects import SimModel
from nengo.simulator_objects import Signal
from nengo.simulator_objects import Encoder
from nengo.simulator_objects import Decoder
#from nengo.simulator_objects import Filter
#from nengo.simulator_objects import Transform
from nengo.simulator import Simulator
from nengo.simulator import register_handler
from nengo.old_api import filter_coefs


def probe_nl(model, nl, dt=0):
    model.transform(1.0, nl.output_signal, nl.output_signal)
    probe = model.probe(nl.output_signal, dt=0)
    return probe


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


def main():

    N_NEURONS = 2000
    N_POSITIONS = 20  # -- position is grid of this many points
    N_BASIS = 8       # -- represent position in terms of this many Fourier components
    clamp_dt = 0.10   # -- holds matrix rows this many secs

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

    clamp_basis = Clamp(np.asarray(Xbasis), dt=clamp_dt)
    clamp_direc = Clamp(np.asarray(Xdirec)[:, None], dt=clamp_dt)
    clamp_basis_out = Clamp(np.asarray(Ybasis), dt=clamp_dt)
    clamp_positions = Clamp(np.asarray(Yposit), dt=clamp_dt)

    # 4.4: Run simulator
    training_model = SimModel()
    #nl_readout = training_model.signal(N_NEURONS)
    training_model.signals.update(
        [basis_signal, position_signal, direction_signal,
         basis_neurons.input_signal, basis_neurons.bias_signal,
         basis_neurons.output_signal,
         clamp_basis.output_signal,
         clamp_direc.output_signal,
         clamp_basis_out.output_signal,
         clamp_positions.output_signal,
        ])
    training_model.nonlinearities.update([basis_neurons,
                                          clamp_basis, clamp_direc,
                                          clamp_basis_out, clamp_positions])
    training_model.encoders.update([enc_basis, enc_direc])
    training_model.decoder(clamp_basis, basis_signal, 1.0)
    training_model.decoder(clamp_direc, direction_signal, 1.0)
    training_model.transform(1.0, direction_signal, direction_signal)
    #training_model.decoder(basis_neurons, nl_readout, 1.0)
    #training_model.transform(1.0, nl_readout, nl_readout)

    simple_filter_decoded_signal(training_model, basis_signal, pstc=.01)
    simple_filter_decoded_signal(training_model, position_signal, pstc=.01)

    p1 = probe_nl(training_model, clamp_direc)
    p2 = probe_nl(training_model, clamp_basis)
    p3 = probe_nl(training_model, basis_neurons)
    p4 = training_model.probe(basis_signal, dt=0)
    p5 = probe_nl(training_model, clamp_basis_out)
    p6 = probe_nl(training_model, clamp_positions)

    req_steps = int(N_POSITIONS * 4 * clamp_dt / training_model.dt)
    print 'Running for steps:', req_steps
    sim = Simulator(training_model)
    sim.run_steps(req_steps)

    A = np.asarray(sim.probe_data(p3))
    d5 = sim.probe_data(p5)
    d6 = sim.probe_data(p6)

    basis_readout_weights, res5, rank5, s5 = np.linalg.lstsq(A, d5, rcond=0.01)
    posit_readout_weights, res6, rank6, s6 = np.linalg.lstsq(A, d6, rcond=0.01)

    dec_basis = Decoder(basis_neurons, basis_signal,
                        weights=basis_readout_weights.T)
    dec_posit = Decoder(basis_neurons, position_signal,
                        weights=posit_readout_weights.T)

    #print sim.probe_outputs[p]

    if 1:
        plt.subplot(1, 4, 1)
        plt.imshow(sim.probe_data(p1), extent=[0, 100, 0, 1], aspect='auto')
        plt.subplot(1, 4, 2)
        plt.imshow(sim.probe_data(p2), extent=[0, 100, 0, 1], aspect='auto')
        plt.subplot(1, 4, 3)
        plt.imshow(sim.probe_data(p3), extent=[0, 100, 0, 1], aspect='auto')
        plt.subplot(1, 4, 4)
        plt.imshow(sim.probe_data(p4), extent=[0, 100, 0, 1], aspect='auto')
        #print sim.probe_data(p4)
        plt.show()

    testing_model = SimModel()
    testing_model.signals.update(
        [basis_signal, position_signal, direction_signal,
         basis_neurons.input_signal, basis_neurons.bias_signal,
         basis_neurons.output_signal,
        ])
    testing_model.nonlinearities.update([basis_neurons,
                                        ])
    testing_model.encoders.update([enc_basis, enc_direc])
    testing_model.decoders.update([dec_basis, dec_posit])
    simple_filter_decoded_signal(testing_model, basis_signal, pstc=.01)
    simple_filter_decoded_signal(testing_model, position_signal, pstc=.01)

    # -- add direction clamp (using feedback for basis)
    #clamp_direc.dt /= 2  # -- slow it down rel to training, test generalization
    testing_model.signals.add(clamp_direc.output_signal)
    testing_model.nonlinearities.add(clamp_direc)
    testing_model.decoder(clamp_direc, direction_signal, 1.0)
    testing_model.transform(1.0, direction_signal, direction_signal)

    tp_basis = testing_model.probe(basis_signal, dt=0)
    tp_posit = testing_model.probe(position_signal, dt=0)
    #import pdb; pdb.set_trace()
    tp_neurons = probe_nl(testing_model, basis_neurons)

    test_sim = Simulator(testing_model)
    test_sim.run_steps(5000)
    
    #print test_sim.probe_outputs.keys()
    #print test_sim.probe_data(tp_neurons)

    if 1:
        plt.subplot(1, 2, 1)
        plt.imshow(test_sim.probe_data(tp_basis), extent=[0, N_BASIS, 0, 1],
                   aspect='auto')
        plt.subplot(1, 2, 2)
        plt.imshow(test_sim.probe_data(tp_posit), extent=[0, N_POSITIONS, 0, 1],
                   aspect='auto')
        #plt.subplot(1, 3, 3)
        #plt.imshow(test_sim.probe_data(tp_neurons))
        plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())

