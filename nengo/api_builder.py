from copy import deepcopy
import logging
from collections import OrderedDict
import numpy as np
import nonlinearities
from itertools import chain

logger = logging.getLogger(__name__)

import decoders
from . import api

from builder import (
    ShapeMismatch,
    SignalView,
    Signal,
    Probe,
    Operator,
    Reset,
    Copy,
    DotInc,
    ProdUpdate,
    SimPyFunc,
    SimLIF,
    SimLIFRate)


def model_index(rval, prefix, model):
    rval[prefix] = model
    for submodel in model.get('models', []):
        model_index(rval=rval,
                    prefix=prefix + (submodel.name,),
                    model=submodel)
    return rval


class Builder(object):
    """A callable class that copies a model and determines the signals
    and operators necessary to simulate that model.

    Builder does this by mapping each high-level object to its associated
    signals and operators one-by-one, in the following order:

      1. Ensembles and Nodes
      2. Probes
      3. Connections

    """

    def __init__(self, model, dt, copy=True):
        # Whether or not we make a deep-copy of the model we're building
        self.copy = copy
        model = model.model  #-- unpack it from the "defining" class
        if self.copy:
            # Make a copy of the model so that we can reuse the non-built model
            logger.info("Copying model")
            memo = {}
            self.model = deepcopy(model, memo)
            self.memo = memo
        else:
            self.model = model
        del model

        self.dt = dt
        self.seed = self.model.get('seed',
                                   np.random.randint(np.iinfo(np.int32).max))
        self.rng = np.random.RandomState(self.seed)

        midx = OrderedDict()
        model_index(midx, (), self.model)
        self.midx = midx

        # The purpose of the build process is to fill up these lists
        self.probes = []
        self.operators = []

        logger.info("Building objects")
        model_objects = list(chain(*[[(m, o) for o in m['objects']]
                                     for m in midx.values()]))
        for m, obj in model_objects:
            getattr(self, 'build_%s' % obj['object_type'])(obj, m)

        logger.info("Building probes")
        model_probes = list(chain(*[[(m, p) for p in m['probes']]
                                    for m in midx.values()]))
        for m, probe in model_probes:
            self.build_probe(probe, m)

        logger.info("Building connections")
        model_conns = list(chain(*[[(m, c) for c in m['connections']]
                                   for m in midx.values()]))
        for m, c in model_conns:
            getattr(self, 'build_%s' % c['connection_type'])(c, m)

        # Set up t and timesteps
        self.operators.append(
            ProdUpdate(Signal(1),
                       Signal(self.dt),
                       Signal(1),
                       self.lookup(self.model,'_t').output_signal))
        self.operators.append(
            ProdUpdate(Signal(1),
                       Signal(1),
                       Signal(1),
                       self.lookup(self.model, '_steps').output_signal))

    def __call__(self, *args, **kwargs):
        return self

    def _get_new_seed(self):
        return self.rng.randint(np.iinfo(np.int32).max)

    def build_ensemble(self, ens, submodel, signal=None):
        if ens.dimensions <= 0:
            raise ValueError(
                'Number of dimensions (%d) must be positive' % ens.dimensions)

        # Create random number generator
        if ens.seed is None:
            ens.seed = self._get_new_seed()
        rng = np.random.RandomState(ens.seed)

        # Generate eval points
        if ens.eval_points is None:
            ens.eval_points = decoders.sample_hypersphere(
                ens.dimensions, ens.EVAL_POINTS, rng) * ens.radius
        else:
            ens.eval_points = np.array(ens.eval_points, dtype=np.float64)
            if ens.eval_points.ndim == 1:
                ens.eval_points.shape = (-1, 1)

        # Set up signal
        if signal is None:
            ens.input_signal = Signal(np.zeros(ens.dimensions),
                                      name=ens.name + ".signal")
        else:
            # Assume that a provided signal is already in the model
            ens.input_signal = signal
            ens.dimensions = ens.input_signal.size

        #reset input signal to 0 each timestep (unless this ensemble has
        #a view of a larger signal -- generally meaning it is an ensemble
        #in an ensemble array -- in which case something else will be
        #responsible for resetting)
        if ens.input_signal.base == ens.input_signal:
            self.operators.append(Reset(ens.input_signal))

        getattr(self, 'build_%s' % ens.neurons.neuron_type)(ens.neurons, ens)
        default_encoders_fn = getattr(
            self,
            'default_encoders_%s' % ens.neurons.neuron_type)

        # Set up encoders
        if ens.encoders is None:
            ens.encoders = default_encoders_fn(ens.neurons, ens, rng)
        else:
            ens.encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if ens.encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    " in this case %s." % (ens.encoders.shape, enc_shape))

            norm = np.sum(ens.encoders * ens.encoders, axis=1)[:, np.newaxis]
            ens.encoders /= np.sqrt(norm)

        if ens.neurons.neuron_type == 'direct':
            ens._scaled_encoders = ens.encoders
        else:
            ens._scaled_encoders = ens.encoders * (
                ens.neurons.gain / ens.radius)[:, np.newaxis]
        self.operators.append(DotInc(Signal(ens._scaled_encoders),
                                     ens.input_signal,
                                     ens.neurons.input_signal,
                                     tag='encoders'))

        # Output is neural output
        ens.output_signal = ens.neurons.output_signal

        # # Set up probes, but don't build them (done explicitly later)
        # # Note: Have to set it up here because we only know these things
        # #       (dimensions, n_neurons) at build time.
        # for probe in ens.probes['decoded_output']:
        #     probe.dimensions = ens.dimensions
        # for probe in ens.probes['spikes']:
        #     probe.dimensions = ens.n_neurons
        # for probe in ens.probes['voltages']:
        #     probe.dimensions = ens.n_neurons

    def build_passthrough(self, ptn, submodel):
        ptn.input_signal = Signal(np.zeros(ptn.dimensions),
                                  name=ptn.name + ".signal")
        ptn.output_signal = ptn.input_signal

        #reset input signal to 0 each timestep
        self.operators.append(Reset(ptn.input_signal))

    def build_node(self, node, submodel):
        if not callable(node.output):
            if isinstance(node.output, (int, float, long, complex)):
                node.output_signal = Signal([node.output], name=node.name)
            else:
                node.output_signal = Signal(node.output, name=node.name)
        else:
            node.input_signal = Signal(np.zeros(node.dimensions),
                                       name=node.name + ".signal")
            n_out = np.asarray(node.output(0)).size
            node.output_signal = Signal(np.zeros(n_out),
                                       name=node.name + ".output")

            #reset input signal to 0 each timestep
            self.operators.append(Reset(node.input_signal))
            self.operators.append(SimPyFunc(node.output_signal,
                                            node.input_signal,
                                            node.output))
        print 'build_node: %s.%s' % (submodel.name, node.name)

    def lookup(self, submodel, key):
        try:
            return api._rec_model_lookup(submodel, key)
        except KeyError:
            try:
                return api._rec_model_lookup(self.model, key)
            except KeyError:
                raise KeyError(key)

    def build_probe(self, pdict, submodel):
        obj = self.lookup(submodel, pdict.signame)
        sig = obj.output_signal # -- installed by build_node
        if pdict.filter > 0:
            decay, inc = self.filter_coefs(pdict.filter, self.dt)
            # -- need a filtered signal to record
            pdict.input_signal = Signal(
                np.zeros(sig.shape),
                name=pdict.signame)
            self.operators.append(ProdUpdate(
                Signal(inc), sig, Signal(decay), pdict.input_signal))
        else:
            # -- just record straight from the original
            pdict.input_signal = sig

        self.probes.append(Probe(pdict.input_signal, pdict.sample_every))

    @staticmethod
    def filter_coefs(pstc, dt):
        assert dt > 0
        if pstc <= 0:
            raise ValueError('Invalid filter pstc', pstc)
        decay = np.exp(-dt / pstc)
        return decay, (1.0 - decay)

    def _filtered_signal(self, signal, filter):
        name = signal.name + ".filtered(%f)" % filter
        filtered = Signal(np.zeros(signal.size), name=name)
        o_coef, n_coef = self.filter_coefs(pstc=filter, dt=self.dt)
        self.operators.append(ProdUpdate(
            Signal(n_coef), signal, Signal(o_coef), filtered))
        return filtered

    def build_connection(self, conn, submodel):
        conn.input_signal = self.lookup(submodel, conn.pre + ".output_signal")
        conn.output_signal = self.lookup(submodel, conn.post + ".input_signal")
        if conn.modulatory:
            # Make a new signal, effectively detaching from post
            conn.output_signal = Signal(np.zeros(conn.dimensions),
                                        name=conn.name + ".mod_output")

        if self.lookup(submodel, conn.post).object_type == 'neurons':
            conn.transform *= conn.post.gain[:, np.newaxis]

        # Set up filter
        if conn.filter is not None and conn.filter > self.dt:
            conn.input_signal = self._filtered_signal(
                conn.input_signal, conn.filter)

        # Set up transform
        try:
            self.operators.append(
                DotInc(
                    Signal(conn.transform),
                    conn.input_signal,
                    conn.output_signal,
                    tag='connection'))
        except:
            print 'Building connection', conn
            raise

    def build_decodedconnection(self, conn, submodel):
        pre = self.lookup(submodel, conn.pre)
        post = self.lookup(submodel, conn.post)
        assert pre.object_type == 'ensemble'
        conn.input_signal = pre.output_signal
        conn.output_signal = post.input_signal
        if conn.modulatory:
            # Make a new signal, effectively detaching from post,
            # but still performing the decoding
            conn.output_signal = Signal(np.zeros(conn.dimensions),
                                        name=conn.name + ".mod_output")
        if post.object_type == 'neurons':
            # XXX WORKS?
            conn.transform *= post.gain[:, np.newaxis]
        dt = self.dt

        # A special case for Direct mode.
        # In Direct mode, rather than do decoders, we just
        # compute the function and make a direct connection.
        if pre.neurons.neuron_type == 'direct':
            # XXX WORKS?
            if conn.function is None:
                conn.signal = conn.input_signal
            else:
                name = conn.name + ".pyfunc"
                conn.pyfunc = nonlinearities.PythonFunction(
                    fn=conn.function, n_in=conn.input_signal.size, name=name)
                self.build_pyfunc(conn.pyfunc)
                self.operators.append(DotInc(
                    conn.input_signal, Signal(1.0), conn.pyfunc.input_signal))
                conn.signal = conn.pyfunc.output_signal

            # Set up filter
            if conn.filter is not None and conn.filter > dt:
                conn.signal = self._filtered_signal(conn.signal, conn.filter)

        else:
            # For normal decoded connections...
            conn.signal = Signal(np.zeros(post.dimensions), name=conn.name)

            # Set up decoders
            if conn._decoders is None:
                eval_points = conn.eval_points
                if eval_points is None:
                    eval_points = pre.eval_points

                rates_fn = getattr(self, 'rates_' + pre.neurons.neuron_type)

                activities = rates_fn(pre.neurons,
                                      np.dot(eval_points,
                                             pre._scaled_encoders.T))

                if conn.function is None:
                    targets = eval_points
                else:
                    targets = np.array(
                        [conn.function(ep) for ep in eval_points])
                    if len(targets.shape) < 2:
                        targets.shape = targets.shape[0], 1
                conn._decoders = conn.decoder_solver(activities, targets)

            # Set up filter
            if conn.filter is not None and conn.filter > dt:
                o_coef, n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)
                self.operators.append(
                    ProdUpdate(Signal(conn._decoders * n_coef),
                               conn.input_signal,
                               Signal(o_coef),
                               conn.signal))
            else:
                self.operators.append(
                    ProdUpdate(Signal(conn._decoders),
                               conn.input_signal,
                               Signal(0),
                               conn.signal))

        # Set up transform
        self.operators.append(DotInc(
            Signal(conn.transform), conn.signal, conn.output_signal))

    def build_connectionlist(self, conn):
        conn.transform = np.asarray(conn.transform)

        i = 0
        for connection in conn.connections:
            pre_dim = connection.dimensions
            if conn.transform.ndim == 0:
                trans = np.zeros((connection.post.dimensions, pre_dim))
                np.fill_diagonal(trans[i:i+pre_dim,:], conn.transform)
            elif conn.transform.ndim == 2:
                trans = conn.transform[:,i:i+pre_dim]
            else:
                raise NotImplementedError(
                    "Only transforms with 0 or 2 ndims are accepted")
            i += pre_dim
            connection.transform = trans
            self._builders[connection.__class__](connection)

    def build_ensemblearray(self, ea, submodel):
        ea.input_signal = Signal(np.zeros(ea.dimensions),
                                 name=ea.name+".signal")
        self.operators.append(Reset(ea.input_signal))
        dims = ea.dimensions_per_ensemble

        for i, ens in enumerate(ea.ensembles):
            self.build_ensemble(ens, submodel,
                                signal=ea.input_signal[i*dims:(i+1)*dims])

        for probe in ea.probes['decoded_output']:
            probe.dimensions = ea.dimensions

    def build_pyfunc(self, pyfn):
        pyfn.input_signal = Signal(np.zeros(pyfn.n_in),
                                   name=pyfn.name + '.input')
        pyfn.output_signal = Signal(np.zeros(pyfn.n_out),
                                    name=pyfn.name + '.output')
        pyfn.operators = [Reset(pyfn.input_signal),
                          SimPyFunc(output=pyfn.output_signal,
                                    J=pyfn.input_signal,
                                    fn=pyfn.fn)]
        self.operators.extend(pyfn.operators)

    def build_neurons(self, neurons):
        neurons.input_signal = Signal(np.zeros(neurons.n_in),
                                      name=neurons.name + '.input')
        neurons.output_signal = Signal(np.zeros(neurons.n_out),
                                       name=neurons.name + '.output')
        neurons.bias_signal = Signal(neurons.bias, name=neurons.name + '.bias')
        self.operators.append(
            Copy(src=neurons.bias_signal, dst=neurons.input_signal))

    def build_direct(self, direct):
        direct.input_signal = Signal(np.zeros(direct.dimensions),
                                     name=direct.name)
        direct.output_signal = direct.input_signal
        self.operators.append(Reset(direct.input_signal))

    def build_lifbase(self, lif, ens):
        if lif.gain is None or lif.bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples.
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(
                    ens.neurons.n_neurons, rng=self.rng)
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.neurons.n_neurons, rng=self.rng)

            logging.debug("Setting gain and bias on %s", lif.name)
            # XXX if either gain or bias was None, this sets both :/
            assert lif.gain is lif.bias is None
            max_rates = np.asarray(ens.max_rates)
            intercepts = np.asarray(ens.intercepts)
            x = 1.0 / (1 - np.exp(
                (lif.tau_ref - (1.0 / max_rates)) / lif.tau_rc))
            lif.gain = (1 - x) / (intercepts - 1.0)
            lif.bias = 1 - lif.gain * intercepts

    def build_lifrate(self, lif, ens):
        if lif.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be non-negative' % lif.n_neurons)
        self.build_neurons(lif)
        self.build_lifbase(lif, ens)
        self.operators.append(SimLIFRate(output=lif.output_signal,
                                               J=lif.input_signal,
                                               nl=lif))

    def default_encoders_lif(self, lif, ens, rng):
        return decoders.sample_hypersphere(
            ens.dimensions, lif.n_neurons, rng, surface=True)

    def build_lif(self, lif, ens):
        if lif.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be non-negative' % lif.n_neurons)
        self.build_neurons(lif)
        self.build_lifbase(lif, ens)
        lif.voltage = Signal(np.zeros(lif.n_neurons))
        lif.refractory_time = Signal(np.zeros(lif.n_neurons))
        self.operators.append(SimLIF(output=lif.output_signal,
                                           J=lif.input_signal,
                                           nl=lif,
                                           voltage=lif.voltage,
                                           refractory_time=lif.refractory_time))

    def rates_lif(self, lif, J_without_bias):
        """LIF firing rates in Hz

        Parameters
        ---------
        J_without_bias: ndarray of any shape
            membrane currents, without bias voltage
        """
        old = np.seterr(divide='ignore', invalid='ignore')
        try:
            J = J_without_bias + lif.bias
            A = lif.tau_ref - lif.tau_rc * np.log(
                1 - 1.0 / np.maximum(J, 0))
            # if input current is enough to make neuron spike,
            # calculate firing rate, else return 0
            A = np.where(J > 1, 1 / A, 0)
        finally:
            np.seterr(**old)
        return A
