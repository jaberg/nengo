from functools import partial
import copy
from simulator import Simulator
import numpy as np
from objects import Uniform

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
#    def __getattribute__(self, attr):
#        try:
#            return self[attr]
#        except KeyError:
#            raise AttributeError(attr)
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def __deepcopy__(self, memo):
        dc = copy.deepcopy
        rval = DotDict(map(partial(copy.deepcopy, memo=memo), self.items()))
        memo[id(self)] = rval
        return rval


model_stack = []

class defining(object):
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        model_stack.append(self.model)

    def __exit__(self, *args):
        assert [self.model] == model_stack[-1:]
        model_stack.pop()


def _active_model():
    return model_stack[-1]


def _init_model(dct):
    with defining(dct):
        node('t', output=0)
        node('steps', output=0)
        probe('t')


def model(name):
    new_model = DotDict({'name': name})
    _init_model(new_model)
    return defining(new_model)

def submodel(name):
    _active_model().setdefault('models', [])
    new_model = DotDict({'name': name})
    _init_model(new_model)
    _active_model()['models'].append(new_model)
    return defining(new_model)


def integrator(name, recurrent_tau, **ens_args):
    with submodel(name):
        passthrough('in')
        ensemble('integrator', **ens_args)
        connect('integrator', 'integrator', filter=recurrent_tau)
        connect('in', 'integrator', filter=recurrent_tau)


def connect(src, dst, transform=None, filter=None):
    _active_model().setdefault('connections', [])
    _active_model()['connections'].append(DotDict({
        'src': src,
        'dst': dst,
        'transform': transform,
        'filter': filter}))

def LIF(n_neurons, tau_rc=0.02, tau_ref=.002):
    return DotDict({'neuron_type': 'lif',
            'name': 'lif',  #?
            'n_in': n_neurons,
            'n_out': n_neurons,
            'n_neurons': n_neurons,
            'tau_rc': tau_rc,
            'tau_ref': tau_ref,
            'gain': None,
            'bias': None,
            })


def _get_ens_by_name(name, model=None):
    if model is None:
        model = _active_model()
    matching = [e for e in model.get('objects', [])
                if (e['object_type'] == 'ensemble'
                    and e['name'] == name)]
    assert len(matching) < 2
    if len(matching):
        return matching[0]
    else:
        raise KeyError(name)


def ensemble(name, neurons, dimensions, seed=None, radius=1.0):
    """Create an Ensemble"""
    _active_model().setdefault('objects', [])
    _active_model()['objects'].append(DotDict({
        'object_type': 'ensemble',
        'name': name,
        'neurons': neurons,
        'dimensions': dimensions,
        'seed': seed,
        'eval_points': None,
        'EVAL_POINTS': 500,
        'radius': radius,
        'max_rates': Uniform(200, 400),
        'intercepts': Uniform(-1, 1),
        'encoders': None,
        }))


def ensemble_array(*args, **kwargs):
    """Create an EnsembleArray"""
    _active_model().add(EnsembleArray(*args, **kwargs))


def encoders(name, val):
    """Retrieve encoders associated with ensemble
    """
    return _get_ens_by_name(name)['encoders']


def set_encoders(name, val):
    """Assign encoders to `name` (typically an ensemble)
    """
    _get_ens_by_name(name)['encoders'] = val


def n_neurons(name):
    """Return the number of neurons of `name` (typically ensemble)
    """
    _get_ens_by_name(name)['neurons']['n_neurons']


def node(name, output):
    """Create a Node"""
    _active_model().setdefault('objects', [])
    _active_model()['objects'].append(DotDict({
        'object_type': 'node',
        'name': name,
        'output': output,
        'dimensions': np.asarray(output(0)).size if output else 1
        }))


def passthrough(name):
    """Create a Passthrough Node"""
    return node(name, output=None)


def dimensions(signame, model=None):
    return _get_ens_by_name(signame, model).dimensions


def probe(signame, sample_every=0.001, filter=0.01):
    _active_model().setdefault('probes', [])
    _active_model()['probes'].append(DotDict({
        'signame': signame,
        'filter': filter,
        'sample_every': sample_every,
        }))


def simulator(model, dt=0.001, sim_class=Simulator):
    """Get a new simulator object for the model.

    Parameters
    ----------
    dt : float, optional
        Fundamental unit of time for the simulator, in seconds.
    sim_class : child class of `Simulator`, optional
        The class of simulator to be used.
    seed : int, optional
        Random number seed for the simulator's random number generator.
        This random number generator is responsible for creating any random
        numbers used during simulation, such as random noise added to
        neurons' membrane voltages.
    **sim_args : optional
        Arguments to pass to the simulator constructor.

    Returns
    -------
    simulator : `sim_class`
        A new simulator object, containing a copy of the model in its
        current state.
    """
    from api_builder import Builder
    builder = Builder(copy=True)
    builder(model, dt)

    raise NotImplementedError()

