from copy import deepcopy
from functools import partial
from simulator import Simulator
import numpy as np
from objects import Uniform
from decoders import least_squares


class DotDict(dict):
    """Dictionary with attribute access to items
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        rval = DotDict(map(partial(deepcopy, memo=memo), self.items()))
        assert id(self) not in rval
        memo[id(self)] = rval
        return rval


model_stack = []
_model_groups = 'objects', 'probes', 'connections', 'models'


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
    for group in _model_groups:
        dct[group] = []


def _model_lookup(dct, key):
    try:
        return dct[key]
    except KeyError:
        if 'model_type' in dct:
            for group in _model_groups:
                for obj in dct[group]:
                    if obj.name == key:
                        return obj
        raise


def _rec_model_lookup(dct, key):
    for term in key.split('.'):
        #print('_rec_model_lookup %s  %s' % (term, str(dct.keys())))
        dct = _model_lookup(dct, term)
    return dct


def model(name='root', descr=None):
    new_model = DotDict({'name': name,
                         'descr': descr,
                         'model_type': 'root'})
    _init_model(new_model)
    with defining(new_model):
        # -- leading underscores indicate "internal" objects
        node('_t', output=0)
        node('_steps', output=0)
        probe('_t')
    return defining(new_model)


def submodel(name, descr=None):
    new_model = DotDict({'name': name,
                         'descr': descr,
                         'model_type': 'sub'})
    _init_model(new_model)
    _active_model()['models'].append(new_model)
    return defining(new_model)


def integrator(name, recurrent_tau, dimensions, **ens_args):
    with submodel(name):
        passthrough('in', dimensions=dimensions)
        ensemble('integrator', dimensions=dimensions, **ens_args)
        decode_connect('integrator', 'integrator', filter=recurrent_tau)
        encode('in', 'integrator', filter=recurrent_tau)
        #TODO:  alias('integrator', 'out')


def connect(pre, post, transform=None, filter=None, name=None):
    _active_model()['connections'].append(DotDict({
        'name': name,
        'connection_type': 'connection',
        'pre': pre,
        'post': post,
        'transform': transform,
        'modulatory': False,
        'filter': filter}))


def encode(pre, post, transform=None, filter=None, name=None):
    return connect(pre, post, transform, filter, name)


def decode_connect(pre, post, function=lambda x: x,
                   transform=None,
                   filter=None,
                   decoder_solver=least_squares,
                   name=None):
    _active_model()['connections'].append(DotDict({
        'name': name,
        'connection_type': 'decodedconnection',
        'pre': pre,
        'post': post,
        'function': function,
        'transform': transform,
        'eval_points': None,
        'EVAL_POINTS': 500,
        'modulatory': False,
        'decoder_solver': decoder_solver,
        '_decoders': None,
        'filter': filter}))


def LIF(n_neurons, tau_rc=0.02, tau_ref=.002):
    return DotDict({
        'neuron_type': 'lif',
        'name': 'lif',
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
    raise NotImplementedError()


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
    _active_model()['objects'].append(DotDict({
        'object_type': 'node',
        'name': name,
        'output': output,
        'dimensions': np.asarray(output(0)).size if output else 1
        }))


def passthrough(name, dimensions):
    """Create a Passthrough Node"""
    _active_model()['objects'].append(DotDict({
        'object_type': 'passthrough',
        'name': name,
        'dimensions': dimensions,
        }))


def dimensions(signame, model=None):
    return _get_ens_by_name(signame, model).dimensions


def probe(signame, sample_every=0.001, filter=0.01, name=None):
    _active_model()['probes'].append(DotDict({
        'name': name,
        'signame': signame,
        'filter': filter,
        'sample_every': sample_every,
        }))


def simulator(model, dt=0.001, sim_class=Simulator, seed=0):
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
    builder = Builder(model, dt, copy=True)
    return sim_class(builder, seed=seed)

# -- EOF
