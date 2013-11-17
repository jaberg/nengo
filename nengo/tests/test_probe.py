import logging
import time
import numpy as np

import nengo
from nengo.tests.helpers import SimulatorTestCase, unittest
from nengo.api import model, LIF, ensemble, probe, simulator

runtime_logger = logging.getLogger('nengo.tests.runtime')

class TestProbe(SimulatorTestCase):

    def test_long_name(self):
        m = model(descr='test_long_name')
        with m:
            name = ("This is an extremely long name that will test "
                    "if we can access sim data with long names")
            ensemble(name, LIF(10), 1)
            probe(name)

        sim = simulator(m)
        sim.run(0.01)

        self.assertIsNotNone(sim.data(name))

    def test_multirun(self):
        """Test probing the time on multiple runs"""
        rng = np.random.RandomState(2239)

        # set rtol a bit higher, since OCL model.t accumulates error over time
        rtol = 1e-4

        m = model("Multi-run")
        sim = simulator(m)
        dt = sim.builder.dt

        # t_stops = [0.123, 0.283, 0.821, 0.921]
        t_stops = dt * rng.randint(low=100, high=2000, size=10)

        t_sum = 0
        for ti in t_stops:
            sim.run(ti)
            sim_t = sim.data('_t').flatten()
            t = dt * np.arange(len(sim_t)) + dt
            self.assertTrue(np.allclose(sim_t, t, rtol=rtol))

            t_sum += ti
            self.assertTrue(np.allclose(sim_t[-1], t_sum, rtol=rtol))

    def test_dts(self):
        """Test probes with different sampling times."""

        n = 10

        rng = np.random.RandomState(48392)
        dts = 0.001 * rng.randint(low=1, high=100, size=n)
        # dts = 0.001 * np.hstack([2, rng.randint(low=1, high=100, size=n-1)])

        def input_fn(t):
            return [1,2,3,4,5,6,7,8,9]

        model = nengo.Model('test_probe_dts', seed=2891)
        pops = []
        for i, dt in enumerate(dts):
            xi = model.make_node('x%d' % i, output=input_fn)
            model.probe(xi, sample_every=dt)
            pops.append(xi)

        sim = model.simulator(sim_class=self.Simulator)
        simtime = 2.483
        # simtime = 2.484
        dt = sim.model.dt

        timer = time.time()
        sim.run(simtime)
        timer = time.time() - timer
        runtime_logger.debug(
            "Ran %(n)s probes for %(simtime)s sec simtime in %(timer)0.3f sec"
            % locals())

        for i, pop in enumerate(pops):
            t = dt * np.arange(int(np.ceil(simtime / dts[i])))
            x = np.asarray(map(input_fn, t))
            y = sim.data(pop)
            self.assertTrue(len(x) == len(y))
            self.assertTrue(np.allclose(y[1:], x[:-1])) # 1-step delay


    def test_large(self):
        """Test with a lot of big probes. Can also be used for speed."""

        n = 10

        # rng = np.random.RandomState(48392)
        # input_val = rng.normal(size=100).tolist()
        def input_fn(t):
            return [1,2,3,4,5,6,7,8,9]
            # return input_val
            # return [np.sin(t), np.cos(t)]

        model = nengo.Model('test_large_probes', seed=3249)

        pops = []
        for i in xrange(n):
            xi = model.make_node('x%d' % i, output=input_fn)
            model.probe(xi)
            pops.append(xi)
            # Ai = m.make_ensemble('A%d' % i, nengo.LIF(n_neurons), 1)
            # m.connect(xi, Ai)
            # m.probe(Ai, filter=0.1)

        sim = model.simulator(sim_class=self.Simulator)
        simtime = 2.483
        dt = sim.model.dt

        timer = time.time()
        sim.run(simtime)
        timer = time.time() - timer
        runtime_logger.debug(
            "Ran %(n)s probes for %(simtime)s sec simtime in %(timer)0.3f sec"
            % locals())

        t = dt * np.arange(int(np.round(simtime / dt)))
        x = np.asarray(map(input_fn, t))
        for pop in pops:
            y = sim.data(pop)
            self.assertTrue(np.allclose(y[1:], x[:-1])) # 1-step delay


if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
