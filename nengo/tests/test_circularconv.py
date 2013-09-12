import numpy as np

import nengo
from nengo.core import Constant, Signal
from nengo.templates import EnsembleArray
from nengo.networks.circularconvolution import circconv, CircularConvolution
from nengo.tests.helpers import (
    SimulatorTestCase, unittest, assert_allclose, Plotter, rmse)

import logging
logger = logging.getLogger(__name__)

class TestCircularConv(SimulatorTestCase):
    def test_helpers(self):
        """Test the circular convolution helper functions in Numpy"""
        rng = np.random.RandomState(43232)

        dims = 1000
        invert_a = True
        invert_b = False
        x = rng.randn(dims)
        y = rng.randn(dims)
        z0 = circconv(x, y, invert_a=invert_a, invert_b=invert_b)

        dims2 = 2*dims - (2 if dims % 2 == 0 else 1)
        inA = CircularConvolution._input_transform(
            dims, first=True, invert=invert_a)
        inB = CircularConvolution._input_transform(
            dims, first=False, invert=invert_b)
        outC = CircularConvolution._output_transform(dims)

        XY = np.zeros((dims2,2))
        XY += np.dot(inA.reshape(dims2, 2, dims), x)
        XY += np.dot(inB.reshape(dims2, 2, dims), y)

        C = XY[:,0] * XY[:,1]
        z1 = np.dot(outC, C)

        assert_allclose(self, logger, z0, z1)

    # def test_direct(self):
    #     """Test direct-mode circular convolution"""

    #     dims = 1000

    #     rng = np.random.RandomState(8238)
    #     a = rng.randn(dims)
    #     b = rng.randn(dims)
    #     c = circconv(a, b)

    #     m = nengo.Model("")
    #     A = m.add(Constant(value=a))
    #     B = m.add(Constant(value=b))
    #     C = m.add(Signal(n=dims, name="C"))

    #     DirectCircularConvolution(m, A, B, C)

    #     sim = m.simulator(sim_class=self.Simulator)
    #     sim.run_steps(10)
    #     c2 = sim.signals[C]

    #     # assert_allclose(self, logger, c, c2, atol=1e-5, rtol=1e-5)
    #     assert_allclose(self, logger, c, c2, atol=1e-5, rtol=1e-3)

    def _test_circularconv(self, dims=5, neurons_per_product=150):
        rng = np.random.RandomState(42342)

        n_neurons = neurons_per_product * dims
        n_neurons_d = 2 * neurons_per_product * (
            2*dims - (2 if dims % 2 == 0 else 1))
        radius = 1

        a = rng.normal(scale=np.sqrt(1./dims), size=dims)
        b = rng.normal(scale=np.sqrt(1./dims), size=dims)
        c = circconv(a, b)
        self.assertTrue(np.abs(a).max() < radius)
        self.assertTrue(np.abs(b).max() < radius)
        self.assertTrue(np.abs(c).max() < radius)

        ### model
        model = nengo.Model("circular convolution")
        inputA = model.make_node("inputA", output=a)
        inputB = model.make_node("inputB", output=b)
        A = model.add(EnsembleArray(
            'A', nengo.LIF(n_neurons), dims, radius=radius))
        B = model.add(EnsembleArray(
            'B', nengo.LIF(n_neurons), dims, radius=radius))
        C = model.add(EnsembleArray(
            'C', nengo.LIF(n_neurons), dims, radius=radius))
        D = model.add(CircularConvolution(
                'D', neurons=nengo.LIF(n_neurons_d),
                dimensions=A.dimensions, radius=radius))

        inputA.connect_to(A)
        inputB.connect_to(B)
        A.connect_to(D.A)
        B.connect_to(D.B)
        D.connect_to(C)

        model.probe(A, filter=0.03)
        model.probe(B, filter=0.03)
        model.probe(C, filter=0.03)
        model.probe(D, filter=0.03)

        # check FFT magnitude
        d = np.dot(D.transformA, a) + np.dot(D.transformB, b)
        self.assertTrue(np.abs(d).max() < radius)

        ### simulation
        sim = model.simulator(sim_class=self.Simulator)
        sim.run(1.0)

        t = sim.data(model.t).flatten()
        with Plotter(self.Simulator) as plt:
            def plot(sim, a, A, title=""):
                a_ref = np.tile(a, (len(t), 1))
                a_sim = sim.data(A)
                colors = ['b', 'g', 'r', 'c', 'm', 'y']
                for i in xrange(dims):
                    plt.plot(t, a_ref[:,i], '--', color=colors[i])
                    plt.plot(t, a_sim[:,i], '-', color=colors[i])
                plt.title(title)

            plt.subplot(221)
            plot(sim, a, A, title="A")
            plt.subplot(222)
            plot(sim, b, B, title="B")
            plt.subplot(223)
            plot(sim, c, C, title="C")
            plt.subplot(224)
            plot(sim, d, D, title="D")
            plt.savefig('test_circularconv.test_circularconv.pdf')
            plt.close()

        ### results
        a_sim = sim.data(A)[t > 0.5].mean(axis=0)
        b_sim = sim.data(B)[t > 0.5].mean(axis=0)
        c_sim = sim.data(C)[t > 0.5].mean(axis=0)
        d_sim = sim.data(D)[t > 0.5].mean(axis=0)

        rtol, atol = 0.05, 0.01
        self.assertTrue(np.allclose(a, a_sim, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(b, b_sim, rtol=rtol, atol=atol))
        rtol, atol = 0.1, 0.05
        self.assertTrue(np.allclose(d, d_sim, rtol=rtol, atol=atol))
        self.assertTrue(np.allclose(c, c_sim, rtol=rtol, atol=atol))

    def test_small(self):
        return self._test_circularconv(dims=4, neurons_per_product=128)

    def test_med(self):
        return self._test_circularconv(dims=50, neurons_per_product=128)

    # Too much for my poor machine
    # def test_large(self):
    #     return self._test_circularconv(dims=512, neurons_per_product=128)


if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()
