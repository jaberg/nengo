import logging

import numpy as np

import nengo
import nengo.helpers
import nengo.api as ng
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestIntegrator(SimulatorTestCase):
    def test_integrator(self):
        model = ng.model()
        inputs = {0:0, 0.2:1, 1:0, 2:-2, 3:0, 4:1, 5:0}
        with model:
            ng.node('input', nengo.helpers.piecewise(inputs))
            tau = 0.1
            ng.integrator('T', tau,  neurons=ng.LIF(100), dimensions=1)
            ng.connect('input', 'T.in', filter=tau)
            ng.ensemble('A', ng.LIF(100), dimensions=1)
            ng.decode_connect('A', 'A', transform=[[1]], filter=tau)
            ng.connect('input', 'A', transform=[[tau]], filter=tau)
            ng.probe('input')
            ng.probe('A', filter=0.01)
            ng.probe('T.integrator', filter=0.01)
        sim = ng.simulator(model)
        sim.run(6.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data('_t')
            plt.plot(t, sim.data('A'), label='Manual')
            plt.plot(t, sim.data('T.integrator'), label='Template')
            plt.plot(t, sim.data('input'), 'k', label='Input')
            plt.legend(loc=0)
            plt.savefig('test_integrator.test_integrator.pdf')
            plt.close()

        self.assertTrue(rmse(sim.data('A'), sim.data('T.integrator')) < 0.2)

if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
