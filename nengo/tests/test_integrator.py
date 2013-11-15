import logging

import numpy as np

import nengo
import nengo.helpers
import nengo.api as ng
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest


logger = logging.getLogger(__name__)

class TestIntegrator(SimulatorTestCase):
    def test_integrator(self):
        model = {}
        inputs = {0:0, 0.2:1, 1:0, 2:-2, 3:0, 4:1, 5:0}
        with ng.defining(model):
            ng.node('input', nengo.helpers.piecewise(inputs))
            tau = 0.1
            ng.integrator('T', tau,  neurons=ng.LIF(100), dimensions=1)
            ng.connect('Input', 'T.in', filter=tau)
            ng.ensemble('A', ng.LIF(100), dimensions=1)
            ng.connect('A', 'A', transform=[[1]], filter=tau)
            ng.connect('Input', 'A', transform=[[tau]], filter=tau)
            ng.probe('Input')
            ng.probe('A', filter=0.01)
            ng.probe('T.Integrator', filter=0.01)
        print model
        ng.simulator(model)
        sim = model.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(6.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(model.t)
            plt.plot(t, sim.data('A'), label='Manual')
            plt.plot(t, sim.data('T.Integrator'), label='Template')
            plt.plot(t, sim.data('Input'), 'k', label='Input')
            plt.legend(loc=0)
            plt.savefig('test_integrator.test_integrator.pdf')
            plt.close()

        self.assertTrue(rmse(sim.data('A'), sim.data('T.Integrator')) < 0.2)

if __name__ == "__main__":
    nengo.log(debug=True, path='log.txt')
    unittest.main()
