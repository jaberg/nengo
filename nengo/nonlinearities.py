import numpy as np


def lif_rate(dt, J, tau_rc, tau_ref):
    """Compute rates for input current (incl. bias)"""
    old = np.seterr(divide='ignore')
    try:
        j = np.maximum(J - 1, 0.)
        r = dt / (tau_ref + tau_rc * np.log1p(1. / j))
    finally:
        np.seterr(**old)
    return r


def lif_step(dt, J, voltage, refractory_time, spiked, tau_rc, tau_ref):
    # N.B. J here *includes* bias

    # Euler's method
    dV = dt / tau_rc * (J - voltage)

    # increase the voltage, ignore values below 0
    v = np.maximum(voltage + dV, 0)

    # handle refractory period
    post_ref = 1.0 - (refractory_time - dt) / dt

    # set any post_ref elements < 0 = 0, and > 1 = 1
    v *= np.clip(post_ref, 0, 1)

    old = np.seterr(all='ignore')
    try:
        # determine which neurons spike
        # if v > 1 set spiked = 1, else 0
        spiked[:] = (v > 1) * 1.0

        # linearly approximate time since neuron crossed spike threshold
        overshoot = (v - 1) / dV
        spiketime = dt * (1.0 - overshoot)

        # adjust refractory time (neurons that spike get a new
        # refractory time set, all others get it reduced by dt)
        new_refractory_time = spiked * (spiketime + tau_ref) \
                              + (1 - spiked) * (refractory_time - dt)
    finally:
        np.seterr(**old)

    # return an ordered dictionary of internal variables to update
    # (including setting a neuron that spikes to a voltage of 0)

    voltage[:] = v * (1 - spiked)
    refractory_time[:] = new_refractory_time
