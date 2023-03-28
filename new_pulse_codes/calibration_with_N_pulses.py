import os
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qiskit import (
    QuantumCircuit, 
    QuantumRegister, 
    ClassicalRegister, 
    pulse, 
    IBMQ
) 
# This is where we access all of our Pulse features!
from qiskit.circuit import Parameter, Gate
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.providers.ibmq.managed import IBMQJobManager

pulse_dict = {
    "gauss": pulse_lib.Gaussian,
    "lor": pulse_lib.Lorentzian,
    "lor2": pulse_lib.LorentzianSquare,
    "lor3": pulse_lib.LorentzianCube,
    "sq": pulse_lib.Constant,
    "sech": pulse_lib.Sech,
    "sech2": pulse_lib.SechSquare,
    "sin": pulse_lib.Sine,
    "sin2": pulse_lib.SineSquare,
    "sin3": pulse_lib.SineCube,
    "sin4": pulse_lib.SineFourthPower,
    "sin5": pulse_lib.SineFifthPower,
}

backend_full_name = "ibmq_manila"
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
qubit = 0
mem_slot = 0

drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)

provider = IBMQ.load_account()
backend = provider.get_backend(backend_full_name)
print(f"Using {backend_name} backend.")
backend_defaults = backend.defaults()
backend_config = backend.configuration()

center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]# 4962284031.287086 Hz
q_freq = [backend_defaults.qubit_freq_est[q] for q in range(num_qubits)]
dt = backend_config.dt
num_qubits = backend_config.n_qubits

def fit_function(
    x_values, 
    y_values, 
    function, 
    init_params,
    bounds
):
    fitparams, conv = curve_fit(
        function, 
        x_values, 
        y_values, 
        init_params, 
        maxfev=100000, 
        bounds=bounds
    )
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

def linear_func(x, a, b):
    return a * x + b

def run_check(
    closest_amp, amp_span, 
    duration, sigma, pulse_type, remove_bg,
    num_exp=10, N_max=100, N_interval=2):
    amplitudes = np.linspace(
        closest_amp - amp_span / 2,
        closest_amp + amp_span / 2,
        num_exp
    )
    base_circ = QuantumCircuit(num_qubits, 1)
    amp = Parameter("amp")
    N = Parameter("N")
    with pulse.build(backend=backend, default_alignment='sequential', name=f"calibration_with_N_pulses") as sched:
        pulse.set_frequency(q_freq[qubit], drive_chan)
        if pulse_type == "sq" or "sin" in pulse_type:
            pulse_played = pulse_dict[pulse_type](
                duration=duration,
                amp=amp,
                name=pulse_type
            )
        elif pulse_type == "gauss":
            pulse_played = pulse_dict[pulse_type](
                duration=duration,
                amp=amp,
                name=pulse_type,
                sigma=sigma / np.sqrt(2),
                zero_ends=remove_bg
            )
        elif pulse_type in ["lor", "lor2", "lor3"]:
            pulse_played = pulse_dict[pulse_type](
                duration=duration,
                amp=amp,
                name=pulse_type,
                gamma=sigma,
                zero_ends=remove_bg
            )
        else:
            pulse_played = pulse_dict[pulse_type](
                duration=duration,
                amp=amp,
                name=pulse_type,
                sigma=sigma,
                zero_ends=remove_bg
            )
        for _ in range(N):
            pulse.play(pulse_played, drive_chan)

    custom_gate = Gate("N_pulses", 1, [amp, N])
    base_circ.append(custom_gate, [qubit])
    base_circ.measure(qubit, 0)
    base_circ.add_calibration(custom_gate, (qubit,), sched, [amp, N])
    circs = [
        base_circ.assign_parameters(
            {amp: a, N: n},
            inplace=False
        ) for a in amplitudes for n in range(0, N_max + 1, N_interval)
    ]

    max_experiments_per_job = 100
    num_shots = 1024

    job_manager = IBMQJobManager()
    job = job_manager.run(
        circs,
        backend=backend,
        name="N pulses calibration",
        max_experiments_per_job=max_experiments_per_job,
        shots=num_shots
    )
    frequency_sweep_results = job.results()

    sweep_values = []
    for i in range(len(circs)):
        counts = frequency_sweep_results.get_counts(i)["1"]
        sweep_values.append(counts / num_shots)
    print(N, np.array(sweep_values).reshape(np.round(N_max / N_interval + 1).astype(np.int64), len(amplitudes)))
    return N, np.array(sweep_values).reshape(np.round(N_max / N_interval + 1).astype(np.int64), len(amplitudes))


def find_least_variation(x, ys, init_params=[1, 1], bounds=[[-100, -100],[100, 100]]):
    y_fits = []
    for y in ys:
        par, y_fit = fit_function(x, y, linear_func, init_params, bounds)
        y_fits.append(y_fit)
    y_fits = np.array(y_fits)
    diff = np.mean(np.abs(ys - y_fits), axis=1)
    closest_idx = np.argmin(diff)
    return closest_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-q", "--qubit", default=0, type=int,
        help="The number of the qubit to be used.")
    parser.add_argument("-s", "--sigma", default=180, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-T", "--duration", default=2256, type=int,
        help="Lorentz duration parameter")
    parser.add_argument("-c", "--cutoff", default=0.5, type=float,
        help="Cutoff parameter in PERCENT of maximum amplitude of Lorentzian")
    parser.add_argument("-rb", "--remove_bg", default=1, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-cp", "--control_param", default="width", type=str,
        help="States whether width or duration is the controlled parameter")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=2048, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-ne", "--num_experiments", default=100, type=int,
        help="Number of experiments with different detuning each.")
    parser.add_argument("-b", "--backend", default="manila", type=str,
        help="The name of the backend to use in the experiment (one of perth, lagos, nairobi, \
        oslo, jakarta, manila, quito, belem, lima).")
    parser.add_argument("-sp", "--span", default=0.005, type=float,
        help="The span of the detuning sweep as a fraction of the driving frequency.")
    args = parser.parse_args()
