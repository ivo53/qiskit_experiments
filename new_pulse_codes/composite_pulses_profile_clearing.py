import os
import sys
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
# from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit_ibm_provider import IBMProvider

current_dir = os.path.dirname(__file__)
package_path = os.path.abspath(os.path.split(current_dir)[0])
sys.path.insert(0, package_path)

from utils.run_jobs import run_jobs
import common.pulse_types as pt

pulse_dict = {
    "gauss": [pt.Gaussian, pt.LiftedGaussian],
    "lor": [pt.Lorentzian, pt.LiftedLorentzian],
    "lor2": [pt.Lorentzian2, pt.LiftedLorentzian2],
    "lor3": [pt.Lorentzian3, pt.LiftedLorentzian3],
    "sq": [pt.Constant, pt.Constant],
    "sech": [pt.Sech, pt.LiftedSech],
    "sech2": [pt.Sech2, pt.LiftedSech2],
    "sin": [pt.Sine, pt.Sine],
    "sin2": [pt.Sine2, pt.Sine2],
    "sin3": [pt.Sine3, pt.Sine3],
    "sin4": [pt.Sine4, pt.Sine4],
    "sin5": [pt.Sine5, pt.Sine5],
    "demkov": [pt.Demkov, pt.LiftedDemkov],
}

COMP_PARAMS = {
    3: {"alpha": 0.6399, "phases": [1.8442, 1.0587]},
    5: {"alpha": 0.45, "phases": [1.9494, 0.5106, 1.31]},
    7: {"alpha": 0.2769, "phases": [1.6803, 0.2724, 0.8255, 1.6624]},
    9: {"alpha": 0.2947, "phases": [0.2711, 1.1069, 1.5283, 0.1283, 0.9884]},
    11: {"alpha": 0.2985, "phases": [1.7377, 0.1651, 0.9147, 0.1510, 0.9331, 1.6415]},
    13: {"alpha": 0.5065, "phases": [0.0065, 1.7755, 0.7155, 0.5188, 0.2662, 1.2251, 1.3189]},
    15: {"alpha": 0.3213, "phases": [1.2316, 0.9204, 0.2043, 1.9199, 0.8910, 0.7381, 1.9612, 1.3649]},
}

def get_calib_params(
    backend, pulse_type, 
    sigma, duration,
    remove_bg
):
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    calib_dir = os.path.join(file_dir, "calibrations", backend)
    params_file = os.path.join(calib_dir, "actual_params.csv")
    
    if os.path.isfile(params_file):
        param_df = pd.read_csv(params_file)
    df = param_df[param_df.apply(
            lambda row: row["pulse_type"] == pulse_type and \
                row["duration"] == duration and \
                row["sigma"] == sigma and \
                row["rb"] == remove_bg, axis=1)]
    # print(df)
    idx = 0 
    if df.shape[0] > 1:
        idx = input("More than one identical calibrations found! "
                    "Which do you want to use? ")
    elif df.shape[0] < 1:
        raise ValueError("No entry found!")
    ser = df.iloc[idx]
    l = ser.at["l"]
    p = ser.at["p"]
    x0 = ser.at["x0"]

    return l, p, x0

def initialize_backend(backend):
    backend_full_name = "ibm_" + backend 
        # if backend in ["perth", "lagos", "nairobi", "oslo"] \
        #     else "ibmq_" + backend
    GHz = 1.0e9 # Gigahertz
    MHz = 1.0e6 # Megahertz
    us = 1.0e-6 # Microseconds
    ns = 1.0e-9 # Nanoseconds
    qubit = 0
    mem_slot = 0

    drive_chan = pulse.DriveChannel(qubit)
    # meas_chan = pulse.MeasureChannel(qubit)
    # acq_chan = pulse.AcquireChannel(qubit)
    
    backend_name = backend
    # provider = IBMQ.load_account()
    backend = IBMProvider().get_backend(backend_full_name)
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]# 4962284031.287086 Hz
    num_qubits = backend_config.n_qubits

    q_freq = [backend_defaults.qubit_freq_est[q] for q in range(num_qubits)]
    dt = backend_config.dt

    return backend, drive_chan, num_qubits, q_freq

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
    amp_span,
    duration, sigma, pulse_type, remove_bg,
    num_exp=10, N_max=100,
    N_interval=2, max_exp_per_job=50,
    num_shots=1024, backend="manila",
    l=100, p=0.5, x0=0,
    closest_amp=None
):
    backend, drive_chan, num_qubits, q_freq = initialize_backend(backend)
    if closest_amp is None:
        closest_amp = -np.log(1 - np.pi / l) / p + x0
    amplitudes = np.linspace(
        closest_amp - amp_span / 2,
        closest_amp + amp_span / 2,
        num_exp
    )
    Ns = np.arange(0, N_max + N_interval / 2, N_interval, dtype="int64")

    def add_circ(amp, duration, sigma, freq, Ns, qubit=0):
        amp = Parameter("amp")
        phi = Parameter("phi")
        with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area_with_N_pulses") as sched:
            dur_dt = duration
            pulse.set_frequency(freq, drive_chan)
            if pulse_type == "sq" or "sin" in pulse_type:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp * np.exp(1j * phi * np.pi),
                    name=pulse_type
                )
            elif pulse_type == "gauss":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp * np.exp(1j * phi * np.pi),
                    name=pulse_type,
                    sigma=sigma / np.sqrt(2)
                )
            elif pulse_type in ["lor", "lor2", "lor3"]:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp * np.exp(1j * phi * np.pi),
                    name=pulse_type,
                    sigma=sigma,
                )
            else:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp * np.exp(1j * phi * np.pi),
                    name=pulse_type,
                    sigma=sigma,
                )
            pulse.play(pulse_played, drive_chan)
        base_circ = QuantumCircuit(num_qubits, len(Ns))
        custom_gate = Gate("N_pulses", 1, [])
        for _ in range(N):
            base_circ.append(custom_gate, [qubit])
        base_circ.measure(qubit, 0)
        base_circ.add_calibration(custom_gate, (qubit,), sched, [])
        return base_circ


    circs = [add_circ(a, duration, sigma, q_freq[qubit], Ns) for a in amplitudes]

    max_experiments_per_job = 100
    num_shots = 1024

    sweep_values, job_ids = run_jobs(circs, backend, duration *len(Ns)  , num_shots_per_exp=num_shots)

    print(Ns, np.array(sweep_values).reshape(np.round(N_max / N_interval + 1).astype(np.int64), len(amplitudes)))
    return Ns, np.array(sweep_values).reshape(np.round(N_max / N_interval + 1).astype(np.int64), len(amplitudes))


def find_least_variation(x, ys, init_params=[1, 1], bounds=[[-100, -100],[100, 100]]):
    # y_fits = []
    # for y in ys:
    #     par, y_fit = fit_function(x, y, linear_func, init_params, bounds)
    #     y_fits.append(y_fit)
    # y_fits = np.array(y_fits)
    # diff = np.mean(np.abs(ys - y_fits[:, None]), axis=1)
    diff = np.sum(ys, 1)
    closest_idx = np.argmin(diff)
    return closest_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-q", "--qubit", default=0, type=int,
        help="The number of the qubit to be used.")
    parser.add_argument("-a", "--closest_amp", default=None, type=float,
        help="First amp to use (should be close to PI area).")   
    parser.add_argument("-s", "--sigma", default=180, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-T", "--duration", default=2256, type=int,
        help="Pulse duration parameter")
    parser.add_argument("-N", "--N_max", default=100, type=int,
        help="Maximum N number of pulses to use in the experiment.")
    parser.add_argument("-Ni", "--N_interval", default=2, type=int,
        help="Interval between N number of pulses.")
    parser.add_argument("-rb", "--remove_bg", default=1, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=2048, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-ne", "--num_experiments", default=100, type=int,
        help="Number of experiments with different detuning each.")
    parser.add_argument("-b", "--backend", default="manila", type=str,
        help="The name of the backend to use in the experiment (one of perth, lagos, nairobi, \
        oslo, jakarta, manila, quito, belem, lima).")
    parser.add_argument("-sp", "--span", default=0.01, type=float,
        help="The span of the amplitude sweep as a fraction of the amplitude value.")
    parser.add_argument("-iter", "--num_iterations", default=3, type=int,
        help="The number of optimization iterations of the algorithm.")
    args = parser.parse_args()

    pulse_type = args.pulse_type
    qubit = args.qubit
    closest_amp = args.closest_amp
    sigma = args.sigma
    duration = args.duration
    N_max = args.N_max
    N_interval = args.N_interval
    remove_bg = args.remove_bg
    max_experiments_per_job = args.max_experiments_per_job
    num_shots = args.num_shots
    num_exp = args.num_experiments
    backend = args.backend
    amp_span = args.span
    num_iterations = args.num_iterations

    l, p, x0 = get_calib_params(
        backend, pulse_type, 
        sigma, duration,
        remove_bg
    )
    for i in range(num_iterations):
        x, ys = run_check(
            amp_span / (10**i), 
            duration, sigma, 
            pulse_type, remove_bg,
            num_exp=num_exp,
            N_max=N_max, N_interval=N_interval,
            max_exp_per_job=max_experiments_per_job,
            num_shots=num_shots, 
            backend=backend,
            l=l, p=p, x0=x0,
            closest_amp=closest_amp)
        index = find_least_variation(x, ys)
        closest_amp = x[index]
