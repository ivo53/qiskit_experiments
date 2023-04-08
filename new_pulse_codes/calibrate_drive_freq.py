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
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit_ibm_provider import IBMProvider

current_dir = os.path.dirname(__file__)
package_path = os.path.abspath(os.path.split(current_dir)[0])
sys.path.insert(0, package_path)

from pulse_types import *


def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

pulse_dict = {
    "gauss": [Gaussian, LiftedGaussian],
    "lor": [Lorentzian, LiftedLorentzian],
    "lor2": [Lorentzian2, LiftedLorentzian2],
    "lor3": [Lorentzian3, LiftedLorentzian3],
    "sq": [Constant, Constant],
    "sech": [Sech, LiftedSech],
    "sech2": [Sech2, LiftedSech2],
    "sin": [Sine, Sine],
    "sin2": [Sine2, Sine2],
    "sin3": [Sine3, Sine3],
    "sin4": [Sine4, Sine4],
    "sin5": [Sine5, Sine5],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-q", "--qubit", default=0, type=int,
        help="The number of the qubit to be used.")
    # parser.add_argument("-l", "--l", default=10, type=float,
    #     help="Parameter l in Rabi oscillations fit")
    # parser.add_argument("-p", "--p", default=0.5, type=float,
    #     help="Parameter p in Rabi oscillations fit")
    # parser.add_argument("-x0", "--x0", default=0.005, type=float,
    #     help="Parameter x0 in Rabi oscillations fit")
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

    pulse_type = args.pulse_type
    # l = args.l
    # p = args.p
    # x0 = args.x0
    sigma = args.sigma
    duration = get_closest_multiple_of_16(args.duration)
    cutoff = args.cutoff
    remove_bg = args.remove_bg
    control_param = args.control_param
    max_experiments_per_job = args.max_experiments_per_job
    num_shots = args.num_shots
    span = args.span
    num_experiments = args.num_experiments
    qubit = args.qubit
    backend = args.backend
    backend_name = backend
    backend = "ibm_" + backend \
        if backend in ["perth", "lagos", "nairobi", "oslo"] \
            else "ibmq_" + backend
    
    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    calib_dir = os.path.join(file_dir, "calibrations")
    save_dir = os.path.join(calib_dir, current_date)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    data_folder = os.path.join(file_dir, "data", backend_name, "calibration", current_date)
    # if not os.path.isdir(data_folder):
    #     os.mkdir(data_folder)
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    params_file = os.path.join(calib_dir, "actual_params.csv")
    if os.path.isfile(params_file):
        param_df = pd.read_csv(params_file)
    df = param_df[param_df.apply(
            lambda row: row["pulse_type"] == pulse_type and \
                row["duration"] == duration and \
                row["sigma"] == sigma and \
                row["rb"] == remove_bg, axis=1)]
    print(df)
    if df.shape[0] != 1:
        raise ValueError("More than one identical entry found!")
    ser = df.iloc[0]
    l = ser.at["l"]
    p = ser.at["p"]
    x0 = ser.at["x0"]
    print(l, p, x0)
    # unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
    GHz = 1.0e9 # Gigahertz
    MHz = 1.0e6 # Megahertz
    us = 1.0e-6 # Microseconds
    ns = 1.0e-9 # Nanoseconds
    mem_slot = 0

    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)

    provider = IBMQ.load_account()
    backend = provider.get_backend(backend)
    # backend = provider.get_backend("ibmq_qasm_simulator")
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()
    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    num_qubits = backend_config.n_qubits
    if num_qubits <= qubit:
        raise ValueError("Qubit index out of range for this system.")
    span = span * center_frequency_Hz
    frequencies = np.arange(center_frequency_Hz - span / 2, 
                            center_frequency_Hz + span / 2,
                            span / num_experiments)
    dt = backend_config.dt
    # print(dt)
    amp = -np.log(1 - np.pi / l) / p + x0

    freq = Parameter('freq')
    with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_freq") as sched:
        dur_dt = duration
        pulse.set_frequency(freq, drive_chan)
        if pulse_type == "sq" or "sin" in pulse_type:
            pulse_played = pulse_dict[pulse_type][remove_bg](
                duration=duration,
                amp=amp,
                name=pulse_type
            )
        elif pulse_type == "gauss":
            pulse_played = pulse_dict[pulse_type][remove_bg](
                duration=duration,
                amp=amp,
                name=pulse_type,
                sigma=sigma / np.sqrt(2),
            )
        elif pulse_type in ["lor", "lor2", "lor3"]:
            pulse_played = pulse_dict[pulse_type][remove_bg](
                duration=duration,
                amp=amp,
                name=pulse_type,
                sigma=sigma,
            )
        else:
            pulse_played = pulse_dict[pulse_type][remove_bg](
                duration=duration,
                amp=amp,
                name=pulse_type,
                sigma=sigma,
            )
        pulse.play(pulse_played, drive_chan)

    # Create gate holder and append to base circuit
    custom_gate = Gate("pi_pulse", 1, [freq])
    base_circ = QuantumCircuit(num_qubits, 1)
    base_circ.append(custom_gate, [qubit])
    base_circ.measure(qubit, 0)
    base_circ.add_calibration(custom_gate, (qubit,), sched, [freq])
    circs = [base_circ.assign_parameters(
            {freq: f},
            inplace=False
        ) for f in frequencies]

    job_manager = IBMQJobManager()
    job = job_manager.run(
        circs,
        backend=backend,
        name="Frequency calibration",
        max_experiments_per_job=max_experiments_per_job,
        shots=num_shots
    )
    frequency_sweep_results = job.results()

    sweep_values = []
    for i in range(len(circs)):
        counts = frequency_sweep_results.get_counts(i)["1"]
        sweep_values.append(counts / num_shots)


    frequencies_GHz = frequencies / GHz

    data = {"frequency_ghz": frequencies_GHz, "transition_probability": (sweep_values)}
    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(
            data_folder, 
            date.strftime("%H%M%S") + f"_calibration.csv"
        ),
        index=False
    )
    plt.figure(1)
    plt.scatter(frequencies_GHz, (sweep_values), color='black') # plot real part of sweep values
    plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    plt.title("Drive Frequency Calibration Curve")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Transition Probability")
    plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_dur_{duration}_s_{int(sigma)}_frequency_sweep.png"))
    # plt.show()

    ## fit curve


    def fit_function(x_values, y_values, function, init_params):
        fitparams, conv = curve_fit(function, x_values, y_values, init_params)#, maxfev=100000)
        y_fit = function(x_values, *fitparams)
        
        return fitparams, y_fit

    fit_params, y_fit = fit_function(frequencies_GHz,
                                    np.real(sweep_values), 
                                    lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                    [0.3, 4.975, 0.2, 0] # initial parameters for curve_fit
                                    )
    plt.figure(2)
    plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
    plt.plot(frequencies_GHz, y_fit, color='red')
    plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    plt.title("Fitted Drive Frequency Calibration Curve")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Measured Signal [a.u.]")
    plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_frequency_sweep_fitted.png"))
    plt.show()

    A, rough_qubit_frequency, B, C = fit_params
    rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
    print(f"We've updated our qubit frequency estimate from "
        f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")
