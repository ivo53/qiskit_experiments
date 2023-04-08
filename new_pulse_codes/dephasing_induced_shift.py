import os
import sys
import pickle
import argparse
import itertools
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from qiskit import pulse, IBMQ, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, Gate
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor
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
    numplus8 = (num + 8).astype(int)
    return numplus8 - (numplus8 % 16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    # parser.add_argument("-G", "--lorentz_G", default=180, type=float,
    #     help="Lorentz width (gamma) parameter")    
    parser.add_argument("-is", "--initial_sigma", default=32, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-fs", "--final_sigma", default=32000, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-nsig", "--num_sigma", default=100, type=int,
        help="Pulse width (sigma) parameter")
    # parser.add_argument("-T", "--duration", default=2256, type=int,
    #     help="Pulse duration parameter")
    # parser.add_argument("-c", "--cutoff", default=0.5, type=float,
    #     help="Cutoff parameter in PERCENT of maximum amplitude of Lorentzian")
    parser.add_argument("-rb", "--remove_bg", default=0, type=int,
        help="Whether to drop the background (and thus discontinuities) "
            "of the pulse (0 or 1).")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=2048, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-b", "--backend", default="manila", type=str,
        help="The name of the backend to use in the experiment (one of nairobi, \
        oslo, manila, quito, belem, lima).")
    parser.add_argument("-ia", "--initial_amp", default=0.001, type=float,
        help="The initial amplitude of the area sweep.")
    parser.add_argument("-fa", "--final_amp", default=0.2, type=float,
        help="The final amplitude of the area sweep.")
    parser.add_argument("-ne", "--num_experiments", default=100, type=int,
        help="The number of amplitude datapoints to evaluate.")
    parser.add_argument("-l", "--l", default=100, type=float,
        help="The initial value of the l fit param.")
    parser.add_argument("-p", "--p", default=0.5, type=float,
        help="The initial value of the p fit param.")
    args = parser.parse_args()
    initial_sigma = args.initial_sigma
    final_sigma = args.final_sigma
    num_sigma = args.num_sigma
    initial_amp = args.initial_amp
    final_amp = args.final_amp
    num_shots_per_exp = args.num_shots
    num_exp = args.num_experiments
    backend = args.backend
    max_experiments_per_job = args.max_experiments_per_job
    remove_bg = bool(args.remove_bg)
    pulse_type = args.pulse_type
    l = args.l
    p = args.p
    backend_name = backend
    backend = "ibm_" + backend \
        if backend in ["perth", "lagos", "nairobi", "oslo"] \
            else "ibmq_" + backend
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
    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    calib_dir = os.path.join(file_dir, "calibrations")
    save_dir = os.path.join(file_dir, "plots", backend_name, "dephasing_induced_shift", current_date)
    data_folder = os.path.join(file_dir, "data", backend_name, "dephasing_induced_shift", current_date)
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    # unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
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
    backend = provider.get_backend(backend)
    # backend_name = str(backend)
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    dt = backend_config.dt
    num_qubits = backend_config.n_qubits

    rough_qubit_frequency = center_frequency_Hz # 4962284031.287086 Hz

    ## set params
    fit_crop = 1#.8
    amplitudes = np.linspace(
        initial_amp, 
        final_amp, 
        num_exp
    )
    fit_crop_parameter = int(fit_crop * len(amplitudes))

    sigmas = np.linspace(initial_sigma, final_sigma, num_sigma)
    durations = get_closest_multiple_of_16(sigmas.astype("int")) if "sin" in pulse_type else get_closest_multiple_of_16(sigmas.astype("int") * 8)

    print(f"The resonant frequency is assumed to be {np.round(rough_qubit_frequency / GHz, 5)} GHz.")
    print(f"The dephasing experiment will start from amp {amplitudes[0]} "
    f"and end at {amplitudes[-1]} with approx step {np.round((final_amp - initial_amp)/num_exp,3)},"
    f"from sigma {sigmas[0]} "
    f"and till {sigmas[-1]} with approx step {np.round((final_sigma - initial_sigma)/num_sigma,0)},"
    f"from duration {durations[0]} "
    f"and till {durations[-1]} with approx step {np.round(8 * (final_sigma - initial_sigma)/num_sigma,0)}.")

    amp = Parameter('amp')
    duration = Parameter('duration')
    sigma = Parameter('sigma')
    with pulse.build(backend=backend, default_alignment='sequential', name="dephasing_inducing_gate") as sched:
        pulse.set_frequency(rough_qubit_frequency, drive_chan)
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
    
    long_gate = Gate("long_gate", 1, params=[sigma, duration, amp])
    base_circ = QuantumCircuit(num_qubits, 1)    
    base_circ.append(long_gate, [qubit])
    base_circ.measure(qubit, 0)
    base_circ.add_calibration(long_gate, (qubit,), sched, [sigma, duration, amp])
    circs = [
        [
            base_circ.assign_parameters(
                {sigma: s, duration: d, amp: a},
                inplace=False
            ) for s, d in zip(sigmas, durations)
        ] for a in amplitudes
    ]
    circs = list(itertools.chain.from_iterable(circs))

    job_manager = IBMQJobManager()
    long_job = job_manager.run(
        circs,
        backend=backend,
        shots=num_shots_per_exp,
        max_experiments_per_job=max_experiments_per_job
    )
    long_results = long_job.results()

    long_values = []

    for i in range(len(circs)):
        try:
            counts = long_results.get_counts(i)["1"]
        except KeyError:
            counts = 0
        long_values.append(counts / num_shots_per_exp)


    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    project_dir = os.path.join(file_dir, "plots", backend_name, "dephasing_induced_shift")
    save_dir = os.path.join(project_dir, current_date)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    data_folder = os.path.join(file_dir, "data", backend_name, "dephasing_induced_shift", current_date)
    # if not os.path.isdir(data_folder):
    #     os.mkdir(data_folder)
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    long_values = np.array(long_values).reshape(len(amplitudes), len(sigmas))
    amps = np.tile(amplitudes, (len(sigmas)))
    ss = np.tile(amplitudes, (len(amplitudes)))
    results_df = pd.DataFrame({"amps": amps, "sigmas": ss, "probs": long_values})

    results_df.to_pickle(os.path.join(data_folder, f"deph_induced_shift_amp-{date.strftime('%H%M%S')}.pkl"))

    for i, am in enumerate(amplitudes):
        plt.figure(i)
        plt.plot(sigmas, long_values[i], "bx")
        plt.xlabel("Pulse width $\sigma$ [dt]")
        plt.ylabel("Transition Probability")
        plt.title(f"Dephasing-induced decoherence for amplitude {am.round(3)}")
        plt.savefig(os.path.join(save_dir, f"deph_induced_shift_amp-{am.round(3)}.png").replace("\\","/"))
        plt.close()
