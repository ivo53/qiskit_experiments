import os
import sys
import argparse
from datetime import datetime
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qiskit import (
    QuantumCircuit, 
    pulse
) 
from qiskit.circuit import Parameter, Gate
from qiskit_ibm_provider import IBMProvider

current_dir = os.path.dirname(__file__)
package_path = os.path.abspath(os.path.split(current_dir)[0])
sys.path.insert(0, package_path)

from utils.run_jobs import run_jobs
import pulse_types as pt


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

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

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
        print("No entry found!")
        command = [
            "python", 
            f"{os.path.join(file_dir, 'new_pulse_codes', 'calibrate_area.py')}", 
            f"-sv 1 -ne 100 -ns 1024 -b {backend} -T {duration} -s {sigma} -rb {remove_bg} -ia 0.001 -fa 0.2"
        ]
        subprocess.run(command, check=True)
        param_df = pd.read_csv(params_file)
        df = param_df[param_df.apply(
                lambda row: row["pulse_type"] == pulse_type and \
                    row["duration"] == duration and \
                    row["sigma"] == sigma and \
                    row["rb"] == remove_bg, axis=1)]

    ser = df.iloc[idx]
    l = ser.at["l"]
    p = ser.at["p"]
    x0 = ser.at["x0"]

    return l, p, x0

def initialize_backend(backend):
    backend_full_name = "ibm_" + backend \
        if backend in ["perth", "lagos", "nairobi", "oslo"] \
            else "ibmq_" + backend
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

def run_check(
    duration, sigma, pulse_type, remove_bg,
    num_exp=10, N_max=100,
    N_interval=2, max_exp_per_job=50,
    num_shots=1024, backend="manila",
    l=100, p=0.5, x0=0,
    closest_amp=None,
    eps=0.
):
    backend, drive_chan, num_qubits, q_freq = initialize_backend(backend)
    if closest_amp is None:
        closest_amp = -np.log(1 - (np.pi / 2 - eps) / l) / p + x0
    # amplitudes = np.linspace(
    #     closest_amp - amp_span / 2,
    #     closest_amp + amp_span / 2,
    #     num_exp
    # )
    # Ns = np.arange(0, N_max + N_interval / 2, N_interval, dtype="int64")

    def add_circ(amp, duration, sigma, freq, N, qubit=0):
        with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area_with_N_pulses") as sched:
            dur_dt = duration
            pulse.set_frequency(freq, drive_chan)
            if pulse_type == "sq" or "sin" in pulse_type:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type
                )
            elif pulse_type == "gauss":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type,
                    sigma=sigma / np.sqrt(2)
                )
            elif pulse_type in ["lor", "lor2", "lor3"]:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type,
                    sigma=sigma,
                )
            else:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type,
                    sigma=sigma,
                )
            pulse.play(pulse_played, drive_chan)
    
        base_circ = QuantumCircuit(num_qubits, 1)
        custom_gate = Gate("N_pulses", 1, [])
        for _ in range(N):
            base_circ.append(custom_gate, [qubit])
        base_circ.measure(qubit, 0)
        base_circ.add_calibration(custom_gate, (qubit,), sched, [])
        return base_circ
    circs = [add_circ(closest_amp, duration, sigma, q_freq[qubit], N) for N in range(N_max)]

    sweep_values, job_ids = run_jobs(circs, backend, duration, num_shots_per_exp=num_shots)

    print(np.arange(N_max), np.array(sweep_values))
    return np.arange(N_max), np.array(sweep_values), closest_amp

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
    parser.add_argument("-T", "--duration", default=2256, type=float,
        help="Pulse duration parameter")
    parser.add_argument("-N", "--N_max", default=100, type=int,
        help="Maximum N number of pulses to use in the experiment.")
    parser.add_argument("-Ni", "--N_interval", default=2, type=int,
        help="Interval between N number of pulses.")
    parser.add_argument("-rb", "--remove_bg", default=1, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=4096, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-ne", "--num_experiments", default=100, type=int,
        help="Number of experiments with different detuning each.")
    parser.add_argument("-b", "--backend", default="manila", type=str,
        help="The name of the backend to use in the experiment (one of perth, lagos, nairobi, \
        oslo, jakarta, manila, quito, belem, lima).")
    parser.add_argument("-iter", "--num_iterations", default=3, type=int,
        help="The number of optimization iterations of the algorithm.")
    args = parser.parse_args()

    pulse_type = args.pulse_type
    qubit = args.qubit
    closest_amp = args.closest_amp
    sigma = args.sigma
    duration = get_closest_multiple_of_16(round(args.duration))
    N_max = args.N_max
    N_interval = args.N_interval
    remove_bg = args.remove_bg
    max_experiments_per_job = args.max_experiments_per_job
    num_shots = args.num_shots
    num_exp = args.num_experiments
    backend = args.backend
    num_iterations = args.num_iterations

    l, p, x0 = get_calib_params(
        backend, pulse_type, 
        sigma, duration,
        remove_bg
    )
    Ns, vals, amp = run_check(
        duration, sigma, 
        pulse_type, remove_bg,
        num_exp=num_exp, N_max=N_max, N_interval=N_interval, 
        max_exp_per_job=max_experiments_per_job,
        num_shots=num_shots, backend=backend,
        l=l, p=p, x0=x0
    )

    vals = 1 - vals

    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    save_dir = os.path.join(file_dir, "plots", backend, "repeated_half_pi", current_date)
    data_folder = os.path.join(file_dir, "data", backend, "repeated_half_pi", current_date)
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    ax = fig.add_subplot()
    ax.plot(Ns, vals, color='black', marker="P", label="Transition Probability", linewidth=0.)
    plt.savefig(os.path.join(save_dir, ))
    plt.close()

    k = 2
    
    while 42:
        four_m_plus_one = [Ns[1::4], vals[1::4]]
        four_m_minus_one = [Ns[3::4], vals[3::4]]
        threshold = 0.15
        is_bigger_threshold = np.abs(four_m_plus_one[1][1:] - four_m_minus_one[1][:-1]) > threshold

        idx = is_bigger_threshold.view(bool).argmax() // is_bigger_threshold.itemsize
        m = idx if is_bigger_threshold[idx] else -1

        eps = np.arccos(np.sqrt(vals[4 * m + 1])) / (m * k + 1) - np.pi / (2 * k * (m * k + 1))

        if eps / (np.pi / 2) < 0.001:
            break

        # amp *= (1 - eps / (np.pi / 2))

        Ns, vals, amp = run_check(
            duration, sigma, 
            pulse_type, remove_bg,
            num_exp=num_exp, N_max=N_max, N_interval=N_interval, 
            max_exp_per_job=max_experiments_per_job,
            num_shots=num_shots, backend=backend,
            l=l, p=p, x0=x0, eps=eps
        )

        vals = 1 - vals

    fig = plt.figure(constrained_layout=True, figsize=(10,6))
    ax = fig.add_subplot()
    ax.plot(Ns, vals, color='black', marker="P", label="Transition Probability", linewidth=0.)
    plt.savefig(save_dir)
    plt.show()