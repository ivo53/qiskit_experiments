import os
import sys
import pickle
from datetime import datetime

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from qiskit.tools.jupyter import *
from qiskit import QuantumCircuit
from qiskit import pulse                  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter, Gate # This is Parameter Class for variable parameters.
from qiskit_ibm_provider import IBMProvider

# current_dir = os.path.dirname(__file__)
# package_path = os.path.abspath(os.path.split(current_dir)[0])
# sys.path.insert(0, package_path)

import qiskit_experiments.pulse_types as pt
from utils.run_jobs import run_jobs

pulse_dict = {
    "gauss": [pt.Gaussian, pt.LiftedGaussian],
    "lor": [pt.Lorentzian, pt.LiftedLorentzian],
    "lor2": [pt.Lorentzian2, pt.LiftedLorentzian2],
    "lor3_2": [pt.Lorentzian3_2, pt.LiftedLorentzian3_2],
    "lor3": [pt.Lorentzian3, pt.LiftedLorentzian3],
    "lor2_3": [pt.Lorentzian2_3, pt.LiftedLorentzian2_3],
    "lor3_4": [pt.Lorentzian3_4, pt.LiftedLorentzian3_4],
    "lor3_5": [pt.Lorentzian3_5, pt.LiftedLorentzian3_5],
    "sq": [pt.Constant, pt.Constant],
    "sech": [pt.Sech, pt.LiftedSech],
    "sech2": [pt.Sech2, pt.LiftedSech2],
    "sin": [pt.Sine, pt.Sine],
    "sin2": [pt.Sine2, pt.Sine2],
    "sin3": [pt.Sine3, pt.Sine3],
    "sin4": [pt.Sine4, pt.Sine4],
    "sin5": [pt.Sine5, pt.Sine5],
    "demkov": [pt.Demkov, pt.LiftedDemkov],
    "drag": [pt.Drag, pt.LiftedDrag],
    "ipN": [pt.InverseParabola, pt.InverseParabola],
    "fcq": [pt.FaceChangingQuadratic, pt.FaceChangingQuadratic],
}

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

def get_calib_params(
        calib_dir, 
        pulse_type, 
        duration, sigma, 
        remove_bg, 
        N, beta):
    params_file = os.path.join(calib_dir, "actual_params.csv")
    if os.path.isfile(params_file):
        param_df = pd.read_csv(params_file)
    if pulse_type not in ["ipN", "fcq"]:
        df = param_df[param_df.apply(
            lambda row: row["pulse_type"] == pulse_type and \
                row["duration"] == duration and \
                row["sigma"] == sigma and \
                row["rb"] == remove_bg, axis=1)]
    elif pulse_type == "ipN":
        df = param_df[param_df.apply(
            lambda row: row["pulse_type"] == pulse_type and \
                row["duration"] == duration and \
                row["sigma"] == sigma and \
                row["rb"] == remove_bg and \
                row["N"] == N, axis=1)]
    elif pulse_type == "fcq":
        df = param_df[param_df.apply(
            lambda row: row["pulse_type"] == pulse_type and \
                row["duration"] == duration and \
                row["sigma"] == sigma and \
                row["rb"] == remove_bg and \
                row["beta"] == beta, axis=1)]
    if df.shape[0] > 1:
        raise ValueError("More than one identical entry found!")
    elif df.shape[0] == 0:
        raise ValueError("No entries found!")
    ser = df.iloc[0]

    return ser.at["l"], ser.at["p"], ser.at["x0"]


def get_amp_for(area, l, p, x0):
    return -np.log(1 - area / l) / p + x0

def initialize_backend(backend):
    backend_full_name = "ibm_" + backend
    drive_chan = pulse.DriveChannel(qubit)
    # meas_chan = pulse.MeasureChannel(qubit)
    # acq_chan = pulse.AcquireChannel(qubit)
    
    backend_name = backend
    # provider = IBMQ.load_account()
    backend = IBMProvider().get_backend(backend_full_name)
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()
    print("Warning: dt =",backend_config.dt)
    num_qubits = backend_config.n_qubits

    q_freq = [backend_defaults.qubit_freq_est[q] for q in range(num_qubits)]
    # dt = backend_config.dt

    return backend, drive_chan, num_qubits, q_freq

def add_circ(backend, drive_chan, pulse_type, amp, duration, sigma, remove_bg, freq, qubit=0):
    with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area") as sched:
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
                sigma=sigma / np.sqrt(2)
            )
        elif pulse_type == "ipN":
            pulse_played = pulse_dict[pulse_type][remove_bg](
                duration=duration,
                amp=amp,
                N=N,
                name=pulse_type
            )
        elif pulse_type == "fcq":
            pulse_played = pulse_dict[pulse_type][remove_bg](
                duration=duration,
                amp=amp,
                beta=beta,
                name=pulse_type
            )
        elif "lor" in pulse_type:
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
    pi_gate = Gate("power_spectre", 1, [])
    base_circ = QuantumCircuit(qubit+1, 1)
    base_circ.append(pi_gate, [qubit])
    base_circ.measure(qubit, 0)
    base_circ.add_calibration(pi_gate, (qubit,), sched, [])
    return base_circ


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-q", "--qubit", default=0, type=int,
        help="The number of the qubit to be used.")
    parser.add_argument("-s", "--sigma", default=192, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-T", "--duration", default=2256, type=float,
        help="Lorentz duration parameter")
    parser.add_argument("-rb", "--remove_bg", default=0, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-ns", "--num_shots", default=2048, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-b", "--backend", default="kyoto", type=str,
        help="The name of the backend to use in the experiment (one of kyoto, osaka).")
    parser.add_argument("-rA", "--resolution_A", default=100, type=int,
        help="Resolution in the amplitude axis.")
    parser.add_argument("-rD", "--resolution_D", default=100, type=int,
        help="Resolution in the detuning axis.")
    parser.add_argument("-cp", "--cut_param", default=0.2, type=float,
        help="Cutoff point as amp value as a fraction of maximum amplitude.")
    parser.add_argument("-a", "--max_amp", default=0.5, type=float,
        help="Maximum amplitude to reach in the sweep.")
    parser.add_argument("-N", "--N", default=1, type=int,
        help="The order of inverse parabola pulse(in case of inv. parabola).")
    parser.add_argument("-be", "--beta", default=0, type=float,
        help="The beta parameter for the face changing quadratic function.")
    parser.add_argument("-sp", "--frequency_span", default=18, type=float,
        help="Frequency span in MHz units.")
    args = parser.parse_args()

    pulse_type = args.pulse_type
    sigma = args.sigma
    duration = get_closest_multiple_of_16(round(args.duration))
    remove_bg = args.remove_bg
    num_shots = args.num_shots
    qubit = args.qubit
    resolution = (args.resolution_A, args.resolution_D)
    cut_param = args.cut_param
    # a_max = args.max_amp
    N = float(args.N)
    beta = args.beta
    frequency_span = args.frequency_span
    backend = args.backend
    backend_name = backend

    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    save_dir = os.path.join(
        file_dir,
        "plots",
        f"{backend_name}",
        "power_broadening (narrowing)",
        f"{pulse_type}_pulses",
        current_date
    )
    folder_name = os.path.join(
        save_dir,
        date.strftime("%H%M%S")
    ).replace("\\", "/")
    data_folder = os.path.join(
        file_dir,
        "data",
        f"{backend_name}",
        "power_broadening (narrowing)",
        f"{pulse_type}_pulses",
        current_date,
        date.strftime("%H%M%S")
    ).replace("\\", "/")
    make_all_dirs(data_folder)
    make_all_dirs(folder_name)
    calib_dir = os.path.join(file_dir, "calibrations", backend_name)
    l, p, x0 = get_calib_params(calib_dir, pulse_type, duration, sigma, remove_bg, N, beta)
    
    backend, drive_chan, num_qubits, q_freq =  initialize_backend(backend)

    center_frequency_Hz = q_freq[qubit]

    GHz, MHz, us, ns = 1.0e9, 1.0e6, 1.0e-6, 1.0e-9 # Gigahertz, Megahertz, Microseconds, Nanoseconds

    print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

    backend_full_name = "ibm_" + backend_name
        # if backend_name in ["perth", "lagos", "nairobi", "oslo", "kyoto", "brisbane"] \
        #     else "ibmq_" + backend_name

    frequency_span_Hz = frequency_span * MHz #5 * MHz #if cut_param < 1 els e 1.25 * MHz
    frequency_step_Hz = np.round(frequency_span_Hz / resolution[1], 3) #(1/4) * MHz
    
    # We will sweep 20 MHz above and 20 MHz below the estimated frequency
    frequency_min = center_frequency_Hz - frequency_span_Hz / 2
    frequency_max = center_frequency_Hz + frequency_span_Hz / 2
    # Construct an np array of the frequencies for our experiment
    frequencies_GHz = np.linspace(frequency_min / GHz, 
                                frequency_max / GHz, 
                                resolution[1])

    a_max = get_amp_for(10 * np.pi, l, p, x0)
    amplitudes = np.linspace(0.001, a_max, resolution[0]).round(3)

    assert len(amplitudes) == resolution[0], "amplitudes error"
    assert len(frequencies_GHz) == resolution[1], "frequencies error"
    
    print(f"The gamma factor is G = {sigma} and cut param is {cut_param}, \
    compared to G_02 = {sigma} at cut param 0.2.")
    print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
    in steps of {frequency_step_Hz / MHz} MHz.")
    print(f"The amplitude will go from {amplitudes[0]} to {amplitudes[-1]} in steps of {np.round(a_max / resolution[0], 4)}.")

    frequencies_Hz = frequencies_GHz * GHz

    drive_chan = pulse.DriveChannel(qubit)

    circs = [add_circ(
        backend, drive_chan, 
        pulse_type, a, duration, 
        sigma, remove_bg, f, qubit
        ) for a in amplitudes for f in frequencies_Hz]

    num_circs = len(circs)
    # num_shots = 1024

    transition_probability, job_ids = run_jobs(circs, backend, duration, num_shots_per_exp=num_shots)

    transition_probability = np.array(transition_probability).reshape(len(amplitudes), len(frequencies_GHz))
    # print("JobsID:", job_ids)

    ## save final data
    freq_offset = (frequencies_Hz - center_frequency_Hz) / 10**6
    y, x = np.meshgrid(amplitudes, freq_offset)
    # z = np.reshape(transition_probability, (len(frequencies_Hz),len(amplitudes)))

    with open(os.path.join(data_folder, f"{duration}dt_cutparam-{cut_param}_tr_prob.pkl").replace("\\","/"), 'wb') as f1:
        pickle.dump(transition_probability, f1)
    with open(os.path.join(data_folder, f"{duration}dt_cutparam-{cut_param}_areas.pkl").replace("\\","/"), 'wb') as f2:
        pickle.dump(amplitudes, f2)
    with open(os.path.join(data_folder, f"{duration}dt_cutparam-{cut_param}_detunings.pkl").replace("\\","/"), 'wb') as f3:
        pickle.dump((frequencies_Hz - center_frequency_Hz), f3)

    for i, am in enumerate(amplitudes):
        plt.figure(i)
        plt.plot(freq_offset, transition_probability[i], "bx")
        plt.xlabel("Detuning [MHz]")
        plt.ylabel("Transition Probability")
        plt.title(f"Lorentzian Freq Offset - Amplitude {am.round(3)}")
        plt.savefig(os.path.join(folder_name, f"lor_amp-{am.round(3)}.png").replace("\\","/"))
        plt.close()

    fig, ax = plt.subplots(figsize=(5,4))

    c = ax.pcolormesh(x, y, transition_probability.T, vmin=0, vmax=1)
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    ax.set_ylabel('Rabi Freq. [a.u.]')
    ax.set_xlabel('Detuning [MHz]')
    plt.axhline(y = 0.5, color = 'w', linestyle = '--')
    plt.axhline(y = 0.9, color = 'w', linestyle = '--')
    plt.savefig(
        os.path.join(
            save_dir,
            f"{date.strftime('%H%M%S')}_{pulse_type}_pwr_spctr_duration-{duration}dt_sigma-{sigma}_N-{N}_beta-{beta}.png"
        ).replace("\\","/")
    )
    plt.show()