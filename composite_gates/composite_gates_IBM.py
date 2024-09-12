import os
import sys
import argparse
import pickle
from copy import deepcopy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from qiskit.ignis.verification import randomized_benchmarking_seq

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

# COMP_PARAMS = {
#     3: {"alpha": 0.6399, "phases": [1.8442, 1.0587]},
#     5: {"alpha": 0.45, "phases": [1.9494, 0.5106, 1.31]},
#     7: {"alpha": 0.2769, "phases": [1.6803, 0.2724, 0.8255, 1.6624]},
#     9: {"alpha": 0.2947, "phases": [0.2711, 1.1069, 1.5283, 0.1283, 0.9884]},
#     11: {"alpha": 0.2985, "phases": [1.7377, 0.1651, 0.9147, 0.1510, 0.9331, 1.6415]},
#     13: {"alpha": 0.5065, "phases": [0.0065, 1.7755, 0.7155, 0.5188, 0.2662, 1.2251, 1.3189]},
#     15: {"alpha": 0.3213, "phases": [1.2316, 0.9204, 0.2043, 1.9199, 0.8910, 0.7381, 1.9612, 1.3649]},
# }
# COMP_PARAMS = {
#     3: [{"alpha": 1, "phases": [0, 1/2]}],
#     5: [
#         {"alpha": 1, "phases": [0, 5/6, 2/6]}, 
#         {"alpha": 1, "phases": [0, 11/6, 2/6]}
#     ],
#     7: [
#         {"alpha": 1, "phases": [0, 11/12, 10/12, 17/12]}, 
#         {"alpha": 1, "phases": [0, 1/12, 14/12, 19/12]}
#     ],
#     9: [
#         {"alpha": 1, "phases": [0, 0.366, 0.638, 0.435, 1.697]},
#         {"alpha": 1, "phases": [0, 0.634, 1.362, 0.565, 0.303]}
#     ],
#     11: [
#         {"alpha": 1, "phases": np.array([0, 11, 10, 23, 1, 19]) / 12}, 
#         {"alpha": 1, "phases": np.array([0, 1, 14, 13, 23, 17]) / 12}
#     ],
#     13: [
#         {"alpha": 1, "phases": np.array([0, 9, 42, 11, 8, 37, 2]) / 24}, 
#         {"alpha": 1, "phases": np.array([0, 33, 42, 35, 8, 13, 2]) / 24}
#     ],
#     25: [
#         {"alpha": 1, "phases": np.array([0, 5, 2, 5, 0, 11, 4, 1, 4, 11, 2, 7, 4]) / 6}, 
#         {"alpha": 1, "phases": np.array([0, 11, 2, 11, 0, 5, 4, 7, 4, 5, 2, 1, 4]) / 6}
#     ],
# }

COMP_PARAMS = {
    "X": {
        1: [{"alpha": 1, "phases": [1/2]}],
        3: [{"alpha": 1, "phases": [1/6, 5/6]}],
        5: [{"alpha": 1, "phases": [0.0672, 0.3854, 1.1364]}],
        7: [{"alpha": 1, "phases": [0.2560, 1.6839, 0.5933, 0.8306]}],
        9: [{"alpha": 1, "phases": [0.3951, 1.2211, 0.7806, 1.9335, 0.4580]}],
        11: [{"alpha": 1, "phases": [0.7016, 1.1218, 1.8453, 0.9018, 0.3117, 0.1699]}],
        13: [{"alpha": 1, "phases": [0.1200, 0.3952, 1.5643, 0.0183, 0.9219, 0.4975,1.1096]}],
        15: [{"alpha": 1, "phases": [0.5672, 1.4322, 0.9040, 0.2397, 0.9118, 0.5426, 1.6518, 0.1406]}],
        17: [{"alpha": 1, "phases": [0.3604, 1.1000, 0.7753, 1.6298, 1.2338, 0.2969, 0.6148, 1.9298, 0.4443]}]
    },
    "H": {
        3: [{"alpha": 0.6399, "phases": [0.8442, 0.0587]}],
        5: [{"alpha": 0.45, "phases": [1.9494, 0.5106, 1.3179]}],
        7: [{"alpha": 0.2769, "phases": [1.6803, 0.2724, 0.8255, 1.6624]}],
        9: [{"alpha": 0.2947, "phases": [1.2711, 0.1069, 0.5283, 1.1283, 1.9884]}],
        11: [{"alpha": 0.2985, "phases": [1.7377, 0.1651, 0.9147, 0.1510, 0.9331, 1.6415]}],
        13: [{"alpha": 0.5065, "phases": [0.0065, 1.7755, 0.7155, 0.5188, 0.2662, 1.2251, 1.3189]}],
        15: [{"alpha": 0.3132, "phases": [1.2316, 0.9204, 0.2043, 1.9199, 0.8910, 0.7381, 1.9612, 1.3649]}],
    }
}

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_calib_params(
    backend, qubit,
    pulse_type, 
    sigma, duration,
    remove_bg
):
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    calib_dir = os.path.join(file_dir, "calibrations", backend, str(qubit))
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
    duration, sigma, 
    pulse_type, remove_bg, N=5, variant=0, gate_name=None,
    delay_int=None, delay_min=32, delay_max=3200,
    amp_int=None, amp_min=0, amp_max=2,
    det_int=None, det_min=-1e6, det_max=1e6,
    intensity_int=32, intensity_min=32, intensity_max=3200, 
    max_exp_per_job=50,
    num_shots=1024, backend="manila",
    l=100, p=0.5, x0=0,
    qubit=0,
    closest_amp=None,
    nseeds=5,
    lengths=5
):
    backend, drive_chan, num_qubits, q_freq = initialize_backend(backend)
    if closest_amp is None:
        closest_amp = -np.log(1 - np.pi / l) / p + x0
    # amplitudes = np.linspace(
    #     closest_amp - amp_span / 2,
    #     closest_amp + amp_span / 2,
    #     num_exp
    # )
    # Ns = np.arange(0, N_max + N_interval / 2, N_interval, dtype="int64")
    natural_freq = q_freq[qubit]
    delays = [0] if delay_int is None else np.arange(delay_min, delay_max, delay_int)
    frequencies = [natural_freq] if det_int is None else np.arange(natural_freq + det_min, natural_freq + det_max, det_int)
    intensities = [1] if amp_int is None else np.arange(amp_min, amp_max, amp_int)
    
    amplitude_values = np.ones((N))
    alpha = COMP_PARAMS[N][variant]["alpha"] if gate_name is None else COMP_PARAMS[gate_name][N][variant]["alpha"]
    phases = COMP_PARAMS[N][variant]["phases"] if gate_name is None else COMP_PARAMS[gate_name][N][variant]["phases"]
    amplitude_values[0] = alpha
    amplitude_values[-1] = alpha
    amplitude_values *= closest_amp
    phase_values = np.empty((N))
    phase_values[:int(N/2) + 1] = np.array(phases)
    phase_values[int(N/2) + 1:] = np.array(phases)[::-1][1:]
    
    freq = Parameter("freq")
    rabi_intensity = Parameter("rabi_intensity")
    delay = Parameter("delay")
    with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area_with_N_pulses") as sched:
        dur_dt = duration
        pulse.set_frequency(freq, drive_chan)
        for idx, (amp, phi) in enumerate(zip(amplitude_values, phase_values)):
            if pulse_type == "sq" or "sin" in pulse_type:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=rabi_intensity * amp * np.exp(1j * phi * np.pi),
                    name=pulse_type
                )
            elif pulse_type == "gauss":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=rabi_intensity * amp * np.exp(1j * phi * np.pi),
                    name=pulse_type,
                    sigma=sigma / np.sqrt(2)
                )
            elif pulse_type in ["lor", "lor2", "lor3"]:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=rabi_intensity * amp * np.exp(1j * phi * np.pi),
                    name=pulse_type,
                    sigma=sigma,
                )
            else:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=rabi_intensity * amp * np.exp(1j * phi * np.pi),
                    name=pulse_type,
                    sigma=sigma,
                )
            pulse.play(pulse_played, drive_chan)
            if idx < len(amplitude_values) - 1:
                pulse.delay(delay, drive_chan)


    # circs = []
    # for d in delays:
    #     # amps = []
    #     # phis = []
    #     # for i in range(N):
    #     #     amps.append(Parameter(f"amp{i}"))
    #     #     phis.append(Parameter(f"phi{i}"))
    #     amplitude_values = np.ones((N))
    #     amplitude_values[0] = COMP_PARAMS[N]["alpha"]
    #     phase_values = np.empty((N))
    #     phase_values[:int(N/2) + 1] = np.array(COMP_PARAMS[N]["phases"])
    #     phase_values[int(N/2) + 1:] = np.array(COMP_PARAMS[N]["phases"])[::-1][1:]

        # base_circ = QuantumCircuit(num_qubits, 1)
        # cp_sched, comp_pulses = [], [None] * N
        # for i in range(N):
        #     comp_pulses[i] = Gate(f"cp{i}", 1, [])
        #     cp_sched.append(add_pulse(amplitude_values[i] * closest_amp, phase_values[i], d))
        #     base_circ.append(comp_pulses[i], [qubit])

        # base_circ.measure(qubit, 0)
        # for i in range(N):
        #     base_circ.add_calibration(comp_pulses[i], (qubit,), cp_sched[i], [])
        # params_dict = {}
        # # for i in range(N):
        # #     params_dict[amps[i]] = amplitude_values[i] * closest_amp
        # #     params_dict[phis[i]] = phase_values[i]

    # base_circ = QuantumCircuit(num_qubits, 1)
    comp_gate = Gate("composite_series", 1, [freq, rabi_intensity, delay])
    # base_circ.append(comp_gate, [qubit])
    # base_circ.measure(qubit, 0)
    # base_circ.add_calibration(comp_gate, (qubit,), sched, [freq, rabi_intensity, delay])
    
    # Generate RB circuits
    rb_circuits, _ = randomized_benchmarking_seq(nseeds=nseeds, length_vector=lengths, n_qubits=1)

    circs = []
    # Add custom gate to the RB circuits
    for rb_circ in rb_circuits:
        rb_circ.append(comp_gate, [0])  # Append your custom gate
        rb_circ.add_calibration(comp_gate, (qubit,), sched, [freq, rabi_intensity, delay])

        circs.extend(
            [
            rb_circ.assign_parameters({freq: q, rabi_intensity: i, delay: d}, inplace=False) 
            for q in frequencies
            for i in intensities
            for d in delays
            ]
        )
        # for d in delays:
        #     params_dict_copy = params_dict.copy()
        #     params_dict_copy[delay] = d
        #     print(params_dict_copy)
        #     circs.append(
        #         base_circ.assign_parameters(
        #             params_dict_copy, inplace=False
        #         )
        #     )

    # num_shots = 1024
    sweep_values, job_ids = run_jobs(circs, backend, duration * len(delays), num_shots_per_exp=num_shots)
    print(sweep_values)
    plt.scatter(intensities, sweep_values)
    plt.show()
    # print(Ns, np.array(sweep_values).reshape(np.round(N_max / N_interval + 1).astype(np.int64), len(amplitudes)))
    return frequencies, intensities, delays, np.array(sweep_values)


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
    parser.add_argument("-N", "--N", default=5, type=int,
        help="Number of composite pulses to use.")
    parser.add_argument("-v", "--variant", default=0, type=int,
        help="Composite pulses variant to use (0 or 1).")
    parser.add_argument("-gt", "--gate_name", default=None, type=str,
        help="Gate to compose using CPs.")
    parser.add_argument("-di", "--delay_int", default=None, type=int,
        help="Delay number step between the composite pulses.")
    parser.add_argument("-dmn", "--delay_min", default=32, type=int,
        help="Minimum delay between the composite pulses.")
    parser.add_argument("-dmx", "--delay_max", default=1000, type=int,
        help="Maximum delay between the composite pulses.")
    parser.add_argument("-ai", "--amp_int", default=None, type=float,
        help="Interval of the amplitude of the composite pulses.")
    parser.add_argument("-amn", "--amp_min", default=0, type=float,
        help="Minimum amplitude of the composite pulses.")
    parser.add_argument("-amx", "--amp_max", default=2, type=float,
        help="Maximum amplitude of the composite pulses.")
    parser.add_argument("-deti", "--det_int", default=None, type=float,
        help="Interval of the detuning of the composite pulses.")
    parser.add_argument("-detmn", "--det_min", default=-1e6, type=float,
        help="Initial detuning of the composite pulses.")
    parser.add_argument("-detmx", "--det_max", default=1e6, type=float,
        help="Final detuning of the composite pulses.")
    parser.add_argument("-rb", "--remove_bg", default=1, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=256, type=int,
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
    N = args.N
    variant = args.variant
    gate_name = args.gate_name
    delay_int = args.delay_int
    delay_min = args.delay_min
    delay_max = args.delay_max
    amp_int = args.amp_int
    amp_min = args.amp_min
    amp_max = args.amp_max
    det_int = args.det_int
    det_min = args.det_min
    det_max = args.det_max
    remove_bg = args.remove_bg
    max_experiments_per_job = args.max_experiments_per_job
    num_shots = args.num_shots
    num_exp = args.num_experiments
    backend = args.backend
    amp_span = args.span
    num_iterations = args.num_iterations

    dt_now = datetime.now()

    l, p, x0 = get_calib_params(
        backend, qubit, pulse_type, 
        sigma, duration,
        remove_bg
    )
    # for i in range(num_iterations):
    frequencies, intensities, delays, values = run_check(
        # amp_span / (10**i), 
        duration, sigma, 
        pulse_type, remove_bg,
        N=N, variant=variant, gate_name=gate_name,
        delay_int=delay_int, delay_min=delay_min, 
        delay_max=delay_max,
        amp_int=amp_int, amp_min=amp_min, 
        amp_max=amp_max, det_int=det_int, 
        det_min=det_min, det_max=det_max,
        max_exp_per_job=max_experiments_per_job,
        num_shots=num_shots, 
        backend=backend,
        l=l, p=p, x0=x0,
        closest_amp=closest_amp)
        # index = find_least_variation(intensities, values)
        # closest_amp = intensities[index]

    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    time = dt_now.strftime("%H%M%S")
    date = dt_now.strftime("%Y-%m-%d")
    plot_save_dir = os.path.join(file_dir, "plots", backend, str(qubit), "composites", date)
    data_save_dir = os.path.join(file_dir, "data", backend, str(qubit), "composites", date)
    make_all_dirs(plot_save_dir.replace("\\","/"))
    make_all_dirs(data_save_dir.replace("\\","/"))
    with open(os.path.join(data_save_dir, f"{time}_composites_{duration}dt_{N}pulses_v{variant}.pkl").replace("\\","/"), 'wb') as f:
        pickle.dump((frequencies, intensities, delays, values), f)
    print("Save to pickle successful!")
    
    params = np.array([(q, i, d) for q in frequencies for i in intensities for d in delays])
    # print(params)
    is_variable = [None] * 3
    for i in range(len(params[0])):
        is_variable[i] = False if (params[:, i] == np.roll(params[:, i], 1)).all() else True
    
    variable_dict = {
        0: "Detuning [MHz]",
        1: "Amplitude [arb. units]",
        2: "Delays [dt]"
    }
    num_variables = np.sum(is_variable)
    if num_variables == 1:
        variable_type = np.where(is_variable)[0][0]
        variable = params[:, variable_type]
    elif num_variables == 2:
        variable_type = [np.where(is_variable)[0][0], np.where(is_variable)[0][1]]
        variable = [params[:, variable_type[0]], params[:, variable_type[1]]]
    else:
        print("Too few/many variables, choose 1 or 2 manually.")

    if num_variables == 1:
        fig, ax = plt.subplots()
        sc = ax.scatter(variable, values)
        ax.set_xlabel(variable_dict[variable_type])
        ax.set_ylabel("Transition probability")
        plt.show()
        plt.savefig(os.path.join(plot_save_dir, f"{time}_composites_{variable_dict[variable_type]}_{duration}dt_{N}pulses_v{variant}.png").replace("\\","/"))
    
    if num_variables == 2:
        fig, ax = plt.subplots(1, 1)
        c = ax.pcolormesh(variable[0], variable[1], values.reshape(len(variable[0]), len(variable[1])))
        ax.set_xlabel(variable_dict[variable_type[0]])
        ax.set_ylabel(variable_dict[variable_type[1]])
        fig.colorbar(c, ax=ax)
        plt.show()
        plt.savefig(os.path.join(plot_save_dir, f"{time}_composites_{variable_dict[variable_type[0]]},{variable_dict[variable_type[1]]}_{duration}dt_{N}pulses_v{variant}.png").replace("\\","/"))
    print("Figure saved successfully!")
