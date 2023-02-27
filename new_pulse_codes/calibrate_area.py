import os
import pickle
import argparse
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

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    # parser.add_argument("-G", "--lorentz_G", default=180, type=float,
    #     help="Lorentz width (gamma) parameter")    
    parser.add_argument("-s", "--sigma", default=180, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-T", "--duration", default=2256, type=int,
        help="Lorentz duration parameter")
    # parser.add_argument("-c", "--cutoff", default=0.5, type=float,
    #     help="Cutoff parameter in PERCENT of maximum amplitude of Lorentzian")
    parser.add_argument("-rb", "--remove_bg", default=0, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-cp", "--control_param", default="width", type=str,
        help="States whether width or duration is the controlled parameter")
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
    args = parser.parse_args()
    # cutoff = args.cutoff
    lor_G = args.sigma
    duration = get_closest_multiple_of_16(args.duration)
    sigma = args.sigma
    initial_amp = args.initial_amp
    final_amp = args.final_amp
    num_shots_per_exp = args.num_shots
    num_exp = args.num_experiments
    ctrl_param = args.control_param
    backend = args.backend
    max_experiments_per_job = args.max_experiments_per_job
    remove_bg = bool(args.remove_bg)
    pulse_type = args.pulse_type
    backend_name = backend
    backend = "ibm_" + backend \
        if backend in ["perth", "lagos", "nairobi", "oslo"] \
            else "ibmq_" + backend
    pulse_dict = {
        "gauss": pulse_lib.Gaussian,
        "lor": pulse_lib.Lorentzian,
        "lor2": pulse_lib.LorentzianSquare,
        "lor3": pulse_lib.LorentzianCube,
        "sq": pulse_lib.Constant,
        "sech": pulse_lib.Sech,
        "sech2": pulse_lib.SechSquare
    }
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


    rough_qubit_frequency = center_frequency_Hz # 4962284031.287086 Hz

    ## set params
    fit_crop = 1#.8
    amplitudes = np.linspace(
        initial_amp, 
        final_amp, 
        num_exp
    )
    fit_crop_parameter = int(fit_crop * len(amplitudes))

    print(f"The resonant frequency is assumed to be {np.round(rough_qubit_frequency / GHz, 5)} GHz.")
    print(f"The area calibration will start from amp {amplitudes[0]} "
    f"and end at {amplitudes[-1]} with approx step {(final_amp - initial_amp)/num_exp}.")

    amp = Parameter('amp')
    with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area") as sched:
        dur_dt = duration
        pulse.set_frequency(rough_qubit_frequency, drive_chan)
        if pulse_type == "sq":
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type
            )
        elif pulse_type == "gauss":
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type,
                sigma=sigma / np.sqrt(2),
                zero_ends=remove_bg
            )
        elif pulse_type in ["lor", "lor2", "lor3"]:
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type,
                gamma=sigma,
                zero_ends=remove_bg
            )
        else:
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type,
                sigma=sigma,
                zero_ends=remove_bg
            )
        pulse.play(pulse_played, drive_chan)

    # Create gate holder and append to base circuit
    pi_gate = Gate("rabi", 1, [amp])
    base_circ = QuantumCircuit(1, 1)
    base_circ.append(pi_gate, [0])
    base_circ.measure(0, 0)
    base_circ.add_calibration(pi_gate, (qubit,), sched, [amp])
    circs = [
        base_circ.assign_parameters(
                {amp: a},
                inplace=False
        ) for a in amplitudes]

    # rabi_schedule = qiskit.schedule(circs[-1], backend)
    # rabi_schedule.draw(backend=backend)
    
    job_manager = IBMQJobManager()
    pi_job = job_manager.run(
        circs,
        backend=backend,
        shots=num_shots_per_exp,
        max_experiments_per_job=max_experiments_per_job
    )

    pi_sweep_results = pi_job.results()

    pi_sweep_values = []

    for i in range(len(circs)):
        try:
            counts = pi_sweep_results.get_counts(i)["1"]
        except KeyError:
            counts = 0
        pi_sweep_values.append(counts / num_shots_per_exp)

    # print(amplitudes, np.real(pi_sweep_values))
    plt.figure(3)
    plt.scatter(amplitudes, np.real(pi_sweep_values), color='black') # plot real part of sweep values
    plt.title("Rabi Calibration Curve")
    plt.xlabel("Amplitude [a.u.]")
    plt.ylabel("Transition Probability")
    plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_pi_amp_sweep.png"))
    datapoints = np.vstack((amplitudes, np.real(pi_sweep_values)))
    with open(os.path.join(data_folder, f"area_calibration_{date.strftime('%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(datapoints, f)
    ## fit curve
    def fit_function(x_values, y_values, function, init_params):
        fitparams, conv = curve_fit(
            function, 
            x_values, 
            y_values, 
            init_params, 
            maxfev=100000, 
            bounds=(
                [-0.53, 10, 1, 0, 0.45], 
                [-.47, 100, 100, 1000, 0.55]
            )
        )
        y_fit = function(x_values, *fitparams)
        
        return fitparams, y_fit


    rabi_fit_params, _ = fit_function(
        amplitudes[: fit_crop_parameter],
        np.real(pi_sweep_values[: fit_crop_parameter]), 
        lambda x, A, k, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * (x - x0))))) + B,
        [-0.47273362, 70.14552555, 10, 0.5, 0, 0.47747625]
        # lambda x, A, k, B: A * (np.cos(k * x)) + B,
        # [-0.5, 50, 0.5]
    )

    print(rabi_fit_params)
    A, k, l, p, B = rabi_fit_params
    pi_amp = ((l - np.sqrt(l ** 2 + 4 * k * np.pi)) / (2 * k)) ** 2 #np.pi / (k)
    half_amp = ((l - np.sqrt(l ** 2 + 2 * k * np.pi)) / (2 * k)) ** 2 #np.pi / (k)

    detailed_amps = np.arange(amplitudes[0], amplitudes[-1], amplitudes[-1] / 2000)
    extended_y_fit = A * (np.cos(k * (detailed_amps) + l * (1 - np.exp(- p * detailed_amps)))) + B

    ## create pandas series to keep calibration info
    param_dict = {
        "date": date.strftime("%Y-%m-%d"),
        "time": date.strftime("%H:%M:%S"),
        "pulse_type": pulse_type,
        "A": A,
        "k": k,
        "l": l,
        "p": p,
        "B": B,
        "drive_freq": rough_qubit_frequency,
        "pi_duration": dur_dt,
        "sigma": sigma,
        "pi_amp": pi_amp,
        "half_amp": half_amp
    }
    print(param_dict)
    param_series = pd.Series(param_dict)
    params_file = os.path.join(calib_dir, "params.csv")
    if os.path.isfile(params_file):
        param_df = pd.read_csv(params_file)
        param_df = pd.concat([param_df, param_series.to_frame().T], ignore_index=True)
        param_df.to_csv(params_file, index=False)
    else:
        param_series.to_frame().T.to_csv(params_file, index=False)

    plt.figure(4)
    plt.scatter(amplitudes, np.real(pi_sweep_values), color='black')
    plt.plot(detailed_amps, extended_y_fit, color='red')
    plt.xlim([min(amplitudes), max(amplitudes)])
    plt.title("Fitted Rabi Calibration Curve")
    plt.xlabel("Amplitude [a.u.]")
    plt.ylabel("Transition Probability")
    # plt.ylabel("Measured Signal [a.u.]")
    plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_pi_amp_sweep_fitted.png"))

    plt.show()