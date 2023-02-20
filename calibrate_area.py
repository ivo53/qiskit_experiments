import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from scipy.optimize import curve_fit
from scipy.special import lambertw
from qiskit import pulse, IBMQ, assemble, execute 
# This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-l", "--l", default=10, type=float,
        help="Parameter l in Rabi oscillations fit")
    parser.add_argument("-p", "--p", default=0.5, type=float,
        help="Parameter p in Rabi oscillations fit")
    parser.add_argument("-G", "--lorentz_G", default=180, type=float,
        help="Lorentz width (gamma) parameter")    
    parser.add_argument("-T", "--lorentz_T", default=2256, type=int,
        help="Lorentz duration parameter")
    parser.add_argument("-c", "--cutoff", default=0.5, type=float,
        help="Cutoff parameter in PERCENT of maximum amplitude of Lorentzian")
    parser.add_argument("-rb", "--remove_bg", default=0, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-cp", "--control_param", default="width", type=str,
        help="States whether width or duration is the controlled parameter")
    parser.add_argument("-ne", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-b", "--backend", default="ibmq_armonk", type=str,
        help="The name of the backend to use in the experiment.")
    args = parser.parse_args()
    cutoff = args.cutoff
    lor_G = args.lorentz_G
    lor_T = get_closest_multiple_of_16(args.lorentz_T)
    ctrl_param = args.control_param
    backend = args.backend
    remove_bg = bool(args.remove_bg)
    backend_short_name = backend.split("_")[1]
    assert 100. > cutoff > 0., "Cutoff percentage MUST be between 0 and 100."
    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    calib_dir = os.path.join(file_dir, "calibrations")
    save_dir = os.path.join(calib_dir, current_date)
    data_folder = os.path.join(file_dir, "data", backend_short_name, "calibration", current_date).replace("\\", "/")
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    # unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
    GHz = 1.0e9 # Gigahertz
    MHz = 1.0e6 # Megahertz
    us = 1.0e-6 # Microseconds
    ns = 1.0e-9 # Nanoseconds
    qubit = 0

    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)

    provider = IBMQ.load_account()
    backend = provider.get_backend(backend)
    # backend = provider.get_backend("ibmq_qasm_simulator")
    backend_name = str(backend)
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    dt = backend_config.dt
    # measurement 
    meas_map_idx = None
    for i, measure_group in enumerate(backend_config.meas_map):
        if qubit in measure_group:
            meas_map_idx = i
            break
    assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
    inst_sched_map = backend_defaults.instruction_schedule_map
    measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

    rough_qubit_frequency = 4.97173 * GHz

    ## set params
    pulse_type = args.pulse_type
    fit_crop = 1#.8
    min_amplitude, max_amplitude = 0.001, .99#0.55#.65
    num_exp = 100
    amplitudes = np.linspace(
        min_amplitude,
        max_amplitude,
        num_exp
    )
    fit_crop_parameter = int(fit_crop * len(amplitudes))

    schedules = [] 
    for amp in amplitudes:
        ## gauss pulse
        sigma = 0.05 #us
        gauss_sigma = get_closest_multiple_of_16(sigma * us / dt)
        gauss_dur = get_closest_multiple_of_16(8 * sigma * us / dt)

        gauss_t_dt = np.arange(gauss_dur)
        gauss_mu = gauss_dur / 2
        gauss_amps = amp * np.exp(-(gauss_t_dt - gauss_mu) ** 2 / (2 * gauss_sigma ** 2))
        if remove_bg:
            gauss_amps -= gauss_amps[0]
        gauss = pulse_lib.Waveform(samples=gauss_amps, name="gauss pulse")
        # gauss = pulse_lib.gaussian(
        #     duration=gauss_dur,
        #     sigma=gauss_sigma,
        #     amp=amp,
        #     name='pi_sweep_gauss_pulse'
        # )

        ## sq pulse
        dur = .5 # us
        duration = get_closest_multiple_of_16(dur * us / dt)
        sq = pulse_lib.constant(
            duration=duration,
            amp=amp,
            name='pi_sweep_square_pulse'
        )
        ## Sine
        sin_dur_dt = 960
        sin_w = np.pi / sin_dur_dt
        sin_t_dt = np.arange(sin_dur_dt)
        sin_A, sin_B = 0.8, 0
        sin_amps = amp * np.sin(sin_w * sin_t_dt) + sin_B
        sine = pulse_lib.Waveform(samples=sin_amps, name="sine pulse")

        ## Sine^2
        sin2_dur_dt = 1120
        sin2_w = np.pi / sin2_dur_dt
        sin2_t_dt = np.arange(sin2_dur_dt)
        sin2_B = 0
        sin2_amps = amp * (np.sin(sin2_w * sin2_t_dt)) ** 2 + sin2_B
        sine2 = pulse_lib.Waveform(samples=sin2_amps, name="sine^2 pulse")

        ## Sine^3
        sin3_dur_dt = 1280
        sin3_w = np.pi / sin3_dur_dt
        sin3_t_dt = np.arange(sin3_dur_dt)
        sin3_B = 0.
        sin3_amps = amp * (np.sin(sin3_w * sin3_t_dt)) ** 3 + sin3_B
        sine3 = pulse_lib.Waveform(samples=sin3_amps, name="sine^3 pulse")

        ## Lorentzian
        if ctrl_param == "width":
            G = lor_G
            lor_dur_dt = 2 * G * np.sqrt(100 / cutoff - 1)
            lor_dur_dt = get_closest_multiple_of_16(lor_dur_dt)
        elif ctrl_param == "duration":
            lor_dur_dt = lor_T
            G = lor_dur_dt / (2 * np.sqrt(100 / cutoff - 1))
        # lor_dur_dt = 2576
        lor_t_dt = np.arange(lor_dur_dt)
        lor_amps = amp / (((lor_t_dt - lor_dur_dt / 2) / G) ** 2 + 1)
        if remove_bg:
            lor_amps -= lor_amps[0]
        lor = pulse_lib.Waveform(samples=lor_amps, name="lorentzian pulse")
        
        ## Lorentzian^2
        if ctrl_param == "width":
            G2 = lor_G
            lor2_dur_dt = 2 * G2 * np.sqrt(100 / cutoff - 1)
            lor2_dur_dt = get_closest_multiple_of_16(lor2_dur_dt)
        elif ctrl_param == "duration":
            lor2_dur_dt = lor_T
            G2 = lor2_dur_dt / (2 * np.sqrt(100 / cutoff - 1))
        # lor2_dur_dt = 2576
        # G2 = 350
        lor2_t_dt = np.arange(lor2_dur_dt)
        lor2_amps = amp / (((lor2_t_dt - lor2_dur_dt / 2) / G2) ** 2 + 1) ** 2 
        if remove_bg:
            lor2_amps -= lor2_amps[0]
        lor2 = pulse_lib.Waveform(samples=lor2_amps, name="lorentzian^2 pulse")
        
        ## Sech
        sech_dur_dt = 1280
        sech_k = 8.52457502
        sech_w = 0.0095
        sech_B = 0
        sech_t_dt = np.arange(sech_dur_dt)
        sech_amps = amp / np.cosh(sech_w * (sech_t_dt - sech_dur_dt / 2)) + sech_B
        if remove_bg:
            sech_amps -= sech_amps[0]
        sech = pulse_lib.Waveform(samples=sech_amps, name="sech pulse")
        
        ## Sech^2
        sech2_dur_dt = 1280
        sech2_w = 0.005
        sech2_B = 0
        sech2_t_dt = np.arange(sech2_dur_dt)
        sech2_amps = 1 * amp / (np.cosh(sech2_w * (sech2_t_dt - sech2_dur_dt / 2))) ** 2 + sech2_B
        if remove_bg:
            sech2_amps -= sech2_amps[0]
        sech2 = pulse_lib.Waveform(samples=sech2_amps, name="sech^2 pulse")

        ## Demkov
        demkov_dur_dt = 2576
        demkov_t_dt = np.arange(demkov_dur_dt)
        demkov_B = 250
        demkov_amps = amp * np.exp(-np.abs(demkov_t_dt - demkov_dur_dt / 2) / demkov_B)
        if remove_bg:
            demkov_amps -= demkov_amps[0]
        demkov = pulse_lib.Waveform(samples=demkov_amps, name="demkov pulse")
        ##
        pulses = {
            "gauss": gauss,
            "sq": sq,
            "sine": sine,
            "sine2": sine2,
            "sine3": sine3,
            "lorentz": lor,
            "lorentz2": lor2,
            "sech": sech,
            "sech2": sech2,
            "demkov": demkov
        }

        pi_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {amp}")
        pi_schedule += Play(pulses[pulse_type], drive_chan)
        pi_schedule += measure << pi_schedule.duration
        schedules.append(pi_schedule)

    num_shots_per_exp = 2048

    # pi_job = execute(
    #     schedules,
    #     backend=backend, 
    #     meas_level=2,
    #     meas_return='avg',
    #     memory=True,
    #     shots=num_shots_per_exp,
    #     schedule_los=[{drive_chan: rough_qubit_frequency}] * num_exp
    # )
    job_manager = IBMQJobManager()
    pi_job = job_manager.run(
        schedules,
        backend=backend,
        shots=num_shots_per_exp,
        schedule_los={drive_chan: rough_qubit_frequency},
        max_experiments_per_job=args.max_experiments_per_job
    )

    # pi_job = backend.run(rabi_pi_qobj)
    # job_monitor(pi_job)

    # pi_sweep_results = pi_job.result(timeout=120)

    pi_sweep_results = pi_job.results()

    pi_sweep_values = []
    # for i in range(len(pi_sweep_results.results)):
    for i in range(len(schedules)):
        # # Get the results from the ith experiment
        # results = pi_sweep_results.get_memory(i)*1e-14
        # # Get the results for `qubit` from this experiment
        # pi_sweep_values.append(results[qubit])

        # length = len(pi_sweep_results.get_memory(i))
        # res = pi_sweep_results.get_memory(i)
        # _, counts = np.unique(res, return_counts=True)
        # Get the results for `qubit` from this experiment
        counts = pi_sweep_results.get_counts(i)["1"]
        pi_sweep_values.append(counts / num_shots_per_exp)

    if ctrl_param == "width":
        c_p = G
    elif ctrl_param == "duration": 
        if pulse_type == "lorentz":
            c_p = lor_dur_dt
        elif pulse_type == "lorentz2":
            c_p = lor2_dur_dt

    time = date.strftime("%H%M%S")

    amps_name = f"{pulse_type}_{time}_amps.pkl" if pulse_type not in ["lorentz", "lorentz2"] else \
        f"{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_{time}_amps.pkl"

    tr_prob_name = f"{pulse_type}_{time}_tr_prob.pkl" if pulse_type not in ["lorentz", "lorentz2"] else \
        f"{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_{time}_tr_prob.pkl"

    with open(os.path.join(data_folder, amps_name).replace("\\","/"), 'wb') as f1:
        pickle.dump(amplitudes, f1)
    with open(os.path.join(data_folder, tr_prob_name).replace("\\","/"), 'wb') as f2:
        pickle.dump(np.real(pi_sweep_values), f2)

    plt.figure(3)
    plt.scatter(amplitudes, np.real(pi_sweep_values), color='black') # plot real part of sweep values
    # plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    plt.title("Rabi Calibration Curve")
    plt.xlabel("Amplitude [a.u.]")
    plt.ylabel("Transition Probability")
    # plt.ylabel("Measured Signal [a.u.]")
    plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_pi_amp_sweep.png"))

    ## fit curve
    def fit_function(x_values, y_values, function, init_params):
        fitparams, conv = curve_fit(
            function, 
            x_values, 
            y_values, 
            init_params, 
            maxfev=100000, 
            bounds=(
                [-0.53, 1, 0, 0.45], 
                [-.47, 200, 1000, 0.55]
            )
        )
        y_fit = function(x_values, *fitparams)
        
        return fitparams, y_fit


    rabi_fit_params, _ = fit_function(
        amplitudes[: fit_crop_parameter],
        np.real(pi_sweep_values[: fit_crop_parameter]), 
        lambda x, A, l, p, B: A * (np.cos(l * (1 - np.exp(- p * x)))) + B,
        [-.5, args.l, args.p, 0.5]
        # lambda x, A, k, B: A * (np.cos(k * x)) + B,
        # [-0.5, 50, 0.5]
    )

    print(rabi_fit_params)
    A, l, p, B = rabi_fit_params

    pi_amp = - np.log(1 - np.pi / l) / p
    half_amp = - np.log(1 - np.pi / (2 * l)) / p

    detailed_amps = np.arange(amplitudes[0], amplitudes[-1], amplitudes[-1] / 2000)
    extended_y_fit = A * (np.cos(l * (1 - np.exp(- p * detailed_amps)))) + B

    ## create pandas series to keep calibration info
    param_dict = {
        "date": date.strftime("%Y-%m-%d"),
        "time": date.strftime("%H:%M:%S"),
        "pulse_type": pulse_type,
        "remove_bg": remove_bg,
        "A": A,
        "l": l,
        "p": p,
        "B": B,
        "drive_freq": rough_qubit_frequency,
        "pi_duration": pulses[pulse_type].duration,
        "pi_amp": pi_amp,
        "half_amp": half_amp,
        "backend": backend_name
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
    fig_name = date.strftime("%H%M%S") + f"_{pulse_type}_pi_amp_sweep_fitted.png" if pulse_type not in ["lorentz", "lorentz2"] else \
        date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_pi_amp_sweep_fitted.png"
    plt.savefig(os.path.join(save_dir, fig_name))

    plt.show()