import argparse
import os
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qiskit import pulse, IBMQ, assemble, execute 
# This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
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
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str)
    parser.add_argument("-fs", "--freq_span", default=30, type=float)
    parser.add_argument("-cp", "--control_param", default="width", type=str)
    parser.add_argument("-c", "--cutoff", default=0.5, type=float)
    parser.add_argument("-G", "--lorentz_G", default=180, type=float)
    parser.add_argument("-T", "--lorentz_T", default=2256, type=int)
    parser.add_argument("-ne", "--num_experiments_per_job", default=100, type=int)
    parser.add_argument("-nf", "--num_freq", default=100, type=int)
    parser.add_argument("-a", "--area", default="pi", type=str)
    args = parser.parse_args()

    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    calib_dir = os.path.join(file_dir, "calibrations")
    save_dir = os.path.join(calib_dir, current_date)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    data_folder = os.path.join(file_dir, "data", "armonk", "calibration", current_date)
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

    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)

    provider = IBMQ.load_account()
    backend = provider.get_backend("ibmq_armonk")
    # backend = provider.get_backend("ibmq_qasm_simulator")
    backend_name = str(backend)
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    span = args.freq_span * MHz # 0.004 * center_frequency_Hz
    num_sweep_experiments = args.num_freq
    frequencies = np.arange(center_frequency_Hz - span / 2, 
                            center_frequency_Hz + span / 2,
                            span / num_sweep_experiments)
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


    pulse_type = args.pulse_type

    ## Gauss
    drive_sigma = 0.05 # us
    drive_samples = 8 * drive_sigma # us
    gauss_l = 62.84861205556516
    gauss_p = 0.2336463810098461
    gauss_pi_amp = - np.log(1 - np.pi / gauss_l) / gauss_p
    gauss_half_amp = - np.log(1 - np.pi / (2 * gauss_l)) / gauss_p
    drive_sigma = get_closest_multiple_of_16(drive_sigma * us / dt)
    drive_samples = get_closest_multiple_of_16(drive_samples * us/ dt)
    gauss = pulse_lib.gaussian(duration=drive_samples,
                            sigma=drive_sigma,
                            amp=gauss_pi_amp,
                            name='freq_sweep_excitation_pulse')

    ## Square
    sq_l = 134.51518325444695
    sq_p = 0.4478258714863412
    sq_pi_amp = - np.log(1 - np.pi / sq_l) / sq_p
    sq_half_amp = - np.log(1 - np.pi / (2 * sq_l)) / sq_p
    sq_dur = 0.5 #us
    sq_dur = get_closest_multiple_of_16(sq_dur * us / dt)
    sq = pulse_lib.Constant(
        duration=sq_dur,
        amp=sq_pi_amp,
        name='freq_sweep_excitation_pulse'
    )

    ## Sine
    sin_l = 52.357224097109786
    sin_p = 0.3078839508818245
    sin_dur_dt = 960
    sin_w = np.pi / sin_dur_dt
    sin_t_dt = np.arange(sin_dur_dt)
    sin_pi_amp = - np.log(1 - np.pi / sin_l) / sin_p
    sin_half_amp = - np.log(1 - np.pi / (2 * sin_l)) / sin_p
    sin_B = 0
    sin_amps = sin_pi_amp * np.sin(sin_w * sin_t_dt) + sin_B
    sine = pulse_lib.Waveform(samples=sin_amps, name="sine pulse")

    ## Sine^2
    sin2_l = 53.89822071260082
    sin2_p = 0.2732390971650457
    sin2_dur_dt = 1120
    sin2_w = np.pi / sin2_dur_dt
    sin2_t_dt = np.arange(sin2_dur_dt)
    sin2_pi_amp = - np.log(1 - np.pi / sin2_l) / sin2_p
    sin2_half_amp = - np.log(1 - np.pi / (2 * sin2_l)) / sin2_p
    sin2_B = 0
    sin2_amps = sin2_pi_amp * (np.sin(sin2_w * sin2_t_dt)) ** 2 + sin2_B
    sine2 = pulse_lib.Waveform(samples=sin2_amps, name="sine^2 pulse")

    ## Sine^3
    sin3_l = 58.53704682384882
    sin3_p = 0.2426385511427126
    sin3_dur_dt = 1280
    sin3_w = np.pi / sin3_dur_dt
    sin3_t_dt = np.arange(sin3_dur_dt)
    sin3_pi_amp = - np.log(1 - np.pi / sin3_l) / sin3_p
    sin3_half_amp = - np.log(1 - np.pi / (2 * sin3_l)) / sin3_p
    sin3_B = 0
    sin3_amps = sin3_pi_amp * (np.sin(sin3_w * sin3_t_dt)) ** 3 + sin3_B
    sine3 = pulse_lib.Waveform(samples=sin3_amps, name="sine^3 pulse")

    ## Lorentzian
    area = args.area
    cutoff = args.cutoff
    ctrl_param = args.control_param
    if ctrl_param == "width":
        G = args.lorentz_G
        lor_dur_dt = 2 * G * np.sqrt(100 / cutoff - 1)
        lor_dur_dt = get_closest_multiple_of_16(lor_dur_dt)
    elif ctrl_param == "duration":
        lor_dur_dt = get_closest_multiple_of_16(args.lorentz_T)
        G = lor_dur_dt / (2 * np.sqrt(100 / cutoff - 1))
    
    sought_date = current_date
    params_df = pd.read_csv(os.path.join(file_dir, "calibrations", "params.csv"))
    params_df = params_df[params_df["date"] == f"{sought_date}"]
    params_df = params_df[params_df["cutoff"] == cutoff]
    params_df = params_df[params_df["G"] == G]
    params_df = params_df[params_df["pi_duration"] == lor_dur_dt]
    params_df.sort_values(by="time", ascending=False, inplace=True)
    l = params_df.iloc[0, :]["l"]
    p = params_df.iloc[0, :]["p"]
    ## width 20220621
    # lor_l = {0.4: 56.77199428658362, 0.5: 54.91731471693372, 0.6: 56.26892763432354, 
    #     1.5: 58.85915834379367, 1.75: 56.2150878976182, 2: 56.80061636900623, 
    #     2.25: 59.02321480892749, 2.5: 60.76913526188999,
    #     0.7: 55.120183437340856, 1.9: 57.04537246610142, 2.1: 56.68569368707239}
    # lor_p = {0.4: 0.21104817681807553, 0.5: 0.2178857841466114, 0.6: 0.2112529225915634, 
    #     1.5: 0.194746619911249, 1.75: 0.2034756022278653, 2: 0.1994700022660682, 
    #     2.25: 0.1906740666928195, 2.5: 0.1835084266636036,
    #     0.7: 0.2151091276858692, 1.9: 0.1992296794887094, 2.1: 0.19946357826054156}
    ## duration 20220616
    # lor_l = {0.2: 91.39007278512425, 0.5: 93.45232177943862, 1: 78.77197364751918, 
    #     2: 89.32620119308197, 5: 88.28364023430483, 10: 93.75305942618854, 
    #     20: 20.75379254882979, 30: 20.29610432235783, 50: 15.883102793312124}
    # lor_p = {0.2: 0.1292315293383349, 0.5: 0.1244469529325799, 1: 0.145878761296941,
    #     2: 0.1240125793539657, 5: 0.1176810848138511, 10: 0.1017532400671358,
    #     20: 0.4386204785513318, 30: 0.3978699140119533, 50: 0.39994682516041763}
    ## width
    # lor_l = {0.2: 14.63068824145528, 0.5: 13.952286313454197, 1: 60.93722878197972, 
    #     2: 54.99316701884349, 5: 73.55394505096632, 10: 87.98513448856242, 
    #     20: 103.93357411037624, 30: 111.06850344447228, 50: 115.45161766441902}
    # lor_p = {0.2: 0.2804979578817719, 0.5: 0.485348788505724, 1: 0.1423067863399656,
    #     2: 0.2222151070267957, 5: 0.2515045873299424, 10: 0.2841674835644741,
    #     20: 0.321484724418223, 30: 0.3538873617515627, 50: 0.41462025376828227}
    # l, p = lor_l[cutoff], lor_p[cutoff]
    lor_pi_amp = - np.log(1 - np.pi / l) / p
    lor_half_amp = - np.log(1 - np.pi / (2 * l)) / p
    lor_3pi_amp = - np.log(1 - 3 * np.pi / (l)) / p
    lor_5pi_amp = - np.log(1 - 5 * np.pi / (l)) / p
    areas = {
        "pi": lor_pi_amp,
        "3pi": lor_3pi_amp,
        "5pi": lor_5pi_amp,
        "half": lor_half_amp,
    }
    if area == "3pi":
        assert lor_3pi_amp < 1, "Max amplitude is 1 (3pi)"
    elif area == "5pi":
        assert lor_5pi_amp < 1, "Max amplitude is 1 (5pi)"
    lor_t_dt = np.arange(lor_dur_dt)
    lor_amps = areas[area] / (((lor_t_dt - lor_dur_dt / 2) / G) ** 2 + 1) 
    lor = pulse_lib.Waveform(samples=lor_amps, name="lorentzian pulse")
    
    # ## Lorentzian^2
    # if ctrl_param == "width":
    #     G2 = args.lorentz_G
    #     lor2_dur_dt = 2 * G2 * np.sqrt(100 / cutoff - 1)
    #     lor2_dur_dt = get_closest_multiple_of_16(lor2_dur_dt)
    # elif ctrl_param == "duration":
    #     lor2_dur_dt = get_closest_multiple_of_16(args.lorentz_T)
    #     G2 = lor2_dur_dt / (2 * np.sqrt(100 / cutoff - 1))
    # lor2_l = {0.2: None, 0.5: None, 1: None, 2: None, 5: None, 10: None, 20: None, 30: None, 50: None}
    # lor2_p = {0.2: None, 0.5: None, 1: None, 2: None, 5: None, 10: None, 20: None, 30: None, 50: None}
    # lor2_pi_amp = - np.log(1 - np.pi / lor2_l[cutoff]) / lor2_p[cutoff]
    # lor2_half_amp = - np.log(1 - np.pi / (2 * lor2_l[cutoff])) / lor2_p[cutoff]
    # lor2_t_dt = np.arange(lor2_dur_dt)
    # lor2_amps = lor2_pi_amp / (((lor2_t_dt - lor2_dur_dt / 2) / G2) ** 2 + 1) ** 2
    # lor2 = pulse_lib.Waveform(samples=lor2_amps, name="lorentzian^2 pulse")

    ## Sech
    sech_l = 20.93027598991279
    sech_p = 0.4299824030518138
    sech_dur_dt = 1280
    sech_pi_amp = - np.log(1 - np.pi / sech_l) / sech_p
    sech_half_amp = - np.log(1 - np.pi / (2 * sech_l)) / sech_p
    sech_w = 0.0095
    sech_B = 0
    sech_t_dt = np.arange(sech_dur_dt)
    sech_amps = sech_pi_amp / np.cosh(sech_w * (sech_t_dt - sech_dur_dt / 2)) + sech_B
    sech = pulse_lib.Waveform(samples=sech_amps, name="sech pulse")

    ## Sech^2
    sech2_l = 70.76612319503813
    sech2_p = 0.1441850799053603
    sech2_dur_dt = 1280
    sech2_w = 0.005
    sech2_B = 0
    sech2_pi_amp = - np.log(1 - np.pi / sech2_l) / sech2_p
    sech2_half_amp = - np.log(1 - np.pi / (2 * sech2_l)) / sech2_p
    sech2_t_dt = np.arange(sech2_dur_dt)
    sech2_amps = sech2_pi_amp / (np.cosh(sech2_w * (sech2_t_dt - sech2_dur_dt / 2))) ** 2 + sech2_B
    sech2 = pulse_lib.Waveform(samples=sech2_amps, name="sech^2 pulse")

    ## Demkov
    demkov_l = 82.50739898903716
    demkov_p = 0.1573706940073794
    demkov_pi_amp = - np.log(1 - np.pi / demkov_l) / demkov_p
    demkov_half_amp = - np.log(1 - np.pi / (2 * demkov_l)) / demkov_p
    demkov_dur_dt = 2576
    demkov_t_dt = np.arange(demkov_dur_dt)
    demkov_B = 250
    demkov_amps = demkov_pi_amp * np.exp(-np.abs(demkov_t_dt - demkov_dur_dt / 2) / demkov_B)
    demkov = pulse_lib.Waveform(samples=demkov_amps, name="demkov pulse")

    ##

    pulses = {
        "gauss": gauss,
        "sq": sq,
        "sine": sine,
        "sine2": sine2,
        "sine3": sine3,
        "lorentz": lor,
        "sech": sech,
        "sech2": sech2,
        "demkov": demkov
    }
    schedule = pulse.Schedule(name='Frequency sweep')
    schedule += Play(pulses[pulse_type], drive_chan)
    schedule += measure << schedule.duration

    schedule_frequencies = [{drive_chan: freq} for freq in frequencies]
    # schedule.draw()

    num_shots_per_frequency = 2048
    job_manager = IBMQJobManager()
    jobs = job_manager.run(
        [schedule for _ in range(len(schedule_frequencies))],
        backend=backend, 
        # meas_level=2,
        # memory=True,
        # meas_return='single',
        shots=num_shots_per_frequency,
        schedule_los=schedule_frequencies,
        max_experiments_per_job=args.num_experiments_per_job
    )

    # job = backend.run(frequency_sweep_program)
    # job_monitor(job)
    # frequency_sweep_results = job.result(timeout=120)
    frequency_sweep_results = jobs.results()

    sweep_values = []
    for i in range(len(schedule_frequencies)):
        # # Get the results from the ith experiment
        # length = len(frequency_sweep_results.get_memory(i))
        # res = frequency_sweep_results.get_memory(i)#*1e-14
        # _, counts = np.unique(res, return_counts=True)

        # # Get the results for `qubit` from this experiment
        # # sweep_values.append(res[qubit])
        # sweep_values.append(counts[1] / length)
        counts = frequency_sweep_results.get_counts(i)["1"]
        sweep_values.append(counts / num_shots_per_frequency)

    frequencies_GHz = frequencies / GHz
    lorentz_pulses = ["lorentz", "lorentz2"]
    if ctrl_param == "width":
        c_p = G
    elif ctrl_param == "duration": 
        if pulse_type == "lorentz":
            c_p = lor_dur_dt
        elif pulse_type == "lorentz2":
            c_p = None#lor2_dur_dt
    
    csv_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_calibration.csv" if pulse_type not in lorentz_pulses else \
        date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_calibration.csv"

    data = {"frequency_ghz": frequencies_GHz, "transition_probability": (sweep_values)}
    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(
            data_folder, 
            csv_name
        ),
        index=False
    )
    plt.figure(1)
    plt.scatter(frequencies_GHz, (sweep_values), color='black') # plot real part of sweep values
    plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    plt.title(f"{pulse_type.capitalize()} Frequency Curve (Area {area}, Cutoff {cutoff}%)")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Transition Probability")
    fig1_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep.png" if pulse_type not in lorentz_pulses else \
        date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep.png"

    plt.savefig(os.path.join(save_dir, fig1_name))
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
    plt.title(f"Fitted {pulse_type.capitalize()} Frequency Curve (Area {area})")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Measured Signal [a.u.]")
    fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
        date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
    plt.savefig(os.path.join(save_dir, fig2_name))
    plt.show()

    A, rough_qubit_frequency, B, C = fit_params
    rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
    print(f"We've updated our qubit frequency estimate from "
        f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")
