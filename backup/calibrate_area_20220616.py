import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import lambertw
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

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

rough_qubit_frequency = 4.97173 * GHz

## set params
pulse_type = "sq"
fit_crop = 1#.8
min_amplitude, max_amplitude = 0.001, .999#0.55#.65
num_exp = 500
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
    gauss = pulse_lib.gaussian(
        duration=gauss_dur,
        sigma=gauss_sigma,
        amp=amp,
        name='pi_sweep_gauss_pulse'
    )

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
    lor_dur_dt = 2576
    lor_A, lor_B = 0.5, 0
    G = 150
    lor_t_dt = np.arange(lor_dur_dt)
    lor_amps = G * amp * G / ((lor_t_dt - lor_dur_dt / 2) ** 2 + G ** 2) + lor_B
    lor = pulse_lib.Waveform(samples=lor_amps, name="lorentzian pulse")
    
    ## Lorentzian^2
    lor2_dur_dt = 2576
    lor2_A, lor2_B = 0.5, 0
    G2 = 350
    lor2_t_dt = np.arange(lor2_dur_dt)
    lor2_amps = G2 * G2 * amp * (G2 / ((lor2_t_dt - lor2_dur_dt / 2) ** 2 + G2 ** 2)) ** 2 + lor2_B
    lor2 = pulse_lib.Waveform(samples=lor2_amps, name="lorentzian^2 pulse")
    
    ## Sech
    sech_dur_dt = 1280
    sech_k = 8.52457502
    sech_w = 0.0095
    sech_B = 0
    sech_t_dt = np.arange(sech_dur_dt)
    sech_amps = amp / np.cosh(sech_w * (sech_t_dt - sech_dur_dt / 2)) + sech_B
    sech = pulse_lib.Waveform(samples=sech_amps, name="sech pulse")
    
    ## Sech^2
    sech2_dur_dt = 1280
    sech2_w = 0.005
    sech2_B = 0
    sech2_t_dt = np.arange(sech2_dur_dt)
    sech2_amps = 1 * amp / (np.cosh(sech2_w * (sech2_t_dt - sech2_dur_dt / 2))) ** 2 + sech2_B
    sech2 = pulse_lib.Waveform(samples=sech2_amps, name="sech^2 pulse")

    ## Demkov
    demkov_dur_dt = 2576
    demkov_t_dt = np.arange(demkov_dur_dt)
    demkov_B = 250
    demkov_amps = amp * np.exp(-np.abs(demkov_t_dt - demkov_dur_dt / 2) / demkov_B)
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
    schedule_los={drive_chan: rough_qubit_frequency}
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

print(amplitudes, np.real(pi_sweep_values))
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
            [-0.53, 10, 1, 0, 0.45], 
            [-.47, 100, 100, 1000, 0.55]
        )
    )
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit


rabi_fit_params, _ = fit_function(
    amplitudes[: fit_crop_parameter],
    np.real(pi_sweep_values[: fit_crop_parameter]), 
    lambda x, A, k, l, p, B: A * (np.cos(k * (x) + l * (1 - np.exp(- p * x)))) + B,
    [-.5, 36.4, 16, 2, 0.5]
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
    "pi_duration": duration,
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