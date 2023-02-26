import os
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

## create folder where plots are saved
file_dir = os.path.dirname(__file__)
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
calib_dir = os.path.join(file_dir, "calibrations")
save_dir = os.path.join(calib_dir, current_date)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
data_folder = os.path.join(file_dir, "data", "calibration", current_date)
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

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
span = 0.008 * center_frequency_Hz
num_sweep_experiments = 100
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

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

drive_sigma = 0.05 # us
drive_samples = 8 * drive_sigma # us
drive_amp = 0.1
drive_sigma = get_closest_multiple_of_16(drive_sigma * us / dt)
drive_samples = get_closest_multiple_of_16(drive_samples * us/ dt)
pulse_type_freq_sweep = "gauss"

## Gauss
gauss = pulse_lib.gaussian(duration=drive_samples,
                           sigma=drive_sigma,
                           amp=drive_amp,
                           name='freq_sweep_excitation_pulse')

## Square
k = 22.56983823
pi_amp = np.pi / k
sq_dur = 0.2 #us
sq_dur = get_closest_multiple_of_16(sq_dur * us / dt)
sq = pulse_lib.constant(
    duration=sq_dur,
    amp=pi_amp,
    name='freq_sweep_excitation_pulse'
)

## Sine
sin_dur_dt = 960
sin_w = np.pi / sin_dur_dt
sin_t_dt = np.arange(sin_dur_dt)
sin_k = 15.52313743
sin_A = np.pi / sin_k
sin_B = 0
sin_amps = sin_A * np.sin(sin_w * sin_t_dt) + sin_B
sine = pulse_lib.Waveform(samples=sin_amps, name="sine pulse")

## Sine^2
sin2_dur_dt = 1120
sin2_w = np.pi / sin2_dur_dt
sin2_t_dt = np.arange(sin2_dur_dt)
sin2_k = 14.19073647
sin2_A = np.pi / sin2_k
sin2_B = 0
sin2_amps = sin2_A * (np.sin(sin2_w * sin2_t_dt)) ** 2 + sin2_B
sine2 = pulse_lib.Waveform(samples=sin2_amps, name="sine^2 pulse")

## Sine^3
sin3_dur_dt = 1280
sin3_w = np.pi / sin3_dur_dt
sin3_t_dt = np.arange(sin3_dur_dt)
sin3_k = 13.72311431
sin3_A = np.pi / sin3_k
sin3_B = 0
sin3_amps = sin3_A * (np.sin(sin3_w * sin3_t_dt)) ** 3 + sin3_B
sine3 = pulse_lib.Waveform(samples=sin3_amps, name="sine^3 pulse")

## Lorentzian
lor_dur_dt = 640
k_lor = 15.59393961
lor_A = 1000 * np.pi / k_lor
lor_B = 0
G = 1000
lor_t_dt = np.arange(lor_dur_dt)
lor_amps = lor_A * G / ((lor_t_dt - lor_dur_dt / 2) ** 2 + G ** 2) + lor_B
lor = pulse_lib.Waveform(samples=lor_amps, name="lorentzian pulse")

## Sech
sech_dur_dt = 1280
sech_k = 8.52457502
sech_A = np.pi / sech_k
sech_w = 0.0095
sech_B = 0
sech_t_dt = np.arange(sech_dur_dt)
sech_amps = sech_A / np.cosh(sech_w * (sech_t_dt - sech_dur_dt / 2)) + sech_B
sech = pulse_lib.Waveform(samples=sech_amps, name="sech pulse")

## Sech^2
sech2_dur_dt = 1280
sech2_w = 0.005
sech2_B = 0
sech2_k = 10.31068256
sech2_A = np.pi / sech2_k
sech2_t_dt = np.arange(sech2_dur_dt)
sech2_amps = sech2_A / (np.cosh(sech2_w * (sech2_t_dt - sech2_dur_dt / 2))) ** 2 + sech2_B
sech2 = pulse_lib.Waveform(samples=sech2_amps, name="sech^2 pulse")

##

pulses = {
    "gauss": gauss,
    "sq": sq,
    "sine": sine,
    "sine2": sine2,
    "sine3": sine3,
    "lorentz": lor,
    "sech": sech,
    "sech2": sech2
}
schedule = pulse.Schedule(name='Frequency sweep')
schedule += Play(pulses[pulse_type_freq_sweep], drive_chan)
schedule += measure << schedule.duration

schedule_frequencies = [{drive_chan: freq} for freq in frequencies]
# schedule.draw()

num_shots_per_frequency = 2048
job = execute(
    schedule,
    backend=backend, 
    meas_level=2,
    memory=True,
    meas_return='single',
    shots=num_shots_per_frequency,
    schedule_los=schedule_frequencies
)

# job = backend.run(frequency_sweep_program)
job_monitor(job)
frequency_sweep_results = job.result(timeout=120)

sweep_values = []
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    length = len(frequency_sweep_results.get_memory(i))
    res = frequency_sweep_results.get_memory(i)#*1e-14
    _, counts = np.unique(res, return_counts=True)

    # Get the results for `qubit` from this experiment
    # sweep_values.append(res[qubit])
    sweep_values.append(counts[1] / length)

frequencies_GHz = frequencies / GHz

data = {"frequency_ghz": frequencies_GHz, "transition_probability": (sweep_values)}
df = pd.DataFrame(data)
df.to_csv(
    os.path.join(
        data_folder, 
        date.strftime("%H%M%S") + "_gauss_calibration.csv"
    ),
    index=False
)
plt.figure(1)
plt.scatter(frequencies_GHz, (sweep_values), color='black') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.title("Drive Frequency Calibration Curve")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Transition Probability")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + "_frequency_sweep.png"))
# plt.show()

## fit curve


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)#, maxfev=100000)
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

fit_params, y_fit = fit_function(frequencies_GHz,
                                 np.real(sweep_values), 
                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [0.3, 4.975, 0.1, -10.5] # initial parameters for curve_fit
                                )
plt.figure(2)
plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
plt.plot(frequencies_GHz, y_fit, color='red')
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.title("Fitted Drive Frequency Calibration Curve")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Signal [a.u.]")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + "_frequency_sweep_fitted.png"))
# plt.show()

A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print(f"We've updated our qubit frequency estimate from "
      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")

# pi pulse
k = 56.1210057
pi_amp_01 = np.pi / k

pi_pulse_01 = pulse_lib.gaussian(
    duration=drive_samples,
    amp=pi_amp_01, 
    sigma=drive_sigma,
    name='pi_pulse_01'
)

# sideband pulse
sigma_us = 0.08
sigma_base_12 = get_closest_multiple_of_16(sigma_us*us/dt)
dur_base_12 = get_closest_multiple_of_16(8*sigma_us*us/dt)

base_12_pulse = pulse_lib.gaussian(
    duration=dur_base_12,
    sigma=sigma_base_12,
    amp=0.9,
    name='base_12_pulse'
)
num_freqs = 75
freqs = 4.6225 * GHz + np.linspace(-5 *MHz, 5*MHz, num_freqs)
t_samples = np.linspace(0, dt*dur_base_12, dur_base_12)
two_schedules = []
for freq in freqs:
    two_schedule = pulse.Schedule(name='Frequency sweep')
    sine_pulse = np.sin(2*np.pi*(freq-rough_qubit_frequency)*t_samples) # no amp for the sine
    sideband_pulse = pulse_lib.Waveform(
        np.multiply(
            np.real(
                base_12_pulse.samples
            ), 
            sine_pulse
        ),
        name='sideband_pulse'
    )
    two_schedule += Play(pi_pulse_01, drive_chan)
    two_schedule += Play(sideband_pulse, drive_chan) << two_schedule.duration
    two_schedule += measure << two_schedule.duration
    two_schedules.append(two_schedule)

two_job = execute(
    two_schedules,
    backend=backend, 
    meas_level=1,
    meas_return='avg',
    shots=num_shots_per_frequency,
    schedule_los=[{drive_chan: rough_qubit_frequency} for _ in range(num_freqs)]
)
job_monitor(two_job)

frequency_sweep_results_2 = two_job.result(timeout=120)
sweep_values_2 = []
for i in range(len(frequency_sweep_results_2.results)):
    # Get the results from the ith experiment
    res2 = frequency_sweep_results_2.get_memory(i)*1e-14
    # Get the results for `qubit` from this experiment
    sweep_values_2.append(res2[qubit])

freqs_GHz = freqs / GHz
plt.figure(3)
plt.scatter(freqs_GHz, np.abs(sweep_values_2), color='black') # plot real part of sweep values
plt.xlim([min(freqs_GHz), max(freqs_GHz)])
plt.title("Drive Frequency Level 2 Calibration Curve")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Transition Probability")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + "_frequency_sweep_2.png"))
# plt.show()

## fit curve

fit_params, y_fit = fit_function(
    freqs_GHz,
    np.abs(sweep_values_2), 
    lambda x, A, q_freq, B, C: (A) * (B**2 / ((x - q_freq)**2 + B**2)) + C,
    [0.3, 4.62, 0.005, -10.6] # initial parameters for curve_fit
)
plt.figure(4)
plt.scatter(freqs_GHz, np.abs(sweep_values_2), color='black')
plt.plot(freqs_GHz, y_fit, color='red')
plt.xlim([min(freqs_GHz), max(freqs_GHz)])
plt.title("Fitted Drive Frequency Level 2 Calibration Curve")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Signal [a.u.]")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + "_frequency_sweep_2_fitted.png"))
plt.show()

print(fit_params)