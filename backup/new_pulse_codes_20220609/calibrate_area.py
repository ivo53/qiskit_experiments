import os
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import lambertw
from qiskit import pulse, IBMQ, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

backend_name = "belem"

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

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
qubit = 2
mem_slot = 0

drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)

provider = IBMQ.load_account()
backend = provider.get_backend(f"ibmq_{backend_name}")
# backend_name = str(backend)
print(f"Using {backend_name} backend.")
backend_defaults = backend.defaults()
backend_config = backend.configuration()

center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
dt = backend_config.dt

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

rough_qubit_frequency = center_frequency_Hz #4.97173 * GHz

## set params
pulse_type = "sq"
fit_crop = 1#.8
min_amplitude, max_amplitude = 0.001, .999#0.55#.65
num_exp = 200
amplitudes = np.linspace(
    min_amplitude, 
    max_amplitude, 
    num_exp
)
fit_crop_parameter = int(fit_crop * len(amplitudes))

dur_dt = 2256

print(f"The resonant frequency is assumed to be {rough_qubit_frequency}.")
print(f"The area calibration will start from amp {amplitudes[0]} "
f"and end at {amplitudes[-1]} with approx step {max_amplitude/num_exp}.")

# Create base circuit
q = QuantumRegister(1)
c = ClassicalRegister(1)
base_circ = QuantumCircuit(q, c)
base_circ.x(0)

circs = []
amp = Parameter('amp')
with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area") as sched:
    pulse.set_frequency(rough_qubit_frequency, drive_chan)
    pulse.play(
        pulse_lib.Constant(
            duration=dur_dt,
            amp=amp,
            name="sq_pulse"
        ),
        drive_chan
    )
    pulse.measure(
        qubits=[qubit], 
        registers=[pulse.MemorySlot(mem_slot)]
    )
for a in amplitudes:
    current_sched = sched.assign_parameters(
            {amp: a},
            inplace=False
    )
    circ_copy = deepcopy(base_circ)
    circ_copy.add_calibration("x", [qubit], current_sched)
    circs.append(circ_copy)

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
    circs,
    backend=backend,
    shots=num_shots_per_exp,
)

# pi_job = backend.run(rabi_pi_qobj)
# job_monitor(pi_job)

# pi_sweep_results = pi_job.result(timeout=120)

pi_sweep_results = pi_job.results()

pi_sweep_values = []
# for i in range(len(pi_sweep_results.results)):
for i in range(len(circs)):
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
    "pi_duration": dur_dt,
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