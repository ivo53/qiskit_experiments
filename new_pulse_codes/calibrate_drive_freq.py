from copy import deepcopy
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qiskit import (
    QuantumCircuit, 
    QuantumRegister, 
    ClassicalRegister, 
    pulse, 
    IBMQ
) 
# This is where we access all of our Pulse features!
from qiskit.circuit import Parameter
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.providers.ibmq.managed import IBMQJobManager
backend_name = "manila"

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
qubit = 0
mem_slot = 0

drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)

provider = IBMQ.load_account()
backend = provider.get_backend(f"ibmq_{backend_name}")
# backend = provider.get_backend("ibmq_qasm_simulator")
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

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

sigma_dt = 160
dur_dt = 8 * sigma_dt
amp = 0.2

# create base circuit
q = QuantumRegister(1)
c = ClassicalRegister(1)
base_circ = QuantumCircuit(q, c)
base_circ.x(0)

circs = []
freq = Parameter('freq')
with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_freq") as sched:
    pulse.set_frequency(freq, drive_chan)
    pulse.play(
        pulse_lib.Gaussian(
            duration=dur_dt,
            amp=amp,
            sigma=sigma_dt,
            name="Gaussian pulse"
        ),
        drive_chan
    )
    pulse.measure(
        qubits=[qubit], 
        registers=[pulse.MemorySlot(mem_slot)]
    )

for f in frequencies:
    current_sched = sched.assign_parameters(
        {freq: f},
        inplace=False
    )
    circ = deepcopy(base_circ)
    circ.add_calibration("x", [qubit], current_sched)
    circs.append(circ)



num_shots = 2048
max_experiments_per_job = 100

job_manager = IBMQJobManager()
job = job_manager.run(
    circs,
    backend=backend,
    name="Frequency calibration",
    max_experiments_per_job=max_experiments_per_job,
    shots=num_shots
)
frequency_sweep_results = job.results()

sweep_values = []
for i in range(len(circs)):
    counts = frequency_sweep_results.get_counts(i)["1"]
    sweep_values.append(counts / num_shots)


frequencies_GHz = frequencies / GHz

data = {"frequency_ghz": frequencies_GHz, "transition_probability": (sweep_values)}
df = pd.DataFrame(data)
df.to_csv(
    os.path.join(
        data_folder, 
        date.strftime("%H%M%S") + f"_calibration.csv"
    ),
    index=False
)
plt.figure(1)
plt.scatter(frequencies_GHz, (sweep_values), color='black') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.title("Drive Frequency Calibration Curve")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Transition Probability")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_frequency_sweep.png"))
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
plt.title("Fitted Drive Frequency Calibration Curve")
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Signal [a.u.]")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_frequency_sweep_fitted.png"))
plt.show()

A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print(f"We've updated our qubit frequency estimate from "
      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")
