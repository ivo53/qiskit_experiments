import os
import sys
import pickle
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
# from qiskit.tools.jupyter import *
from qiskit import IBMQ, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import pulse                  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter, Gate # This is Parameter Class for variable parameters.
from qiskit.pulse import library as pulse_lib
from qiskit.scheduler import measure
from qiskit import assemble
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager
from qiskit_ibm_provider import IBMProvider

current_dir = os.path.dirname(__file__)
package_path = os.path.abspath(os.path.split(current_dir)[0])
sys.path.insert(0, package_path)

from pulse_types import *
from utils.run_jobs import run_jobs

backend_name = "manila"

def make_all_dirs(path):
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
save_dir = os.path.join(
    file_dir,
    "plots",
    f"{backend_name}",
    "power_broadening (narrowing)",
    "lorentz_pulses",
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
    "lorentz_pulses",
    current_date,
    date.strftime("%H%M%S")
).replace("\\", "/")
make_all_dirs(data_folder)
make_all_dirs(folder_name)

backend_full_name = "ibm_" + backend_name \
    if backend_name in ["perth", "lagos", "nairobi", "oslo"] \
        else "ibmq_" + backend_name

backend = IBMProvider().get_backend(backend_full_name)
backend_config = backend.configuration()
print(f"Using backend {backend_full_name}")

dt = backend_config.dt
print(f"Sampling time: {dt*1e9} ns")

backend_defaults = backend.defaults()

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc.)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
# We will find the qubit frequency for the following qubit.
qubit = 0

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

# Convert seconds to dt
def get_dt_from(sec):
    return get_closest_multiple_of_16(sec/dt)

# The sweep will be centered around the estimated qubit frequency.
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
                                                                    # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")



# Drive pulse parameters (us = microseconds)
dur_dt = 30 * 16#1117 * 16 #525 * 16 #644 * 16 #483 * 16 #5152

resolution = (100,100)#(5, 10)

cut_param = 0.2
G = dur_dt / (2 * np.sqrt((100 / cut_param) - 1)) # 400
G_02 = dur_dt / (2 * np.sqrt((100 / 0.2) - 1)) # gamma factor at cut param 0.2

a_02 = 0.901 # max amp at cut param 0.2
a_max = a_02 * (G_02 * np.arctan(dur_dt / G_02)) / (G * np.arctan(dur_dt / G))

a_max = 0.5
# a_step = np.round(a_max / resolution[0], 3)

frequency_span_Hz = 18 * MHz #5 * MHz #if cut_param < 1 els e 1.25 * MHz
frequency_step_Hz = np.round(frequency_span_Hz / resolution[1], 3) #(1/4) * MHz

max_experiments_per_job = 100

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz, 
                            frequency_max / GHz, 
                            frequency_step_Hz / GHz)

amplitudes = np.linspace(0., a_max + 1e-3, resolution[0]).round(3)

print(f"The gamma factor is G = {G} and cut param is {cut_param}, \
compared to G_02 = {G_02} at cut param 0.2.")
print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
in steps of {frequency_step_Hz / MHz} MHz.")
print(f"The amplitude will go from {amplitudes[0]} to {amplitudes[-1]} in steps of {np.round(a_max / resolution[0], 4)}.")

frequencies_Hz = frequencies_GHz * GHz

# Create base circuit
# base_circ.measure(0, 0)
# base_circ.draw(output='mpl')
drive_chan = pulse.DriveChannel(qubit)

def add_circ(amp, freq):
    with pulse.build(backend=backend, default_alignment='sequential', name="lorentz_2d") as sched:
        pulse.set_frequency(freq, drive_chan)
        pulse.play(
            Lorentzian(
                duration=dur_dt,
                amp=amp,
                sigma=G,
                name='lor_pulse'
            ),
            drive_chan
        )
    lorentz = Gate("lorentz", 1, [])
    base_circ = QuantumCircuit(5, 1)
    base_circ.append(lorentz, [qubit])
    base_circ.measure(qubit, 0)
    base_circ.add_calibration(lorentz, (qubit,), sched, [])
    return base_circ

# Create the base schedule
# Start with drive pulse acting on the drive channel
# circs = []
# freq_off = Parameter('freq_off')
# amp = Parameter('amp')
# with pulse.build(backend=backend, default_alignment='sequential', name="lorentz_2d") as sched:
    
#     pulse.set_frequency(freq_off, drive_chan)
#     pulse.play(
#         Lorentzian(
#             duration=dur_dt,
#             amp=amp,
#             sigma=G,
#             name='lor_pulse'
#         ),
#         drive_chan
#     )
#     # Create the frequency settings for the sweep (MUST BE IN HZ)
# frequencies_Hz = frequencies_GHz*GHz

# lorentz = Gate("lorentz", 1, [amp, freq_off])
# base_circ = QuantumCircuit(5, 1)
# base_circ.append(lorentz, [qubit])
# base_circ.measure(qubit, 0)
# base_circ.add_calibration(lorentz, (qubit,), sched, [amp, freq_off])
# circs = [
#     base_circ.assign_parameters(
#             {amp: a, freq_off: f},
#             inplace=False
#     ) for a in amplitudes for f in frequencies_Hz]

# for a in amplitudes:
#     for f in frequencies_Hz - center_frequency_Hz:
#         current_sched = sched.assign_parameters(
#                 {freq_off: f, amp: a},
#                 inplace=False
#         )
#         circ_copy = deepcopy(base_circ)
#         circ_copy.add_calibration("x", [qubit], current_sched)
#         circs.append(circ_copy)


circs = [add_circ(a, f) for a in amplitudes for f in frequencies_Hz]

# num_schedules = len(schedules)
num_circs = len(circs)
num_shots = 1024

# job_manager = IBMQJobManager()

# jobs = job_manager.run(
#     circs,
#     backend=backend, 
#     shots=num_shots,
#     max_experiments_per_job=max_experiments_per_job,
#     name="Lorentzian Power Narrowing"
# )
# results = jobs.results()
# transition_probability = []
# for i in range(num_circs):
#     print(results.get_counts(i))
#     try:
#         transition_probability.append(results.get_counts(i)["1"] / num_shots)
#     except KeyError:
#         transition_probability.append(1 - results.get_counts(i)["0"] / num_shots)

transition_probability, job_ids = run_jobs(circs, backend, dur_dt, num_shots_per_exp=num_shots)

transition_probability = np.array(transition_probability).reshape(len(amplitudes), len(frequencies_GHz))
# job_set_id = jobs.job_set_id()
print("JobsID:", job_ids)

## save final data
freq_offset = (frequencies_Hz - center_frequency_Hz) / 10**6
y, x = np.meshgrid(amplitudes, freq_offset)
# z = np.reshape(transition_probability, (len(frequencies_Hz),len(amplitudes)))

with open(os.path.join(data_folder, f"{dur_dt}dt_cutparam-{cut_param}_tr_prob.pkl").replace("\\","/"), 'wb') as f1:
    pickle.dump(transition_probability, f1)
with open(os.path.join(data_folder, f"{dur_dt}dt_cutparam-{cut_param}_areas.pkl").replace("\\","/"), 'wb') as f2:
    pickle.dump(amplitudes, f2)
with open(os.path.join(data_folder, f"{dur_dt}dt_cutparam-{cut_param}_detunings.pkl").replace("\\","/"), 'wb') as f3:
    pickle.dump((frequencies_Hz - center_frequency_Hz), f3)

for i, am in enumerate(amplitudes):
    plt.figure(i)
    plt.plot(freq_offset, transition_probability[:, i], "bx")
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
        f"{date.strftime('%H%M%S')}_lorentz_pwr_nrw_duration-{dur_dt}dt_g-{G}_cutparam-{cut_param}.png"
    ).replace("\\","/")
)
plt.show()