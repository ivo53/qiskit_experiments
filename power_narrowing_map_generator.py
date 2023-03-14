import os
import pickle
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from qiskit import IBMQ

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

exp_date = "2023-03-12"
exp_time = "101213" #"234849" #"004901" #"015426" #"090904"
backend_name = "manila"

pulse_type = "rect"
file_dir = os.path.dirname(__file__)

save_dir = os.path.join(
    file_dir,
    "plots",
    f"{backend_name}",
    "power_broadening (narrowing)",
    f"{pulse_type}_pulses",
    exp_date
)
folder_name = os.path.join(
    save_dir,
    exp_time
).replace("\\", "/")
make_all_dirs(folder_name)
data_folder = os.path.join(
    file_dir,
    "data",
    f"{backend_name}",
    "power_broadening (narrowing)",
    f"{pulse_type}_pulses",
    exp_date,
    exp_time
).replace("\\", "/")
make_all_dirs(data_folder)

dur_dt = 800
cut_param = 0.2
G = dur_dt / (2 * np.sqrt((100 / cut_param) - 1))

GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
provider = IBMQ.load_account()
backend = provider.get_backend(f"ibmq_{backend_name}")
backend_defaults = backend.defaults()
qubit = 0 
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]

# center_frequency_Hz = 4962284031.287086
resolution = (100,100)
a_max = 0.5
# a_step = np.round(a_max / resolution[0], 7)
amplitudes = np.linspace(0., a_max + 1e-3, resolution[0]).round(3)
frequency_span_Hz = 25 * MHz #5 * MHz #if cut_param < 1 els e 1.25 * MHz
frequency_step_Hz = np.round(frequency_span_Hz / resolution[1], 3) #(1/4) * MHz

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz, 
                            frequency_max / GHz, 
                            frequency_step_Hz / GHz)

print(amplitudes.shape, frequencies_GHz.shape)
os.listdir(data_folder)
with open(os.path.join(data_folder, f"{dur_dt}dt_tr_prob.pkl").replace("\\","/"), 'rb') as f1:
    transition_probability = pickle.load(f1)
with open(os.path.join(data_folder, f"{dur_dt}dt_areas.pkl").replace("\\","/"), 'rb') as f2:
    amplitudes = pickle.load(f2)
with open(os.path.join(data_folder, f"{dur_dt}dt_detunings.pkl").replace("\\","/"), 'rb') as f3:
    freq_offset = pickle.load(f3)

y, x = np.meshgrid(amplitudes, freq_offset)

print(transition_probability.shape,x.shape, y.shape)
transition_probability = transition_probability.flatten().reshape(len(amplitudes), len(frequencies_GHz))
for i, am in enumerate(amplitudes):
    plt.figure(i)
    plt.plot(freq_offset, transition_probability.T[:, i], "bx")
    plt.xlabel("Detuning [MHz]")
    plt.ylabel("Transition Probability")
    plt.title(f"{pulse_type.capitalize()} Freq Offset - Amplitude {am.round(3)}")
    plt.savefig(os.path.join(folder_name, f"{pulse_type}_amp-{am.round(3)}.png").replace("\\","/"))
    plt.close()

fig, ax = plt.subplots(figsize=(5,4))

# print(transition_probability.shape)
# print(x.shape)
# print(y.shape)
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
        f"{exp_time}_{pulse_type}_pwr_nrw_duration-{dur_dt}dt.pdf"
    ).replace("\\","/")
    , format="pdf"
)
plt.show()