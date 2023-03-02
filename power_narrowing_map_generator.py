import os
import pickle
from datetime import datetime
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

exp_date = "2023-03-02"
exp_time = "090904"
backend_name = "manila"

file_dir = os.path.dirname(__file__)

save_dir = os.path.join(
    file_dir,
    "plots",
    f"{backend_name}",
    "power_broadening (narrowing)",
    "lorentz_pulses",
    exp_date
)

folder_name = os.path.join(
    save_dir,
    exp_time
).replace("\\", "/")

data_folder = os.path.join(
    file_dir,
    "data",
    f"{backend_name}",
    "power_broadening (narrowing)",
    "lorentz_pulses",
    exp_date,
    exp_time
).replace("\\", "/")

dur_dt = 4800
cut_param = 0.5
G = dur_dt / (2 * np.sqrt((100 / cut_param) - 1))

with open(os.path.join(data_folder, f"{dur_dt}dt_cutparam-{cut_param}_tr_prob.pkl").replace("\\","/"), 'rb') as f1:
    transition_probability = pickle.load(f1)
with open(os.path.join(data_folder, f"{dur_dt}dt_cutparam-{cut_param}_areas.pkl").replace("\\","/"), 'rb') as f2:
    amplitudes = pickle.load(f2)
with open(os.path.join(data_folder, f"{dur_dt}dt_cutparam-{cut_param}_detunings.pkl").replace("\\","/"), 'rb') as f3:
    freq_offset = pickle.load(f3)

y, x = np.meshgrid(amplitudes, freq_offset)

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
        f"{exp_time}_lorentz_pwr_nrw_duration-{dur_dt}dt_g-{G}_cutparam-{cut_param}.png"
    ).replace("\\","/")
)
plt.show()