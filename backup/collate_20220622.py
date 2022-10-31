import os
from datetime import datetime
from matplotlib.markers import MarkerStyle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

AREA_VALUES = {
    "half": 0.5 * np.pi,
    "pi": np.pi,
    "3pi": 3 * np.pi,
    "5pi": 5 * np.pi,
    "7pi": 7 * np.pi,
}
TIMES = {
    "width": { # 2022-06-21
        0.2: "184609",
        0.5: "184614",
        1: "184618",
        2: "184621",
        5: "184624",
        10: "184627",
        20: "184632",
        30: "184637",
        50: "184641",
    },
    # "width": { # 2022-06-16
    #     0.2: "213127",
    #     0.5: "213130",
    #     1: "213134",
    #     2: "213138",
    #     5: "203210",
    #     10: "203216",
    #     20: "203221",
    #     30: "203224",
    #     50: "203228",
    # },
    "duration": { # 2022-06-17
        0.2: "003736",
        0.5: "003740",
        1: "003748",
        2: "003803",
        5: "003807",
        10: "003815",
        20: "003822",
        30: "003826",
        50: "003831",
    }
}
pulse_type = "lorentz"
date = "2022-06-21"
area = "pi"
ctrl_param = "width"
cutoff = 0.2
center_freq = 4.97172
file_dir = os.path.dirname(__file__)
data_folder = os.path.join(file_dir, "data", "armonk", "calibration", date)
data_files = os.listdir(data_folder)
csv_files = []
for i in TIMES[ctrl_param].keys():
    for d in data_files:
        if d.startswith(TIMES[ctrl_param][i]):
            csv_files.append(d)
            break

pulse_type = d.split("_")[1]

dfs = [pd.read_csv(os.path.join(data_folder, csv_file)) for csv_file in csv_files]

freqs = [dfs[i]["frequency_ghz"].to_numpy() for i in range(len(dfs))]
vals = [dfs[i]["transition_probability"].to_numpy() for i in range(len(dfs))]
det_mhz = (np.array(freqs, dtype="object") - center_freq) * 1e3
# fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [5, 1, 1]})
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(9,6))
for i, cutoff in enumerate(TIMES[ctrl_param].keys()):
    current_ax = ax[int(i//3)][int(i%3)]
    current_ax.scatter(det_mhz[i], vals[i], color='blue', marker="+", linewidths=0.7)
    current_ax.set_xlim([min(det_mhz[i]), max(det_mhz[i])])
    if i // 3 == 2:
        current_ax.set_xlabel("Detuning [MHz]", fontsize=8)
    if i % 3 == 0:
        current_ax.set_ylabel("Transition Probability", fontsize=8)
    current_ax.set_title(f"Cutoff Amplitude {cutoff}%", fontsize=9)
    major_xticks = np.arange(0, det_mhz[i][-1], 10)
    major_xticks = np.concatenate((-major_xticks[::-1][:-1], major_xticks))
    minor_xticks = np.arange(0, det_mhz[i][-1], 2)
    minor_xticks = np.concatenate((-minor_xticks[::-1][:-1], minor_xticks))
    major_yticks = np.arange(0, 1.01, 0.5)
    minor_yticks = np.arange(0, 1.01, 0.1)

    current_ax.set_xticks(major_xticks)
    current_ax.set_xticks(minor_xticks, minor="True")
    current_ax.set_yticks(major_yticks)
    current_ax.set_yticks(minor_yticks, minor="True")
    current_ax.grid(which="both")
    current_ax.grid(which='minor', alpha=0.2)
    current_ax.grid(which='major', alpha=0.5)
    
fig.suptitle(f"Frequency Profile of {pulse_type.capitalize()}-shaped Pulses of Area {area.capitalize()} with Different Cutoff Amplitudes")
# fig.label("Frequency [GHz]")


# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
# plt.savefig(os.path.join(save_dir, fig2_name))
plt.show()