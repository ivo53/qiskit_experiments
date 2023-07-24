import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numerical_solutions import ndsolve_lorentz_spectre

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "manila"
save = 1


# times = {
#     0.5: ["2023-06-03", "120038"],
#     1: ["2023-06-03", "160754"],
#     2: ["2023-06-03", "161333"],
#     3: ["2023-06-03", "022027"],
#     5: ["2023-06-03", "022625"],
#     7.5: ["2023-06-03", "023608"],
#     15: ["2023-06-03", "120636"],
#     20: ["2023-06-03", "115516"],
#     50: ["2023-06-03", "222957"],
# }

# # time = ["2023-06-06", "201236"] # lor 3/4, sigma = 96, dur = 4128
# time = ["2023-06-06", "201245"] # lor 3/4, sigma = 96, dur = 5008
# # time = ["2023-06-04", "011733"] # lor, sigma = 96, dur = 2704
# # time = ["2023-06-07", "023054"] # lor2_3, sigma = 48, dur = 5104
# # time = ["2023-07-02", "002750"] # lor2, sigma = 96, dur = 704
# # time = ["2023-03-12", "101213"] # rect sigma = 800, dur = 800

times = {
    "lor2": ["2023-07-04", "192453"],
    "lor": ["2023-07-04", "192414"],
    "lor3_4": ["2023-07-04", "192347"],
    "lor2_3": ["2023-07-04", "192350"], 
}
params = {
    "lor2": [(24 + 8/9) * 1e-9, (181 + 2/3) * 1e-9],
    "lor": [(24 + 8/9) * 1e-9, (704) * 1e-9],
    "lor3_4": [(10 + 2/3) * 1e-9, (728 + 8/9) * 1e-9],
    "lor2_3": [(10 + 2/3) * 1e-9, (1134 + 2/9) * 1e-9],
}
powers = [2, 1, 3/4, 2/3]

## create folder where plots are saved
file_dir = os.path.dirname(__file__)

def save_dir(date, time, pulse_type):
    save_dir = os.path.join(
        file_dir,
        "plots",
        f"{backend_name}",
        "power_broadening (narrowing)",
        f"{pulse_type}_pulses",
        date
    )
    folder_name = os.path.join(
        save_dir,
        time
    ).replace("\\", "/")
    return save_dir, folder_name

def data_folder(date, time, pulse_type):
    return os.path.join(
        file_dir,
        "data",
        f"{backend_name}",
        "power_broadening (narrowing)",
        f"{pulse_type}_pulses",
        date,
        time
    ).replace("\\", "/")

l, p, x0 = 419.1631352890144, 0.0957564968883284, 0.0003302995697281
T = 192 * 2/9 * 1e-9

det_limits = 15
numerical_dets, numerical_tr_probs = [], []
for i in range(4):
    area_dets, area_probs = [], []
    for j in [0,1,3]:
        numer_det, numerical_tr_prob = ndsolve_lorentz_spectre(
            params[list(times.keys())[i]][0],
            params[list(times.keys())[i]][1],
            -det_limits * 1e6 * 2 * np.pi, 100,
            d_end=det_limits * 1e6 * 2 * np.pi,
            num_t=1000,
            pulse_area=(2*j+1) * np.pi,
            lor_power=powers[i]
        )
        area_dets.append(numer_det / (1e6 * 2 * np.pi))
        area_probs.append(numerical_tr_prob)
    numerical_dets.append(area_dets)
    numerical_tr_probs.append(area_probs)

tr_probs, amps, dets = [], [], []
for i, k in times.items():
    files = os.listdir(data_folder(k[0], k[1], i))
    with open(os.path.join(data_folder(k[0], k[1], i), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob = pickle.load(f1)
    with open(os.path.join(data_folder(k[0], k[1], i), files[0]), 'rb') as f2:
        amp = l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T)
    with open(os.path.join(data_folder(k[0], k[1], i), files[1]), 'rb') as f3:
        det = pickle.load(f3) / 1e6
    tr_probs.append(tr_prob)
    amps.append(amp)
    dets.append(det)

colors = [
    "red",
    "brown",
    "green",
    "blue",
    "purple"
]

markers = [
    "o",
    "*",
    "X",
    "D",
    "P"
]
sizes = 15 * np.ones(5)

# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(12, 9))
gs0 = fig.add_gridspec(2, 2, width_ratios=[1, 1])
# Adjust the layout
fig.tight_layout()
# Generate datetime
date = datetime.now()

for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs0[i, j])
        for idx, order, t in zip([0,1,2], [0,1,3], tr_probs[2*i+j][1::2][[0,1,3]]):
            ax.scatter(dets[2*i+j], t, c=colors[idx],linewidth=0,marker=markers[idx], s=sizes[idx], label=f"Pulse area {2*order+1}$\pi$")
            ax.plot(numerical_dets[2*i+j][idx], numerical_tr_probs[2*i+j][idx], c=colors[idx],linewidth=0.2, label=f"Simulation - area {2*order+1}$\pi$")
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # ax.set_xticks([-12, -9, -6, -3, 0, 3, 6, 9, 12])
        ax.set_xticks(np.arange(-det_limits, det_limits+0.001, 3))
        y_minor_ticks = np.arange(0, 1.01, 0.05)
        # x_minor_ticks = np.arange(-12, 12+0.001, 1)
        x_minor_ticks = np.arange(-det_limits, det_limits+0.001, 1)
        ax.set_yticks(y_minor_ticks, minor="True")
        ax.set_xticks(x_minor_ticks, minor="True")
        ax.set_ylim((-0.025,1))
        # ax.set_xlim((-13,13))
        ax.set_xlim((-det_limits - 1, det_limits + 1))
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.6)
        ax.set_ylabel("Transition Probability")
        ax.set_xlabel("Detuning (MHz)")
        # ax.legend()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"block_1,3,7pi_{list(times.keys())}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

# Save the fig
if save:
    plt.savefig(os.path.join(save_folder, fig_name))

# Display the plot
plt.show()
