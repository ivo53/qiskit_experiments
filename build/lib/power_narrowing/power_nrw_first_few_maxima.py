import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "manila"
pulse_type = "lor3_4"
sigma = 96
dur = 5008
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

# time = ["2023-06-06", "201236"] # lor 3/4, sigma = 96, dur = 4128
time = ["2023-06-06", "201245"] # lor 3/4, sigma = 96, dur = 5008
# time = ["2023-06-04", "011733"] # lor, sigma = 96, dur = 2704
# time = ["2023-06-07", "023054"] # lor2_3, sigma = 48, dur = 5104
# time = ["2023-07-02", "002750"] # lor2, sigma = 96, dur = 704
# time = ["2023-03-12", "101213"] # rect sigma = 800, dur = 800

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

files = os.listdir(data_folder(time[0], time[1], pulse_type))
with open(os.path.join(data_folder(time[0], time[1], pulse_type), files[2]).replace("\\","/"), 'rb') as f1:
    tr_prob = pickle.load(f1)
with open(os.path.join(data_folder(time[0], time[1], pulse_type), files[0]), 'rb') as f2:
    amp = l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T)
with open(os.path.join(data_folder(time[0], time[1], pulse_type), files[1]), 'rb') as f3:
    det = pickle.load(f3) / 1e6

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
det_limits = 12

# Create a 3x3 grid of subplots with extra space for the color bar
fig, ax = plt.subplots(figsize=(6,5))
for i, t in zip(np.arange(len(amp[1::2])), tr_prob[1::2]):
    ax.scatter(det, t, c=colors[i],linewidth=0,marker=markers[i], s=sizes[i], label=f"Pulse area {2*i+1}$\pi$")
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
ax.legend()
fig.tight_layout()
# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"first_5_{pulse_type}_sigma_{sigma}_duration_{dur}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

# Save the fig
if save:
    plt.savefig(os.path.join(save_folder, fig_name))

# Display the plot
plt.show()
