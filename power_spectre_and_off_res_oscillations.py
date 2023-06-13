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
pulse_type = "rect"
sigma = 800
dur = 800

fixed_detuning = 3 # MHz
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

# time = ["2023-06-06", "013505"] # lor3_4 sigma = 96,dur = 5008
# time = ["2023-06-03", "120038"] # lorentz sigma = 96,dur = 2704
# time = ["2023-06-06", "202254"] # lor2_3 sigma = 48, dur = 5104
time = ["2023-03-12", "101213"] # rect sigma = 800, dur = 800

amp_end_idx = 100
amp_map_end_idx = 100
detuning_start_idx, detuning_end_idx = 0,100
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

det_idx_0, det_idx_1 = np.argsort(np.abs(np.abs(det) - fixed_detuning))[[0,1]]


# Generate datetime
date = datetime.now()
# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

fig0, ax0 = plt.subplots(figsize=(8, 6))
cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
ax0.grid()
ax0.scatter(amp[:amp_end_idx], tr_prob[:amp_end_idx, det_idx_0], linewidth=0, cmap=cmap)
ax0.set_xlabel("Rabi Freq. Amplitude (MHz)")
ax0.set_ylabel("Transition Probability")
# Set fig name
fig_name = f"rabi_oscillations_detuning_{(-1) ** (int(det_idx_0 < det_idx_1)) * fixed_detuning}_{pulse_type}_sigma_{sigma}_duration_{dur}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

# Save off-resonant Rabi oscillations
# plt.savefig(os.path.join(save_folder, fig_name))
plt.show()
plt.close()

fig0, ax0 = plt.subplots(figsize=(8, 6))
cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
ax0.grid()
ax0.scatter(amp[:amp_end_idx], tr_prob[:amp_end_idx, det_idx_1], linewidth=0, cmap=cmap)
ax0.set_xlabel("Rabi Freq. Amplitude (MHz)")
ax0.set_ylabel("Transition Probability")
# Set fig name
fig_name = f"rabi_oscillations_detuning_{(-1) ** (-int(det_idx_1 < det_idx_0)) * fixed_detuning}_{pulse_type}_sigma_{sigma}_duration_{dur}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
# Save off-resonant Rabi oscillations
# plt.savefig(os.path.join(save_folder, fig_name))
plt.show()
plt.close()

# Create a 3x3 grid of subplots with extra space for the color bar
fig, ax = plt.subplots(figsize=(7,6))

# Iterate over each subplot and plot the color map
im = ax.pcolormesh(det[detuning_start_idx:detuning_end_idx], amp[:amp_map_end_idx], tr_prob[:amp_map_end_idx, detuning_start_idx:detuning_end_idx], vmin=0, vmax=1, cmap=cmap)
ax.set_xlabel('Detuning (MHz)')
ax.set_ylabel('Rabi Freq. Amplitude (MHz)')

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Transition Probability')

# Adjust the layout
fig.tight_layout()

# Set fig name
fig_name = f"power_spectre_{pulse_type}_sigma_{sigma}_duration_{dur}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

# Save the fig
plt.savefig(os.path.join(save_folder, fig_name))

# Display the plot
plt.show()
