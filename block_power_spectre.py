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
# pulse_type = "lor"
# pulse_type2 = "lorentz"
fixed_detuning = [5, 2, 2, 2] # MHz
intervals = [200, 100, 100, 100]
save_osc, save_map = 1, 1
times = {
    "lor2": ["2023-07-04", "192920"],
    "lor": ["2023-07-04", "024124"],
    "lor3_4": ["2023-07-04", "193308"],
    "lor2_3": ["2023-07-04", "193649"],
}

## create folder where plots are saved
file_dir = os.path.dirname(__file__)
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

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

tr_prob, amp, det = [], [], []
for k, t in times.items():
    files = os.listdir(data_folder(t[0], t[1], k))
    with open(os.path.join(data_folder(t[0], t[1], k), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob.append(pickle.load(f1))
    with open(os.path.join(data_folder(t[0], t[1], k), files[0]), 'rb') as f2:
        amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
    with open(os.path.join(data_folder(t[0], t[1], k), files[1]), 'rb') as f3:
        det.append(pickle.load(f3)/1e6)

# tr_prob[0] = tr_prob[0][:94]
# amp[0] = amp[0][:94]
tr_prob[1] = tr_prob[1][:62]
amp[1] = amp[1][:62]
# tr_prob[2] = tr_prob[2][:67]
# amp[2] = amp[2][:67]
# tr_prob[3] = tr_prob[3][:72]
# amp[3] = amp[3][:72]
# Set up the backend to use the EPS file format
# plt.switch_backend('ps')


start_idx, end_idx = [], []
for i in range(4):
    start_idx.append(np.argmin(np.abs(det[i] - det[1][0])))
    end_idx.append(np.argmin(np.abs(det[i] - det[1][-1])))
for i in range(4):
    if i == 1:
        continue
    tr_prob[i] = tr_prob[i][:, start_idx[i]:end_idx[i]]
    det[i] = det[i][start_idx[i]:end_idx[i]]

det_indices_0, det_indices_1, max_amp, max_amp_minor = [], [], [], []
for i in range(4):
    det_idx_0 = np.argsort(np.abs(det[i] + fixed_detuning[i]))[0]
    det_idx_1 = np.argsort(np.abs(det[i] - fixed_detuning[i]))[1]
    det_indices_0.append(det_idx_0)
    det_indices_1.append(det_idx_1)
    max_amp.append(intervals[i] * np.floor(amp[i][-1] / intervals[i]))
    max_amp_minor.append(intervals[i] / 4 * np.ceil(amp[i][-1] / (intervals[i] / 4)))

fig0 = plt.figure(figsize=(12, 9))
gs0 = fig0.add_gridspec(2, 2, width_ratios=[1, 1])
# Adjust the layout
fig0.tight_layout()
# Generate datetime
date = datetime.now()

for i in range(2):
    for j in range(2):
        ax0 = fig0.add_subplot(gs0[i, j])
        cmap0 = plt.cm.get_cmap('cividis')  # Choose a colormap
        ax0.scatter(amp[2*i+j], tr_prob[2*i+j][:, det_indices_0[2*i+j]], marker="p", cmap=cmap0)
        ax0.set_xticks(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int))
        ax0.set_xticks(np.round(np.arange(0, max_amp_minor[2*i+j] + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax0.set_xticklabels(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int), fontsize=15)
        ax0.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.2), 1))
        ax0.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.1), 1), minor=True)
        ax0.set_yticklabels(np.round(np.arange(0, .8 + 1e-3, 0.2), 1), fontsize=15)
        ax0.grid(which='minor', alpha=0.2)
        ax0.grid(which='major', alpha=0.6)
        if i == 1:
            ax0.set_xlabel('Detuning (MHz)', fontsize=15)
        if j == 0:
            ax0.set_ylabel('Transition Probability', fontsize=15)
        # Set fig name
fig_name = f"rabi_oscillations_detuning_{(-1) * fixed_detuning}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
if save_osc:
    plt.savefig(os.path.join(save_folder, fig_name))
plt.show()

fig1 = plt.figure(figsize=(12, 9))
gs1 = fig1.add_gridspec(2, 2, width_ratios=[1, 1])
# Adjust the layout
fig1.tight_layout()

for i in range(2):
    for j in range(2):
        ax1 = fig1.add_subplot(gs1[i, j])
        cmap1 = plt.cm.get_cmap('cividis')  # Choose a colormap
        ax1.scatter(amp[2*i+j], tr_prob[2*i+j][:, det_indices_1[2*i+j]], marker="p", cmap=cmap1)
        ax1.set_xticks(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int))
        ax1.set_xticks(np.round(np.arange(0, max_amp_minor[2*i+j] + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax1.set_xticklabels(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int), fontsize=15)
        ax1.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.2), 1))
        ax1.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.1), 1), minor=True)
        ax1.set_yticklabels(np.round(np.arange(0, .8 + 1e-3, 0.2), 1), fontsize=15)
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.6)
        if i == 1:
            ax1.set_xlabel('Detuning (MHz)', fontsize=15)
        if j == 0:
            ax1.set_ylabel('Transition Probability', fontsize=15)
        # Set fig name
fig_name = f"rabi_oscillations_detuning_{fixed_detuning}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
if save_osc:
    plt.savefig(os.path.join(save_folder, fig_name))
plt.show()
        
# Create a 2x2 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.1])
# Adjust the layout
fig.tight_layout()

# Iterate over each subplot and plot the color map
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, j])
        cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
        im = ax.pcolormesh(det[2*i+j], amp[2*i+j], tr_prob[2*i+j], vmin=0, vmax=1, cmap=cmap)
        ax.set_yticks(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int))
        ax.set_yticks(np.round(np.arange(0, max_amp_minor[2*i+j] - intervals[2*i+j] / 4 + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax.set_yticklabels(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int), fontsize=15)

        tick_locations = np.arange(-15, 15.01, 3)
        tick_locations[-1] = 14.85
        ax.set_xticks(tick_locations)
        ax.set_xticks(np.arange(-14.9, 15.01, 1).astype(int), minor=True)
        ax.set_xticklabels(np.round(np.arange(-15,15.01, 3)).astype(int), fontsize=15)
        if i == 1:
            ax.set_xlabel('Detuning (MHz)', fontsize=15)
        if j == 0:
            ax.set_ylabel('Rabi Freq. Amplitude (MHz)', fontsize=15)

# Create a separate subplot for the color bar
cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Transition probability', fontsize=15)


# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"


# Save the fig
if save_map:
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
plt.show()
