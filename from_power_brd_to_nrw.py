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
pulse_type = "lor"
pulse_type2 = "lorentz"

times = {
    0.5: ["2023-06-03", "120038"],
    1: ["2023-06-03", "160754"],
    2: ["2023-06-03", "161333"],
    3: ["2023-06-03", "022027"],
    5: ["2023-06-03", "022625"],
    7.5: ["2023-06-03", "023608"],
    15: ["2023-06-03", "120636"],
    20: ["2023-06-03", "115516"],
    50: ["2023-06-03", "222957"],
}

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

tr_prob, amp, det = [], [], []
for k, t in times.items():
    if int(t[1]) > 150000:
        files = os.listdir(data_folder(t[0], t[1], pulse_type))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type), files[2]).replace("\\","/"), 'rb') as f1:
            tr_prob.append(pickle.load(f1))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type), files[0]), 'rb') as f2:
            amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type), files[1]), 'rb') as f3:
            det.append(pickle.load(f3)/1e6)
    else:
        files = os.listdir(data_folder(t[0], t[1], pulse_type2))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type2), files[2]).replace("\\","/"), 'rb') as f1:
            tr_prob.append(pickle.load(f1))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type2), files[0]), 'rb') as f2:
            amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type2), files[1]), 'rb') as f3:
            det.append(pickle.load(f3)/1e6)

tr_prob[7] = tr_prob[7][:, 20:-20]
det[7] = det[7][20:-20]
tr_prob[6] = tr_prob[6][:, 20:-20]
det[6] = det[6][20:-20]

# Set up the backend to use the EPS file format
# plt.switch_backend('ps')

# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.1])

# Generate some random data for the color maps
data = np.random.rand(10, 10)

# Iterate over each subplot and plot the color map
for i in range(3):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
        im = ax.pcolormesh(det[::-1][3*i+j], amp[::-1][3*i+j], tr_prob[::-1][3*i+j], vmin=0, vmax=1, cmap=cmap)
        if i == 2:
            ax.set_xlabel('Detuning (MHz)')
        if j == 0:
            ax.set_ylabel('Rabi Freq. Amplitude (MHz)')

# Create a separate subplot for the color bar
cax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Transition probability')

# Adjust the layout
fig.tight_layout()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"from_brd_to_nrw_{pulse_type}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"


# Save the fig
plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
plt.show()
