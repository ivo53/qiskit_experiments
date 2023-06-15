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

times = {
    "sq": ["2023-06-12", "230439"],
    "lorentz": ["2023-06-03", "120038"],
    "lor3_4": ["2023-06-06", "013505"],
    "lor2_3": ["2023-06-06", "202254"],
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
    files = os.listdir(data_folder(t[0], t[1], k))
    with open(os.path.join(data_folder(t[0], t[1], k), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob.append(pickle.load(f1))
    with open(os.path.join(data_folder(t[0], t[1], k), files[0]), 'rb') as f2:
        amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
    with open(os.path.join(data_folder(t[0], t[1], k), files[1]), 'rb') as f3:
        det.append(pickle.load(f3)/1e6)

tr_prob[2] = tr_prob[2][:70]
amp[2] = amp[2][:70]
tr_prob[3] = tr_prob[3][:70]
amp[3] = amp[3][:70]
# Set up the backend to use the EPS file format
# plt.switch_backend('ps')

# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.1])

# Iterate over each subplot and plot the color map
for i in range(2):
    for j in range(2):
        ax = fig.add_subplot(gs[i, j])
        cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
        im = ax.pcolormesh(det[2*i+j], amp[2*i+j], tr_prob[2*i+j], vmin=0, vmax=1, cmap=cmap)
        if i == 1:
            ax.set_xlabel('Detuning (MHz)')
        if j == 0:
            ax.set_ylabel('Rabi Freq. Amplitude (MHz)')

# Create a separate subplot for the color bar
cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Transition probability')

# Adjust the layout
fig.tight_layout()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"


# Save the fig
# plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
plt.show()
