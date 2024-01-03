import os
import pickle
from datetime import datetime

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib; matplotlib.use('Agg')

times = {
    "sq": ["2023-12-08", "175156"],
    "sin": ["2023-12-08", "130503"],
}
backend_name = "kyoto"
save_fig = 1
file_dir = os.path.dirname(__file__)

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

l, p, x0 = 1.25116799e+02, 3.01445760e-01, 8.23116906e-03
T = 192 * 5e-10
tr_prob, amp, det = [], [], []
for k, t in times.items():
    files = os.listdir(data_folder(t[0], t[1], k))
    with open(os.path.join(data_folder(t[0], t[1], k), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob.append(pickle.load(f1))
    with open(os.path.join(data_folder(t[0], t[1], k), files[0]), 'rb') as f2:
        amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * (2*T/np.pi)))
    with open(os.path.join(data_folder(t[0], t[1], k), files[1]), 'rb') as f3:
        det.append(pickle.load(f3) * 2 * np.pi / 1e6)

data = pd.read_csv("C:/Users/Ivo/Documents/Wolfram Mathematica/sine.csv", header=None).to_numpy()
interval_det, interval_amp = 100, 100
det.append(np.arange(-2.3e2, 2.3001e2, 0.05e2))
amp.append(np.arange(0, 5.001e2, 1e1))
tr_prob.append(data[:, 2].reshape(len(det[2]), len(amp[2])).T)
fig = plt.figure(figsize=(13.6,6), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
for idx in range(2):
    a = amp[idx + 1]
    d = det[idx + 1]
    tr = tr_prob[idx + 1]
    ax = fig.add_subplot(gs[0, idx])
    im = ax.imshow(tr, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_xlabel('Detuning (MHz)', fontsize=18)
    if idx == 0:
        ax.set_ylabel('Amplitude (MHz)', fontsize=18)
    max_amp = interval_amp * np.floor(a[-1] / interval_amp)
    max_det = interval_det * np.floor(d[-1] / interval_det)
    ax.set_xlim(
        (
            len(tr[0])-int(np.round(max_det / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2) * 19/18), 
            int(np.round(max_det / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2) * 19/18)
        )
    )
    ax.set_ylim(
        (
            0,
            int(np.round(max_amp / a[-1] * (len(tr) - 1) * (5/4 - 1/4 * bool(idx))))
        )
    )
    xticks = np.linspace(
        len(tr[0])-int(np.round(max_det / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2)), 
        int(np.round(max_det / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2)), 
        int(2 * max_det / interval_det + 1)
    )
    yticks = np.linspace(
        0, 
        int(np.round(max_amp / a[-1] * (len(tr) - 1))), 
        int(max_amp / interval_amp + 1)
    )
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.linspace(-max_det, max_det, int(2 * max_det / interval_det + 1)).round(0), fontsize=18)
    ax.set_yticklabels(np.linspace(0, max_amp, int(max_amp / interval_amp + 1)).round(0), fontsize=18)
    
# Create a separate subplot for the color bar
cax = fig.add_subplot(gs[:, 2])
cbar = fig.colorbar(im, cax=cax)
# cbar.set_label('Transition probability', fontsize=15)
cbar.ax.tick_params(labelsize=18)

# plt.show()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "finite_pulses")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"2d_sine_exp_sim_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

if save_fig:
    # Save the fig
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

