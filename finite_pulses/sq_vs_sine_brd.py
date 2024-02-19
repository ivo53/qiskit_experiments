import os
import pickle
from datetime import datetime

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
# import matplotlib; matplotlib.use('Agg')

from ..transition_line_profile_functions import *

FIT_FUNCTIONS = {
    "lorentzian": [lorentzian],
    "constant": [rabi],
    "rabi": [rabi],
    "gauss": [gauss_rlzsm, gauss_dappr], #gauss_rzconj,
    "rz": [sech_rlzsm, sech_dappr], #rz,
    "sech": [sech_rlzsm, sech_dappr], #rz,
    "demkov": [demkov_rlzsm, demkov_dappr],
    "sech2": [sech2_rlzsm, sech2_dappr], #sech_sq,
    "sinc2": [sinc2],
    "sin": [sin_rlzsm, sin_dappr],
    "sin2": [sin2_rlzsm, sin2_dappr],
    "sin3": [sin3_rlzsm, sin3_dappr],
    "sin4": [sin4_rlzsm, sin4_dappr],
    "sin5": [sin5_rlzsm, sin5_dappr],
    "lor": [lor_rlzsm, lor_dappr],
    "lor2": [lor2_rlzsm, lor2_dappr],
    "lor3": [lor3_rlzsm, lor3_dappr],
}

times = {
    "sq": ["2023-12-08", "175156"],
    "sin": ["2023-12-08", "130503"],
}
backend_name = "kyoto"
save_fig = 0
# save_fig = 1
file_dir = os.path.dirname(__file__) +  "/.."

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


colors = [
    "red",
    "blue",
    "#9D7725",
    "brown",
    "purple"
]

fig = plt.figure(figsize=(14,12), layout="constrained")
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1, 0.04, 0.08])
# cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
linewidths = np.empty(5)
for i in range(2):
    a = amp[i]
    d = det[i]
    tr = tr_prob[i]
    ax = fig.add_subplot(gs[0, i])
    raw_idx = np.where(tr[:, int(0.5 * tr.shape[1])]>0.9)[0]
    raw_idx_divided, ll = [], []
    for idx in raw_idx:
        # print(len(ll))
        if len(ll) > 0:
            if idx - ll[-1] == 1:
                ll.append(idx)
            else:
                raw_idx_divided.append(ll)
                ll = []
                ll.append(idx)
        else: 
            ll.append(idx)
    else:
        raw_idx_divided.append(ll)

    final_idx = []
    for idx in raw_idx_divided:
        final_idx.append(idx[np.argmax(tr[idx, int(0.5 * tr.shape[1])])])
    final_idx = np.array(final_idx)

    # im = ax.imshow(tr, cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)
    max_amp = interval_amp[i] * np.floor(lim_amp[i] / interval_amp[i])
    max_det = interval_det[i] * np.floor(lim_det[i] / interval_det[i])
    
    #### START OF FITS
    f, axis = plt.subplots(1, 5,figsize=(12,3))
    s = 192
    dur = 192
    sd = []
    for idx_area, idx_amp in enumerate(final_idx):
        ax.scatter(d, tr[idx_amp], "")
        init_params, lower, higher = [
            [0,0.5,0.5,0.32],
            [-10,0,0,0.1],
            [10,1,1,.5]
        ]
        ff = FIT_FUNCTIONS["sin"][1]
        detun = d
        detun[detun==0] = 0.0001
        fitparams, tr_fit, perr = fit_function(
            d, tr[idx_amp], ff,
            init_params=init_params,
            lower=lower,
            higher=higher,
            sigma=s, duration=dur, time_interval=0.5e-9,
            remove_bg=True, area=(2 * idx_area + 1) * np.pi
        )
        # print(fitparams, perr)
        sd.append(perr[0] / (2 * np.pi))
        ef = np.linspace(detun[0], detun[-1], 5000)
        extended_tr_fit = ff(ef, *fitparams)
        axis[idx_area].scatter(detun, tr[idx_amp], marker="x")
        axis[idx_area].plot(ef, extended_tr_fit, color="r")
        linewidths[idx_area] = 2 * np.abs(ef[np.argmin(np.abs(extended_tr_fit - (np.amax(extended_tr_fit) + np.amin(extended_tr_fit)) / 2))])
    ##
    ## END OF FITS

    ax.set_xlim(
        (
            len(tr[0])-int(np.round(lim_det[i] / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2)), 
            int(np.round(lim_det[i] / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2))
        )
    )
    ax.set_ylim(
        (
            0,
            int(np.round(lim_amp[i] / a[-1] * (len(tr) - 1)))
        )
    )

    xticks = np.linspace(
        len(tr[0])-int(np.round(max_det / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2)), 
        int(np.round(max_det / d[-1] * (len(tr[0]) / 2 - 1) + len(tr[0]) / 2)), 
        int(2 * max_det / interval_det[i] + 1)
    )
    yticks = np.linspace(
        0, 
        int(np.round(max_amp / a[-1] * (len(tr) - 1))), 
        int(max_amp / interval_amp[i] + 1)
    )
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    if j == 0:
        ax.set_yticklabels([int(tick) for tick in np.linspace(0, max_amp, int(max_amp / interval_amp[i] + 1)).round(0)], fontsize=18)
        ax.set_ylabel('Amplitude (MHz)', fontsize=18)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('Detuning (MHz)', fontsize=18)
    ax.set_xticklabels([int(tick) for tick in np.linspace(-max_det, max_det, int(2 * max_det / interval_det[i] + 1)).round(0)], fontsize=18)

print(linewidths)
# # Create a separate subplot for the color bar
# cax = fig.add_subplot(gs[:, 3])
# cbar = fig.colorbar(im, cax=cax)
# # cbar.set_label('Transition probability', fontsize=15)
# cbar.ax.tick_params(labelsize=18)

plt.show()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "finite_pulses")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"2d_sine_exp_sim_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

if save_fig:
    # Save the fig
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

