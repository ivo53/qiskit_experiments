import os
import pickle
from datetime import datetime
import numpy as np
# import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from transition_line_profile_functions import *

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

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

backend_name = "kyoto"
save_fig = 0

times = {
    "sq": ["2023-12-08", "175156"],
    "sin": ["2023-12-08", "130503"],
}
full_pulse_type = ["Rect.", "Sine"]
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
s = 192 * 0.5 * 1e-9
dur = s

tr_prob, amp, det = [], [], []
for k, t in times.items():
    files = os.listdir(data_folder(t[0], t[1], k))
    with open(os.path.join(data_folder(t[0], t[1], k), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob.append(pickle.load(f1))
    with open(os.path.join(data_folder(t[0], t[1], k), files[0]), 'rb') as f2:
        amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
    with open(os.path.join(data_folder(t[0], t[1], k), files[1]), 'rb') as f3:
        det.append(pickle.load(f3) * 2 * np.pi / 1e6)

# tr_prob[7] = tr_prob[7][:, 20:-20]
# det[7] = det[7][20:-20]
# tr_prob[6] = tr_prob[6][:, 20:-20]
# det[6] = det[6][20:-20]

intervals_amp = [100, 100]
intervals_det = [100, 100]
# Set up the backend to use the EPS file format
# plt.switch_backend('ps')

# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(13.6,6), layout="constrained")
fig2, ax2 = plt.subplots(1, 1, figsize=(8,6), layout="constrained")
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.02, 0.07])

# Generate some random data for the color maps
data = np.random.rand(10, 10)

# Iterate over each subplot and plot the color map
for idx in range(2):
    max_amp = intervals_amp[idx] * np.floor(amp[idx][-1] / intervals_amp[idx])
    max_det = intervals_det[idx] * np.floor(det[idx][-1] / intervals_det[idx])
    # max_amp_minor = intervals[3*i+j] / 4 * np.ceil(amp[3*i+j][-1] / (intervals[3*i+j] / 4))

    ax = fig.add_subplot(gs[0, idx])
    cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
    # im = ax.pcolormesh(det[::-1][3*i+j], amp[::-1][3*i+j], tr_prob[::-1][3*i+j], vmin=0, vmax=1, cmap=cmap)
    # cmap = plt.get_cmap('twilight_r')
    
    # Plot the color map
    im = ax.imshow(tr_prob[idx], cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)
    
    raw_idx = np.where(tr_prob[idx][:, int(0.5 * tr_prob[idx].shape[1])]>0.9)[0]
    raw_idx_divided, ll = [], []
    for i in raw_idx:
        # print(len(ll))
        if len(ll) > 0:
            if i - ll[-1] == 1:
                ll.append(i)
            else:
                raw_idx_divided.append(ll)
                ll = []
                ll.append(i)
        else: 
            ll.append(i)
    else:
        raw_idx_divided.append(ll)

    final_idx = []
    for i in raw_idx_divided:
        final_idx.append(i[np.argmax(tr_prob[idx][i, int(0.5 * tr_prob[idx].shape[1])])])
    final_idx = np.array(final_idx)

    if idx==1:
        sd = []
        for idx_area, idx_amp in enumerate(final_idx):
            init_params, lower, higher = [
                [0,0.3,0.2,0.32],
                [-10,0,0,0.1],
                [10,1,1,.5]
            ]
            ff = FIT_FUNCTIONS["sin"][0]
            d = det[idx] * 1e6
            fitparams, tr_fit, perr = fit_function(
                d, tr_prob[idx][idx_amp], ff,
                init_params=init_params,
                lower=lower,
                higher=higher,
                sigma=s, duration=dur,
                remove_bg=True, area=(2 * idx_area + 1) * np.pi
            )
            # print(fitparams, perr)
            sd.append(perr[0] / (2 * np.pi))
            ef = np.linspace(d[0], d[-1], 5000)
            extended_tr_fit = ff(ef, *fitparams)
            ax2.plot(d, tr_prob[idx][idx_amp])
            ax2.plot(ef, extended_tr_fit)
            print(tr_fit)
            # plt.show()

    # # Add a rectangle in the top right corner
    rect = Rectangle((0.8, 0.875), 0.16, 0.1, transform=ax.transAxes,
                    color='#DEDA8D', alpha=0.7)
    ax.add_patch(rect)
    text = full_pulse_type[idx]
    # # Add text inside the rectangle
    # text = str(list(times.keys())[::-1][3*i+j]) + "%"
    # if (i==2 and j!=2) or (i==1 and j!=0):
    #     text = " " + text
    ax.text(0.83 - (1 - idx) * 0.01, 0.91, text, transform=ax.transAxes,
        color='black', fontsize=18)

    xticks = np.linspace(
        len(tr_prob[idx][0])-int(np.round(max_det / det[idx][-1] * (len(tr_prob[idx][0]) / 2 - 1) + len(tr_prob[idx][0]) / 2)), 
        int(np.round(max_det / det[idx][-1] * (len(tr_prob[idx][0]) / 2 - 1) + len(tr_prob[idx][0]) / 2)), 
        int(2 * max_det / intervals_det[idx] + 1)
    )
    # print(xticks)
    # print(max_det)
    xticklabels = np.round(np.arange(-max_det, max_det + 1e-2, intervals_det[idx]), 1).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=18)

    yticks = np.linspace(
        0, 
        int(np.round(max_amp / amp[idx][-1] * (len(tr_prob[idx]) - 1))), 
        int(max_amp / intervals_amp[idx] + 1)
    )
    yticklabels = np.round(np.arange(0, max_amp + 1, intervals_amp[idx]), 0).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels((yticklabels).astype(int), fontsize=18)
    # if i == 2:
    ax.set_xlabel('Detuning (MHz)', fontsize=18)
    if idx == 0:
        ax.set_xlim((det[1][0], det[1][-1]))
        ax.set_ylabel('Peak Rabi Freq. (MHz)', fontsize=16)
# fig.patch.set_facecolor('white')
# fig.patch.set_alpha(1)


# Create a separate subplot for the color bar
cax = fig.add_subplot(gs[:, 3])
cbar = fig.colorbar(im, cax=cax)
# cbar.set_label('Transition probability', fontsize=15)
cbar.ax.tick_params(labelsize=18)

# Adjust the layout
# fig.tight_layout()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "finite_pulses")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"from_brd_to_nrw_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"


if save_fig:
    # Save the fig
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
plt.show()
