import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from numerical_solutions import ndsolve_lorentz_rabi_osc, ndsolve_lorentz_map
from transition_line_profile_functions import *

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "manila"
# pulse_type = "lor"
# pulse_type2 = "lorentz"
fixed_detuning = [30, 25, 12.5, 12.5, 12.5, 12.5] # MHz
# fixed_detuning = [5, 4, 2, 2, 2, 2] # MHz
intervals = [200, 150, 100, 100, 100, 100]
save_osc, save_map = 1, 0
times = {
    "lor2": ["2023-07-04", "192920"],
    "lor3_2": ["2023-08-24", "101937"],
    "lor": ["2023-07-04", "024124"],
    "lor3_4": ["2023-07-04", "193308"],
    "lor2_3": ["2023-07-04", "193649"],
    "lor3_5": ["2023-07-03", "001114"], 
}
params = {
    "lor2": [(24 + 8/9) * 1e-9, (181 + 2/3) * 1e-9],
    "lor3_2": [(24 + 8/9) * 1e-9, (286.81 + 5/900) * 1e-9],
    "lor": [(24 + 8/9) * 1e-9, (704) * 1e-9],
    "lor3_4": [(10 + 2/3) * 1e-9, (728 + 8/9) * 1e-9],
    "lor2_3": [(10 + 2/3) * 1e-9, (1134 + 2/9) * 1e-9],
    "lor3_5": [(7 + 1/9) * 1e-9, (1176 + 8/9) * 1e-9], 
}
powers = [2, 3/2, 1, 3/4, 2/3, 3/5]
powers_latex = [" $L^2$", "$L^{3/2}$", "  $L$", "$L^{3/4}$", "$L^{2/3}$", "$L^{3/5}$"]
pulse_names = list(params.keys())

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
        det.append(pickle.load(f3) / 1e6)
# tr_prob[0] = tr_prob[0][:94]
# amp[0] = amp[0][:94]
tr_prob[2] = tr_prob[2][:62] # was [1]
amp[2] = amp[2][:62]
tr_prob[5] = tr_prob[5][:85] # was [1]
amp[5] = amp[5][:85]
# tr_prob[2] = tr_prob[2][:67]
# amp[2] = amp[2][:67]
# tr_prob[3] = tr_prob[3][:72]
# amp[3] = amp[3][:72]
# Set up the backend to use the EPS file format
# plt.switch_backend('ps')

# Set intervals
intervals_det = [1.5, 1.5, 1.5, 4, 4, 6]

include_35 = False
if include_35:
    tr_prob.pop(0)
    amp.pop(0)
    det.pop(0)
    intervals_det.pop(0)
    powers_latex.pop(0)
    pulse_names.pop(0)
    powers.pop(0)
    fixed_detuning.pop(0)
    intervals.pop(0)
    params.pop(list(params.keys())[0])

start_idx, end_idx = [], []
for i in range(6):
    start_idx.append(np.argmin(np.abs(det[i] - det[1][0])))
    end_idx.append(np.argmin(np.abs(det[i] - det[1][-1])))

numerical_amps, numerical_tr_probs = [], []
for i in range(6):
    # A_range_minus, numerical_tr_prob_minus = ndsolve_lorentz_rabi_osc(
    #     params[list(times.keys())[i]][0],
    #     params[list(times.keys())[i]][1],
    #     0, 100, -fixed_detuning[i] * 2 * np.pi * 1e6,
    #     A_end=amp[i][-1] * 1e6,
    #     num_t=1000,
    #     lor_power=powers[i]
    # )
    # A_range_plus, numerical_tr_prob_plus = ndsolve_lorentz_rabi_osc(
    #     params[list(times.keys())[i]][0],
    #     params[list(times.keys())[i]][1],
    #     0, 100, fixed_detuning[i] * 2 * np.pi * 1e6,
    #     A_end=amp[i][-1] * 1e6,
    #     num_t=1000,
    #     lor_power=powers[i]
    # )
    # numerical_amps.append([A_range_minus * 1e-6, A_range_plus * 1e-6])
    # numerical_tr_probs.append([numerical_tr_prob_minus, numerical_tr_prob_plus])
    if i == 2:
        continue
    tr_prob[i] = tr_prob[i][:, start_idx[i]:end_idx[i]]
    det[i] = det[i][start_idx[i]:end_idx[i]]

det_indices_0, det_indices_1, max_amp, max_amp_minor = [], [], [], []
for i in range(6):
    det_idx_0 = np.argsort(np.abs(det[i] + fixed_detuning[i]))[0]
    det_idx_1 = np.argsort(np.abs(det[i] - fixed_detuning[i]))[1]
    det_indices_0.append(det_idx_0)
    det_indices_1.append(det_idx_1)
    max_amp.append(intervals[i] * np.floor(amp[i][-1] / intervals[i]))
    max_amp_minor.append(intervals[i] / 4 * np.ceil(amp[i][-1] / (intervals[i] / 4)))

fig0 = plt.figure(figsize=(12, 15), layout="constrained")
gs0 = fig0.add_gridspec(3, 2, width_ratios=[1, 1])
# Adjust the layout
# fig0.tight_layout()
# Generate datetime
date = datetime.now()
marker_size = 10
initial, initial_min, initial_max = [], [], []
for i in range(3):
    for j in range(2):
        # fit_params, y_fit, err = fit_function(
        #     amp[2*i+j],
        #     tr_prob[2*i+j][:, det_indices_0[2*i+j]],
        #     list(times.keys())[2*i+j],
        #     initial, initial_min, initial_max,
        #     s, dur
        # )

        ax0 = fig0.add_subplot(gs0[i, j])
        cmap0 = plt.cm.get_cmap('cividis')  # Choose a colormap
        ax0.plot(amp[2*i+j], tr_prob[2*i+j][:, det_indices_0[2*i+j]], marker="p", ms=marker_size)
        # ax0.plot(numerical_amps[2*i+j][0], numerical_tr_probs[2*i+j][0])

        # Add a rectangle in the top right corner
        rect = Rectangle((0.85, 0.89), 0.13, 0.11, transform=ax0.transAxes,
                        color='#003399', alpha=0.7)
        ax0.add_patch(rect)
        
        # Add text inside the rectangle
        text = powers_latex[2*i+j]
        ax0.text(0.87, 0.91, text, transform=ax0.transAxes,
            color='white', fontsize=18)

        ax0.set_xticks(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int))
        ax0.set_xticks(np.round(np.arange(0, max_amp_minor[2*i+j] + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax0.set_xticklabels(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int), fontsize=18)
        ax0.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.2), 1))
        ax0.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.1), 1), minor=True)
        ax0.set_yticklabels(np.round(np.arange(0, .8 + 1e-3, 0.2), 1), fontsize=18)
        ax0.set_ylim(0, 0.8)
        ax0.grid(which='minor', alpha=0.2)
        ax0.grid(which='major', alpha=0.6)
        if i == 2:
            ax0.set_xlabel('Peak Rabi Frequency (MHz)', fontsize=18)
        if j == 0:
            ax0.set_ylabel('Transition Probability', fontsize=18)
# Set fig name
fig_name = f"rabi_oscillations_detuning_{(-1) * fixed_detuning}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
if save_osc:
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")
# plt.show()


fig1 = plt.figure(figsize=(12, 15), layout="constrained")
gs1 = fig1.add_gridspec(3, 2, width_ratios=[1, 1])
# Adjust the layout
# fig1.tight_layout()

for i in range(3):
    for j in range(2):
        ax1 = fig1.add_subplot(gs1[i, j])
        cmap1 = plt.cm.get_cmap('cividis')  # Choose a colormap
        ax1.plot(amp[2*i+j], tr_prob[2*i+j][:, det_indices_1[2*i+j]], marker="p", ms=marker_size)
        # ax1.plot(numerical_amps[2*i+j][1], numerical_tr_probs[2*i+j][1])

        # Add a rectangle in the top right corner
        rect = Rectangle((0.85, 0.89), 0.13, 0.11, transform=ax1.transAxes,
                        color='#003399', alpha=0.7)
        ax1.add_patch(rect)
        
        # Add text inside the rectangle
        text = powers_latex[2*i+j]
        ax1.text(0.87, 0.91, text, transform=ax1.transAxes,
            color='white', fontsize=18)

        ax1.set_xticks(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int))
        ax1.set_xticks(np.round(np.arange(0, max_amp_minor[2*i+j] + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax1.set_xticklabels(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])).astype(int), fontsize=18)
        ax1.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.2), 1))
        ax1.set_yticks(np.round(np.arange(0, .8 + 1e-3, 0.1), 1), minor=True)
        ax1.set_yticklabels(np.round(np.arange(0, .8 + 1e-3, 0.2), 1), fontsize=18)
        ax1.set_ylim(0, 0.8)
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.6)
        if i == 2:
            ax1.set_xlabel('Peak Rabi Frequency (MHz)', fontsize=18)
        if j == 0:
            ax1.set_ylabel('Transition Probability', fontsize=18)
        # Set fig name
fig_name = f"rabi_oscillations_detuning_{fixed_detuning}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
if save_osc:
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")
# plt.show()
# exit()
#  # Create a 2x2 grid of subplots with extra space for the color bar
# fig_sim = plt.figure(figsize=(12, 9))
# gs_sim = fig_sim.add_gridspec(2, 3, width_ratios=[1, 1, 0.1])
# tr_probss, A_ranges, d_ranges = [], [], []
# for i in range(2):
#     for j in range(2):
#         sigma_temp = params[pulse_names[2*i +j]][0] * 10**6
#         d_range, A_range, tr_probs = ndsolve_lorentz_map(
#             params[list(times.keys())[2 * i + j]][0],
#             params[list(times.keys())[2 * i + j]][1],
#             det[2*i+j][0]*2*np.pi*1e6, 100, det[2*i+j][-1]*2*np.pi*1e6,    
#             num_t=1000,
#             A_num=100,
#             max_pulse_area=10 * np.pi,
#             lor_power=powers[2 * i + j]
#         )
#         tr_probss.append(tr_probs)
#         A_ranges.append(A_range)
#         d_ranges.append(d_range)
#         ax_sim = fig_sim.add_subplot(gs_sim[i, j])
#         cmap_sim = plt.cm.get_cmap('cividis')  # Choose a colormap
#         im_sim = ax_sim.pcolormesh(d_range, A_range, tr_probs, vmin=0, vmax=1, cmap=cmap_sim)

#         # Add a rectangle in the top right corner
#         rect = Rectangle((0.8, 0.85), 0.12, 0.1, transform=ax_sim.transAxes,
#                         color='#DEDA8D', alpha=0.7)
#         ax_sim.add_patch(rect)
        
#         # Add text inside the rectangle
#         text = powers_latex[2*i+j]
#         ax_sim.text(0.82, 0.86, text, transform=ax_sim.transAxes,
#             color='black', fontsize=18)

#         ax_sim.set_yticks(np.round(np.arange(0, max_amp[2*i+j] * 1e6 + 1e-3, intervals[2*i+j] * 1e6)).astype(int))
#         ax_sim.set_yticks(np.round(np.arange(0, max_amp_minor[2*i+j] * 1e6 - intervals[2*i+j] * 1e6 / 4 + 1e-3, intervals[2*i+j] * 1e6 / 4)).astype(int), minor=True)
#         ax_sim.set_yticklabels(np.round(np.arange(0, max_amp[2*i+j] * 1e6 + 1e-3, intervals[2*i+j] * 1e6)).astype(int), fontsize=18)

#         tick_locations = np.arange(-15, 15.01, 5)
#         tick_locations = np.arange(-0.15, 0.1501, 0.05)
#         # ax_sim.set_xlim(tick_locations[0], tick_locations[-1])
#         tick_locations[-1] = 14.85
#         ax_sim.set_xticks(tick_locations)
#         ax_sim.set_xticks(np.arange(-15, 14.9, 1).astype(int), minor=True)
#         ax_sim.set_xticklabels(np.round(tick_locations * sigma_temp,2), fontsize=18)
#         if i == 1:
#             ax_sim.set_xlabel('Detuning (in units of $1/\\tau$)', fontsize=18)
#         if j == 0:
#             ax_sim.set_ylabel('Rabi Freq. Amplitude (MHz)', fontsize=18)



# # Create a separate subplot for the color bar
# cax_sim = fig_sim.add_subplot(gs_sim[:, 2])
# cbar_sim = fig_sim.colorbar(im_sim, cax=cax_sim)
# cbar_sim.set_label('\nSimulated Transition probability', fontsize=18)
# cbar_sim.ax.tick_params(labelsize=14)

# # Adjust the layout
# fig_sim.tight_layout()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()


data_save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing", "data")
make_all_dirs(data_save_folder)

# # Set fig name
# fig_name_sim = f"sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
# tr_probs_name_sim = f"tr_probs_sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pkl"
# dets_name_sim = f"det_sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pkl"
# amps_name_sim = f"amp_sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pkl"


# # Save the simulation
# if save_map:
#     plt.savefig(os.path.join(save_folder, fig_name_sim), format="pdf")

# with open(os.path.join(data_save_folder, tr_probs_name_sim), mode="wb") as f:
#     pickle.dump(tr_probss, f)
# with open(os.path.join(data_save_folder, dets_name_sim), mode="wb") as g:
#     pickle.dump(d_ranges, g)
# with open(os.path.join(data_save_folder, amps_name_sim), mode="wb") as h:
#     pickle.dump(A_ranges, h)


# Create a 2x2 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(12,15), layout="constrained")
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.1])

# Iterate over each subplot and plot the color map
for i in range(3):
    for j in range(2):
        max_det = intervals_det[2*i+j] * np.floor(det[2*i+j][-1] / intervals_det[2*i+j])
        sigma_temp = params[pulse_names[2*i+j]][0] * 10**6
        d = det[2*i+j] * sigma_temp
        a = amp[2*i+j]
        ax = fig.add_subplot(gs[i, j])
        cmap = plt.cm.get_cmap('cividis') # Choose a colormap
        im = ax.imshow(tr_prob[2*i+j], cmap=cmap, aspect="auto", origin="lower", vmin=0, vmax=1)
        # Add a rectangle in the top right corner
        rect = Rectangle((0.8, 0.86), 0.15, 0.14, transform=ax.transAxes,
                        color='#DEDA8D', alpha=0.7)
        ax.add_patch(rect)
        
        # Add text inside the rectangle
        text = powers_latex[2*i+j]
        ax.text(0.82, 0.885, text, transform=ax.transAxes,
            color='black', fontsize=18)

        yticks = np.linspace(
            0, 
            int(np.round(max_amp[2*i+j] / a[-1] * (len(tr_prob[2*i+j]) - 1))), 
            int(max_amp[2*i+j] / intervals[2*i+j] + 1)
        )
        # ax_sim.set_yticks(np.round(np.arange(0, max_amp + 1e-3, intervals[2*i+j])).astype(int))
        # ax_sim.set_yticks(np.round(np.arange(0, max_amp_minor - intervals[2*i+j] / 4 + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax.set_yticks(np.round(yticks).astype(int))
        ax.set_yticklabels(np.round(np.arange(0, max_amp[2*i+j] + 1e-3, intervals[2*i+j])/100).astype(int), fontsize=18)
        
        tick_labels = np.arange(
            -max_det, 
            max_det + 0.01, 
            intervals_det[2*i+j]
        )
        remainders = np.modf(tick_labels.round(1))[0]
        tick_labels_formatted = [int(t) if r == 0 else np.round(t, 1) for r, t in zip(remainders, tick_labels)]

        tick_locations = np.linspace(
            len(tr_prob[2*i+j][0]) - int(np.round(max_det / det[2*i+j][-1] * (len(tr_prob[2*i+j][0]) / 2 - 1) + len(tr_prob[2*i+j][0]) / 2)),
            int(np.round(max_det / det[2*i+j][-1] * (len(tr_prob[2*i+j][0]) / 2 - 1) + len(tr_prob[2*i+j][0]) / 2)),
            len(tick_labels)
        )
        # print(max_det)
        # print(tick_locations)
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels_formatted, fontsize=18)

        if i == 2:
            ax.set_xlabel("Detuning (MHz)", fontsize=18)
        
        ax2nd = ax.secondary_xaxis('top')
        # ax_sim.set_xlim(tick_locations[0], tick_locations[-1])
        # tick_locations[-1] = 14.85
        lim_delta_tau = 0.09
        
        tick_labels2 = np.arange(-lim_delta_tau, lim_delta_tau + 1e-3, lim_delta_tau / 2)
        tick_locations2 = np.linspace(
            len(tr_prob[2*i+j][0]) - int(np.round(lim_delta_tau / d[-1] * (len(tr_prob[2*i+j][0]) / 2 - 1) + len(tr_prob[2*i+j][0]) / 2)), 
            int(np.round(lim_delta_tau / d[-1] * (len(tr_prob[2*i+j][0]) / 2 - 1) + len(tr_prob[2*i+j][0]) / 2)), 
            len(tick_labels2)
        )

        ax2nd.set_xticks(tick_locations2)
        ax.set_xlim(tick_locations2[0], tick_locations2[-1])

        if i == 0:
            ax2nd.set_xlabel('Detuning (in units of $1/\\tau$)', fontsize=18)
            ax2nd.set_xticklabels(np.round(tick_labels2,3), fontsize=18)
            # if j==0:
            #     ax2nd.set_xticklabels(np.round(tick_labels2,3), fontsize=18)
            # else:
            #     tick_labels2 = np.round(tick_labels2,3).tolist()
            #     tick_labels2[0] = ""
            #     ax2nd.set_xticklabels(tick_labels2, fontsize=18)
        else:
            ax2nd.set_xticklabels([])

        if j == 0:
            ax.set_ylabel('Peak Rabi Freq. (100 MHz)', fontsize=18)

# Create a separate subplot for the color bar
cax = fig.add_subplot(gs[:, 2])
# Adjust the position of the colorbar
# divider = plt.make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust 'size' and 'pad' as needed
# cbar = plt.colorbar(im, cax=cax)

cbar = fig.colorbar(im, cax=cax)
# cbar.set_label('\nTransition probability', fontsize=18)
cbar.ax.tick_params(labelsize=18)

# Adjust the layout
# fig.tight_layout()

# Set fig name
fig_name = f"block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

# Save the fig
if save_map:
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
# plt.show()
