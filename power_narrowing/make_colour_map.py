import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
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

with open("C:/Users/Ivo/Documents/qiskit_codes/paper_ready_plots/power_narrowing/data/amp_sim_block_power_spectre_lor2,lor,lor3_4,lor2_3_20230815_172100.pkl", "rb") as f1:
    amps = pickle.load(f1)
with open("C:/Users/Ivo/Documents/qiskit_codes/paper_ready_plots/power_narrowing/data/det_sim_block_power_spectre_lor2,lor,lor3_4,lor2_3_20230815_172100.pkl", "rb") as f2:
    dets_init = pickle.load(f2)
with open("C:/Users/Ivo/Documents/qiskit_codes/paper_ready_plots/power_narrowing/data/tr_probs_sim_block_power_spectre_lor2,lor,lor3_4,lor2_3_20230815_172100.pkl", "rb") as f3:
    tr_probs = pickle.load(f3)

file_dir = os.path.dirname(__file__)

times = {
    "lor2": ["2023-07-04", "192920"],
    "lor": ["2023-07-04", "024124"],
    "lor3_4": ["2023-07-04", "193308"],
    "lor2_3": ["2023-07-04", "193649"],
}
params = {
    "lor2": [(24 + 8/9) * 1e-9, (181 + 2/3) * 1e-9],
    "lor": [(24 + 8/9) * 1e-9, (704) * 1e-9],
    "lor3_4": [(10 + 2/3) * 1e-9, (728 + 8/9) * 1e-9],
    "lor2_3": [(10 + 2/3) * 1e-9, (1134 + 2/9) * 1e-9],
}
pulse_names = list(params.keys())
powers = [2, 1, 3/4, 2/3]
powers_latex = [" $L^2$", "  $L$", "$L^{3/4}$", "$L^{2/3}$"]
intervals = [200e6, 100e6, 100e6, 100e6]
intervals_det = [2.5e6, 2.5e6, 5e6, 5e6]

save_map = 1

dets_init = np.array(dets_init) / (2 * np.pi)

dets = dets_init * np.array(list(params.values()))[:, 0][:, None]

fig_sim = plt.figure(figsize=(12,10), layout="constrained")
gs_sim = fig_sim.add_gridspec(2, 3, width_ratios=[1, 1, 0.1])
 
for i in range(2):
    for j in range(2):
        max_amp = intervals[2*i+j] * np.floor(amps[2*i+j][-1] / intervals[2*i+j])
        max_amp_minor = intervals[2*i+j] / 4 * np.ceil(amps[2*i+j][-1] / (intervals[2*i+j] / 4))
        max_det = intervals_det[2*i+j] * np.floor(dets_init[2*i+j][-1] / intervals_det[2*i+j])
        cmap_sim = plt.cm.get_cmap('cividis')  # Choose a colormap
        ax_sim = fig_sim.add_subplot(gs_sim[i, j])
        # im_sim = ax_sim.pcolormesh(dets[2*i+j], amps[2*i+j], tr_probs[2*i+j], vmin=0, vmax=1, cmap=cmap_sim)
        im_sim = ax_sim.imshow(tr_probs[2*i+j], cmap=cmap_sim, aspect="auto", origin="lower", vmin=0, vmax=1)
        # Add a rectangle in the top right corner
        rect = Rectangle((0.8, 0.86), 0.15, 0.14, transform=ax_sim.transAxes,
                        color='#DEDA8D', alpha=0.7)
        ax_sim.add_patch(rect)
        
        # Add text inside the rectangle
        text = powers_latex[2*i+j]
        ax_sim.text(0.82, 0.885, text, transform=ax_sim.transAxes,
            color='black', fontsize=18)

        yticks = np.linspace(
            0, 
            int(np.round(max_amp / amps[2*i+j][-1] * (len(tr_probs[2*i+j]) - 1))), 
            int(max_amp / intervals[2*i+j] + 1)
        )
        # ax_sim.set_yticks(np.round(np.arange(0, max_amp + 1e-3, intervals[2*i+j])).astype(int))
        # ax_sim.set_yticks(np.round(np.arange(0, max_amp_minor - intervals[2*i+j] / 4 + 1e-3, intervals[2*i+j] / 4)).astype(int), minor=True)
        ax_sim.set_yticks(np.round(yticks).astype(int))
        ax_sim.set_yticklabels(np.round(np.arange(0, max_amp / 1e6 + 1e-3, intervals[2*i+j] / 1e6)/100).astype(int), fontsize=18)
        
        tick_labels = (np.arange(
            -max_det, 
            max_det + 0.01, 
            intervals_det[2*i+j]
        ) / (1e6))
        # ax_sim.set_xlim(tick_locations[0], tick_locations[-1])
        # tick_locations[-1] = 14.85
        tick_locations = np.linspace(
            len(tr_probs[2*i+j][0]) - int(np.round(max_det / dets_init[2*i+j][-1] * (len(tr_probs[2*i+j][0]) / 2 - 1) + len(tr_probs[2*i+j][0]) / 2)),
            int(np.round(max_det / dets_init[2*i+j][-1] * (len(tr_probs[2*i+j][0]) / 2 - 1) + len(tr_probs[2*i+j][0]) / 2)),
            len(tick_labels)
            # len(tr_probs[2*i+j][0]) - int(np.round(0.15 / dets[2*i+j][-1] * (len(tr_probs[2*i+j][0]) / 2 - 1) + len(tr_probs[2*i+j][0]) / 2)), 
            # int(np.round(0.15 / dets[2*i+j][-1] * (len(tr_probs[2*i+j][0]) / 2 - 1) + len(tr_probs[2*i+j][0]) / 2)), 
            # 7
        )

        ax_sim.set_xticks(tick_locations)
        ax_sim.set_xticklabels(np.round(tick_labels, 2), fontsize=18)

        if j == 0:
            ax_sim.set_ylabel('Peak Rabi Freq. (100 MHz)', fontsize=18)
        if i == 1:
            ax_sim.set_xlabel('Detuning (MHz)', fontsize=18)

        ax_sim_2 = ax_sim.secondary_xaxis('top')

        tick_labels2 = np.arange(-0.15, 0.1501, 0.05)
        # ax_sim.set_xlim(tick_locations[0], tick_locations[-1])
        # tick_locations[-1] = 14.85
        tick_locations2 = np.linspace(
            len(tr_probs[2*i+j][0]) - int(np.round(0.15 / dets[2*i+j][-1] * (len(tr_probs[2*i+j][0]) / 2 - 1) + len(tr_probs[2*i+j][0]) / 2)), 
            int(np.round(0.15 / dets[2*i+j][-1] * (len(tr_probs[2*i+j][0]) / 2 - 1) + len(tr_probs[2*i+j][0]) / 2)), 
            len(tick_labels2)
        )

        ax_sim_2.set_xticks(tick_locations2)
        ax_sim.set_xlim(tick_locations2[0], tick_locations2[-1])
        # ax_sim.set_xticks(np.arange(tick_locations[0], tick_locations[-1] + 1e-3, 0.01), minor=True)
        if i == 0:
            ax_sim_2.set_xlabel('Detuning (in units of $1/\\tau$)', fontsize=18)
            ax_sim_2.set_xticklabels(np.round(tick_labels2, 2), fontsize=18)
        else:
            ax_sim_2.set_xticklabels([])

colorbar_ticks = np.round(np.arange(0, 1.01, 0.2),1)
# Create a separate subplot for the color bar
cax_sim = fig_sim.add_subplot(gs_sim[:, 2])
cbar_sim = fig_sim.colorbar(im_sim, cax=cax_sim)
cbar_sim.set_ticks(colorbar_ticks)
# cbar_sim.set_label('Simulated Transition probability', fontsize=18)
cbar_sim.ax.tick_params(labelsize=18)

# Adjust the layout
# fig_sim.tight_layout()

# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()


data_save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing", "data")
make_all_dirs(data_save_folder)

# Set fig name
fig_name_sim = f"sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"
tr_probs_name_sim = f"tr_probs_sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pkl"
dets_name_sim = f"det_sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pkl"
amps_name_sim = f"amp_sim_block_power_spectre_{','.join([k for k in times.keys()])}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pkl"


# Save the simulation
if save_map:
    plt.savefig(os.path.join(save_folder, fig_name_sim), format="pdf")

# plt.show()