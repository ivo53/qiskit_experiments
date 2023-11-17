import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from transition_line_profile_functions import *

center_freqs = {
    "armonk": 4971730000 * 1e-6,
    "quito": 5300687845.108738 * 1e-6,
    "manila": 4962287341.966912 * 1e-6,
    "perth": 5157543429.014656 * 1e-6,
    "lima": 5029749743.642724 * 1e-6
}

model_name_dict = {
    "rabi": ["Rabi", "Sinc$^2$"], 
    "rz": ["Rosen-Zener rLZSM", "Double Approx"], 
    "sech": ["Rosen-Zener rLZSM", "Double Approx"], 
    "gauss": ["Gaussian rLZSM", "Double Approx"], 
    "demkov": ["Demkov rLZSM", "Double Approx"], 
    "sech2": ["Sech$^2$ rLZSM", "Double Approx"],
    "sin": ["Sine rLZSM", "Double Approx"],
    "sin2": ["Sine$^2$", "Double Approx"],
    "sin3": ["Sine$^3$", "Double Approx"],
    "sin4": ["Sine$^4$", "Double Approx"],
    "sin5": ["Sine$^5$", "Double Approx"],
    "lor": ["Lorentzian rLZSM", "Double Approx"],
    "lor2": ["Lorentzian$^2$ rLZSM", "Double Approx"],
    "lor3": ["Lorentzian$^3$ rLZSM", "Double Approx"],
}

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

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

file_dir = os.path.dirname(__file__)

save_dir = os.path.join(
    file_dir,
    "paper_ready_plots",
    "finite_pulses"
).replace("\\", "/")

def data_folder(date):
    return os.path.join(
        file_dir,
        "data",
        f"{backend_name}",
        "calibration",
        date
    ).replace("\\", "/")

backend_name = "quito"
# pulse_types = ["lor", "lor2", "demkov", "sech", "sech2", "gauss"]
pulse_types = ["sin"]
save = 1

times = {
    "lor_192": [["2023-04-27", "135516"],["2023-04-27", "135524"],["2023-04-27", "135528"],["2023-04-27", "135532"]],
    "lor2_192": [["2023-04-27", "135609"],["2023-04-27", "135613"],["2023-04-27", "135616"],["2023-04-27", "135619"]],
    "lor3_192": [["2023-04-27", "135622"],["2023-04-27", "135625"],["2023-04-27", "135628"],["2023-04-27", "135631"]],
    "sech_192": [["2023-04-27", "135702"],["2023-04-27", "135706"],["2023-04-27", "135710"],["2023-04-27", "135714"]],
    "sech2_192": [["2023-04-27", "135730"],["2023-04-27", "135734"],["2023-04-27", "135736"],["2023-04-27", "135738"]],
    "gauss_192": [["2023-04-27", "135639"],["2023-04-27", "135644"],["2023-04-27", "135648"],["2023-04-27", "135651"]],
    "demkov_192": [["2023-04-27", "135803"],["2023-04-27", "135807"],["2023-04-27", "135811"],["2023-04-27", "135815"]],
    "sin_192": [["2023-04-27", "135342"]],
    "sin2_192": [["2023-04-27", "135402"]],
    "sin3_192": [["2023-04-27", "135404"]],
    "sin4_192": [["2023-04-27", "135406"]],
    "sin5_192": [["2023-04-27", "135408"]],
}

durations = {
    192: 0,
    224: 1,
    256: 2,
    320: 3
}

s = 192
dur = 192 # get_closest_multiple_of_16(round(957.28))
tr_probs, dets = [], []

for pulse_type in pulse_types:
    full_pulse_name = pulse_type if dur is None else "_".join([pulse_type, str(s)])
    print(full_pulse_name)
    dur_idx = durations [dur] if dur is not None else 0
    date = times[full_pulse_name][dur_idx][0]
    time = times[full_pulse_name][dur_idx][1]
    fit_func = pulse_type # full_pulse_name if "_" not in full_pulse_name else full_pulse_name.split("_")[0]
    # baseline_fit_func = "sinc2" if pulse_type in ["rabi", "constant"] else "lorentzian"
    comparison = 0 # 0 or 1, whether to have both curves
    log_plot = 0

    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)


    l, p, x0 = 419.1631352890144, 0.0957564968883284, 0.0003302995697281
    T = 192 * 2/9 * 1e-9

    tr_prob, det = [], []
    for t in times[full_pulse_name]:
        files = os.listdir(data_folder(t[0]))
        for file in files:
            if file.startswith(t[1]) and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(data_folder(t[0]), file))
        tr_prob.append(df["transition_probability"].to_numpy())
        det.append((df["frequency_ghz"].to_numpy()))
    tr_probs.append(tr_prob[dur_idx])
    dets.append(det[dur_idx])
# print(dets)
dets = np.array(dets) * 1e3
tr_probs = np.array(tr_probs)
dets_subtracted = dets - dets.mean(1)[:, None]
# print(dets_subtracted)
colors = ["r", "g"]
params = [
    [
        [0.1,0.3,0.2,0.32],
        [-10,0,0,0.2],
        [10,1,1,.5]
    ],
    [
        [0.1,0.3,0.2],
        [-10,0,0],
        [10,1,1]
    ]
]

num_figures = len(pulse_types)
num_columns = 3 if num_figures % 3 == 0 else 1
num_rows = int(num_figures / num_columns)

font_size, width, height = 16, 8, 6
# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(num_columns * width, num_rows * height), layout="constrained")
gs0 = fig.add_gridspec(3 * num_rows, num_columns, height_ratios=[1, 0.1, 0.1] * num_rows, width_ratios=[1] * num_columns)
# Generate datetime
date = datetime.now()

for i in range(num_rows):
    for j in range(num_columns):
        d = dets_subtracted[num_columns*i+j]
        tr = tr_probs[num_columns*i+j]
        efs, ex_tr_fits, tr_fits = [], [], []
        current_pulse_shape = pulse_types[num_columns*i+j]
        ffs = FIT_FUNCTIONS[current_pulse_shape]
        init_params, lower, higher = params[1] if len(ffs) == 1 else params[0]
        for ff in ffs:
            fitparams, tr_fit, perr = fit_function(
                d * 2 * np.pi, tr, ff, 
                init_params=init_params,
                lower=lower,
                higher=higher,
                sigma=s, duration=dur,
                remove_bg=True, area=np.pi
            )
            print(fitparams)
            ef = np.linspace(d[0], d[-1], 5000) * 2 * np.pi
            extended_tr_fit = ff(ef, *fitparams)
            efs.append(ef)
            tr_fits.append(tr_fit)
            ex_tr_fits.append(extended_tr_fit)

        ax = fig.add_subplot(gs0[3*i, j])
        ax.scatter(d, tr, marker="p", label="Measured Data")
        for idx, ex_tr_fit in enumerate(ex_tr_fits):
            ax.plot((ef - ef.mean()) / (2 * np.pi), ex_tr_fit, color=colors[idx], label=model_name_dict[pulse_types[num_columns*i+j]][idx])
        ax.legend(fontsize=font_size)
        ax.set_xticklabels([])
        ax.set_yticklabels([item.get_text() for item in ax.get_yticklabels()], fontsize=font_size)
        ax1 = fig.add_subplot(gs0[3*i+1, j])
        ax2 = fig.add_subplot(gs0[3*i+2, j])
        ax1.scatter(d, tr_fits[0] - tr, c="r", marker="x")
        ax1.set_ylim((-0.03, 0.03))
        ax1.set_xticklabels([])
        ax2.scatter(d, tr_fits[1] - tr, c="g", marker="x")
        ax2.set_ylim((-0.03, 0.03))
        if j == 0:
            ax.set_ylabel("Transition Probability", fontsize=font_size)
            ax1.set_yticklabels([item.get_text() for item in ax1.get_yticklabels()], fontsize=0.9 * font_size)
            ax2.set_yticklabels([item.get_text() for item in ax2.get_yticklabels()], fontsize=0.9 * font_size)
        else:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        if i == num_rows - 1:
            xlabels = [item.get_text() for item in ax2.get_xticklabels()]
            ax2.set_xticklabels(xlabels, fontsize=font_size)
            ax2.set_xlabel("Detuning (MHz)", fontsize=font_size)
        else:
            ax2.set_xticklabels([])


# plt.show()
if save:
    plt.savefig(os.path.join(save_dir, f"two_fits_intermixed_dur-{dur}dt_s-{s}dt_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"), format="pdf")