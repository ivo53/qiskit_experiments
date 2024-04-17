import os
import re
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from common.transition_line_profile_functions import *

center_freqs = {
    "armonk": 4971730000 * 1e-6,
    "quito": 5300687845.108738 * 1e-6,
    "manila": 4962287341.966912 * 1e-6,
    "perth": 5157543429.014656 * 1e-6,
    "lima": 5029749743.642724 * 1e-6
}

model_short_name_dict = ["Split Model", "Integrated Model"]
model_name_dict = {
    "rabi": ["Rabi", "Sinc$^2$", "Rabi Model"], 
    "rz": ["Rosen-Zener Split Model", "Integrated Model", "Sech"], 
    "sech": ["Rosen-Zener Split Model", "Integrated Model", "Sech"], 
    "gauss": ["Gaussian Split Model", "Integrated Model", "Gaussian"], 
    "demkov": ["Demkov Split Model", "Integrated Model", "Exponential"], 
    "sech2": ["Sech$^2$ Split Model", "Integrated Model", "Sech$^2$"],
    "sin": ["Sine Split Model", "Integrated Model", "Sine"],
    "sin2": ["Sine$^2$", "Integrated Model", "Sine$^2$"],
    "sin3": ["Sine$^3$", "Integrated Model", "Sine$^3$"],
    "sin4": ["Sine$^4$", "Integrated Model", "Sine$^4$"],
    "sin5": ["Sine$^5$", "Integrated Model", "Sine$^5$"],
    "lor": ["Lorentzian Split Model", "Integrated Model", "Lorentzian"],
    "lor2": ["Lorentzian$^2$ Split Model", "Integrated Model", "Lorentzian$^2$"],
    "lor3": ["Lorentzian$^3$ Split Model", "Integrated Model", "Lorentzian$^3$"],
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

file_dir = "/".join(os.path.dirname(__file__).split("\\")[:-1])
save_dir = os.path.join(
    file_dir,
    "paper_ready_plots",
    "finite_pulses"
).replace("\\", "/")

def mae(vals):
    return np.mean(np.abs(vals))

def data_folder(date):
    return os.path.join(
        file_dir,
        "data",
        f"{backend_name}",
        "calibration",
        date
    ).replace("\\", "/")

backend_name = "quito"
pulse_types = ["sin", "lor", "lor2", "sech", "sech2", "gauss"]
# pulse_types = ["sin"]
# pulse_types = ["lor2"] * 4
# pulse_types = ["demkov"]
# pulse_types = ["lor2", "demkov"]
both_models = 0
save = 0
# save = 1

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
# durs = [192, 224, 256, 320]
durs = [192] * len(pulse_types) # get_closest_multiple_of_16(round(957.28))
# durs = [320, 192]
tr_probs, dets = [], []

for pulse_type, dur in zip(pulse_types, durs):
    full_pulse_name = pulse_type if dur is None else "_".join([pulse_type, str(s)])
    # print(full_pulse_name)
    dur_idx = durations[dur] if dur is not None else 0
    date = times[full_pulse_name][dur_idx][0]
    time = times[full_pulse_name][dur_idx][1]
    fit_func = pulse_type # full_pulse_name if "_" not in full_pulse_name else full_pulse_name.split("_")[0]
    # baseline_fit_func = "sinc2" if pulse_type in ["rabi", "constant"] else "lorentzian"
    comparison = 0 # 0 or 1, whether to have both curves
    log_plot = 0


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
colors = ["#118ab3", "r"] if both_models else ["#118ab3", "r"]
params = [
    [
        [4,0.3,0.2,0.32],
        [-10,0,0,0.1],
        [10,1,1,.4]
    ],
    [
        [0.1,0.3,0.2],
        [-10,0,0],
        [10,1,1]
    ]
]

num_figures = len(pulse_types)
num_columns = np.maximum(int(num_figures / 2), 1) # 3 if num_figures % 3 == 0 else 2
num_rows = int(num_figures / num_columns)
print(num_columns, num_rows)
num_residual_axes = 2 if both_models else 1
font_size, width, height = 16, 8, 6
# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(num_columns * width, num_rows * height), layout="constrained")
gs0 = fig.add_gridspec((1 + num_residual_axes) * num_rows, num_columns, height_ratios=([1] + [0.1] * num_residual_axes) * num_rows, width_ratios=[1] * num_columns)
# Generate datetime
date = datetime.now()
maes, sdrfs = [], []
for i in range(num_rows):
    ma, sdr = [], []
    for j in range(num_columns):
        d = dets_subtracted[num_columns*i+j]
        tr = tr_probs[num_columns*i+j]
        efs, ex_tr_fits, tr_fits = [], [], []
        current_pulse_shape = pulse_types[num_columns*i+j]
        ffs = FIT_FUNCTIONS[current_pulse_shape]
        init_params, lower, higher = params[1] if len(ffs) == 1 else params[0]
        sd = []
        dur = durs[num_columns * i + j]
        for ff in ffs:
            fitparams, tr_fit, perr = fit_function(
                d * 2 * np.pi, tr, ff,
                init_params=init_params,
                lower=lower,
                higher=higher,
                sigma=s, duration=dur, time_interval=2e-9/9,
                remove_bg=True, area=np.pi
            )
            print(fitparams, perr)
            sd.append(perr[0] / (2 * np.pi))
            ef = np.linspace(d[0], d[-1], 5000) * 2 * np.pi
            extended_tr_fit = ff(ef, *fitparams)
            efs.append(ef)
            tr_fits.append(tr_fit)
            ex_tr_fits.append(extended_tr_fit)
        dur_ns = np.round(dur * 2 / 9, 2)
        ax = fig.add_subplot(gs0[3*i, j]) if both_models else fig.add_subplot(gs0[2*i, j])
        ax.scatter(d, tr, marker="p", color="black", label=f"Measured Data")
        for idx, ex_tr_fit in enumerate(ex_tr_fits):
            if not both_models and idx == 0:
                continue
            ax.plot((ef - ef.mean()) / (2 * np.pi), ex_tr_fit, color=colors[idx], label=model_short_name_dict[idx])#label=model_name_dict[pulse_types[num_columns*i+j]][idx])
        legend_text = f"{model_name_dict[pulse_types[num_columns * i + j]][2]} {dur_ns}ns" if len(pulse_types) < 5 \
            else f"{model_name_dict[pulse_types[num_columns * i + j]][2]} Shape"
        legend = ax.legend(fontsize=font_size, title_fontsize=font_size, title=legend_text)
        legend.get_title().set_fontweight('bold')
        if both_models:
            ax1 = fig.add_subplot(gs0[3*i+1, j])
            ax1.scatter(d, tr_fits[0] - tr, c=colors[0], marker="x")
            ax1.set_ylim((-0.03, 0.03))
            ax1.set_xticklabels([])
            ax2 = fig.add_subplot(gs0[3*i+2, j])
        else:
            ax2 = fig.add_subplot(gs0[2*i+1, j])
        ax2.scatter(d, tr_fits[1] - tr, c=colors[1], marker="x")
        ax2.set_ylim((-0.03, 0.03))
        ax.set_xlim((-80, 80))
        ax.set_ylim((-0.02,1))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        if both_models:
            ax1.set_xlim((-80, 80))
        ax2.set_xlim((-80, 80))
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        ylabels = [item.get_text() for item in ax.get_yticklabels()]
        minor_xlabels = np.linspace(
            float(re.sub(r'[^\x00-\x7F]+','-', xlabels[0])), 
            float(re.sub(r'[^\x00-\x7F]+','-', xlabels[-1])), 
            (len(xlabels) - 1) * 2 + 1
        )
        minor_ylabels = np.linspace(
            float(re.sub(r'[^\x00-\x7F]+','-', ylabels[0])), 
            float(re.sub(r'[^\x00-\x7F]+','-', ylabels[-1])), 
            (len(ylabels) - 1) * 2 + 1
        )
        # print(xlabels, minor_xlabels)
        ax.set_xticks(minor_xlabels, minor="True")
        if both_models:
            ax1.set_xticks(minor_xlabels, minor="True")
        ax2.set_xticks(minor_xlabels, minor="True")
        ax.set_xticklabels([])
        ax.set_yticks(minor_ylabels, minor="True")
        ax.set_yticklabels([item.get_text() for item in ax.get_yticklabels()], fontsize=font_size)
        if j == 0:
            ax.set_ylabel("Transition Probability", fontsize=font_size)
            if both_models:
                ax1.set_yticklabels([item.get_text() for item in ax1.get_yticklabels()], fontsize=0.9 * font_size)
            ax2.set_yticklabels([item.get_text() for item in ax2.get_yticklabels()], fontsize=0.9 * font_size)
        else:
            if both_models:
                ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        if i == num_rows - 1:
            ax2.set_xticklabels(xlabels, fontsize=font_size)
            ax2.set_xlabel("Detuning (MHz)", fontsize=font_size)
        else:
            ax2.set_xticklabels([])
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.6)
        if both_models:
            ax1.grid(which='minor', alpha=0.2)
            ax1.grid(which='major', alpha=0.6)
        ax2.grid(which='minor', alpha=0.2)
        ax2.grid(which='major', alpha=0.6)
        ma.append([mae(tr_fits[0] - tr), mae(tr_fits[1] - tr)])
        sdr.append(sd)
    maes.append(ma)
    sdrfs.append(sdr)

maes = np.array(maes)
sdrfs = np.array(sdrfs)
print(maes, sdrfs)
# plt.show()
if save:
    mae_csv_file_path1 = os.path.join(save_dir, f"MAE_split_dur-{dur}dt_s-{s}dt_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.txt")
    mae_csv_file_path2 = os.path.join(save_dir, f"MAE_intermixed_dur-{dur}dt_s-{s}dt_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.txt")
    sdrf_csv_file_path1 = os.path.join(save_dir, f"SDRF_split_dur-{dur}dt_s-{s}dt_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.txt")
    sdrf_csv_file_path2 = os.path.join(save_dir, f"SDRF_intermixed_dur-{dur}dt_s-{s}dt_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.txt")

    plt.savefig(os.path.join(save_dir, f"two_fits_intermixed_dur-{dur}dt_s-{s}dt_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"), format="pdf")

    np.savetxt(mae_csv_file_path1, maes[:, :, 0], delimiter=',')
    np.savetxt(sdrf_csv_file_path1, sdrfs[:, :, 0], delimiter=',')
    np.savetxt(mae_csv_file_path2, maes[:, :, 1], delimiter=',')
    np.savetxt(sdrf_csv_file_path2, sdrfs[:, :, 1], delimiter=',')
