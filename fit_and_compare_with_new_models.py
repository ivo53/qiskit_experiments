import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, root
from scipy.integrate import quad, quad_vec
from scipy.misc import derivative
import mpmath as mp

from transition_line_profile_functions import *


# times = {
#     "gauss": [["2022-06-16", "174431"]],
#     "constant": [["2022-06-16", "174532"]],
#     "rabi": [["2022-06-16", "174532"]],
#     "sine": [["2022-06-16", "174541"]],
#     "sine2": [["2022-06-16", "174543"]],
#     "sine3": [["2022-06-16", "174546"]],
#     "rz": [["2022-06-16", "174403"]],
#     "sech2": [["2022-06-16", "174406"]],
#     "demkov": [["2022-06-16", "174352"]],
#     "lor_192": [["2023-02-28", "025139"],["2023-02-28", "025146"],["2023-02-28", "020906"],["2023-02-28", "020911"],["2023-02-28", "020917"],["2023-02-28", "020925"]],
#     "lor2_192": [["2023-02-28", "025156"],["2023-02-28", "025202"],["2023-02-28", "021915"],["2023-02-28", "021921"],["2023-02-28", "021925"],["2023-02-28", "021933"]],
#     "lor3_192": [["2023-02-28", "025228"],["2023-02-28", "025233"],["2023-02-28", "022002"],["2023-02-28", "022009"],["2023-02-28", "022017"],["2023-02-28", "022026"]],
#     "sech_192": [["2023-02-28", "025719"],["2023-02-28", "025724"],["2023-02-28", "012715"],["2023-02-28", "012720"],["2023-02-28", "012726"],["2023-02-28", "012733"]],
#     "sech2_192": [["2023-02-28", "025731"],["2023-02-28", "025737"],["2023-02-28", "012801"],["2023-02-28", "012806"],["2023-02-28", "012810"],["2023-02-28", "012818"]],
#     "gauss_192": [["2023-02-28", "025701"],["2023-02-28", "025706"],["2023-02-28", "012839"],["2023-02-28", "012843"],["2023-02-28", "012848"],["2023-02-28", "012857"]],
#     "sin_192": [["2023-02-28", "025352"]],
#     "sin2_192": [["2023-02-28", "025357"]],
#     "sin3_192": [["2023-02-28", "025401"]],
#     "sin4_192": [["2023-02-28", "025405"]],
#     "sin5_192": [["2023-02-28", "025410"]],
# }

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

# durations = {
#     192: 0,
#     384: 1,
#     768: 2,
#     1152: 3,
#     1920: 4,
#     3840: 5
# }

durations = {
    192: 0,
    224: 1,
    256: 2,
    320: 3
}
 
# date = "2022-06-16"
area = "pi"
backend_name = "quito"
s = 192
dur = 192
pulse_type = "sin"
pulse_type = pulse_type if s is None else "_".join([pulse_type, str(s)])
dur_idx = durations [dur] if dur is not None else 0
date = times[pulse_type][dur_idx][0]
time = times[pulse_type][dur_idx][1]
fit_func = pulse_type if "_" not in pulse_type else pulse_type.split("_")[0]
# baseline_fit_func = "sinc2" if pulse_type in ["rabi", "constant"] else "lorentzian"
comparison = 0 # 0 or 1, whether to have both curves

FIT_FUNCTIONS = {
    "lorentzian": lorentzian,
    "constant": rabi,
    "rabi": rabi,
    "gauss": [gauss_rlzsm, gauss_dappr], #gauss_rzconj,
    "rz": [sech_rlzsm, sech_dappr], #rz,
    "sech": [sech_rlzsm, sech_dappr], #rz,
    "demkov": [demkov,],
    "sech2": [sech2_rlzsm, sech2_dappr], #sech_sq,
    "sinc2": sinc2,
    "sin": [sin_rlzsm, sin_dappr],
    "sin2": [sin2,],
    "sin3": [sin3,],
    "sin4": [sin4,],
    "sin5": [sin5,],
    "lor": [lor_rlzsm, lor_dappr],
    "lor2": [lor2_rlzsm, lor2_dappr],
    "lor3": [lor3_rlzsm, lor3_dappr],
}
file_dir = os.path.dirname(__file__)
data_folder = os.path.join(file_dir, "data", backend_name, "calibration", date)
data_files = os.listdir(data_folder)
# center_freq = 4962284031.287086 * 1e-6
center_freq = 5300687845.108738 * 1e-6
for d in data_files:
    if d.startswith(times[pulse_type][dur_idx][1]):
        csv_file = d
        break

# pulse_type = d.split("_")[1]
# pulse_type = "rabi" if pulse_type in ["sq", "constant"] else pulse_type
# pulse_type = "rz" if pulse_type in ["sech", "rosenzener"] else pulse_type

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

df = pd.read_csv(os.path.join(data_folder, csv_file))
freq = df["frequency_ghz"].to_numpy() * 1e3
vals = df["transition_probability"].to_numpy()
detuning = 2 * np.pi * (freq - center_freq)
extended_freq = np.linspace(detuning[0], detuning[-1], 5000)

def fit_once(
    detuning, vals, fit_func,
    args, args_min, args_max,
    ef=None
):
    # initial = [0, 0.4, 0.4, 0.4] if fit_func in ["sech2"] else [0.1, 0, 0]
    # initial_min = [-3, 0.3, 0.3, 0] if fit_func in ["sech2"] else [-3, 0, 0]
    # initial_max = [3, 0.5, 0.6, 1] if fit_func in ["sech2"] else [3, 0.5, 0.6]
    # initial = [0.1, 0, 0]
    # initial_min = [-3, 0, 0]
    # initial_max = [3, 0.5, 0.6]
    fit_params, y_fit, err = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func][1],
        # initial, initial_min, initial_max,
        args, args_min, args_max,
        s, dur
    )
    y_fit = FIT_FUNCTIONS[fit_func][1](detuning, *fit_params)
    ##
    ##
    if comparison:
        baseline_fit_params, baseline_y_fit, baseline_err = fit_function(
            detuning,
            vals, 
            FIT_FUNCTIONS[fit_func][0],
            args, args_min, args_max,
            s, dur

            # [1, 0, 1], # initial parameters for curve_fit
            # [0, -10, 0],
            # [10, 10, 1]
        )
    ##
    # print(fit_params, "\n", baseline_fit_params)
    ef = extended_freq
    extended_y_fit = FIT_FUNCTIONS[fit_func][0](ef, *fit_params)
    similarity_idx = np.sum(np.abs(y_fit - vals))
    overfitting_idx = np.mean(np.abs(np.diff(extended_y_fit)))
    overfitting = overfitting_idx > 0.1
    if overfitting:
        print("Strong overfitting present.")
        exit(1)

    if comparison:
        baseline_extended_y_fit = FIT_FUNCTIONS[fit_func][1](ef, *baseline_fit_params)
        baseline_similarity_idx = np.sum(np.abs(baseline_y_fit - vals))
        baseline_overfitting_idx = np.mean(np.abs(np.diff(baseline_extended_y_fit)))
        baseline_overfitting = baseline_overfitting_idx > 0.1
    # print(overfitting_idx, baseline_overfitting_idx)
    if comparison:
        return (similarity_idx, 
            y_fit, 
            extended_y_fit, 
            fit_params,
            err), \
           (baseline_similarity_idx, 
            baseline_y_fit, 
            baseline_extended_y_fit, 
            baseline_fit_params,
            baseline_err)
    return ((similarity_idx, 
            y_fit, 
            extended_y_fit, 
            fit_params,
            err),)

iargs = [0.1, 0, 0, 0.24] if dur < 5 * s else [0.1, 0, 0]
minargs = [-3, 0, 0, 0.04] if dur < 5 * s else [-3, 0, 0]
maxargs = [3, 0.5, 0.6, 0.5] if dur < 5 * s else [3, 0.5, 0.6]

fit = fit_once(
    detuning, vals, fit_func,
    args=iargs, 
    args_min=minargs,
    args_max=maxargs
)
if comparison:
    fit, baseline = fit
    baseline_similarity_idx, baseline_y_fit, \
        baseline_extended_y_fit, baseline_fit_params, baseline_err = baseline
else:
    fit = fit[0]
similarity_idx, y_fit, extended_y_fit, fit_params, err = fit
print(fit_params)
# print(baseline_fit_params)
# print(err)
# print(baseline_err)
dof = len(vals) - len(fit_params)
residuals = y_fit - vals
err_res = np.sqrt(np.sum(residuals ** 2) / dof)

print(model_name_dict[fit_func][0])
print("rLZSM SI:", similarity_idx)
q_freq_model = fit_params[0] / (2 * np.pi)
q_freq_err_model = err[0] / (2 * np.pi)
if comparison:
    print("Double approx SI:", baseline_similarity_idx)
    q_freq_bl = baseline_fit_params[-2] / (2 * np.pi)
    q_freq_err_bl = baseline_err[-2] / (2 * np.pi)
print(q_freq_model, "+-", q_freq_err_model)
if comparison:
    print(q_freq_bl, "+-", q_freq_err_bl)
    print("deviation:", q_freq_model - q_freq_bl, "+-", np.sqrt(q_freq_err_model ** 2 + q_freq_err_bl ** 2))
print(f"Error on residuals: {err_res}")
scaled_ef = extended_freq / (2 * np.pi)
scaled_det = detuning / (2 * np.pi)


date = datetime.now()
txt_name = pulse_type + "_" + str(dur) + "_" + str(comparison) + "_" + date.strftime("%Y%m%d") + ".txt"
with open(os.path.join(data_folder, txt_name), "w") as file:
    file.write(model_name_dict[fit_func][0] + "\n")
    file.write("Double Approx SI: " + str(similarity_idx) + "\n")
    if comparison:
        file.write("rLZSM SI: " + str(baseline_similarity_idx) + "\n")
    file.write(str(q_freq_model) + "+-" + str(q_freq_err_model) + "\n")
    if comparison:
        file.write(str(q_freq_bl) + "+-" + str(q_freq_err_bl) + "\n")
        file.write("deviation: " + str(q_freq_model - q_freq_bl) + "+-" + str(np.sqrt(q_freq_err_model ** 2 + q_freq_err_bl ** 2)) + "\n")
    file.write(f"Error on residuals: {str(err_res)}" + "\n")


if comparison:
    baseline_dof = len(vals) - len(baseline_fit_params)
    baseline_residuals = baseline_y_fit - vals
    baseline_err_res = np.sqrt(np.sum(baseline_residuals ** 2) / baseline_dof)

figsize = (10,7) if comparison else (10,6)
fig = plt.figure(constrained_layout=True, figsize=figsize)
gs = fig.add_gridspec(figsize[1], 1)
ax0 = fig.add_subplot(gs[:5, :])
ax0.semilogy(scaled_det, vals, color='black', marker="P", label="Measured values")
if comparison:
    ax0.semilogy(scaled_ef, baseline_extended_y_fit, color='blue', label=f"{model_name_dict[fit_func][0]} model fit")
ax0.semilogy(scaled_ef, extended_y_fit, color='red', label=f"{model_name_dict[fit_func][1]} model fit")
ax0.set_xlim(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10)
# Create a legend object and customize the order of the labels
handles, labels = plt.gca().get_legend_handles_labels()
order = [0, 2, 1] if comparison else [0, 1] # Change the order of the labels here
ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

major_interval = 20.
minor_interval = 0.5 if pulse_type=="rabi" else 2.5
major_xticks = np.round(np.arange(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10 + 1e-1, major_interval),1)
# x_limit = np.floor(np.abs(extended_freq[0]) / 5) * 5
# x_interval = np.round(x_limit / 5) if pulse_type == "rabi" else np.round(x_limit / 6)
# x_small_interval = np.round(x_interval / 3) if pulse_type == "rabi" else np.round(x_limit / 30)
# # print(x_limit)
# major_xticks = np.round(np.arange(-x_limit, x_limit + 1e-3, x_interval),0)
major_xticks[major_xticks>-0.01] = np.abs(major_xticks[major_xticks>-0.01])
minor_xticks = np.round(np.arange(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10 + 1e-1, minor_interval),1)
# minor_xticks = np.round(np.arange(-x_limit, x_limit + 1e-3, x_small_interval),0)
major_yticks = np.arange(0, 1.01, 0.2).round(1)
minor_yticks = np.arange(0, 1.01, 0.1).round(1)

ax0.set_xticks(major_xticks)
ax0.set_xticklabels([])
ax0.set_xticks(minor_xticks, minor="True")
# ax0.set_yticks(major_yticks)
# ax0.set_yticklabels(major_yticks, fontsize=16)
# ax0.set_yticks(minor_yticks, minor="True")
ax0.grid(which='minor', alpha=0.3)
ax0.grid(which='major', alpha=0.6)

# ax0.set_title(f"{pulse_type.capitalize()} Model Frequency Curve" \
# # - SI = {np.round(similarity_idx, 2)} vs BSI = \
# # {np.round(baseline_similarity_idx, 2)}"
# , fontsize=22)
ax0.set_ylabel("Transition Probability", fontsize=20)

ax = fig.add_subplot(gs[5:, :])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False,\
    right=False)
fig.text(0.015, 0.2, 'Residuals', ha='center', va='center', rotation='vertical', fontsize=20)

if comparison:
    limit_num = np.ceil(np.amax([max(baseline_y_fit - vals), np.abs(min(baseline_y_fit - vals))]) / 0.05) * 0.05
    tick_interval = 0.1 if limit_num > 0.1 else 0.05
    if limit_num > 0.1:
        y_ticks_res_minor = np.arange(-limit_num, limit_num + 1e-3, 0.05).round(2)
else:
    limit_num = np.ceil(np.amax([max(y_fit - vals), np.abs(min(y_fit - vals))]) / 0.05) * 0.05
    tick_interval = 0.1 if limit_num > 0.1 else 0.05
    if limit_num > 0.1:
        y_ticks_res_minor = np.arange(-limit_num, limit_num + 1e-3, 0.05).round(2)


y_ticks_res = np.arange(-limit_num, limit_num + 1e-3, tick_interval).round(2)

ax1 = fig.add_subplot(gs[5:6, :])
ax1.set_ylim(-limit_num, limit_num)
ax1.set_xlim(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10)

ax1.set_xticks(major_xticks)
if comparison:
    ax1.set_xticklabels([])
else:
    ax1.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
ax1.set_xticks(minor_xticks, minor="True")
ax1.set_yticks(y_ticks_res)
ax1.set_yticklabels(y_ticks_res, fontsize=13)
if limit_num > 0.1:
    ax1.set_yticks(y_ticks_res_minor, minor=True)
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.6)
else:
    ax1.grid()
ax1.errorbar(scaled_det, y_fit - vals, yerr=err_res * np.ones(scaled_det.shape), fmt="+", color="r")
# print(major_xticks)
if comparison:
    ax2 = fig.add_subplot(gs[6:, :], sharey=ax1)
    ax2.set_xlim(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10)
    ax2.set_xticks(major_xticks)
    ax2.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
    ax2.set_xticks(minor_xticks, minor="True")
    ax2.set_yticklabels(y_ticks_res, fontsize=13)
    ax2.errorbar(scaled_det, baseline_y_fit - vals, yerr=baseline_err_res * np.ones(scaled_det.shape), fmt="+", color="b")
    if limit_num > 0.1:
        ax2.set_yticks(y_ticks_res_minor, minor=True)
        ax2.grid(which='minor', alpha=0.3)
        ax2.grid(which='major', alpha=0.6)
    else:
        ax2.grid()
plt.xlabel("Detuning [MHz]", fontsize=20)

# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
save_dir = os.path.join(file_dir, "paper_ready_plots")

fig_name = pulse_type + "_" + str(dur) + "_" + str(comparison) + "_" + date.strftime("%Y%m%d") + ".pdf"
plt.savefig(os.path.join(save_dir, fig_name), format="pdf")
plt.show()