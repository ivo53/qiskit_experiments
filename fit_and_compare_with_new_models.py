import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as sp
from scipy.optimize import curve_fit, root
from scipy.integrate import quad, quad_vec
from scipy.misc import derivative
import mpmath as mp

from transition_line_profile_functions import *


times = {
    "gauss": [["2022-06-16", "174431"]],
    "constant": [["2022-06-16", "174532"]],
    "rabi": [["2022-06-16", "174532"]],
    "sine": [["2022-06-16", "174541"]],
    "sine2": [["2022-06-16", "174543"]],
    "sine3": [["2022-06-16", "174546"]],
    "rz": [["2022-06-16", "174403"]],
    "sech2": [["2022-06-16", "174406"]],
    "demkov": [["2022-06-16", "174352"]],
    "lor_192": [["2023-02-28", "015053"],["2023-02-28", "015100"],["2023-02-28", "020906"],["2023-02-28", "020911"],["2023-02-28", "020917"],["2023-02-28", "020925"]],
    "lor2_192": [["2023-02-28", "021904"],["2023-02-28", "021910"],["2023-02-28", "021915"],["2023-02-28", "021921"],["2023-02-28", "021925"],["2023-02-28", "021933"]],
    "lor3_192": [["2023-02-28", "021942"],["2023-02-28", "021954"],["2023-02-28", "022002"],["2023-02-28", "022009"],["2023-02-28", "022017"],["2023-02-28", "022026"]],
    "sech_192": [["2023-02-28", "025719"],["2023-02-28", "025724"],["2023-02-28", "012715"],["2023-02-28", "012720"],["2023-02-28", "012726"],["2023-02-28", "012733"]],
    "sech2_192": [["2023-02-28", "025731"],["2023-02-28", "025737"],["2023-02-28", "012801"],["2023-02-28", "012806"],["2023-02-28", "012810"],["2023-02-28", "012818"]],
    "gauss_192": [["2023-02-28", "025701"],["2023-02-28", "025706"],["2023-02-28", "012839"],["2023-02-28", "012843"],["2023-02-28", "012848"],["2023-02-28", "012857"]],
    "sin_192": [["2023-02-28", "025352"]],
    "sin2_192": [["2023-02-28", "025357"]],
    "sin3_192": [["2023-02-28", "025401"]],
    "sin4_192": [["2023-02-28", "025405"]],
    "sin5_192": [["2023-02-28", "025410"]],
}

durations = {
    192: 0,
    384: 1,
    768: 2,
    1152: 3,
    1920: 4,
    3840: 5
}

# date = "2022-06-16"
area = "pi"
backend_name = "manila"
s = 192
dur = 192
pulse_type = "sech2"
pulse_type = pulse_type if s is None else "_".join([pulse_type, str(s)])
dur_idx = durations [dur] if dur is not None else 0
date = times[pulse_type][dur_idx][0]
time = times[pulse_type][dur_idx][1]
fit_func = pulse_type if "_" not in pulse_type else pulse_type.split("_")[0]
baseline_fit_func = "sinc2" if pulse_type in ["rabi", "constant"] else "lorentzian"


FIT_FUNCTIONS = {
    "lorentzian": lorentzian,
    "constant": rabi,
    "rabi": rabi,
    "gauss": gauss_rzconj,
    "rz": rz,
    "sech": rz,
    "demkov": demkov,
    "sech2": sech_sq,
    "sinc2": sinc2,
    "sin": sin_alt,
    "sin2": sin2,
    "sin3": sin3,
    "sin4": sin4,
    "sin5": sin4,
}
file_dir = os.path.dirname(__file__)
data_folder = os.path.join(file_dir, "data", backend_name, "calibration", date)
data_files = os.listdir(data_folder)
center_freq = 4962284031.287086 * 1e-6
for d in data_files:
    if d.startswith(times[pulse_type][dur_idx][1]):
        csv_file = d
        break

# pulse_type = d.split("_")[1]
# pulse_type = "rabi" if pulse_type in ["sq", "constant"] else pulse_type
# pulse_type = "rz" if pulse_type in ["sech", "rosenzener"] else pulse_type

model_name_dict = {
    "rabi": ["Rabi", "Sinc$^2$"], 
    "rz": ["Rosen-Zener", "Lorentzian"], 
    "sech": ["Rosen-Zener", "Lorentzian"], 
    "gauss": ["Gaussian", "Lorentzian"], 
    "demkov": ["Demkov", "Lorentzian"], 
    "sech2": ["Sech$^2$", "Lorentzian"],
    "sin": ["Sine", "Lorentzian"],
    "sin2": ["Sine$^2$", "Lorentzian"],
    "sin3": ["Sine$^3$", "Lorentzian"],
    "sin4": ["Sine$^4$", "Lorentzian"],
    "sin5": ["Sine$^5$", "Lorentzian"],
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
    initial = [0.1, 0, 0]
    initial_min = [-3, 0, 0]
    initial_max = [3, 0.5, 0.6]
    fit_params, y_fit, err = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func],
        initial, initial_min, initial_max,
        s, dur
    )
    y_fit = FIT_FUNCTIONS[fit_func](detuning, *fit_params)
    ##
    ##
    baseline_fit_params, baseline_y_fit, baseline_err = fit_function(
        detuning,
        vals, 
        FIT_FUNCTIONS[baseline_fit_func],
        [2, 0, 1, 0], # initial parameters for curve_fit
        [0, -10, 0, 0],
        [100, 10, 100, 0.5],
        s, dur

        # [1, 0, 1], # initial parameters for curve_fit
        # [0, -10, 0],
        # [10, 10, 1]
    )
    ##
    # print(fit_params, "\n", baseline_fit_params)
    if ef is not None: 
        ef = sech2_extended_freq
    else:
        ef = extended_freq
    extended_y_fit = FIT_FUNCTIONS[fit_func](ef, *fit_params)
    baseline_extended_y_fit = FIT_FUNCTIONS[baseline_fit_func](ef, *baseline_fit_params)

    similarity_idx = np.sum(np.abs(y_fit - vals))
    baseline_similarity_idx = np.sum(np.abs(baseline_y_fit - vals))

    overfitting_idx = np.mean(np.abs(np.diff(extended_y_fit)))
    baseline_overfitting_idx = np.mean(np.abs(np.diff(baseline_extended_y_fit)))
    overfitting = overfitting_idx > 0.1
    baseline_overfitting = baseline_overfitting_idx > 0.1
    # print(overfitting_idx, baseline_overfitting_idx)
    if overfitting:
        print("Strong overfitting present.")
        exit(1)
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

fit, baseline = fit_once(
    detuning, vals, fit_func,
    args=[0, 1, 1e6], 
    args_min=[-3, .99, 1e4],
    args_max=[3, 1., 1e8]
)
similarity_idx, y_fit, extended_y_fit, fit_params, err = fit
baseline_similarity_idx, baseline_y_fit, \
    baseline_extended_y_fit, baseline_fit_params, baseline_err = baseline
# print(fit_params)
# print(baseline_fit_params)
# print(err)
# print(baseline_err)

print(model_name_dict[fit_func][0])
print("Model SI:", similarity_idx)
print("Baseline SI:", baseline_similarity_idx)
q_freq_model = fit_params[0] / (2 * np.pi)
q_freq_bl = baseline_fit_params[-2] / (2 * np.pi)
q_freq_err_model = err[0] / (2 * np.pi)
q_freq_err_bl = baseline_err[-2] / (2 * np.pi)
print(q_freq_model, "+-", q_freq_err_model)
print(q_freq_bl, "+-", q_freq_err_bl)
print("deviation:", q_freq_model - q_freq_bl, "+-", np.sqrt(q_freq_err_model ** 2 + q_freq_err_bl ** 2))
# for o in np.linspace(5e5, 1e9, 500):
#     fit_, baseline_ = fit_once(
#         detuning, vals, fit_func, 0, 1, o,
#         q_freq_min=-3, egamma_min=.9, O_min=1e4,
#         q_freq_max=3, egamma_max=1., O_max=1e8)
#     similarity_idx_, y, prms = fit_
#     print("o =", o, "sim =", similarity_idx_, "rabi =", prms[-1])

scaled_ef = extended_freq / (2 * np.pi)
scaled_det = detuning / (2 * np.pi)

fig = plt.figure(constrained_layout=True, figsize=(10,7))
gs = fig.add_gridspec(7, 1)
ax0 = fig.add_subplot(gs[:5, :])
ax0.scatter(scaled_det, vals, color='black', marker="P")
ax0.plot(scaled_ef, baseline_extended_y_fit, color='blue')
ax0.plot(scaled_ef, extended_y_fit, color='red')
ax0.set_xlim(scaled_ef[0], -scaled_ef[0])
ax0.legend([
    "Measured values", 
    f"{model_name_dict[fit_func][1]} fit", 
    f"{model_name_dict[fit_func][0]} model analytical fit"
])

major_interval = 7.9
minor_interval = 0.5 if pulse_type=="rabi" else 1.975
major_xticks = np.round(np.arange(scaled_ef[0], -scaled_ef[0] + 1e-1, major_interval),1)
# x_limit = np.floor(np.abs(extended_freq[0]) / 5) * 5
# x_interval = np.round(x_limit / 5) if pulse_type == "rabi" else np.round(x_limit / 6)
# x_small_interval = np.round(x_interval / 3) if pulse_type == "rabi" else np.round(x_limit / 30)
# # print(x_limit)
# major_xticks = np.round(np.arange(-x_limit, x_limit + 1e-3, x_interval),0)
major_xticks[major_xticks>-0.01] = np.abs(major_xticks[major_xticks>-0.01])
minor_xticks = np.round(np.arange(scaled_ef[0], -scaled_ef[0] + 1e-1, minor_interval),1)
# minor_xticks = np.round(np.arange(-x_limit, x_limit + 1e-3, x_small_interval),0)
major_yticks = np.arange(0, 1.01, 0.2).round(1)
minor_yticks = np.arange(0, 1.01, 0.1).round(1)

ax0.set_xticks(major_xticks)
ax0.set_xticklabels([])
ax0.set_xticks(minor_xticks, minor="True")
ax0.set_yticks(major_yticks)
ax0.set_yticklabels(major_yticks, fontsize=16)
ax0.set_yticks(minor_yticks, minor="True")
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

limit_num = np.ceil(np.amax([max(baseline_y_fit - vals), np.abs(min(baseline_y_fit - vals))]) / 0.05) * 0.05
tick_interval = 0.1 if limit_num > 0.1 else 0.05
if limit_num > 0.1:
    y_ticks_res_minor = np.arange(-limit_num, limit_num + 1e-3, 0.05).round(2)

y_ticks_res = np.arange(-limit_num, limit_num + 1e-3, tick_interval).round(2)

err = 1 / np.sqrt(2048)
ax1 = fig.add_subplot(gs[5:6, :])
ax1.set_ylim(-limit_num, limit_num)
ax1.set_xlim(scaled_ef[0], -scaled_ef[0])

ax1.set_xticks(major_xticks)
ax1.set_xticklabels([])
ax1.set_xticks(minor_xticks, minor="True")
ax1.set_yticks(y_ticks_res)
ax1.set_yticklabels(y_ticks_res, fontsize=13)
if limit_num > 0.1:
    ax1.set_yticks(y_ticks_res_minor, minor=True)
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.6)
else:
    ax1.grid()
ax1.errorbar(scaled_det, y_fit - vals, yerr=err * np.ones(scaled_det.shape), fmt="+", color="r")
# print(major_xticks)
ax2 = fig.add_subplot(gs[6:, :], sharey=ax1)
ax2.set_xlim(scaled_ef[0], -scaled_ef[0])
ax2.set_xticks(major_xticks)
ax2.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
ax2.set_xticks(minor_xticks, minor="True")
ax2.set_yticklabels(y_ticks_res, fontsize=13)
ax2.errorbar(scaled_det, baseline_y_fit - vals, yerr=err * np.ones(scaled_det.shape), fmt="+", color="b")
if limit_num > 0.1:
    ax2.set_yticks(y_ticks_res_minor, minor=True)
    ax2.grid(which='minor', alpha=0.3)
    ax2.grid(which='major', alpha=0.6)
else:
    ax2.grid()
plt.xlabel("Detuning [MHz]", fontsize=20)

# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
save_dir = "C:/Users/Ivo/Documents/PhD Documents/Sine Pulses"
date = datetime.now()
fig_name = pulse_type + "_" + date.strftime("%Y%m%d") + ".pdf"
plt.savefig(os.path.join(save_dir, fig_name), format="pdf")
plt.show()