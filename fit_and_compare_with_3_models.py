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

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

center_freqs = {
    "armonk": 4971730000 * 1e-6,
    "quito": 5300687845.108738 * 1e-6,
    "manila": 4962287341.966912 * 1e-6,
    "perth": 5157543429.014656 * 1e-6,
    "lima": 5029749743.642724 * 1e-6
}

# times = {
#     "gauss": ["2022-06-16", "174431"],
#     "constant": ["2022-06-16", "174532"],
#     "rabi": ["2022-06-16", "174532"],
#     "sine": ["2022-06-16", "174541"],
#     "sine2": ["2022-06-16", "174543"],
#     "sine3": ["2022-06-16", "174546"],
#     "rz": ["2022-06-16", "174403"],
#     "sech2": ["2022-06-16", "174406"],
#     "demkov": ["2022-06-16", "174352"],
# }

times = {
    # "lor_192": [["2023-04-27", "135516"],["2023-04-27", "135524"],["2023-04-27", "135528"],["2023-04-27", "135532"]],
    # "lor2_192": [["2023-04-27", "135609"],["2023-04-27", "135613"],["2023-04-27", "135616"],["2023-04-27", "135619"]],
    # "lor3_192": [["2023-04-27", "135622"],["2023-04-27", "135625"],["2023-04-27", "135628"],["2023-04-27", "135631"]],
    # "sech_192": [["2023-04-27", "135702"],["2023-04-27", "135706"],["2023-04-27", "135710"],["2023-04-27", "135714"]],
    # "sech2_192": [["2023-04-27", "135730"],["2023-04-27", "135734"],["2023-04-27", "135736"],["2023-04-27", "135738"]],
    # "gauss_192": [["2023-04-27", "135639"],["2023-04-27", "135644"],["2023-04-27", "135648"],["2023-04-27", "135651"]],
    # "demkov_192": [["2023-04-27", "135803"],["2023-04-27", "135807"],["2023-04-27", "135811"],["2023-04-27", "135815"]],
    # "sin_192": [["2023-04-27", "135342"]],
    # "sin2_192": [["2023-04-27", "135402"]],
    # "sin3_192": [["2023-04-27", "135404"]],
    # "sin4_192": [["2023-04-27", "135406"]],
    # "sin5_192": [["2023-04-27", "135408"]],
    
    # # "gauss": ["2023-05-20", "025844"],
    # "constant": ["2023-05-20", "104446"],
    # "rabi": ["2023-05-20", "104446"],
    # # "rz": ["2023-05-20", "025817"],
    # # "sech2": ["2023-05-20", "025829"],
    # # "demkov": ["2023-05-20", "101530"]
    #
    # "constant": ["2023-05-25", "221645"],
    # "rabi": ["2023-05-25", "221645"],
    # "rz": ["2023-05-25", "221729"],
    # "demkov": ["2023-05-25", "015554"],
    # "sech2": ["2023-05-25", "221740"],
    # "gauss": ["2023-05-25", "221752"],
    #
    # "constant": ["2023-05-25", "225504"],
    # "rabi": ["2023-05-25", "225504"],
    # "rz": ["2023-05-25", "225550"],
    # "demkov": ["2023-05-25", "225637"],
    # "sech2": ["2023-05-25", "225600"],
    # "gauss": ["2023-05-25", "225625"],
    # "lor": ["2023-05-25", "225648"],
    # "lor2": ["2023-05-25", "225524"],
    # "lor3": ["2023-05-25", "225532"],
    #
    # "constant": ["2023-05-26", "003832"],
    # "rabi": ["2023-05-26", "003832"],
    # "rz": ["2023-05-26", "003032"],
    # "demkov": ["2023-05-26", "003102"],
    # "sech2": ["2023-05-26", "003041"],
    # "gauss": ["2023-05-26", "003052"],
    # "lor": ["2023-05-26", "003113"],
    # "lor2": ["2023-05-26", "003010"],
    # "lor3": ["2023-05-26", "003023"],
    #
    "constant": ["2023-05-26", "021927"],
    "rabi": ["2023-05-26", "021927"],
    "rz": ["2023-05-26", "012657"],
    "demkov": ["2023-05-26", "012735"],
    "sech2": ["2023-05-26", "012710"],
    "gauss": ["2023-05-26", "012723"],
    "lor": ["2023-05-26", "003113"],
    "lor2": ["2023-05-26", "012617"],
    "lor3": ["2023-05-26", "012642"],
}

# durations = {
#     "gauss": 957.28,
#     "demkov": 2386.41,
#     "rz": 2652.58,
#     "sech2": 1459.18,
#     "rabi": 192,
# }
durations = {
    "gauss": 504.63,
    "demkov": 1326.29,
    "rz": 1459.37,
    "sech2": 796.18,
    "rabi": 96,
    "lor": 1910.38
    # "gauss": 957.28,
    # "demkov": 2386.41,
    # "rz": 2652.58,
    # "sech2": 1459.18,
    # "rabi": 192,
    # # "gauss": 957.28,
    # # "demkov": 2386.41,
    # # "rz": 2652.58,
    # # "sech2": 1459.18,
    # # "rabi": 192,
    # # "demkov": 3390.92,
    # # "rz": 3834.53,
}

 
# date = "2022-06-16"
area = "pi"
backend_name = "perth"
s = 96
pulse_type = "sech2"
dur = get_closest_multiple_of_16(round(durations[pulse_type]))
# pulse_type = pu
# lse_type if dur is None else "_".join([pulse_type, str(s)])
# dur_idx = durations [dur] if dur is not None else 0
date = times[pulse_type][0]
time = times[pulse_type][1]
fit_func = pulse_type
second_fit = 1 # 0 or 1, whether to have both curves
comparison = 1 # 0 or 1, whether to have a Lorentzian fit for comparison
log_plot = 0 # 0 or 1, whether to plot transition probability in a logarithmic plot
central_fraction = 1.
every_nth = 1

FIT_FUNCTIONS = {
    "lorentzian": [lorentzian],
    "constant": [rabi, None, lorentzian],
    "rabi": [rabi, None, lorentzian],
    "gauss": [gauss, gauss_rzconj, lorentzian],
    "rz": [rz, None, lorentzian], #rz,
    "sech": [rz, None, lorentzian], #rz,
    "demkov": [demkov, demkov_rzconj, lorentzian],
    "sech2": [sech_sq, None, lorentzian], #sech_sq,
    "sinc2": [sinc2],
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
for d in data_files:
    if d.startswith(times[pulse_type][1]):
        csv_file = d
        break
center_freq = center_freqs[backend_name]
# pulse_type = d.split("_")[1]
# pulse_type = "rabi" if pulse_type in ["sq", "constant"] else pulse_type
# pulse_type = "rz" if pulse_type in ["sech", "rosenzener"] else pulse_type

model_name_dict = {
    "rabi": ["Rabi", None, "Lorentzian"], 
    "rz": ["Rosen-Zener Exact Soln", None, "Lorentzian"], 
    "sech": ["Rosen-Zener Exact Soln", None, "Lorentzian"], 
    "gauss": ["Gaussian DDP", "Gaussian RZConj", "Lorentzian"], 
    "demkov": ["Demkov Exact Soln", "Demkov RZConj", "Lorentzian"], 
    "sech2": ["Sech$^2$ RZConj", None, "Lorentzian"],
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

length = len(freq)
middle_length = round(central_fraction * length)
start = (length - middle_length) // 2 
end = start + middle_length

freq = freq[start:end:every_nth]
vals = vals[start:end:every_nth]

detuning = 2 * np.pi * (freq - center_freq)
extended_freq = np.linspace(detuning[0], detuning[-1], 5000)

def fit_once(
    detuning, vals, fit_func,
    args, args_min, args_max,
    ef=None
):
    initial = [0.1, 0.5, 0.5]
    initial_min = [-5, 0, 0]
    initial_max = [5, 1, 1]
    # initial = [0.1, 0, 0]
    # initial_min = [-3, 0, 0]
    # initial_max = [3, 0.5, 0.6]
    fit_params, y_fit, err = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func][0],
        initial, initial_min, initial_max,
        # args, args_min, args_max,
        s, dur
    )
    y_fit = FIT_FUNCTIONS[fit_func][0](detuning, *fit_params)
    ##
    ##
    if second_fit:
        second_fit_params, second_y_fit, second_err = fit_function(
            detuning,
            vals, 
            FIT_FUNCTIONS[fit_func][1],
            initial, initial_min, initial_max,
            # args, args_min, args_max,
            s, dur
        )
    if comparison:
        baseline_fit_params, baseline_y_fit, baseline_err = fit_function(
            detuning,
            vals,
            FIT_FUNCTIONS[fit_func][2],
            [4, 0, 1, 0], # initial parameters for curve_fit
            [0, -10, 0, -1],
            [200, 10, 10, 2],
            # args, args_min, args_max,
            s, dur
        )
    ##
    # print(fit_params, "\n", baseline_fit_params)
    ef = extended_freq
    extended_y_fit = FIT_FUNCTIONS[fit_func][0](ef, *fit_params)
    similarity_idx = np.mean(np.abs(y_fit - vals))
    overfitting_idx = np.mean(np.abs(np.diff(extended_y_fit)))
    overfitting = overfitting_idx > 0.1


    if second_fit:
        second_extended_y_fit = FIT_FUNCTIONS[fit_func][1](ef, *second_fit_params)
        second_similarity_idx = np.mean(np.abs(second_y_fit - vals))
        second_overfitting_idx = np.mean(np.abs(np.diff(second_extended_y_fit)))
        second_overfitting = second_overfitting_idx > 0.1
        if second_overfitting:
            print("Strong overfitting present on second fit.")
            exit(1)
    if comparison:
        baseline_extended_y_fit = FIT_FUNCTIONS[fit_func][2](ef, *baseline_fit_params)
        baseline_similarity_idx = np.mean(np.abs(baseline_y_fit - vals))
        baseline_overfitting_idx = np.mean(np.abs(np.diff(baseline_extended_y_fit)))
        baseline_overfitting = baseline_overfitting_idx > 0.1
        if baseline_overfitting:
            print("Strong overfitting present on Lorentzian fit.")
            exit(1)

    if overfitting:
        print("Strong overfitting present.")
        exit(1)

    # print(overfitting_idx, baseline_overfitting_idx)
    first_return = (
        similarity_idx, 
        y_fit, 
        extended_y_fit, 
        fit_params,
        err
    )
    second_return = (                    
        second_similarity_idx, 
        second_y_fit, 
        second_extended_y_fit, 
        second_fit_params,
        second_err
    ) if second_fit else None
    baseline_return = (                    
        baseline_similarity_idx, 
        baseline_y_fit, 
        baseline_extended_y_fit, 
        baseline_fit_params,
        baseline_err
    ) if comparison else None
    return (first_return, second_return, baseline_return)

if FIT_FUNCTIONS[fit_func][1] is None:
    second_fit = 0

iargs = [0.1, 0, 0, 0.24] if dur is not None and dur < 5 * s else [0.1, 0, 0]
minargs = [-3, 0, 0, 0.04] if dur is not None and dur < 5 * s else [-3, 0, 0]
maxargs = [3, 0.5, 0.6, 0.5] if dur is not None and dur < 5 * s else [3, 0.5, 0.6]

fit, second, baseline = fit_once(
    detuning, vals, fit_func,
    args=iargs, 
    args_min=minargs,
    args_max=maxargs
)

similarity_idx, y_fit, extended_y_fit, fit_params, err = fit
if second is not None:
    second_similarity_idx, second_y_fit, second_extended_y_fit, second_fit_params, second_err = second
if baseline is not None:
    baseline_similarity_idx, baseline_y_fit, baseline_extended_y_fit, baseline_fit_params, baseline_err = baseline
print(fit_params)
# print(baseline_fit_params)
# print(err)
# print(baseline_err)
dof = len(vals) - len(fit_params)
residuals = y_fit - vals
err_res = np.sqrt(np.sum(residuals ** 2) / dof)

print(model_name_dict[fit_func][0])
print("Model MAE:", similarity_idx)
q_freq_model = fit_params[0] / (2 * np.pi)
q_freq_err_model = err[0] / (2 * np.pi)
if second_fit:
    print("Second Model MAE:", second_similarity_idx)
    q_freq_sec = second_fit_params[0] / (2 * np.pi)
    q_freq_err_sec = second_err[0] / (2 * np.pi)
if comparison:
    print("Lorentzian MAE:", baseline_similarity_idx)
    q_freq_bl = baseline_fit_params[-2] / (2 * np.pi)
    q_freq_err_bl = baseline_err[-2] / (2 * np.pi)
print(q_freq_model, "+-", q_freq_err_model)
if second_fit:
    print(q_freq_sec, "+-", q_freq_err_sec)
    print("deviation with 2nd fit:", q_freq_model - q_freq_sec, "+-", np.sqrt(q_freq_err_model ** 2 + q_freq_err_sec ** 2))
if comparison:
    print(q_freq_bl, "+-", q_freq_err_bl)
    print("deviation with Lorentzian:", q_freq_model - q_freq_bl, "+-", np.sqrt(q_freq_err_model ** 2 + q_freq_err_bl ** 2))
print(f"Error on residuals: {err_res}")
scaled_ef = extended_freq / (2 * np.pi)
scaled_det = detuning / (2 * np.pi)


date = datetime.now()
txt_name = pulse_type + "_" + str(dur) + "_" + str(comparison) + "_" + date.strftime("%Y%m%d") + "_" + date.strftime("%H%M%S") + ".txt"
with open(os.path.join(data_folder, txt_name), "w") as file:
    file.write(model_name_dict[fit_func][0] + "\n")
    file.write(f"{model_name_dict[fit_func][0]} MAE: " + str(similarity_idx) + "\n")
    if second_fit:
        file.write(f"{model_name_dict[fit_func][1]} MAE: " + str(second_similarity_idx) + "\n")
    if comparison:
        file.write(f"{model_name_dict[fit_func][2]} MAE: " + str(baseline_similarity_idx) + "\n")
    file.write(str(q_freq_model) + "+-" + str(q_freq_err_model) + "\n")
    if second_fit:
        file.write(str(q_freq_sec) + "+-" + str(q_freq_err_sec) + "\n")
        file.write("deviation with 2nd fit: " + str(q_freq_model - q_freq_sec) + "+-" + str(np.sqrt(q_freq_err_model ** 2 + q_freq_err_sec ** 2)) + "\n")
    if comparison:
        file.write(str(q_freq_bl) + "+-" + str(q_freq_err_bl) + "\n")
        file.write("deviation with Lorentzian: " + str(q_freq_model - q_freq_bl) + "+-" + str(np.sqrt(q_freq_err_model ** 2 + q_freq_err_bl ** 2)) + "\n")
    file.write(f"Error on residuals: {str(err_res)}" + "\n")


if second_fit:
    second_dof = len(vals) - len(second_fit_params)
    second_residuals = second_y_fit - vals
    second_err_res = np.sqrt(np.sum(second_residuals ** 2) / second_dof)
if comparison:
    baseline_dof = len(vals) - len(baseline_fit_params)
    baseline_residuals = baseline_y_fit - vals
    baseline_err_res = np.sqrt(np.sum(baseline_residuals ** 2) / baseline_dof)

if second_fit and comparison:
    figsize = (10,8)
elif second_fit or comparison:
    figsize = (10,7)
else:
    figsize = (10,6)
fig = plt.figure(constrained_layout=True, figsize=figsize)
gs = fig.add_gridspec(figsize[1], 1)
ax0 = fig.add_subplot(gs[:5, :])
if log_plot:
    ax0.semilogy(scaled_det, vals, color='black', marker="P", label="Measured values", linewidth=0.)
else:
    ax0.plot(scaled_det, vals, color='black', marker="P", label="Measured values", linewidth=0.)
if second_fit:
    if log_plot:
        ax0.semilogy(scaled_ef, second_extended_y_fit, color='green', label=f"{model_name_dict[fit_func][1]} model fit")
    else:
        ax0.plot(scaled_ef, second_extended_y_fit, color='green', label=f"{model_name_dict[fit_func][1]} model fit")
if comparison:
    if log_plot:
        ax0.semilogy(scaled_ef, baseline_extended_y_fit, color='blue', label=f"{model_name_dict[fit_func][2]} model fit")
    else:
        ax0.plot(scaled_ef, baseline_extended_y_fit, color='blue', label=f"{model_name_dict[fit_func][2]} model fit")
if log_plot:
    ax0.semilogy(scaled_ef, extended_y_fit, color='red', label=f"{model_name_dict[fit_func][0]} model fit")
else:
    ax0.plot(scaled_ef, extended_y_fit, color='red', label=f"{model_name_dict[fit_func][0]} model fit")
ax0.set_xlim(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10)
# Create a legend object and customize the order of the labels
handles, labels = plt.gca().get_legend_handles_labels()
if comparison:
    order = [0, 3, 1, 2] if second_fit else [0, 2, 1] 
else:
    order = [0, 2, 1] if second_fit else [0, 1] 
ax0.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

major_interval = -np.round(scaled_ef[0]/10)*10 / 4 # 42.5
minor_interval = major_interval / 5
# minor_interval = 0.5 if pulse_type=="rabi" else 2.5
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
if second_fit:
    fig.text(0.013, 0.25, 'Residuals', ha='center', va='center', rotation='vertical', fontsize=20)
else:
    fig.text(0.013, 0.222, 'Residuals', ha='center', va='center', rotation='vertical', fontsize=20)

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
if comparison or second_fit:
    ax1.set_xticklabels([])
else:
    ax1.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
ax1.set_xticks(minor_xticks, minor="True")
ax1.set_yticks(y_ticks_res)
ax1.set_yticklabels(y_ticks_res, fontsize=10)
if limit_num > 0.1:
    ax1.set_yticks(y_ticks_res_minor, minor=True)
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.6)
else:
    ax1.grid()
ax1.errorbar(scaled_det, y_fit - vals, yerr=err_res * np.ones(scaled_det.shape), fmt="+", color="r")
# print(major_xticks)
if second_fit:
    ax2 = fig.add_subplot(gs[6:7, :], sharey=ax1)
    ax2.set_xlim(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10)
    ax2.set_xticks(major_xticks)
    if comparison:
        ax2.set_xticklabels([])
    else:
        ax2.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
    ax2.set_xticks(minor_xticks, minor="True")
    ax2.set_yticklabels(y_ticks_res, fontsize=10)
    ax2.errorbar(scaled_det, second_y_fit - vals, yerr=second_err_res * np.ones(scaled_det.shape), fmt="+", color="g")
    if limit_num > 0.1:
        ax2.set_yticks(y_ticks_res_minor, minor=True)
        ax2.grid(which='minor', alpha=0.3)
        ax2.grid(which='major', alpha=0.6)
    else:
        ax2.grid()
if comparison:
    if second_fit:
        ax3 = fig.add_subplot(gs[7:, :], sharey=ax1)
    else:
        ax3 = fig.add_subplot(gs[6:, :], sharey=ax1)
    ax3.set_xlim(np.round(scaled_ef[0]/10)*10, -np.round(scaled_ef[0]/10)*10)
    ax3.set_xticks(major_xticks)
    ax3.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
    ax3.set_xticks(minor_xticks, minor="True")
    ax3.set_yticklabels(y_ticks_res, fontsize=10)
    ax3.errorbar(scaled_det, baseline_y_fit - vals, yerr=baseline_err_res * np.ones(scaled_det.shape), fmt="+", color="b")
    if limit_num > 0.1:
        ax3.set_yticks(y_ticks_res_minor, minor=True)
        ax3.grid(which='minor', alpha=0.3)
        ax3.grid(which='major', alpha=0.6)
    else:
        ax3.grid()
plt.xlabel("Detuning [MHz]", fontsize=20)

# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
save_dir = os.path.join(file_dir, "paper_ready_plots")

fig_name = pulse_type + "_" + str(dur) + "_" + str(comparison) + "_" + date.strftime("%Y%m%d") + "_" + date.strftime("%H%M%S") + ".pdf"
plt.savefig(os.path.join(save_dir, fig_name), format="pdf")
plt.show()