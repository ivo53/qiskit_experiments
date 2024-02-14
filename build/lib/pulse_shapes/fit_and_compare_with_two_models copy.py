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



times = {
    "gauss": "174431",
    "constant": "174532",
    "rabi": "174532",
    "sine": "174541",
    "sine2": "174543",
    "sine3": "174546",
    "rz": "174403",
    "sech2": "174406",
    "demkov": "174352"
}

date = "2022-06-16"
area = "pi"
pulse_type = "sine"
pulse_type1 = pulse_type + "1"
pulse_type2 = pulse_type + "2"
fit_func = pulse_type
baseline_fit_func = "sinc2" if pulse_type in ["rabi", "constant"] else "lorentzian"


FIT_FUNCTIONS = {
    "lorentzian": lorentzian,
    "constant": rabi,
    "rabi": rabi,
    "gauss": gauss_rlzsm, #gauss_rzconj,
    "rz": sech_rlzsm, #rz,
    "sech": sech_rlzsm, #rz,
    "demkov": demkov,
    "sech2": sech2_rlzsm, #sech_sq,
    "sinc2": sinc2,
    "sin": sin_rlzsm,
    "sin2": sin2,
    "sin3": sin3,
    "sin4": sin4,
    "sin5": sin4,
    "lor": lor_rlzsm,
    "lor2": lor2_rlzsm,
    "lor3": lor3_rlzsm,
}

file_dir = os.path.dirname(__file__)
data_folder = os.path.join(file_dir, "data", "armonk", "calibration", date)
data_files = os.listdir(data_folder)
center_freq = 4.97169 * 1.e3
for d in data_files:
    if d.startswith(times[pulse_type]):
        csv_file = d
        break

pulse_type = d.split("_")[1]
pulse_type = "rabi" if pulse_type in ["sq", "constant"] else pulse_type
pulse_type = "rz" if pulse_type in ["sech", "rosenzener"] else pulse_type

model_name_dict = {
    "gauss1": ["Gaussian DDP", "Lorentzian"], 
    "gauss2": ["Gaussian RZ Conjecture", "Lorentzian"], 
    "demkov1": ["Demkov Bessel Functions", "Lorentzian"], 
    "demkov2": ["Demkov RZ Conjecture", "Lorentzian"], 
    "sine1": ["Sine RZ Conjecture", "Lorentzian"],
    "sine2": ["Sine rLZSM", "Lorentzian"],
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
    fit_func1 = fit_func + "1"
    fit_func2 = fit_func + "2"
    initial = [0.1, 0, 0]
    initial_min = [-3, 0, 0]
    initial_max = [3, 0.5, 0.6]
    fit_params1, y_fit1, err1 = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func1],
        initial, initial_min, initial_max
    )
    y_fit1 = FIT_FUNCTIONS[fit_func1](detuning, *fit_params1)

    fit_params2, y_fit2, err2 = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func2],
        initial, initial_min, initial_max
    )
    y_fit2 = FIT_FUNCTIONS[fit_func2](detuning, *fit_params2)
    ##
    ##
    baseline_fit_params, baseline_y_fit, baseline_err = fit_function(
        detuning,
        vals, 
        FIT_FUNCTIONS[baseline_fit_func],
        [2, 0, 1, 0], # initial parameters for curve_fit
        [0, -10, 0, 0],
        [100, 10, 100, 0.5]

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
    extended_y_fit1 = FIT_FUNCTIONS[fit_func1](ef, *fit_params1)
    extended_y_fit2 = FIT_FUNCTIONS[fit_func2](ef, *fit_params2)
    baseline_extended_y_fit = FIT_FUNCTIONS[baseline_fit_func](ef, *baseline_fit_params)

    similarity_idx1 = np.sum(np.abs(y_fit1 - vals))
    similarity_idx2 = np.sum(np.abs(y_fit2 - vals))
    baseline_similarity_idx = np.sum(np.abs(baseline_y_fit - vals))

    overfitting_idx1 = np.mean(np.abs(np.diff(extended_y_fit1)))
    overfitting_idx2 = np.mean(np.abs(np.diff(extended_y_fit2)))
    baseline_overfitting_idx = np.mean(np.abs(np.diff(baseline_extended_y_fit)))
    overfitting1 = overfitting_idx1 > 0.1
    overfitting2 = overfitting_idx2 > 0.1
    baseline_overfitting = baseline_overfitting_idx > 0.1
    # print(overfitting_idx, baseline_overfitting_idx)
    if overfitting1 or overfitting2 or baseline_overfitting:
        print("Strong overfitting present.")
        exit(1)
    return (similarity_idx1, 
            y_fit1, 
            extended_y_fit1, 
            fit_params1,
            err1), \
            (similarity_idx2, 
            y_fit2, 
            extended_y_fit2, 
            fit_params2,
            err2), \
           (baseline_similarity_idx, 
            baseline_y_fit, 
            baseline_extended_y_fit, 
            baseline_fit_params,
            baseline_err)

fit1, fit2, baseline = fit_once(
    detuning, vals, fit_func,
    args=[0, 1, 1e6], 
    args_min=[-3, .99, 1e4],
    args_max=[3, 1., 1e8]
)

similarity_idx1, y_fit1, extended_y_fit1, fit_params1, err1 = fit1
similarity_idx2, y_fit2, extended_y_fit2, fit_params2, err2 = fit2
baseline_similarity_idx, baseline_y_fit, \
    baseline_extended_y_fit, baseline_fit_params, baseline_err = baseline

print(model_name_dict[pulse_type1][0])
print(model_name_dict[pulse_type2][0])
print("Model 1 SI:", similarity_idx1)
print("Model 2 SI:", similarity_idx2)
print("Baseline SI:", baseline_similarity_idx)
q_freq_model1 = fit_params1[0] / (2 * np.pi)
q_freq_model2 = fit_params2[0] / (2 * np.pi)
q_freq_bl = baseline_fit_params[-2] / (2 * np.pi)
q_freq_err_model1 = err1[0] / (2 * np.pi)
q_freq_err_model2 = err2[0] / (2 * np.pi)
q_freq_err_bl = baseline_err[-2] / (2 * np.pi)
print(q_freq_model1, "+-", q_freq_err_model1)
print(q_freq_model2, "+-", q_freq_err_model2)
print(q_freq_bl, "+-", q_freq_err_bl)
print("deviation1:", q_freq_model1 - q_freq_bl, "+-", np.sqrt(q_freq_err_model1 ** 2 + q_freq_err_bl ** 2))
print("deviation2:", q_freq_model2 - q_freq_bl, "+-", np.sqrt(q_freq_err_model2 ** 2 + q_freq_err_bl ** 2))

scaled_ef = extended_freq / (2 * np.pi)
scaled_det = detuning / (2 * np.pi)

fig = plt.figure(constrained_layout=True, figsize=(10,7))
gs = fig.add_gridspec(8, 1)
ax0 = fig.add_subplot(gs[:5, :])
ax0.scatter(scaled_det, vals, color='black', marker="P")
ax0.plot(scaled_ef, baseline_extended_y_fit, color='blue')
ax0.plot(scaled_ef, extended_y_fit1, color='red')
ax0.plot(scaled_ef, extended_y_fit2, color='green')
ax0.set_xlim(scaled_ef[0], -scaled_ef[0])
ax0.legend([
    "Measured values", 
    f"{model_name_dict[pulse_type1][1]} fit", 
    f"{model_name_dict[pulse_type1][0]} model analytical fit",
    f"{model_name_dict[pulse_type2][0]} model analytical fit"
])

major_interval = 2.5 if pulse_type=="rabi" else 5.
minor_interval = 0.5 if pulse_type=="rabi" else 1.
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
ax1.errorbar(scaled_det, y_fit1 - vals, yerr=err * np.ones(scaled_det.shape), fmt="+", color="r")
# print(major_xticks)
ax2 = fig.add_subplot(gs[6:7, :], sharey=ax1)
ax2.set_xlim(scaled_ef[0], -scaled_ef[0])
ax2.set_xticks(major_xticks)
ax2.set_xticklabels([])
ax2.set_xticks(minor_xticks, minor="True")
ax2.set_yticklabels(y_ticks_res, fontsize=13)
ax2.errorbar(scaled_det, y_fit2 - vals, yerr=err * np.ones(scaled_det.shape), fmt="+", color="g")
if limit_num > 0.1:
    ax2.set_yticks(y_ticks_res_minor, minor=True)
    ax2.grid(which='minor', alpha=0.3)
    ax2.grid(which='major', alpha=0.6)
else:
    ax2.grid()
ax3 = fig.add_subplot(gs[7:, :], sharey=ax1)
ax3.set_xlim(scaled_ef[0], -scaled_ef[0])
ax3.set_xticks(major_xticks)
ax3.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
ax3.set_xticks(minor_xticks, minor="True")
ax3.set_yticklabels(y_ticks_res, fontsize=13)
ax3.errorbar(scaled_det, baseline_y_fit - vals, yerr=err * np.ones(scaled_det.shape), fmt="+", color="b")
if limit_num > 0.1:
    ax3.set_yticks(y_ticks_res_minor, minor=True)
    ax3.grid(which='minor', alpha=0.3)
    ax3.grid(which='major', alpha=0.6)
else:
    ax3.grid()
plt.xlabel("Detuning [MHz]", fontsize=20)

# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
save_dir = "C:/Users/Ivo/Documents/PhD Documents/Sine Pulses"
date = datetime.now()
fig_name = pulse_type + "_" + date.strftime("%Y%m%d") + ".pdf"
plt.savefig(os.path.join(save_dir, fig_name), format="pdf")
plt.show()