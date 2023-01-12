import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.integrate import quad, quad_vec
from scipy.misc import derivative
from scipy.special import ellipk
import mpmath as mp

AREA_VALUES = {
    "half": 0.5 * np.pi,
    "pi": np.pi,
    "3pi": 3 * np.pi,
    "5pi": 5 * np.pi,
    "7pi": 7 * np.pi,
}

RABI_FREQ = {
    "constant": 6266474.70796,
    "rabi": 6266474.70796,
    "rz": 42874911.4203,
    "gauss": 25179780.7441,
    "demkov": 31739880.846,
    "sech2": 35460561.388
}

Tt = {
    "constant": 1504e-9 / 3,
    "rabi": 1504e-9 / 3,
    "rz": (284 + 4/9) * 1e-9,
    "gauss": (398 + 2/9) * 1e-9,
    "demkov": (572 + 4/9) * 1e-9,
    "sech2": (284 + 4/9) * 1e-9
}

SIGMA = {
    "rz": 23.39181 * 1e-9,
    "gauss": (49 + 7/9) * 1e-9,
    "demkov": (55 + 5/9) * 1e-9,
    "sech2": (44 + 4/9) * 1e-9
}


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
pulse_type = "gauss"
fit_func = pulse_type
baseline_fit_func = "sinc2" if pulse_type in ["rabi", "constant"] else "lorentzian"

def fit_function(x_values, y_values, function, init_params, lower, higher):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params, maxfev=1e6, bounds=(lower, higher))
    y_fit = function(x_values, *fitparams)
    perr = np.sqrt(np.diag(conv))
    return fitparams, y_fit, perr

def post_process(P2, eps, delta):
    # eps = eps0 + 1/2 * (1 - eps1)
    # delta = 1/2 * (1 - eps1) * e^(-gamma * t)
    return eps + delta * (2 * P2 - 1)

def with_dephasing(P2, egamma):
    return P2 * egamma - egamma / 2 + 1 / 2

def lorentzian(x, s, A, q_freq, c):
    return A / (((x - q_freq) / s) ** 2 + 1) + c

def rz(x, q_freq, delta, eps):
    T = Tt["rz"]
    O = RABI_FREQ["rz"]
    sigma = SIGMA["rz"]
    D = (x - q_freq) * 1e6
    P2 = np.sin(0.5 * np.pi * O * sigma) ** 2 \
        / np.cosh(0.5 * np.pi * D * sigma) ** 2
    return post_process(P2, eps, delta)

def demkov(x, q_freq, delta, eps):
    T = Tt["demkov"]
    sigma = SIGMA["demkov"]
    omega_0 = RABI_FREQ["demkov"]
    alpha = 0.5 * (x - q_freq) * 1e6 * sigma 
    if not (isinstance(alpha, np.ndarray) or isinstance(alpha, list)):
        alpha = [alpha]
    s_inf = np.pi * omega_0 * sigma
    bessel1 = np.array([complex(mp.besselj(1/2 + 1j * a, s_inf)) for a in alpha])
    bessel2 = np.array([complex(mp.besselj(-1/2 - 1j * a, s_inf)) for a in alpha])
    bessel3 = np.array([complex(mp.besselj(1/2 - 1j * a, s_inf)) for a in alpha])
    bessel4 = np.array([complex(mp.besselj(-1/2 + 1j * a, s_inf)) for a in alpha])
    if len(alpha) == 1:
        alpha = alpha[0]
    # P2 = (np.pi * s_inf / 2) ** 2 * np.abs(bessel1 * bessel2 \
    #      + bessel3 * bessel4) ** 2 / np.cosh(alpha * np.pi) ** 2
    al = (x - q_freq) * 1e6 * sigma
    bessel11 = np.array([complex(mp.besselj(1/2 + 1j * a / 2, s_inf / (2 * np.pi))) for a in al])
    bessel21 = np.array([complex(mp.besselj(-1/2 - 1j * a / 2, s_inf / (2 * np.pi))) for a in al])
    P2 = (s_inf / 4) ** 2 * np.abs(2 * np.real(bessel11 * bessel21)) ** 2 / np.cosh(al * np.pi / 2) ** 2
    return post_process(P2, eps, delta)

def sech_sq(x, q_freq, delta, eps, alpha):
    T = Tt["sech2"]
    sigma = SIGMA["sech2"]
    omega_0 = RABI_FREQ["sech2"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / np.cosh((t) / sigma) ** 2 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-4, 1e-4, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-4, 1e-4, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def gauss_sech2(x, q_freq, delta, eps):
    O = RABI_FREQ["gauss"]
    T = Tt["gauss"]
    sigma = SIGMA["gauss"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return O * mp.exp(-0.5 * ((t - T/2) / sigma) ** 2)
    S = np.float64(mp.quad(f_, [0, T]))
    numerator = np.sin(0.5 * np.sqrt(np.pi) * O * sigma) ** 2
    denomenator = np.cosh(np.pi * D * sigma / (4 * np.sqrt(np.log(O) - np.log(np.abs(D))))) ** 2
    P2 = numerator / denomenator
    return post_process(P2, eps, delta)

def gauss(x, q_freq, delta, eps):
    O = RABI_FREQ["gauss"]
    T = Tt["gauss"]
    sigma = SIGMA["gauss"]
    D = (x - q_freq) * 1e6
    alpha = np.abs(O / D)
    alpha[np.isnan(alpha)] = 10000000
    # print(alpha)
    m, mu, nu = (1.311468, 0.316193, 0.462350)

    ImD = D * sigma * np.sqrt(
        np.sqrt(
            4 * np.log(m * alpha) ** 2 + np.pi ** 2
        ) \
            - 2 * np.log(m * alpha)
    ) / np.sqrt(2)

    ReD = np.sqrt(2) * D * sigma * (
        (np.sqrt(alpha ** 2 + 1) - 1) * \
            np.sqrt(
                0.5 * np.log(
                    alpha ** 2 / ((1 + nu * (np.sqrt(alpha ** 2 + 1) - 1)) ** 2 - 1)
                )
            ) + 0.5 * np.sqrt(
                np.sqrt(
                    np.log(alpha ** 2 / (mu * (2 - mu))) ** 2 + np.pi ** 2
                ) + np.log(alpha ** 2 / (mu * (2 - mu)))
            )
    )
    P2 = np.sin(ReD) ** 2 / np.cosh(ImD) ** 2
    return post_process(P2, eps, delta)

def sinc2(x, s, A, q_freq, c):
    return A * (np.sinc((x - q_freq) / s)) ** 2 + c

def rabi(x, q_freq, delta, eps):
    T = Tt["rabi"]
    O = RABI_FREQ["rabi"]
    T = np.pi/O
    D = (x - q_freq) * 1e6
    P2 = (O ** 2 / (O**2 + (D) ** 2)) * \
        np.sin(0.5 * T * np.sqrt(O**2 + (D) ** 2)) ** 2
    return post_process(P2, eps, delta)

FIT_FUNCTIONS = {
    "lorentzian": lorentzian,
    "constant": rabi,
    "rabi": rabi,
    "gauss": gauss,
    "rz": rz,
    "demkov": demkov,
    "sech2": sech_sq,
    "sinc2": sinc2
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
df = pd.read_csv(os.path.join(data_folder, csv_file))

freq = df["frequency_ghz"].to_numpy() * 1e3
vals = df["transition_probability"].to_numpy()
detuning = 2 * np.pi * (freq - center_freq)
extended_freq = np.linspace(detuning[0], detuning[-1], 5000)

def fit_once(
    detuning, vals, fit_func,
    args, args_min, args_max
):
    initial = [0, 0.4, 0.4, 0.4] if fit_func in ["sech2"] else [0.1, 0, 0]
    initial_min = [-3, 0.3, 0.3, 0] if fit_func in ["sech2"] else [-3, 0, 0]
    initial_max = [3, 0.5, 0.6, 1] if fit_func in ["sech2"] else [3, 0.5, 0.6]
    fit_params, y_fit, err = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func],
        initial, initial_min, initial_max
    )
    y_fit = FIT_FUNCTIONS[fit_func](detuning, *fit_params)
    ##
    ## sech^2 fit
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
    extended_y_fit = FIT_FUNCTIONS[fit_func](extended_freq, *fit_params)
    baseline_extended_y_fit = FIT_FUNCTIONS[baseline_fit_func](extended_freq, *baseline_fit_params)

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
print(fit_params)
print(baseline_fit_params)
print(err)
print(baseline_err)
# for o in np.linspace(5e5, 1e9, 500):
#     fit_, baseline_ = fit_once(
#         detuning, vals, fit_func, 0, 1, o,
#         q_freq_min=-3, egamma_min=.9, O_min=1e4,
#         q_freq_max=3, egamma_max=1., O_max=1e8)
#     similarity_idx_, y, prms = fit_
#     print("o =", o, "sim =", similarity_idx_, "rabi =", prms[-1])

fig = plt.figure(constrained_layout=True, figsize=(10,7))
gs = fig.add_gridspec(7, 1)
ax0 = fig.add_subplot(gs[:5, :])
ax0.scatter(detuning, vals, color='black', marker="P")
ax0.plot(extended_freq, baseline_extended_y_fit, color='blue')
ax0.plot(extended_freq, extended_y_fit, color='red')
ax0.set_xlim(extended_freq[0], -extended_freq[0])

# major_xticks = np.round(np.arange(extended_freq[0], -extended_freq[0] + 1e-1, 5*2*np.pi),1)
x_limit = np.floor(np.abs(extended_freq[0]) / 5) * 5
x_interval = np.round(x_limit / 5) if pulse_type == "rabi" else np.round(x_limit / 6)
x_small_interval = np.round(x_interval / 3) if pulse_type == "rabi" else np.round(x_limit / 30)
# print(x_limit)
major_xticks = np.round(np.arange(-x_limit, x_limit + 1e-3, x_interval),0)
major_xticks[major_xticks>-0.01] = np.abs(major_xticks[major_xticks>-0.01])
# minor_xticks = np.round(np.arange(extended_freq[0], -extended_freq[0] + 1e-1, 2*np.pi),1)
minor_xticks = np.round(np.arange(-x_limit, x_limit + 1e-3, x_small_interval),0)
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

ax0.set_title(f"{pulse_type.capitalize()} Frequency Curve \
- SI = {np.round(similarity_idx, 2)} vs BSI = \
{np.round(baseline_similarity_idx, 2)}", fontsize=22)
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
ax1.set_xlim(extended_freq[0], -extended_freq[0])

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
ax1.errorbar(detuning, y_fit - vals, yerr=err * np.ones(detuning.shape), fmt="+", color="r")
# print(major_xticks)
ax2 = fig.add_subplot(gs[6:, :], sharey=ax1)
ax2.set_xlim(extended_freq[0], -extended_freq[0])
ax2.set_xticks(major_xticks)
ax2.set_xticklabels(np.round(major_xticks, 1), fontsize=16)
ax2.set_xticks(minor_xticks, minor="True")
ax2.set_yticklabels(y_ticks_res, fontsize=13)
ax2.errorbar(detuning, baseline_y_fit - vals, yerr=err * np.ones(detuning.shape), fmt="+", color="b")
if limit_num > 0.1:
    ax2.set_yticks(y_ticks_res_minor, minor=True)
    ax2.grid(which='minor', alpha=0.3)
    ax2.grid(which='major', alpha=0.6)
else:
    ax2.grid()
plt.xlabel("Detuning [$\\times 10^6$ rad/s]", fontsize=20)

# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
# plt.savefig(os.path.join(save_dir, fig2_name))
plt.show()