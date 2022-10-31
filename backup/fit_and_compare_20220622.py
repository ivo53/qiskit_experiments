import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.integrate import quad_vec

def fit_function(x_values, y_values, function, init_params, lower, higher):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params, maxfev=10000, bounds=(lower, higher))
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

def lorentzian(x, A, q_freq, B, C):
    return A / (((x - q_freq) / B)**2 + 1) + C

def inverse_lorentzian(y, A, q_freq, B, C):
    return B * np.sqrt(A / (y - C) - 1) + q_freq

def sech2(x, A, q_freq, B, C):
    x_normalised = (x - q_freq) / B
    return A * (1 / np.cosh(x_normalised)) ** 2 + C

def gauss_sech2(x, A, q_freq, T, B, C):
    x = np.abs(x - q_freq) / B
    return A * (1 / np.cosh(np.pi * T * x / np.log(np.sqrt(1 / x)))) ** 2 + C

def inverse_sech2(y, A, q_freq, B, C, tol=1e-10):
    rhs = A / (y - C)
    arccosh_before = np.log(rhs - np.sqrt(rhs ** 2 - 1))
    arccosh_after = np.log(rhs + np.sqrt(rhs ** 2 - 1))
    idx_before = rhs < 1
    idx_after = rhs >= 1
    arccosh = np.zeros(rhs.shape, dtype=np.float64)
    arccosh[idx_before] = arccosh_before[idx_before]
    arccosh[idx_after] = arccosh_after[idx_after]
    # arccosh[:int(0.5*len(arccosh))] = arccosh_before[:int(0.5*len(arccosh))]
    # arccosh[int(0.5*len(arccosh)):] = arccosh_after[int(0.5*len(arccosh)):]
    # predicted_x_vals = []
    # for n, y_n in enumerate(y):
    #     predicted_x_vals.append(
    #         root(
    #             lambda x: sech2(x, A, q_freq, B, C) - y_n, 
    #             x0=freq[n] + (1e-2 * (np.random.random() - 0.5)),
    #             tol=tol
    #         ).x
    #     )
    # predicted_x_vals = np.array(predicted_x_vals).reshape(len(y_fit))
    # return predicted_x_vals
    return B * arccosh + q_freq

def sinc2(x, A, q_freq, B, C):
    return A * (np.sinc((x - q_freq) / B)) ** 2 + C

def special_sinc(x, A, q_freq, B, C):
    return (A / (1 + ((x - q_freq) / B) ** 2)) * np.sin(0.5 * AREA_VALUES[area] * np.sqrt(1 + ((x - q_freq) / B) ** 2)) ** 2 + C

def inverse_special_sinc(y, A, q_freq, B, C, tol=1e-10):
    predicted_x_vals = []
    for n, y_n in enumerate(y):
        predicted_x_vals.append(
            root(
                lambda x: special_sinc(x, A, q_freq, B, C) - y_n, 
                x0=freq[n] + (1e-2 * (np.random.random() - 0.5)),
                tol=tol
            ).x
        )
    predicted_x_vals = np.array(predicted_x_vals).reshape(len(y_fit))
    return predicted_x_vals

def double_lorentzian(x, A, q_freq, B, C):
    return A / (((x - q_freq) / B)**4 + 1) + C

# def special_double_lorentzian(x, A, q_freq, k, B, C, a):
#     def integrand(t, x, q_freq, A, B):
#         omega = A * np.sin(np.pi * t / B)
#         omega_dot = A * np.pi * np.cos(np.pi * t / B) / B
#         return np.sqrt(omega ** 2 + (x - q_freq) ** 2 + ((x - q_freq) ** 2 * omega_dot ** 2) / (omega ** 2 + (x - q_freq) ** 2) ** 2)
#     v, e = quad_vec(lambda t: integrand(t, x, q_freq, A, B), 0, B)
#     cos_arg = np.array(v)
#     return a * (np.sin(k * np.abs(cos_arg))) ** 2 / (((x - q_freq)**4 * B ** 2) / (np.pi * A) ** 2 + 1) + C

def special_double_lorentzian(x, A, q_freq, k, B, C, D, E):
    return A * (
        np.cos(
            k * np.abs(
                np.sqrt(
                    ((x - q_freq) / B) ** 2 + E + \
                        D * (((x - q_freq) / B) ** 2) / (1 + ((x - q_freq) / B) ** 2) ** 2
                )
            )
        )
    ) ** 2 / \
        ((x - q_freq) ** 4 / B ** 4 + 1) + C

# def special_double_lorentzian(x, A, q_freq, k, B, C, D, E, p, q, s, t):
#     x = (x - q_freq) / B
#     S = np.sqrt(5 * x ** 2 + s)
#     cos_arg = D / x ** 2 + E + 1j * S * t * (np.arctan((p * x + 1j * q)/S) + np.arctan((p * x - 1j * q)/S)) / x ** 2
#     return A * (np.cos(np.abs(cos_arg))) ** 2 / (x ** 4 + 1) + C

def inverse_double_lorentzian(y, A, q_freq, B, C):
    return B * (A / (y - C) - 1) ** (1/4) + q_freq

def triple_lorentzian(x, A, q_freq, B, C):
    return A / (((x - q_freq) / B) ** 6 + 1) + C

def inverse_triple_lorentzian(y, A, q_freq, B, C):
    return B * (A / (y - C) - 1) ** (1/6) + q_freq

def quadruple_lorentzian(x, A, q_freq, B, C):
    return A / (((x - q_freq) / B)**8 + 1) + C

def inverse_quadruple_lorentzian(y, A, q_freq, B, C):
    return B * (A / (y - C) - 1) ** (1/8) + q_freq

FIT_FUNCTIONS = {
    "lorentzian": lorentzian,
    "sech2": sech2,
    "gauss_sech2": gauss_sech2,
    "sinc2": sinc2,
    "special_sinc": special_sinc,
    "double_lorentzian": double_lorentzian,
    "special_double_lorentzian": special_double_lorentzian,
    "triple_lorentzian": triple_lorentzian,
    "quadruple_lorentzian": quadruple_lorentzian    
}
INVERSE_FIT_FUNCTIONS = {
    "lorentzian": inverse_lorentzian,
    "sech2": inverse_sech2,
    # "sinc2": inverse_sinc2,
    "special_sinc": inverse_special_sinc,
    "double_lorentzian": inverse_double_lorentzian,
    "triple_lorentzian": inverse_triple_lorentzian,
    "quadruple_lorentzian": inverse_quadruple_lorentzian    
}
AREA_VALUES = {
    "half": 0.5 * np.pi,
    "pi": np.pi,
    "3pi": 3 * np.pi,
    "5pi": 5 * np.pi,
    "7pi": 7 * np.pi,
}

date = "2022-06-16"
times = {
    "gauss": "174431",
    "sq": "174532",
    "sine": "174541",
    "sine2": "174543",
    "sine3": "174546",
    "sech": "174403",
    "sech2": "174406",
    "demkov": "174352"
}
area = "pi"

pulse_type = "demkov"
fit_func = "sech2"
compare_fit_func = "lorentzian"

file_dir = os.path.dirname(__file__)
data_folder = os.path.join(file_dir, "data", "armonk", "calibration", date)
data_files = os.listdir(data_folder)
center_freq = 4.97169 * 1.e3 
for d in data_files:
    if d.startswith(times[pulse_type]):
        csv_file = d
        break

pulse_type = d.split("_")[1]

df = pd.read_csv(os.path.join(data_folder, csv_file))

freq = df["frequency_ghz"].to_numpy() * 1e3
vals = df["transition_probability"].to_numpy()
detuning = freq - center_freq

## rectangular Rabi fit
fit_params, y_fit = fit_function(
    detuning,
    vals, 
    FIT_FUNCTIONS[fit_func],
    [1, 0, 15, 0], # initial parameters for curve_fit
    [0, -10,-0.05, 0], # upper bounds on parameters
    [1e8, 10, 100, 1] # lower bounds on parameters
    # [1, 4.97, 5e-5, 0.01, 0, .01, 1, .1, 100, 5, 1], # initial parameters for curve_fit
    # [0, 4.9, 0, 0, -0.05, 0, 0, 0, 0, 0, 0], # upper bounds on parameters
    # [1e8, 5.05, 100, 100, 1, 100, 100, 100, 100, 100, 100] # lower bounds on parameters
    # [1, 0, 1, 1, 0, 1, 1], # initial parameters for curve_fit
    # [0, -10, 0, 0, -0.05, 0, 0], # upper bounds on parameters
    # [1e8, 10, 1e8, 100, 1, 1e6, 1e6] # lower bounds on parameters
)
##
## sech^2 fit
compare_fit_params, compare_y_fit = fit_function(
    detuning,
    vals, 
    FIT_FUNCTIONS[compare_fit_func],
    [1, 0, 10, 0], # initial parameters for curve_fit
    [0, -10, 0, 0],
    [10, 10, 100, 1]
)
##
print(fit_params, compare_fit_params)
extended_freq = np.linspace(detuning[0], detuning[-1], 5000)
extended_y_fit = FIT_FUNCTIONS[fit_func](extended_freq, *fit_params)
compare_extended_y_fit = FIT_FUNCTIONS[compare_fit_func](extended_freq, *compare_fit_params)

fig, ax = plt.subplots(3, 1, sharex=True, sharey=False, gridspec_kw={'height_ratios': [5, 1, 1]}, figsize=(10,7))
ax[0].scatter(detuning, vals, color='black')
ax[0].plot(extended_freq, extended_y_fit, color='red')
ax[0].plot(extended_freq, compare_extended_y_fit, color='blue')
ax[0].set_xlim([min(detuning), max(detuning)])
ax[0].set_title(f"Fitted {pulse_type.capitalize()} Frequency Profile (Area {area})")
plt.xlabel("Detuning [MHz]")
ax[0].set_ylabel("Transition Probability")

# freq_major_xticks = np.arange()

ax[0].grid()

ax[1].scatter(detuning, y_fit - vals, color="red", marker="*")
ax[1].grid()
ax[2].grid()
ax[2].scatter(detuning, compare_y_fit - vals, color="blue", marker="*")

# fig2_name = date.strftime("%H%M%S") + f"_{pulse_type}_area_{area}_frequency_sweep_fitted.png" if pulse_type not in lorentz_pulses else \
#     date.strftime("%H%M%S") + f"_{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}_area_{area}_frequency_sweep_fitted.png"
# plt.savefig(os.path.join(save_dir, fig2_name))
plt.show()


exit()
predicted_x_vals = INVERSE_FIT_FUNCTIONS[fit_func](vals, *fit_params, tol=1e-9)
fig2, ax2 = plt.subplots()
ax2.scatter(freq, predicted_x_vals - freq, marker="x", color='red')
ax2.set_xlim([min(freq), max(freq)])
ax2.set_ylim([min(predicted_x_vals - freq), max(predicted_x_vals - freq)])
ax2.set_title(f"Fitted {pulse_type.capitalize()} Frequency Curve (Area {area})")
ax2.set_xlabel("Frequency [GHz]")
ax2.set_ylabel("Transition Probability")
ax2.grid()
plt.show()

compare_predicted_x_vals = INVERSE_FIT_FUNCTIONS[compare_fit_func](vals, *compare_fit_params, tol=1e-9)
fig3, ax3 = plt.subplots()
ax3.scatter(freq, compare_predicted_x_vals - freq, marker="x", color='blue')
ax3.set_xlim([min(freq), max(freq)])
# ax3.set_ylim([min(compare_predicted_x_vals - freq), max(compare_predicted_x_vals - freq)])
ax3.set_title(f"Fitted {pulse_type.capitalize()} Frequency Curve (Area {area})")
ax3.set_xlabel("Frequency [GHz]")
ax3.set_ylabel("Transition Probability")
ax3.grid()
plt.show()