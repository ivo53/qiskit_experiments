import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import scipy.special as sp
from scipy.optimize import curve_fit

import pulse_shapes
from transition_line_profile_functions import rlzsm_approx, double_approx, fit_function

def double_approx(x, q_freq, delta, eps, tau, pulse_type, det, dur, s, rb, dt=5e-10):
    T = dur * dt
    sigma = s * dt
    # omega_0 = pulse_shapes.find_rabi_amp(pulse_type, T, sigma, rb=rb)
    sigma /= T
    # omega_0 *= T
    OMEGA_0 = x * 1e6 * T
    # print(x)
    D = (det - q_freq) * 1e6 * T
    omega = lambda t: pulse_shapes.rabi_freq(t, OMEGA_0, pulse_type, 1, sigma, rb=rb)
    # print(omega(tau))
    beta = np.sqrt(np.pi * np.maximum(omega(tau), 1e-10))
    d = (D / (2 * beta))
    eta = np.abs(D) * sp.ellipeinc(np.pi, -OMEGA_0**2 / D**2) / np.pi
    chi1 = d**2 / 2 + np.angle(sp.gamma(1/2 * (1 + 1j * d**2))) \
        - d**2 / 2 * np.log(d**2 / 2)
    chi2 = -np.pi / 4 - d**2 / 2 - np.angle(sp.gamma(1j * d**2 / 2)) \
        + d**2 / 2 * np.log(d**2 / 2)
    P2 = 1 / 4 * ((1 + np.exp(-np.pi * d**2)) * np.sin(eta / 2 - 2 * chi1)\
        - (1 - np.exp(-np.pi * d**2)) * np.sin(eta / 2 + 2 * chi2)) ** 2
    return eps + delta * (2 * P2 - 1)

def sin_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sin", det_value, dur, s, 0, dt=5e-10)

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "kyoto"
save_fig = 1
dur = 192
s = 192
data = pd.read_csv("C:/Users/Ivo/Documents/Wolfram Mathematica/sine_46mhz.csv", header=None).to_numpy()

times = {
    "sq": ["2023-12-08", "175156"],
    "sin": ["2023-12-08", "130503"],
}
full_pulse_type = ["Rect.", "Sine"]
## create folder where plots are saved
file_dir = os.path.dirname(__file__)

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

l, p, x0 = 1.25116799e+02, 3.01445760e-01, 8.23116906e-03
T1 = 192 * 5e-10

tr_prob, amp, det = [], [], []
amp_au = []
for k, t in times.items():
    files = os.listdir(data_folder(t[0], t[1], k))
    with open(os.path.join(data_folder(t[0], t[1], k), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob.append(pickle.load(f1))
    with open(os.path.join(data_folder(t[0], t[1], k), files[0]), 'rb') as f2:
        amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * (2*T1/np.pi)))
        # amp_au.append(pickle.load(f2))
    with open(os.path.join(data_folder(t[0], t[1], k), files[1]), 'rb') as f3:
        det.append(pickle.load(f3) * 2 * np.pi / 1e6)

# #############
# # start of fit
# def fit_function(x_values, y_values, function, init_params):
#     fitparams, conv = curve_fit(
#         function, 
#         x_values, 
#         y_values, 
#         init_params, 
#         maxfev=100000, 
#         bounds=(
#             [-0.51, 0, 0, -0.01, 0.45], 
#             [-.45, 1e6, 1e3, 0.01, 0.55]
#         )
#     )
#     y_fit = function(x_values, *fitparams)
    
#     return fitparams, y_fit

# fit_crop_parameter = int(1 * len(amp_au[1]))
# rabi_fit_params, _ = fit_function(
#     amp_au[1][: fit_crop_parameter],
#     tr_prob[1][:, int(len(tr_prob[1])/2)][: fit_crop_parameter], 
#     lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * np.abs(x - x0)**0.9)))) + B,
#     [-0.48273362, 55555, 5.72870197e-04, 0.005, 0.47747625]
#     # lambda x, A, k, B: A * (np.cos(k * x)) + B,
#     # [-0.5, 50, 0.5]
# )
# new_amp = np.linspace(amp_au[1][0], amp_au[1][-1], 5000)
# print(rabi_fit_params)
# A, l, p, x0, B = rabi_fit_params
# pi_amp = x0 - np.log(1 - np.pi / l) / p #((l - np.sqrt(l ** 2 + 4 * k * np.pi)) / (2 * k)) ** 2 #np.pi / (k)
# half_amp =  x0 - np.log(1 - np.pi / (2 * l)) / p #((l - np.sqrt(l ** 2 + 2 * k * np.pi)) / (2 * k)) ** 2 #np.pi / (k)
# fig, ax = plt.subplots(1, 1, figsize=(8,6))
# ax.scatter(amp_au[1][: fit_crop_parameter], tr_prob[1][:, int(len(tr_prob[1])/2)][: fit_crop_parameter])
# ax.plot(new_amp, A * (np.cos(l * (1 - np.exp(- p * np.abs(new_amp - x0)**0.9)))) + B)
# plt.show()
# exit()
# ###############

det_value = -46.16 #MHz
d = det[1]
det_idx = int(0.5 * len(d) * (1 + det_value / d[-1]))
tr = tr_prob[1][:, det_idx]
print("Detuning:", d[det_idx])
am = amp[1][1:]
init = [0,0.5,0.5,0.3]
lower = [-0.5,0.4,0.4,0]
higher = [0.5,0.6,0.6,0.5]
a = np.linspace(am[0], am[-1], 5000)
# print(len(am), len(tr))
fitparams, conv = curve_fit(
    sin_dappr,
    am, tr[1:], init, 
    maxfev=1e6, 
    bounds=(lower, higher)
)
y_fit = sin_dappr(a, *fitparams)
print(y_fit[0])
perr = np.sqrt(np.diag(conv))

intervals_amp = [100, 100]
intervals_det = [100, 100]
# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(8,6), layout="constrained")
gs = fig.add_gridspec(1, 1)

ax = fig.add_subplot(gs[0, 0])
# Plot the color map
ax.scatter(am, tr[1:], marker="$\\times$", s=100)
ax.plot(a, y_fit,"r--")
ax.scatter(data[:, 0]/1e6, data[:, 1], c="green", marker=",", s=1)
# # # Add a rectangle in the top right corner
# rect = Rectangle((0.8, 0.875), 0.16, 0.1, transform=ax.transAxes,
#                 color='#DEDA8D', alpha=0.7)
# ax.add_patch(rect)
# text = full_pulse_type[idx]
# # # Add text inside the rectangle
# # text = str(list(times.keys())[::-1][3*i+j]) + "%"
# # if (i==2 and j!=2) or (i==1 and j!=0):
# #     text = " " + text 
# ax.text(0.83 - (1 - idx) * 0.01, 0.91, text, transform=ax.transAxes,
#     color='black', fontsize=18)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.1), minor="True")
ax.set_yticklabels(np.arange(0, 1.01, 0.2).round(1), fontsize=18)
ax.set_xticks(np.arange(0, 501, intervals_amp[1]))
ax.set_xticks(np.arange(0, 501, intervals_amp[1] / 2), minor="True")
ax.set_xticklabels(np.arange(0, 501, intervals_amp[1]), fontsize=18)
ax.set_xlabel('Peak Amplitude (MHz)', fontsize=18)
ax.set_ylabel('Transition Probability', fontsize=18)
ax.grid(which='major', alpha=0.6)
ax.grid(which='minor', alpha=0.2)


# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "finite_pulses")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"sine_offres_rabi_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"


if save_fig:
    # Save the fig
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
# plt.show()
