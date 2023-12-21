import os
import pickle
from datetime import datetime
import numpy as np
# import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from transition_line_profile_functions import rlzsm_approx, double_approx

def double_approx(x, q_freq, delta, eps, tau, pulse_type):
    T = dur * 2e-9 / 9
    sigma = s * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp(pulse_type, T, sigma, rb=rb)
    sigma /= T
    omega_0 *= T
    D = (x - q_freq) * 1e6 * T
    omega = lambda t: pulse_shapes.rabi_freq(t, omega_0, pulse_type, 1, sigma, rb=rb)
    beta = np.sqrt(np.pi * omega(tau))
    d = (D / (2 * beta))
    eta = np.abs(D) * sp.ellipeinc(np.pi, -omega_0**2 / D**2) / np.pi
    chi1 = d**2 / 2 + np.angle(sp.gamma(1/2 * (1 + 1j * d**2))) \
        - d**2 / 2 * np.log(d**2 / 2)
    chi2 = -np.pi / 4 - d**2 / 2 - np.angle(sp.gamma(1j * d**2 / 2)) \
        + d**2 / 2 * np.log(d**2 / 2)
    P2 = 1 / 4 * ((1 + np.exp(-np.pi * d**2)) * np.sin(eta / 2 - 2 * chi1)\
        - (1 - np.exp(-np.pi * d**2)) * np.sin(eta / 2 + 2 * chi2)) ** 2
    return post_process(P2, eps, delta)

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "kyoto"
save_fig = 0

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

l, p, x0 = 419.1631352890144, 0.0957564968883284, 0.0003302995697281
T = 192 * 2/9 * 1e-9

tr_prob, amp, det = [], [], []
for k, t in times.items():
    files = os.listdir(data_folder(t[0], t[1], k))
    with open(os.path.join(data_folder(t[0], t[1], k), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob.append(pickle.load(f1))
    with open(os.path.join(data_folder(t[0], t[1], k), files[0]), 'rb') as f2:
        amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
    with open(os.path.join(data_folder(t[0], t[1], k), files[1]), 'rb') as f3:
        det.append(pickle.load(f3) * 2 * np.pi / 1e6)

det_value = 45 #MHz
d = det[1]
det_idx = int(0.5 * len(d) * (1 + det_value / d[-1]))


fitparams, conv = curve_fit(
    lambda x: double_approx(det_value, q_freq, delta, eps, tau, "sin"), 
    x_values, y_values, init_params, 
    maxfev=1e6, 
    bounds=(lower, higher)
)

intervals_amp = [100, 100]
intervals_det = [100, 100]
tr = tr_prob[1][:, det_idx]
# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(8,6), layout="constrained")
gs = fig.add_gridspec(1, 1)

ax = fig.add_subplot(gs[0, 0])
# Plot the color map
ax.scatter(amp[1], tr, marker="$\pm$", s=100)
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
plt.show()
