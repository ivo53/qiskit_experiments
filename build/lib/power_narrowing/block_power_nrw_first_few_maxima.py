import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# from numerical_solutions import ndsolve_lorentz_spectre
from transition_line_profile_functions import \
fit_function, lor_narrowing, lor2_narrowing, \
lor3_2_narrowing, lor3_4_narrowing, lor2_3_narrowing, lor3_5_narrowing

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def exponential_moving_average(data, alpha, beta):
    ema = [data[0]]  # Initialize with the first data point
    
    for i in range(1, len(data) - 1):
        ema_value = alpha * data[i] + (1 - alpha - beta) * ema[i-1] + beta * data[i+1]
        ema.append(ema_value)
    ema.append(data[-1])
    return np.array(ema)


def fwhm(data_points, x_vals):
    # Find the maximum value and its index in the data
    max_value = max(data_points)

    # Normalize the data so that the maximum value is 1
    normalized_data = np.array(data_points) / max_value

    # Find indices where the data crosses the half-max threshold
    half_max = max_value / 2
    crossing_indices = []
    for i in range(1, len(normalized_data)):
        if normalized_data[i - 1] >= half_max and normalized_data[i] < half_max:
            crossing_indices.append(i - 1)
        elif normalized_data[i - 1] < half_max and normalized_data[i] >= half_max:
            crossing_indices.append(i)

    # Calculate the FWHM as the difference between the two closest half-max points
    fwhm = None
    if len(crossing_indices) >= 2:
        if len(crossing_indices) > 2:
            pass
            # print("Warning: More than 2 crossing indices!")
        fwhm = (x_vals[crossing_indices[-1]] - x_vals[crossing_indices[0]])
    return fwhm

backend_name = "manila"
save = 1


# times = {
#     0.5: ["2023-06-03", "120038"],
#     1: ["2023-06-03", "160754"],
#     2: ["2023-06-03", "161333"],
#     3: ["2023-06-03", "022027"],
#     5: ["2023-06-03", "022625"],
#     7.5: ["2023-06-03", "023608"],
#     15: ["2023-06-03", "120636"],
#     20: ["2023-06-03", "115516"],
#     50: ["2023-06-03", "222957"],
# }

# # time = ["2023-06-06", "201236"] # lor 3/4, sigma = 96, dur = 4128
# time = ["2023-06-06", "201245"] # lor 3/4, sigma = 96, dur = 5008
# # time = ["2023-06-04", "011733"] # lor, sigma = 96, dur = 2704
# # time = ["2023-06-07", "023054"] # lor2_3, sigma = 48, dur = 5104
# # time = ["2023-07-02", "002750"] # lor2, sigma = 96, dur = 704
# # time = ["2023-03-12", "101213"] # rect sigma = 800, dur = 800

times = {
    "lor2": ["2023-07-04", "192453"],
    "lor3_2": ["2023-08-24", "101751"],
    "lor": ["2023-07-04", "192414"],
    "lor3_4": ["2023-07-04", "192347"],
    "lor2_3": ["2023-07-04", "192350"], 
    "lor3_5": ["2023-07-03", "085956"], 
}
params = {
    "lor2": [(24 + 8/9) * 1e-9, (181 + 2/3) * 1e-9],
    "lor3_2": [(24 + 8/9) * 1e-9, (286.81 + 5/900) * 1e-9],
    "lor": [(24 + 8/9) * 1e-9, (704) * 1e-9],
    "lor3_4": [(10 + 2/3) * 1e-9, (728 + 8/9) * 1e-9],
    "lor2_3": [(10 + 2/3) * 1e-9, (1134 + 2/9) * 1e-9],
    "lor3_5": [(7 + 1/9) * 1e-9, (1176 + 8/9) * 1e-9], 
}
fit_funcs = [lor2_narrowing, lor3_2_narrowing, lor_narrowing, lor3_4_narrowing, lor2_3_narrowing, lor3_5_narrowing]
# fit_funcs = [lor2_rzconj, lor_rzconj, lor3_4_rzconj, lor2_3_rzconj]
powers = [2, 3/2, 1, 3/4, 2/3, 3/5]
powers_latex = [" $L^2$", "$L^{3/2}$", "  $L$", "$L^{3/4}$", "$L^{2/3}$", "$L^{3/5}$"]
pulse_names = list(params.keys())

colors = [
    "red",
    "blue",
    "#9D7725",
    "brown",
    "purple"
]

markers = [
    "o",
    "*",
    "X",
    "D",
    "P"
]
sizes = 1.2 * np.array([30,35,35,35,35])
# intervals_det = [2.5, 2.5, 2.5, 6, 6, 8]
# intervals_det_minor = [0.5, 0.5, 0.5, 1.2, 1.2, 1.6]
intervals_det = [15, 15, 15, 35, 35, 50]
intervals_det_minor = [3, 3, 3, 7, 7, 10]

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


tr_probs, amps, dets = [], [], []
for i, k in times.items():
    files = os.listdir(data_folder(k[0], k[1], i))
    with open(os.path.join(data_folder(k[0], k[1], i), files[2]).replace("\\","/"), 'rb') as f1:
        tr_prob = pickle.load(f1)
    with open(os.path.join(data_folder(k[0], k[1], i), files[0]), 'rb') as f2:
        amp = l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T)
    with open(os.path.join(data_folder(k[0], k[1], i), files[1]), 'rb') as f3:
        det = pickle.load(f3) * 2 * np.pi/ 1e6
    tr_probs.append(tr_prob)
    amps.append(amp)
    dets.append(det)

include_35 = False
if include_35:
    tr_probs.pop(0)
    amps.pop(0)
    dets.pop(0)
    intervals_det.pop(0)
    intervals_det_minor.pop(0)
    powers_latex.pop(0)
    pulse_names.pop(0)
    fit_funcs.pop(0)
    powers.pop(0)


initial = [[[7.39387630e-01, 3.32448420e+00, 3.52987554e+07, 4.95231651e+07, 4.04510782e-01, 5.57783505e-01],
[1.22213543e-01, 1.07754576e+00, 3.19285450e+07 ,2.01938417e+07,
 4.73148229e-01 ,4.87233069e-01],
[-4.78643408e-01,  4.08591824e+00 , 2.75889142e+07,  3.77212479e+07,
  3.78151352e-01,  5.79483946e-01]],
[[7.36535691e-01, 1.03643143e+00, 2.63768868e+07, 8.74417445e+06,
 4.73892849e-01, 4.93638563e-01],
[-1.12407126e-02, -1.56136560e+00,  3.31036477e+07,  4.12817667e+07,
  6.06495588e-01,  3.60342462e-01],
[-2.58051944e-01, -4.49488621e+00 , 3.30615384e+07,  4.47501618e+07,
  9.68302963e-01,  1.61334854e-11]],
[[ 2.37707154e-01, -1.67944023e+00 , 8.70552870e+06,  1.44486380e+07,
  5.08058974e-01,  4.64125136e-01],
[-1.75365316e-01, -3.55457923e+00 , 2.81601879e+08,  1.32092672e+07,
  1.10883543e-01,  8.53011891e-01],
[-1.65533696e-01, -4.94361655e-01 , 1.41017205e+07,  1.02478332e+07,
  4.94737037e-01,  4.78277979e-01]],
[[3.62640373e-01, 5.69206563e-01, 3.57851359e+07, 1.58302413e+07,
 4.66835073e-01, 4.93184058e-01],
[-2.49381251e-01, -3.00468219e-01,  1.56248754e+08,  1.01955982e+07,
  3.82641933e-01,  5.79208632e-01],
[ 2.31643658e-02, -8.01753765e-01,  1.09697196e+07,  8.09967524e+06,
  4.89760105e-01,  4.88432296e-01]],
[[ 3.92035232e-01, -2.49133021e+00 , 1.04062444e+07,  1.53150060e+07,
  4.90219168e-01,  4.85224204e-01],
[-8.47314635e-02, -1.27161648e-01,  3.00391574e+07,  6.64032126e+06,
  4.54706902e-01,  5.00646631e-01],
[ 3.42120701e-02, -8.55793563e-01 , 6.64367484e+06 , 4.93892857e+06,
  4.91221193e-01,  4.93140842e-01]],
[[1.46912524e-01, 5.61873623e-01, 3.31302121e+07, 1.01440461e+07,
 4.59571129e-01, 5.17216946e-01],
[-7.84020915e-02 ,-1.81724683e-01,  2.84358178e+07,  5.33481738e+06,
  4.27660356e-01,  5.13259337e-01],
[0, 0.5, 1e8, 2e8, 0.4, 0.5],]]
# init = [0, 0.5, 1e8, 2e8, 0.4, 0.5]
initial_min = [-30, -10, 0, 0, 0, 0]
initial_max = [30, 10, 1e9, 1e9, 1, 1]

third_curve = 3
# FITS (num/analytical)
# det_limits = 15 * sigma_temp
# numerical_dets, numerical_tr_probs = [], []
all_y_fits, all_ext_y_fits, all_efs, moving_averages = [], [], [], []
for i in range(6):
    # area_dets, area_probs = [], []
    y_fits, extended_y_fits, efs, mavg = [], [], [], []
    for j in [0,1,third_curve]:
        # numer_det, numerical_tr_prob = ndsolve_lorentz_spectre(
        #     params[list(times.keys())[i]][0],
        #     params[list(times.keys())[i]][1],
        #     -det_limits * 1e6 * 2 * np.pi, 100,
        #     d_end=det_limits * 1e6 * 2 * np.pi,
        #     num_t=1000,
        #     pulse_area=(2*j+1) * np.pi,
        #     lor_power=powers[i]
        # ) 
        init = initial[i][np.amin([2, j])]
        fit_params, y_fit, err = fit_function(
            dets[i],
            tr_probs[i][1::2][j], 
            fit_funcs[i],
            init, initial_min, initial_max,
            params[pulse_names[i]][0] / (2e-9/9),
            params[pulse_names[i]][1] / (2e-9/9),
            area=(2 * j + 1) * np.pi,
            remove_bg=False
        )
        # print(fit_params)
        # window_size = 3
        # moving_average = np.convolve(tr_probs[i][1::2][j], np.ones(window_size), mode='valid')/window_size
        ef = np.linspace(dets[i][0], dets[i][-1], 5000)
        y_fit = fit_funcs[i](dets[i], *fit_params)
        extended_y_fit = fit_funcs[i](ef, *fit_params)
        moving_average = exponential_moving_average(
            tr_probs[i][1::2][j], 
            0.8, 0.1)
        moving_average = np.interp(np.linspace(dets[i][:][0], dets[i][:][-1], 5000), dets[i][:], tr_probs[i][1::2][j])
        # print(fit_params)
        # print(y_fit)
        # area_dets.append(numer_det / (1e6 * 2 * np.pi))
        # area_probs.append(numerical_tr_prob)
        mavg.append(moving_average)
        y_fits.append(y_fit)
        efs.append(ef)
        extended_y_fits.append(extended_y_fit)
    moving_averages.append(mavg)
    all_y_fits.append(y_fits)
    all_efs.append(efs)
    all_ext_y_fits.append(extended_y_fits)
    # numerical_dets.append(area_dets)
    # numerical_tr_probs.append(area_probs)

# Create a 3x3 grid of subplots with extra space for the color bar
fig = plt.figure(figsize=(12,15), layout="constrained")
gs0 = fig.add_gridspec(3, 2, width_ratios=[1, 1])
# Generate datetime
date = datetime.now()

fwhms, fwhms_abs = [], []
for i in range(3):
    for j in range(2):
        ax = fig.add_subplot(gs0[i, j])
        # if i == 0 and j == 0:
        #     i = 2
        max_det = intervals_det[2*i+j] * np.floor(dets[2*i+j][-1] / intervals_det[2*i+j])        
        max_det_minor = intervals_det_minor[2*i+j] * np.floor(dets[2*i+j][-1] / intervals_det_minor[2*i+j])        
        sigma_temp = params[pulse_names[2*i+j]][0] * 10**6
        det_limit = 0.8#14
        plots_fwhm, plots_fwhm_abs = [], []
        for idx, order, t in zip([0,1,2], [0,1,third_curve], tr_probs[2*i+j][1::2][[0,1,third_curve]]):
            ax.scatter(dets[2*i+j] * sigma_temp, t, c=colors[idx],linewidth=0,marker=markers[idx], s=sizes[idx], label=f"{2*order+1}$\pi$")
            # ax.plot(numerical_dets[2*i+j][idx], numerical_tr_probs[2*i+j][idx], c=colors[idx],linewidth=0.2, label=f"Simulation - area {2*order+1}$\pi$")
            # plots_fwhm.append(fwhm(t, dets[2*i+j] * sigma_temp))
            # plots_fwhm_abs.append(fwhm(t, dets[2*i+j]))
            ## fit FWHM
            ax.plot(all_efs[2*i+j][idx] * sigma_temp, all_ext_y_fits[2*i+j][idx], c=colors[idx],linewidth=0.2)
            plots_fwhm.append(fwhm(all_ext_y_fits[2*i+j][idx], all_efs[2*i+j][idx] * sigma_temp))
            plots_fwhm_abs.append(fwhm(all_ext_y_fits[2*i+j][idx], all_efs[2*i+j][idx]))
            ## moving average FWHM
            # ax.plot(all_efs[2*i+j][idx] * sigma_temp, moving_averages[2*i+j][idx], c=colors[idx],linewidth=0.2, label=f"Fit func - area {2*order+1}$\pi$")
            # plots_fwhm.append(fwhm(moving_averages[2*i+j][idx], all_efs[2*i+j][idx] * sigma_temp))
            # plots_fwhm_abs.append(fwhm(moving_averages[2*i+j][idx], all_efs[2*i+j][idx]))
            
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        # print("maxdet", max_det_minor)
        tick_labels = np.round(np.arange(
            -max_det, 
            max_det + 0.01, 
            intervals_det[2*i+j]
        ), 1)

        tick_locations = np.linspace(
            -max_det * sigma_temp,
            max_det * sigma_temp,
            len(tick_labels)
        )

        ax.set_xticks(tick_locations)
        ax1 = ax.secondary_xaxis('top')

        top_labels = np.linspace(-det_limit, det_limit, 5)
        ax1.set_xticks(np.round(top_labels, 2))
        # Add a rectangle in the top right corner
        rect = Rectangle((0.8, 0.89), 0.12, 0.11, transform=ax.transAxes,
                        color='#003399', alpha=0.7)
        ax.add_patch(rect)
        
        # Add text inside the rectangle
        text = powers_latex[2*i+j]
        ax.text(0.82, 0.91, text, transform=ax.transAxes,
            color='white', fontsize=18)
        if i == 0:
            ax1.set_xlabel('Detuning (in units of $1/\\tau$)', fontsize=18)
            ax1.set_xticklabels(np.round(top_labels, 2), fontsize=18)
        elif i == 1:
            ax1.set_xticklabels([])        
        elif i == 2:
            ax.set_xlabel('Detuning (MHz)', fontsize=18)
            ax1.set_xticklabels([])
        remainders = np.modf(tick_labels.round(1))[0]
        tick_labels_formatted = [int(t) if r == 0 else np.round(t, 1) for r, t in zip(remainders, tick_labels)]
        ax.set_xticklabels(tick_labels_formatted, fontsize=18)    
        if j == 0:
            ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=18)
            ax.set_ylabel('Transition Probability', fontsize=18)
        else:
            ax.set_yticklabels([])
        y_minor_ticks = np.arange(0, 1.01, 0.05)
        x_minor_ticks = sigma_temp * np.arange(-max_det_minor, max_det_minor+0.01, intervals_det_minor[2*i+j])
        ax.set_yticks(y_minor_ticks, minor="True")
        ax.set_xticks(x_minor_ticks, minor="True")
        ax.set_ylim((-0.025,1))
        # ax.set_xlim((-13,13))
        ax.set_xlim((-det_limit - 1e-3, det_limit + 1e-3))
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.6)

        ax.legend()
        fwhms.append(plots_fwhm)
        fwhms_abs.append(plots_fwhm_abs)

# fig.tight_layout()
fwhms_abs = np.array(fwhms_abs)
print(np.array(fwhms))
print(fwhms_abs)
print(fwhms_abs[:, 0] / fwhms_abs[:, 2])
# Set save folder
save_folder = os.path.join(file_dir, "paper_ready_plots", "power_narrowing")

# Generate datetime
date = datetime.now()

# Set fig name
fig_name = f"block_1,3,{2 * third_curve + 1}pi_{list(times.keys())}_{date.strftime('%Y%m%d')}_{date.strftime('%H%M%S')}.pdf"

# Save the fig
if save:
    plt.savefig(os.path.join(save_folder, fig_name), format="pdf")

# Display the plot
# plt.show()
