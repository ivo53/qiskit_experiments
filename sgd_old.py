import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

rough_qubit_frequency = 4962284031.287086 # 4.97173 * 1.e9
backend_name = "manila"
pulse_type = "lor"
dur_dt = 2256

lorentz = ["lor", "lor2", "lor3"]
if isinstance(dur_dt, float) and dur_dt < 3:
    dur_dt = get_closest_multiple_of_16(dur_dt * 1.e-6 / (1.e-9 * 2/9))

if pulse_type in lorentz:
    cutoff = 50.0 # %
    should_exit = False
    should_exit = True
    ctrl_param = "width"

    if ctrl_param == "width":
        G = 180
        dur_dt = 2 * G * np.sqrt(100 / cutoff - 1)
        dur_dt = get_closest_multiple_of_16(dur_dt)
    elif ctrl_param == "duration":
        dur_dt = 2256
        G = dur_dt / (2 * np.sqrt(100 / cutoff - 1))


date_of_cal = None
date_of_cal = "2023-02-26"
date = datetime.now()
current_date = date_of_cal if date_of_cal is not None else date.strftime("%Y-%m-%d")

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

file_dir = os.path.dirname(__file__)
data_folder = os.path.join(file_dir, "data", backend_name, "calibration", current_date)

data_files = os.listdir(data_folder)
if pulse_type not in lorentz:
    key_start = f"{pulse_type}_"
else:
    if ctrl_param == "width":
        c_p = G
    elif ctrl_param == "duration":
        c_p = dur_dt
    key_start = f"{pulse_type}_cutoff_{cutoff}_{ctrl_param}_{c_p}"
print(key_start)
amps, vals = [], []
for d in data_files:
    if d.startswith(key_start):
        if d.endswith("amps.pkl"):
            amps.append(d)
        elif d.endswith("tr_prob.pkl"):
            vals.append(d)
print(amps)
amps = [amps[-1]]
vals = [vals[-1]]
assert len(amps) == 1 and len(vals) == 1, "Expected 1 data file per pulse type"

with open(os.path.join(data_folder, amps[0]), "rb") as f1:
    x = pickle.load(f1)

with open(os.path.join(data_folder, vals[0]), "rb") as f2:
    y = pickle.load(f2)

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(
        function, 
        x_values, 
        y_values, 
        init_params,
        maxfev=100000, 
        bounds=(
            [-0.53, 1, 0, 0.45], 
            [-.43, 200, 100, 0.55]
        )
    )
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit


fitparams, _ = fit_function(
    x, y, 
    # lambda x, A, k, l, p, B: A * (np.cos(k*x + l * (1 - np.exp(-p*x)))) + B,
    lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(-p*(x-x0))))) + B,
    [-.5, 25, .4, 0, 0.5]
)
print(fitparams)
A,l,p,B = fitparams
x_ext = np.linspace(x[0], x[-1], 10000)
# extended_y_fit = A * (np.cos(k*x_ext + l * (1-np.exp(-p*x_ext)))) + B
extended_y_fit = A * (np.cos(l * (1-np.exp(-p*x_ext)))) + B

plt.plot(x, y, "bx")
plt.plot(x_ext, extended_y_fit, color='red')
plt.show()
if should_exit:
    exit()
# fitparams_folder = os.path.join(file_dir, "data", backend_name, "fits", current_date)
pi_amp = - np.log(1 - np.pi / l) / p
half_amp = - np.log(1 - np.pi / (2 * l)) / p
param_dict = {
    "date": date.strftime("%Y-%m-%d"),
    "time": date.strftime("%H:%M:%S"),
    "pulse_type": pulse_type,
    "A": A,
    "l": l,
    "p": p,
    "B": B,
    "drive_freq": rough_qubit_frequency,
    "pi_duration": dur_dt,
    "pi_amp": pi_amp,
    "half_amp": half_amp,
    "cutoff": cutoff,
    "G": G
}

calib_dir = os.path.join(file_dir, "calibrations")

param_series = pd.Series(param_dict)
params_file = os.path.join(calib_dir, "params.csv")
if os.path.isfile(params_file):
    param_df = pd.read_csv(params_file)
    param_df = pd.concat([param_df, param_series.to_frame().T], ignore_index=True)
    param_df.to_csv(params_file, index=False)
else:
    param_series.to_frame().T.to_csv(params_file, index=False)
