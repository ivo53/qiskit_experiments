import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from pandas import cut
from scipy.optimize import curve_fit

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

backend_name = "armonk"
varied = "vary_duration"
cutparam = .6 # %
if varied == "vary_duration":
    # G = 150
    # dur_dt = 2 * G * np.sqrt(100 / cutparam - 1)
    # dur_dt = get_closest_multiple_of_16(dur_dt)
    G = 400
    dur_dt = 10304
elif varied == "vary_width":
    dur_dt = 2256*5
    G = dur_dt / (2 * np.sqrt(100 / cutparam - 1))
date_of_cal = None
# date_of_cal = "2022-06-11"
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
data_folder = os.path.join(file_dir, "data", backend_name, "calibration", varied, current_date)

data_files = os.listdir(data_folder)
print(data_files)
amps, vals = [], []
for d in data_files:
    if d.startswith(f"{dur_dt}dt_cutparam-{cutparam}%_G"):
        if d.endswith("amps.pkl"):
            amps.append(d)
        elif d.endswith("tr_prob.pkl"):
            vals.append(d)
print(amps)
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
            [-.39, 150, 100, 0.55]
        )
    )
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit


fitparams, _ = fit_function(
    x, y, 
    # lambda x, A, k, l, p, B: A * (np.cos(k*x + l * (1 - np.exp(-p*x)))) + B,
    lambda x, A, l, p, B: A * (np.cos(l * (1 - np.exp(-p*x)))) + B,
    [-.5, 60, 0.5, 0.5]
)
print(fitparams)
A,l,p,B = fitparams
x_ext = np.linspace(x[0], x[-1], 10000)
# extended_y_fit = A * (np.cos(k*x_ext + l * (1-np.exp(-p*x_ext)))) + B
extended_y_fit = A * (np.cos(l * (1-np.exp(-p*x_ext)))) + B

plt.plot(x, y, "bx")
plt.plot(x_ext, extended_y_fit, color='red')
plt.show()
exit()
fitparams_folder = os.path.join(file_dir, "data", backend_name, "fits", varied, current_date)
make_all_dirs(fitparams_folder)
with open(os.path.join(fitparams_folder, vals[0].replace("_tr_prob", "") ), "wb") as f3:
    pickle.dump(fitparams, f3)
