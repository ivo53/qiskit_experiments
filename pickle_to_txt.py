import os
import pickle
import numpy as np
import pandas as pd

times = {
    "lor2": ["2023-07-04", "192453"],
    "lor": ["2023-07-04", "192414"],
    "lor3_4": ["2023-07-04", "192347"],
    "lor2_3": ["2023-07-04", "192350"], 
}
params = {
    "lor2": [(24 + 8/9) * 1e-9, (181 + 2/3) * 1e-9],
    "lor": [(24 + 8/9) * 1e-9, (704) * 1e-9],
    "lor3_4": [(10 + 2/3) * 1e-9, (728 + 8/9) * 1e-9],
    "lor2_3": [(10 + 2/3) * 1e-9, (1134 + 2/9) * 1e-9],
}
powers = [2, 1, 3/4, 2/3]
powers_latex = [" $L^2$", "  $L$", "$L^{3/4}$", "$L^{2/3}$"]
pulse_names = list(params.keys())
backend_name = "manila"

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
        det = pickle.load(f3) / 1e6
    tr_probs.append(tr_prob)
    amps.append(amp)
    dets.append(det)

tr_probs = np.array(tr_probs)[:, 1::2]
amps = np.array(amps)
dets = np.array(dets)

print(tr_probs.shape)
print(amps.shape)
print(dets.shape)
print(dets)
i = 0
for ts, d in zip(tr_probs, dets):
    j = 0 
    for t in ts:
        data = np.vstack((d, t))
        # Convert the matrix to a DataFrame
        df = pd.DataFrame(data)
        # # Export DataFrame to a CSV file
        # df.T.to_csv(f'C:/Users/Ivo/Documents/PhD Documents/статии/Power Narrowing/numerical_data/{pulse_names[i]}-{2 * j + 1}pi.csv', index=False)
        # Export DataFrame to an XLSX file
        df.T.to_excel(f'C:/Users/Ivo/Documents/PhD Documents/статии/Power Narrowing/numerical_data/{pulse_names[i]}-{2 * j + 1}pi.xlsx', index=False, engine='openpyxl')

        j += 1
    i += 1