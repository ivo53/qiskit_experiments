import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle


def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "manila"
pulse_types = ["sin", "lor", "lor2", "lor3", "sech", "sech2", "gauss"]
save_fig = 0

times = {
    0.5: ["2023-06-03", "120038"],
    1: ["2023-06-03", "160754"],
    2: ["2023-06-03", "161333"],
    3: ["2023-06-03", "022027"],
    5: ["2023-06-03", "022625"],
    7.5: ["2023-06-03", "023608"],
    15: ["2023-06-03", "120636"],
    20: ["2023-06-03", "115516"],
    50: ["2023-06-03", "222957"],
}

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
    if int(t[1]) > 150000:
        files = os.listdir(data_folder(t[0], t[1], pulse_type))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type), files[2]).replace("\\","/"), 'rb') as f1:
            tr_prob.append(pickle.load(f1))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type), files[0]), 'rb') as f2:
            amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type), files[1]), 'rb') as f3:
            det.append(pickle.load(f3) * 2 * np.pi / 1e6)
    else:
        files = os.listdir(data_folder(t[0], t[1], pulse_type2))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type2), files[2]).replace("\\","/"), 'rb') as f1:
            tr_prob.append(pickle.load(f1))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type2), files[0]), 'rb') as f2:
            amp.append(l * (1 - np.exp(-p * (pickle.load(f2) - x0))) / (1e6 * T))
        with open(os.path.join(data_folder(t[0], t[1], pulse_type2), files[1]), 'rb') as f3:
            det.append(pickle.load(f3) * 2 * np.pi / 1e6)
