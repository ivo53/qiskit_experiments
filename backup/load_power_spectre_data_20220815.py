import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

shape = "lorentz"
# load_date = "2022-04-23"
# load_time = "094423"
# load_date = "2022-06-13"
# load_time = "230218"
load_date = "2022-06-23"
load_time = "192315"
file_dir = file_dir = os.path.dirname(__file__)
data_folder = os.path.join(
    file_dir,
    "data",
    "armonk",
    "power_broadening (narrowing)",
    f"{shape}_pulses", 
    load_date,
    load_time
).replace("\\", "/")

data_files = os.listdir(data_folder)
for d in data_files:
    if d.endswith("tr_prob.pkl"):
        tr_prob_file = d
    elif d.endswith("areas.pkl"):
        areas_file = d
    elif d.endswith("detunings.pkl"):
        detunings_file = d
with open(os.path.join(data_folder, tr_prob_file), 'rb') as f1:
    transition_probabilities = pickle.load(f1, encoding='latin1')
with open(os.path.join(data_folder, areas_file), 'rb') as f2:
    areas = pickle.load(f2, encoding='latin1')
with open(os.path.join(data_folder, detunings_file), 'rb') as f3:
    detunings = pickle.load(f3, encoding='latin1')
print(detunings[65:135])
# span = 20
# from_idx = int((len(transition_probabilities) - span)/2)
# to_idx = int((len(transition_probabilities) + span)/2)

# maximums = np.amax(transition_probabilities[from_idx:to_idx], axis=0)
# print(transition_probabilities.shape)

# m = np.argsort(maximums)[::-1]
# # print(areas[m])
# # print(m)
peaks = [2, 3, 4]
print(
	np.where(detunings < - 2.*10**6)[-1], 
	np.where(detunings > 2.*10**6)[0]
	)
detunings = (detunings + 4e4) * 1e-6

major_xticks = np.arange(detunings[65], -detunings[65], .4).round(1)
print(major_xticks)
minor_xticks = np.arange(detunings[65], -detunings[65], .2).round(1)
major_yticks = np.arange(0,1.001,0.2).round(1)
minor_yticks = np.arange(0,1.001,0.05).round(1)
fig = plt.figure(1, tight_layout=True, figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(detunings[65:135], transition_probabilities[65:135, peaks[0]], "rx")
ax.plot(detunings[65:135], transition_probabilities[65:135, peaks[1]], "bx")
ax.plot(detunings[65:135], transition_probabilities[65:135, peaks[2]], "gx")
ax.set_xlim(detunings[65], -detunings[65])
ax.set_xticks(major_xticks)
ax.set_xticklabels(major_xticks, fontsize=15)
ax.set_xticks(minor_xticks, minor=True)
ax.set_yticks(major_yticks)
ax.set_yticklabels(major_yticks, fontsize=15)
ax.set_yticks(minor_yticks, minor=True)
ax.set_title("Lorentzian Model - Power Narrowing (1, 3 and 5 Pi)", fontsize=20)
ax.set_xlabel("Detuning [MHz]", fontsize=20)
ax.set_ylabel("Transition Probability", fontsize=20)
ax.grid(which="major", alpha=0.6)
ax.grid(which="minor", alpha=0.3)
plt.show()