import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

shape = "lorentz"
load_date = "2022-04-23"
load_time = "094423"

file_dir = file_dir = os.path.dirname(__file__)
data_folder = os.path.join(
    file_dir,
    "data",
    "power_broadening (narrowing)",
    f"{shape}_pulses", 
    load_date,
    load_time
).replace("\\", "/")


with open(os.path.join(data_folder, "tr_prob.pkl"), 'rb') as f1:
    transition_probabilities = pickle.load(f1, encoding='latin1')
with open(os.path.join(data_folder, "areas.pkl"), 'rb') as f2:
    areas = pickle.load(f2, encoding='latin1')
with open(os.path.join(data_folder, "detunings.pkl"), 'rb') as f3:
    detunings = pickle.load(f3, encoding='latin1')

span = 20
from_idx = int((len(transition_probabilities) - span)/2)
to_idx = int((len(transition_probabilities) + span)/2)

maximums = np.amax(transition_probabilities[from_idx:to_idx], axis=0)
print(transition_probabilities.shape)

m = np.argsort(maximums)[::-1]
# print(areas[m])
# print(m)
peaks = [21, 65, 109, 158]
print(
	np.where(detunings < - 2.*10**6)[-1], 
	np.where(detunings > 2.*10**6)[0]
	)
plt.figure(1)
plt.plot(detunings, transition_probabilities[:, peaks[0]], "r.")
plt.plot(detunings, transition_probabilities[:, peaks[1]], "b.")
plt.plot(detunings[21:101], transition_probabilities[21:101, peaks[2]], "g.")
plt.show()