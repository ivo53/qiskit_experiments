import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# file = open("C:\\Users\\Ivo\\Desktop\\experiment_results\\even_x\\2-192-10-20k-5q.txt")
# rechnik = ""
# for line in file:
#     rechnik += line
# rechnik = json.loads(rechnik)


step = 64
num = 100
starting_length = 2
num_qubits = 1

rechnik = pd.read_csv('C:/Users/Ivo/Documents/qiskit_codes/experiment_results/x-2-6338-64-1.0k-1q.csv').fillna(0).to_dict('records')
# print(rechnik)
populations = []

for i in range(len(rechnik)):

    sumi = [[0, 0] for _ in range(num_qubits)]

    for key in rechnik[i]:
        for qubit in range(num_qubits):
            if key[qubit] == "0":
                sumi[qubit][0] += rechnik[i][key]
            elif key[qubit] == "1":
                sumi[qubit][1] += rechnik[i][key]

    means = np.mean(sumi, axis=1)
    # print(means)
    total = np.sum(sumi, axis=1)
    sumi = np.array(sumi)
    print(sumi)
    errors = np.sqrt(np.sum((sumi - means[:, None]) ** 2, axis=1) / sumi.shape[1])
    population = (sumi / total[:, None])
    # print(errors / means)
    populations.append(population)

populations = np.array(populations)
gate_number = np.arange(num) * step + starting_length
diff_colors = ['rx', 'bx', 'gx', 'yx', 'r.']
fig, ax = plt.subplots(num_qubits)
for i in range(num_qubits):
    if num_qubits == 1:
        ax.plot(gate_number, populations[:, i, 0], diff_colors[i])
    else:
        ax[i].plot(gate_number, populations[:, i, 0], diff_colors[i])

plt.show()