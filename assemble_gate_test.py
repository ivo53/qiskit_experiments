import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from qiskit import QuantumCircuit, IBMQ, transpile, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.circuit.library import XGate
from qiskit.providers.ibmq.managed import IBMQJobManager

gate = "sx"
num_orders = 4 if gate == "sx" else 2
date = "2022-05-04"
basis_gates_folder = os.path.join(
    "C:/Users/Ivo/Documents/qiskit_codes/"
    "data/basis_gate_tests/",
    f"{gate}_gate",
    f"{date}"
)

tr_prob = [None] * num_orders
num_gates = [None] * num_orders

for i in range(num_orders):
    files = os.listdir(
        os.path.join(
            basis_gates_folder, 
            f"4k+{i}"
        )
    )
    num_gates_f, tr_prob_f = [], []
    for f in files:
        if f.endswith("num_gates.pkl"):
            num_gates_f.append(f)
        elif f.endswith("tr_prob.pkl"):
            tr_prob_f.append(f)
    num_gates_f = np.sort(num_gates_f)
    tr_prob_f = np.sort(tr_prob_f)
    
    with open(
        os.path.join(
            basis_gates_folder,
            f"4k+{i}",
            num_gates_f[-1]
        ),
        mode="rb"
    ) as file:
        num_gates[i] = pickle.load(file)    
    
    with open(
        os.path.join(
            basis_gates_folder,
            f"4k+{i}",
            tr_prob_f[-1]
        ),
        mode="rb"
    ) as file:
        tr_prob[i] = pickle.load(file)
    
total_num_gates, total_tr_prob = [], []
for i in range(4):
    for n in range(len(num_gates[i])):
        total_num_gates.append(num_gates[i][n])
        total_tr_prob.append(tr_prob[i][n])
total_num_gates = np.array(total_num_gates)
total_tr_prob = np.array(total_tr_prob)
idx = np.argsort(total_num_gates)
total_num_gates = total_num_gates[idx]
total_tr_prob = total_tr_prob[idx]
df = pd.DataFrame({
    "num_gates": total_num_gates,
    "transition_probability": total_tr_prob
})
df.to_csv("csvcsv.csv", index=False)
plt.figure(1)
plt.plot(num_gates[0], tr_prob[0], "r.")
plt.plot(num_gates[1], tr_prob[1], "b.")
plt.plot(num_gates[2], tr_prob[2], "g.")
plt.plot(num_gates[3], tr_prob[3], ".")
plt.show()