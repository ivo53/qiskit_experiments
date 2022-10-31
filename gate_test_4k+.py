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

gate_type = "sx"
four_k_plus = 3

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

## create folder where plots are saved
file_dir = os.path.dirname(__file__)
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
save_dir = os.path.join(
    file_dir,
    "plots",
    "basis_gate_tests",
    f"{gate_type}_gate",
    current_date
).replace("\\", "/")
folder_name = os.path.join(
    save_dir,
    date.strftime("%H%M%S")
).replace("\\", "/")
data_folder = os.path.join(
    file_dir,
    "data",
    "basis_gate_tests",
    f"{gate_type}_gate",
    current_date,
    "4k+{}".format(four_k_plus)
).replace("\\", "/")
make_all_dirs(data_folder)
make_all_dirs(folder_name)


provider = IBMQ.load_account()
backends = (provider.backends(filters=lambda b: b.name() == "ibmq_armonk"))
# backends = backends[2:7]
backend = least_busy(backends)
backend_name = str(backend)
print(f"Using {backend_name} backend.")

# num_circuits = 100
start_num_gates = four_k_plus
end_num_gates = 300

num_qubits = 1
num_shots = 10000
gate_increment = 4

num_gates = np.arange(
    start_num_gates,
    end_num_gates,
    gate_increment
)

num_circuits = len(num_gates)
# qc = QuantumCircuit(num_qubits, num_qubits)
# qc.h(0)
# qc = qc.decompose()
# qc.draw()
# plt.show()
circuits = []
for i in num_gates:
    qc = QuantumCircuit(num_qubits, num_qubits)
    for _ in range(i):
        qc.sx(0)
    qc.measure([0], [0])
    circuits.append(
        transpile(
            qc, 
            backend=backend, 
            optimization_level=0
        )
    )

# h1 = circuits[1].decompose()
# h1.draw()
# plt.show()
# plt.close()
job_manager = IBMQJobManager()

job = job_manager.run(
    circuits,
    backend=backend,
    shots=num_shots,
    # optimization_level=0
)
# job_monitor(job)
results = job.results()

transition_probability = []
for i in range(num_circuits):
    transition_probability.append(
        1 - results.get_counts(i)["0"] / num_shots
    )

with open(os.path.join(data_folder, f"{date.strftime('%H%M%S')}_tr_prob.pkl").replace("\\","/"), 'wb') as f1:
    pickle.dump(transition_probability, f1)
with open(os.path.join(data_folder, f"{date.strftime('%H%M%S')}_num_gates.pkl").replace("\\","/"), 'wb') as f2:
    pickle.dump(num_gates, f2)


plt.figure(1)
plt.plot(num_gates, transition_probability, "b.")
plt.xlabel(f"Number of {gate_type} gates")
plt.ylabel("Transition Probability")
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.savefig(
    os.path.join(
        folder_name,
        f"{date.strftime('%H%M%S')}_4k+{four_k_plus}_{end_num_gates}_{gate_type}_gates.png"
    )
)
plt.show()