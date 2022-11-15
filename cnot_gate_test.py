import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, IBMQ, transpile, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.circuit.library import XGate
from qiskit.providers.ibmq.managed import IBMQJobManager

gate_type = "cnot"

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
    f"{date.strftime('%H%M%S')}"
).replace("\\", "/")
make_all_dirs(data_folder)
make_all_dirs(folder_name)


provider = IBMQ.load_account()
backends = (provider.backends(filters=lambda b: b.configuration().n_qubits == 5))
backend = least_busy(backends)
backend_name = str(backend)
print(f"Using {backend_name} backend.")

# num_circuits = 100
start_num_gates = 0
end_num_gates = 100

num_qubits = 2
num_shots = 10000
gate_increment = 1

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
    qc = QuantumCircuit(
        QuantumRegister(num_qubits),
        ClassicalRegister(num_qubits)
    )
    qc.x(0)
    for _ in range(i):
        qc.cnot(0, 1)
    qc.measure_all()
    circuits.append(
        transpile(
            qc, 
            backend=backend, 
            optimization_level=0
        )
    )

job_manager = IBMQJobManager()

job = job_manager.run(
    circuits,
    backend=backend,
    shots=num_shots,
    optimization_level=0
)
# job_monitor(job)
results = job.results()
print(results.get_counts(0))
transition_probability = [[] for _ in range(num_qubits ** 2)]
for i in range(num_circuits):
    transition_probability[0].append(
        results.get_counts(i)["00 00"] / num_shots
    )
    transition_probability[1].append(
        results.get_counts(i)["01 00"] / num_shots
    )
    transition_probability[2].append(
        results.get_counts(i)["10 00"] / num_shots
    )
    transition_probability[3].append(
        results.get_counts(i)["11 00"] / num_shots
    )

# ## EXECUTE
# job = execute(
#     circuits,
#     backend=backend,
#     shots=num_shots,
#     optimization_level=0
# )
# job_monitor(job)
# results = job.result()
# transition_probability = []
# for i in range(num_circuits):
#     transition_probability.append(
#         1 - results.get_counts(i)["0"] / num_shots
#     )
##

with open(os.path.join(data_folder, "tr_prob.pkl").replace("\\","/"), 'wb') as f1:
    pickle.dump(transition_probability, f1)
with open(os.path.join(data_folder, "num_gates.pkl").replace("\\","/"), 'wb') as f2:
    pickle.dump(num_gates, f2)


plt.figure(1)
plt.plot(num_gates, transition_probability[0], "b.")
plt.plot(num_gates, transition_probability[1], "g.")
plt.plot(num_gates, transition_probability[2], "r.")
try:
    plt.plot(num_gates, transition_probability[3], "y.")
except:
    pass
plt.xlabel(f"Number of {gate_type} gates")
plt.ylabel("Transition Probability")
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.savefig(
    os.path.join(
        folder_name,
        f"{date.strftime('%H%M%S')}_{end_num_gates}_{gate_type}_gates.png"
    )
)
plt.show()