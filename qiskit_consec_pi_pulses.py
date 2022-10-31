import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from qiskit import QuantumCircuit, IBMQ, transpile, execute
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from qiskit.circuit.library import XGate

provider = IBMQ.load_account()
backends = (provider.backends(filters=lambda b: b.name() == "ibmq_armonk"))
# backends = backends[2:7]
backend = least_busy(backends)
backend_name = str(backend)
print(f"Using {backend_name} backend.")

num_circuits = 100
num_qubits = 1

circuits = []
for i in range(num_circuits):
    qc = QuantumCircuit(num_qubits, num_qubits)
    for _ in range(i * 2):
        qc.x(0)
    qc.measure([0], [0])
    circuits.append(qc)

job = execute(
    circuits,
    backend=backend,
    shots=1024,
    optimization_level=0
)
job_monitor(job)
result = job.result()
counts = result.get_counts()
counts_1 = [c["1"] / 1024 for c in counts]

plt.figure(1)
plt.plot(2 * np.arange(num_circuits), counts_1, "bx")
plt.xlabel("Number of x gates")
plt.ylabel("Transition Probability")
plt.show()