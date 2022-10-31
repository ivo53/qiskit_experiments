import numpy as np
from qiskit import QuantumCircuit, IBMQ, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from sympy import comp

provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_armonk")
backend_name = str(backend)
print(f"Using {backend_name} backend.")

# Create a Quantum Circuit acting on the q register
circuits = []
for i in range(24):
    circuits.append(QuantumCircuit(1, 1))

step = 20
for i in range(len(circuits)):
    for _ in range(i * step + 20):
        circuits[i].h(0)
    circuits[i].measure([0], [0])
# for i in range(len(circuits)):
#     for _ in range(i ** 2 * step + 2):
#         circuits[i].h(0)
#     circuits[i].measure([0], [0])

# Add a H gate on qubit 0

# for i in range(5):
#     for _ in range(101):
#         circuit.h(i)


# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
# circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
# circuit.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
# compiled_circuit = transpile(circuit, backend)
compiled_circuits = []
for circuit in circuits:
    compiled_circuits.append(transpile(circuit, backend, optimization_level=0))
# Execute the circuit on the qasm simulator
# job = simulator.run(compiled_circuit, shots=1000)
for compiled_circuit in compiled_circuits:
    job = backend.run(compiled_circuit, shots=20000)
    job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts()
    print("\nTotal count for qubits are:", counts)

# Draw the circuit
# print(circuit.draw())
