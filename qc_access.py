import numpy as np
import pandas as pd
import argparse
from qiskit import QuantumCircuit, IBMQ, transpile
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', 
                        "--num_circuits", 
                        help="Total number of circuits to try.", 
                        type=int, default=20)    
    parser.add_argument('-s', 
                        "--step", 
                        help="Step between number of gates.", 
                        type=int, default=20)
    parser.add_argument('-gt', 
                        "--gate_type", 
                        help="Type of repeated gate to use in the test" 
                             "('h', 'x' currently supported).", 
                        type=str, default="h")
    parser.add_argument('-sl', 
                        "--starting_length", 
                        help="Increment to each gate number "
                        "(num = i * step + starting_length).", 
                        type=int, default=2)
    parser.add_argument('-nq', 
                        "--num_qubits", 
                        help="Number of qubits to test on.", 
                        type=int, default=5)
    parser.add_argument('-ns', 
                        "--num_shots", 
                        help="Number of shots to repeat each circuit.", 
                        type=int, default=20000)
    parser.add_argument('-opt', 
                        "--optim_level", 
                        help="Optimization level.", 
                        type=int, default=0)
    args = parser.parse_args()

    assert args.gate_type in ['h', 'x'], \
        "Only 'h', 'x' gates are supported currently!"
    
    num_circuits = args.num_circuits if args.num_circuits > 0 else None
    step = args.step if args.step > 0 else None
    gate_type = args.gate_type
    starting_length = args.starting_length if args.starting_length > 0 else None
    num_qubits = args.num_qubits if args.num_qubits > 0 else None
    num_shots = args.num_shots if 20000 >= args.num_shots > 0 else None
    optim_level = args.optim_level if 4 > args.optim_level >= 0 else None

    assert num_circuits is not None, "num_circuits must be > 0"
    assert step is not None, "step must be > 0"
    assert starting_length is not None, "starting_length must be > 0"
    assert num_qubits is not None, "num_qubits must be > 0"
    assert num_shots is not None, "num_shots must be > 0 and <= 20000"
    assert optim_level is not None, "optim_level must be >= 0 and < 4"

    provider = IBMQ.load_account()
    backends = (provider.backends())
    backends = backends[2:7]
    backend = least_busy(backends)
    backend_name = str(backend)
    print(f"Using {backend_name} backend.")
    # Use Aer's qasm_simulator
    # simulator = QasmSimulator()

    # Create a Quantum Circuit acting on the q register
    circuits = []
    for i in range(num_circuits):
        circuits.append(QuantumCircuit(num_qubits, num_qubits))

    qubits_list = list(range(num_qubits))
    if gate_type == "x":
        for i in range(len(circuits)):
            for j in range(num_qubits):
                for _ in range(i * step + starting_length):
                    circuits[i].x(j)
            circuits[i].measure(qubits_list, qubits_list)

    elif gate_type == "h":
        for i in range(len(circuits)):
            for j in range(num_qubits):
                for _ in range(i * step + starting_length):
                    circuits[i].h(j)
            circuits[i].measure(qubits_list, qubits_list)

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
        compiled_circuits.append(transpile(circuit, backend, optimization_level=optim_level))
    # Execute the circuit on the qasm simulator
    # job = simulator.run(compiled_circuit, shots=1000)

    job = backend.run(compiled_circuits, shots=num_shots)
    job_monitor(job)

    # Grab results from the job
    result = job.result()

    # Returns counts
    counts = result.get_counts()
    print("\nTotal count for qubits are:", counts)

    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(f"./experiment_results/{gate_type}-"
                     f"{starting_length}-"
                     f"{(num_circuits - 1) * step + starting_length}-"
                     f"{step}-"
                     f"{num_shots / 1000}k-"
                     f"{num_qubits}q-"
                     f"{backend_name}.csv", index=False)
    # Draw the circuit
    # print(circuit.draw())
