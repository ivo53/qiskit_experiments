from qiskit_ibm_runtime import QiskitRuntimeService

token = "0f431d7b9c0326fcd8a649c288252f555279041566f2bcf875dba8ebce9c9449775d51223cf4215db7cfa96628b3e73e59dda2cd3f9d01a948dd4e9adf2601a1"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
