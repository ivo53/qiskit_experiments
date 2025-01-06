from qiskit_ibm_runtime import QiskitRuntimeService

token = "6509a22fbce7376e8a2f48c952df448ad5bdc664061e55886887d68ae3c4283793eae37cf5a39ea36671e3f64f5d783b3f825a3444ff8604a2e9c09e31d68633"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
