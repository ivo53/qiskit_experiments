from qiskit_ibm_runtime import QiskitRuntimeService

token = "026d40d3b330e2d5cbe20fafa27ef44db83119438568eb8506a54901bc6c11975b12e8180801d0862af2d19b427f0f2c06b56918ecc2546e4b88c5eb18af7d04"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
