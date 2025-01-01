from qiskit_ibm_runtime import QiskitRuntimeService

token = "7b94c8c063ab2df03edb1c0131b2970e026e4deef0eb1afea5a202197b63c755e32e0f067739908ffadfc200104a6e458e48ff9bfd4cd462e25e2a6ba45019f5"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
