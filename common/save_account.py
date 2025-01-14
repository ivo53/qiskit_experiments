from qiskit_ibm_runtime import QiskitRuntimeService

token = "50c94178c3193c95491c6d64675a0650f4e7457eea7f23b0799bd043d17c943a4368b24149cf8be172599a0b1ed6c0a610ef5a3672f9eea1f8116bb71f079ed0"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
