from qiskit_ibm_runtime import QiskitRuntimeService

token = "272fe59b094aaaff2fbea6f6ca80fd6c1edc0e2ca8b38ebd0bc0a546964cea99f31d6cf7c6a298c1c58e04852565cb80a598006a59c9f8857b5534b8cbfbb472"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
