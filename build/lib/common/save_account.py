from qiskit_ibm_runtime import QiskitRuntimeService

token = "89ab1af85caccf23ac1e8e323f062fd40b524d812322521f056dde65c4372118fd7646edaff42fb321573774d154a6cfcfc1ae0fac5589c0aadfe17d5ddf6e73"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
