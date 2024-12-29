from qiskit_ibm_runtime import QiskitRuntimeService

token = "e5d2612aaf2d8c180898e41fffc7d3a9e494b6667cfb894b1cc438a6bff8f0202340a30ae2a814645ebcccebc8f529570074006e517ccfb7f14946d687c3a78f"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
