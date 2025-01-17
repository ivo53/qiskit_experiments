from qiskit_ibm_runtime import QiskitRuntimeService

token = "e76c4fcdab093929f59076765b56b6b0ccb72a742453898d8848d60f3751b6d92b5faff36ac233f256b966ef598a7682fe74ad705e54d12a8927241666b99132"

QiskitRuntimeService.save_account(token, channel="ibm_quantum", instance='ibm-q/open/main', overwrite=True, set_as_default=True)
