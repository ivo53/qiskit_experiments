import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import pulse, IBMQ, assemble           # This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
qubit = 0

drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)

provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_armonk")
# backend = provider.get_backend("ibmq_qasm_simulator")
backend_name = str(backend)
print(f"Using {backend_name} backend.")
backend_defaults = backend.defaults()
backend_config = backend.configuration()
dt = backend_config.dt

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

meas_map_idx = None
for i, measure_group in enumerate(backend_config.meas_map):
    if qubit in measure_group:
        meas_map_idx = i
        break
assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
inst_sched_map = backend_defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

rough_qubit_frequency = 4.97171 * GHz
k = 56.1210057
pi_amp = np.pi / k 

duration = get_closest_multiple_of_16(0.5 * us / dt)
half_duration = get_closest_multiple_of_16(0.5 * us / (2 * dt))

num_shots_per_exp = 1024
num_areas = 20
num_freq = 20

min_area, max_area = 0, 9 * np.pi
areas = np.linspace(
    min_area,
    max_area,
    num_areas
)
freq_span = 5.e7 # Hz
pb_freq = np.linspace(
    rough_qubit_frequency - freq_span / 2,
    rough_qubit_frequency + freq_span / 2,
    num_freq
)
pb_frequencies = [{drive_chan: freq} for freq in pb_freq]

pb_signals = []
for area in areas:
    pulse_with_area = pulse_lib.constant(    
        duration=duration,
        amp=area * pi_amp / np.pi,
        name='square_pulse'
    )
    pb_schedule = pulse.Schedule(name=f"Power broadening with square pulse")
    pb_schedule += Play(pulse_with_area, drive_chan)
    pb_schedule += measure << pb_schedule.duration

    pb_qobj = assemble(
        pb_schedule,
        backend=backend,
        meas_level=2,
        memory=True,
        meas_return='avg',
        shots=num_shots_per_exp,
        schedule_los=pb_frequencies
    )

    pb_job = backend.run(pb_qobj)
    job_monitor(pb_job)

    pb_results = pb_job.result(timeout=120)
    pb_values = []
    for i in range(len(pb_results.results)):
        # Get the results from the ith experiment
        # print(pb_results.get_memory(i))
        length = len(pb_results.get_memory(i))
        # results = pb_results.get_memory(i)*1e-14
        _, counts = np.unique(pb_results.get_memory(i), return_counts=True)
        pb_values.append(counts[1] / length)

    pb_signals.append(pb_values)

pb_signals = np.array(pb_signals)
detunings = np.round((pb_freq - rough_qubit_frequency) / GHz, 4)
np.savetxt("./cache/sq_pb/pb_signals.txt", pb_signals)
np.savetxt("./cache/sq_pb/areas.txt", areas)
np.savetxt("./cache/sq_pb/detunings.txt", detunings)
plt.imshow(np.real(pb_signals), cmap='viridis', origin="lower")
plt.xticks(np.arange(len(detunings))[::2], np.round(detunings * 100)[::2])
plt.yticks(np.arange(len(areas))[::2], np.round(areas, 3)[::2])
plt.xlabel(r"Detuning $\Delta$ [0.01 GHz]")
plt.ylabel(r"Pulse area $\Omega$")
plt.title("Power broadening with square pulse")
plt.legend()
plt.show()