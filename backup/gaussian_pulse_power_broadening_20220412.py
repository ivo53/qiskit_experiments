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

rough_qubit_frequency = 4.97187 * GHz
k = 5.4800222
pi_amp = np.pi / k
sq_pulse_duration = 0.5
duration = get_closest_multiple_of_16(0.05 * us / dt)

num_shots_per_exp = 1024
num_areas = 30
num_freq = 30

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
x = 10
sigma = get_closest_multiple_of_16(duration / x)
dur = get_closest_multiple_of_16(10 * duration / x)
pb_signals = []
for area in areas:
    gauss_with_area = pulse_lib.gaussian(    
        duration=dur,
        sigma=sigma,
        amp= (area / np.pi) * (0.5 * us * pi_amp / (sigma * np.sqrt(2 * np.pi))),
        name='gauss_pulse'
    )
    pb_schedule = pulse.Schedule(name=f"Power broadening with square pulse")
    pb_schedule += Play(gauss_with_area, drive_chan)
    pb_schedule += measure << pb_schedule.duration

    pb_qobj = assemble(
        pb_schedule,
        backend=backend,
        meas_level=1,
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
        results = pb_results.get_memory(i)*1e-14
        # Get the results for `qubit` from this experiment
        pb_values.append(results[qubit])

    pb_signals.append(pb_values)

pb_signals = np.array(pb_signals)
detunings = np.round((pb_freq - rough_qubit_frequency) / GHz, 4)
np.savetxt("./cache/gaussian_pb/pb_signals.txt", pb_signals)
np.savetxt("./cache/gaussian_pb/areas.txt", areas)
np.savetxt("./cache/gaussian_pb/detunings.txt", detunings)
plt.imshow(np.real(pb_signals), cmap='viridis', origin="lower")
plt.xticks(np.arange(len(detunings))[::4], np.round(detunings * 100)[::4])
plt.yticks(np.arange(len(areas))[::4], np.round(areas, 3)[::4])
plt.xlabel(r"Detuning $\Delta$ [0.01 GHz]")
plt.ylabel(r"Pulse area $\Omega$")
plt.title("Power broadening with Gaussian pulse")
plt.legend()
plt.show()