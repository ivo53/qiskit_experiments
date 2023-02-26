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
span = 0.001 * rough_qubit_frequency

num_exp = 100
num_shots_per_exp = 1024
num_dd_blocks = 10

half_pi_pulse = pulse_lib.constant(        
    duration=duration,
    amp=pi_amp / 2,
    name='pi/2_square_pulse'
)
pi_pulse = pulse_lib.constant(        
    duration=duration,
    amp=pi_amp,
    name='pi_square_pulse'
)

delays = np.linspace(0, 10 * duration, num_exp)
phases = np.linspace(0, 2 * np.pi, num_exp)
schedules = []
for phi in phases:
    half_pi_pulse_with_phase = pulse_lib.constant(        
        duration=duration,
        amp=pi_amp * np.exp(1.j * phi)/ 2,
        name='pi/2_square_pulse'
    )
    ramsey_schedule = pulse.Schedule(name=f"Ramsey fringe experiment")
    ramsey_schedule += Play(half_pi_pulse, drive_chan)
    # ramsey_schedule += Delay(half_duration, drive_chan)
    for _ in range(num_dd_blocks):
        ramsey_schedule += Delay(half_duration, drive_chan)
        ramsey_schedule += Play(pi_pulse, drive_chan)
        ramsey_schedule += Delay(duration, drive_chan)
        ramsey_schedule += Play(pi_pulse, drive_chan)
        ramsey_schedule += Delay(half_duration, drive_chan)
    # ramsey_schedule += Delay(half_duration, drive_chan)
    ramsey_schedule += Play(half_pi_pulse_with_phase, drive_chan)
    ramsey_schedule += measure << ramsey_schedule.duration
    schedules.append(ramsey_schedule)

ramsey_qobj = assemble(
    schedules,
    backend=backend,
    meas_level=1,
    meas_return='avg',
    shots=num_shots_per_exp,
    schedule_los=[{drive_chan: rough_qubit_frequency}] * num_exp
)

ramsey_job = backend.run(ramsey_qobj)
job_monitor(ramsey_job)

ramsey_results = ramsey_job.result(timeout=120)
ramsey_values = []
for i in range(len(ramsey_results.results)):
    # Get the results from the ith experiment
    results = ramsey_results.get_memory(i)*1e-14
    # Get the results for `qubit` from this experiment
    ramsey_values.append(results[qubit])



plt.figure(1)
plt.scatter(phases, np.real(ramsey_values), color='black') # plot real part of sweep values
# plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Phase [radians]")
plt.ylabel("Measured signal [a.u.]")

plt.show()