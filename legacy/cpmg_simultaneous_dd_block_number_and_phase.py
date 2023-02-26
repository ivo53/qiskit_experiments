import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import pulse, IBMQ, execute           # This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor

sys.setrecursionlimit(10000)

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

fixed_duration = 0.05
unit_duration = get_closest_multiple_of_16(fixed_duration * us / dt)
## set calibration constants
autoset = True # False # to use given values below
rough_qubit_frequency = 4.97195 * GHz
k = 11.16905011

if autoset:
    file_dir = os.path.dirname(__file__)
    calib_dir = os.path.join(file_dir, "calibrations")
    params_file = os.path.join(calib_dir, "params.csv")
    if os.path.isfile(params_file):
        params_df = pd.read_csv(params_file)
        params_dict = params_df.query(f"pi_duration == {unit_duration}").sort_values(by=["date", "time"]).to_dict("records")
        params = params_dict[-1]
        rough_qubit_frequency = params["drive_freq"]
        k = params["k"]
fixed_duration_amp = np.pi / k

## set run variables
pi_pulse_duration = 0.05 # us

pi_unit_duration = get_closest_multiple_of_16(pi_pulse_duration * us / dt)
half_pi_unit_duration = get_closest_multiple_of_16(pi_pulse_duration * us / (2 * dt))

max_num_dd_blocks = 100
min_delay_pulse_ratio = 10
total_dd_duration = 2 * max_num_dd_blocks * \
    pi_pulse_duration * (min_delay_pulse_ratio + 1)

span = 0.001 * rough_qubit_frequency
num_phase_exp = 100
num_shots_per_exp = 1024

phases = np.linspace(0, 2 * np.pi, num_phase_exp)
num_dd_blocks = np.arange(0, max_num_dd_blocks + 1, 5)

# pi amplitude is for fixed duration (0.5 us) so adjust inv proportional
amplitude_factor = fixed_duration / pi_pulse_duration
pi_amp = fixed_duration_amp * amplitude_factor


half_pi_pulse = pulse_lib.constant(        
    duration=pi_unit_duration,
    amp=pi_amp / 2,
    name='pi/2_square_pulse'
)
pi_pulse = pulse_lib.constant(        
    duration=pi_unit_duration,
    amp=pi_amp,
    name='pi_square_pulse'
)

dd_signals = []
for j, n in enumerate(num_dd_blocks):
    schedules = []
    for phi in phases:
        ## adjust delays to keep total DD duration fixed
        if n == 0:
            delay = total_dd_duration
        else:
            delay = (total_dd_duration / (2 * n) - pi_pulse_duration)
        tau = get_closest_multiple_of_16(delay * us / dt)
        half_tau = get_closest_multiple_of_16(delay * us / (2 * dt))

        half_pi_pulse_with_phase = pulse_lib.constant(        
            duration=pi_unit_duration,
            amp=pi_amp * np.exp(1.j * phi)/ 2,
            name='pi/2_square_pulse'
        )
        dd_schedule = pulse.Schedule(name=f"{n} CPMG blocks")
        dd_schedule += Play(half_pi_pulse, drive_chan)
        if n == 0:
            dd_schedule += Delay(tau, drive_chan)
        for _ in range(n):
            dd_schedule += Delay(half_tau, drive_chan)
            dd_schedule += Play(pi_pulse, drive_chan)
            dd_schedule += Delay(tau, drive_chan)
            dd_schedule += Play(pi_pulse, drive_chan)
            dd_schedule += Delay(half_tau, drive_chan)
        dd_schedule += Play(half_pi_pulse_with_phase, drive_chan)
        dd_schedule += measure << dd_schedule.duration
        schedules.append(dd_schedule)

    dd_job = execute(
        schedules,
        backend=backend,
        meas_level=2,
        memory=True,
        meas_return='single',
        shots=num_shots_per_exp,
        schedule_los=[{drive_chan: rough_qubit_frequency}] * num_phase_exp
    )

    job_monitor(dd_job)

    dd_results = dd_job.result(timeout=120)
    dd_values = []
    for i in range(len(dd_results.results)):
        # Get the results from the ith experiment
        # results = dd_results.get_memory(i)*1e-14
        length = len(dd_results.get_memory(i))
        _, counts = np.unique(dd_results.get_memory(i), return_counts=True)
        # Get the results for `qubit` from this experiment
        dd_values.append(counts[1] / length)



    plt.figure(j)
    plt.scatter(phases, np.real(dd_values), color='black') # plot real part of sweep values
    # plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    plt.xlabel("Phase [radians]")
    plt.ylabel("Measured signal [a.u.]")
    plt.title(f"Phase variation with {n * 2} pi pulses in CPMG")
    plt.savefig(f"./plots/cpmg/num_blocks_and_phases/{n}_blocks_phase_variation.png")
    #plt.show()
    plt.close()
    dd_signals.append(np.real(dd_values))

dd_signals = np.array(dd_signals)

# num_dd_blocks = 2 ** np.arange(8)
# phases = np.linspace(0, 2 * np.pi, 10)
# dd_signals = np.random.random((8, 10)).T
fig, ax = plt.subplots()
im = ax.imshow(dd_signals, cmap='viridis', origin="lower")
plt.xticks(np.arange(len(num_dd_blocks)), num_dd_blocks)
plt.yticks(np.arange(len(phases)), np.round(phases, 2))
plt.xlabel("Number of CPMG blocks")
plt.ylabel("Phase [radians]")
plt.title("|1> population for varied numbers of CPMG blocks and phases")
plt.show()