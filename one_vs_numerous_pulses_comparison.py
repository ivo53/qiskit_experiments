import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from qiskit import pulse, IBMQ, execute           # This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

sys.setrecursionlimit(10000)

file_dir = os.path.dirname(__file__)
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
calib_dir = os.path.join(file_dir, "plots", "one_vs_numerous_pulses_comparison")
save_dir = os.path.join(calib_dir, current_date)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
folder_name = os.path.join(save_dir, date.strftime("%H%M%S"))
# if not os.path.isdir(folder_name):
#     os.mkdir(folder_name)

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
num_dd_blocks = 30
# num_dd_blocks = np.arange(2, max_num_dd_blocks + 1, 2)
freq_span_frac = 0.025
min_freq = (1 - freq_span_frac / 2) * rough_qubit_frequency
max_freq = (1 + freq_span_frac / 2) * rough_qubit_frequency
num_exp = 50

frequencies = np.linspace(
    min_freq, 
    max_freq, 
    num_exp
)
pi_pulse_duration = 0.1 # us

pi_unit_duration = get_closest_multiple_of_16(pi_pulse_duration * us / dt)
half_pi_unit_duration = get_closest_multiple_of_16(pi_pulse_duration * us / (2 * dt))
min_delay_pulse_ratio = 10
# total_dd_duration = 2 * max_num_dd_blocks * \
#     pi_pulse_duration * (min_delay_pulse_ratio + 1)


# pi amplitude is for fixed duration (0.05 us) so adjust inv proportional
#amplitude_factor = fixed_duration / pi_pulse_duration
pi_amp = fixed_duration_amp # * amplitude_factor

num_shots_per_exp = 1024

# amplitude_multipliers = np.arange(1.054, 1.064001, 0.0002)

dd_schedules = []

half_pi_pulse = pulse_lib.constant(        
    duration=pi_unit_duration,
    amp=1.06 * pi_amp / 2,
    name='pi/2_square_pulse'
)
pi_pulse = pulse_lib.constant(        
    duration=pi_unit_duration,
    amp=1.06 * pi_amp,
    name='pi_square_pulse'
)
long_pulse = pulse_lib.constant(        
    duration=num_dd_blocks * pi_unit_duration,
    amp=1.06 * pi_amp,
    name='long_square_pulse'
)

dd_schedule1 = pulse.Schedule(name=f"{num_dd_blocks} pi pulses")
for _ in range(num_dd_blocks):
    dd_schedule1 += Play(pi_pulse, drive_chan)
dd_schedule1 += measure << dd_schedule1.duration

dd_schedule2 = pulse.Schedule(name=f"One long pulse")
dd_schedule2 += Play(long_pulse, drive_chan)
dd_schedule2 += measure << dd_schedule2.duration


# ## JOB MANAGER Approach
# job_manager = IBMQJobManager()

# print(len(dd_schedules))
# print(num_exp)
# jobs = job_manager.run(
#     dd_schedules, 
#     backend=backend, 
#     name="Pi Area Fine Calibration", 
#     shots=num_shots_per_exp,
#     schedule_los={drive_chan: rough_qubit_frequency}
# )
# results = jobs.results()
# transition_probability = []
# for i in range(num_exp):
#     transition_probability.append(results.get_counts(i)["1"] / num_shots_per_exp)
# job_set_id = jobs.job_set_id()
# ## 

## EXECUTE Approach
dd_job1 = execute(
    dd_schedule1,
    backend=backend,
    meas_level=2,
    memory=True,
    meas_return='single',
    shots=num_shots_per_exp,
    schedule_los=[{drive_chan: f} for f in frequencies],
    optimization_level=0
)
job_monitor(dd_job1)

dd_results1 = dd_job1.result(timeout=120)
transition_probability1 = []
for i in range(len(dd_results1.results)):
    # Get the results from the ith experiment
    length = len(dd_results1.get_memory(i))
    _, counts = np.unique(dd_results1.get_memory(i), return_counts=True)
    transition_probability1.append(counts[1] / length)
##
## EXECUTE Approach
dd_job2 = execute(
    dd_schedule2,
    backend=backend,
    meas_level=2,
    memory=True,
    meas_return='single',
    shots=num_shots_per_exp,
    schedule_los=[{drive_chan: f} for f in frequencies],
    optimization_level=0
)
job_monitor(dd_job2)

dd_results2 = dd_job2.result(timeout=120)
transition_probability2 = []
for i in range(len(dd_results2.results)):
    # Get the results from the ith experiment
    length = len(dd_results2.get_memory(i))
    _, counts = np.unique(dd_results2.get_memory(i), return_counts=True)
    transition_probability2.append(counts[1] / length)
##

plt.figure(1)
plt.plot((frequencies - rough_qubit_frequency) / GHz, transition_probability1, 'bo')
plt.xlabel("Detuning [GHz]")
plt.ylabel("Transition Probability")
plt.title(f"Transition Probability After {num_dd_blocks} Sequential Pi Pulses")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{num_dd_blocks}_pi_pulses_duration-{pi_pulse_duration}_freq_variation.png"))
# plt.show()

plt.figure(2)
plt.plot((frequencies - rough_qubit_frequency) / GHz, transition_probability2, 'ro')
plt.xlabel("Detuning [GHz]")
plt.ylabel("Transition Probability")
plt.title(f"Transition Probability After One Pulse with Duration of {num_dd_blocks} Pi Pulses")
plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_long_pulse_duration-{num_dd_blocks * pi_pulse_duration}_freq_variation.png"))
plt.show()
