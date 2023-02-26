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
calib_dir = os.path.join(file_dir, "plots", "many_half_pulses_amp_vary")
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
num_dd_blocks = 100
# num_dd_blocks = np.arange(2, max_num_dd_blocks + 1, 2)
min_amp = .98
max_amp = 1.015
amp_span = 0.0001
amplitude_multipliers = np.arange(
    min_amp, 
    max_amp + 1.e-5, 
    amp_span
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

num_exp = len(amplitude_multipliers)
# amplitude_multipliers = np.arange(1.054, 1.064001, 0.0002)

dd_schedules = []

for j, a in enumerate(amplitude_multipliers):

    half_pi_pulse = pulse_lib.constant(        
        duration=pi_unit_duration,
        amp=a * pi_amp / 2,
        name='pi/2_square_pulse'
    )
    pi_pulse = pulse_lib.constant(        
        duration=pi_unit_duration,
        amp=a * pi_amp,
        name='pi_square_pulse'
    )


    dd_schedule = pulse.Schedule(name=f"{num_dd_blocks} half pi pulses")
    # dd_schedule += Play(half_pi_pulse, drive_chan)
    # if n == 0:
        # dd_schedule += Delay(tau, drive_chan)
    for _ in range(num_dd_blocks):
        dd_schedule += Play(half_pi_pulse, drive_chan)
        # dd_schedule += Delay(half_tau, drive_chan)
    # dd_schedule += Play(half_pi_pulse, drive_chan)
    dd_schedule += measure << dd_schedule.duration
    dd_schedules.append(dd_schedule)

# ## JOB MANAGER Approach
# job_manager = IBMQJobManager()

# print(len(dd_schedules))
# print(num_exp)
# jobs = job_manager.run(
#     dd_schedules, 
#     backend=backend, 
#     name="Half Pi Area Fine Calibration", 
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
dd_job = execute(
    dd_schedules,
    backend=backend,
    meas_level=2,
    memory=True,
    meas_return='single',
    shots=num_shots_per_exp,
    schedule_los=[{drive_chan: rough_qubit_frequency}] * num_exp,
    optimization_level=0
)
job_monitor(dd_job)

dd_results = dd_job.result(timeout=120)
transition_probability = []
for i in range(len(dd_results.results)):
    # Get the results from the ith experiment
    # results = dd_results.get_memory(i)*1e-14
    length = len(dd_results.get_memory(i))
    _, counts = np.unique(dd_results.get_memory(i), return_counts=True)
    # Get the results for `qubit` from this experiment
    transition_probability.append(counts[1] / length)
##

plt.figure(1)
plt.plot(amplitude_multipliers, transition_probability, 'bo')
plt.xlabel("Amplitude Multipliers")
plt.ylabel("Transition Probability")

plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{num_dd_blocks}_half_pulses_duration-{pi_pulse_duration}_amp_variation.png"))
plt.show()

# dd_signals.append(np.real(dd_values))

# dd_signals = np.array(dd_signals)
# print(pi_amp)

# values = np.unique(dd_signals.ravel())

# # plt.figure(len(amplitude_multipliers))
# fig, ax = plt.subplots()
# im = ax.imshow(dd_signals, cmap='viridis', origin="lower", interpolation='none')
# plt.xticks(np.arange(len(num_dd_blocks)), num_dd_blocks * 2)
# plt.yticks(np.arange(len(amplitude_multipliers)), np.round(amplitude_multipliers, 4))
# plt.xlabel("Number of Pi pulses")
# plt.ylabel("Amplitude [a.u.]")
# plt.title("Transition probability for varied CPMG blocks numbers and amplitudes")
# bar = plt.colorbar(im)
# bar.set_label('Excited state population')
# plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_duration-{pi_pulse_duration}_block_variation.png"))
# plt.show()