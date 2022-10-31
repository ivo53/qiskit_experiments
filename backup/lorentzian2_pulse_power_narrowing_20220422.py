import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import pulse, IBMQ, execute, assemble  # This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor

def create_folder(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

## create folder where plots are saved
file_dir = os.path.dirname(__file__)
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
create_folder(
    os.path.join(
        file_dir, 
        "plots"
    )
)
create_folder(
    os.path.join(
        file_dir, 
        "plots",
        "power_broadening (narrowing)"
    )
)
save_dir = os.path.join(
    file_dir, 
    "plots", 
    "power_broadening (narrowing)",
    "lorentz2_pulses",
    current_date
)
create_folder(save_dir)
power_broadening_folder = os.path.join(
    file_dir, 
    "data", 
    "power_broadening (narrowing)"
)
create_folder(
    os.path.join(
        power_broadening_folder, 
        "lorentz2_pulses"
    )
)
data_folder = os.path.join(
    power_broadening_folder, 
    "lorentz2_pulses", 
    current_date
)
create_folder(data_folder)
folder_name = os.path.join(
    save_dir, 
    date.strftime("%H%M%S")
)
create_folder(folder_name)
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

rough_qubit_frequency = 4.97173 * GHz
k = 13.93332379 #7.74532549
pi_amp = np.pi / k
G = 350 #100

num_shots_per_exp = 1024
num_areas = 100
num_freq = 100

min_area, max_area = 0, 4.4 * np.pi
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
detunings = np.round((pb_freq - rough_qubit_frequency) / GHz, 4)
pb_frequencies = [{drive_chan: freq} for freq in pb_freq]

dur_dt = 2576
t_dt = np.arange(dur_dt)

pb_signals = []
for j, area in enumerate(areas):
    amp = area * pi_amp / np.pi
    if dur_dt == 0 or amp == 0:
        pb_schedule = pulse.Schedule(name=f"Lorentzian^2 Power Narrowing")
        pb_schedule += measure << pb_schedule.duration
    else:
        lor2_amps = amp * G * G * (G / ((t_dt - dur_dt / 2) ** 2 + G ** 2)) ** 2
        lor2 = pulse_lib.Waveform(lor2_amps, name="Lorentzian^2 Pulse")
        pb_schedule = pulse.Schedule(name=f"Lorentzian^2 Power Narrowing")
        pb_schedule += Play(lor2, drive_chan)
        pb_schedule += measure << pb_schedule.duration

    pb_job = execute(
        pb_schedule,
        backend=backend,
        meas_level=2,
        memory=True,
        meas_return='single',
        shots=num_shots_per_exp,
        schedule_los=pb_frequencies
    )
    # pb_job = backend.run(pb_qobj)
    job_monitor(pb_job)

    pb_results = pb_job.result(timeout=120)
    pb_values = []
    for i in range(len(pb_results.results)):
        # Get the results from the ith experiment
        length = len(pb_results.get_memory(i))
        _, counts = np.unique(pb_results.get_memory(i), return_counts=True)
        pb_values.append(counts[1] / length)

    plt.figure(j)
    plt.scatter(detunings, pb_values, color='black')
    plt.xlabel("Detuning")
    plt.ylabel("Transition Probability")
    plt.title(f"Lorentzian^2 Pulse with Rabi Amp {amp}")
    plt.savefig(
        os.path.join(
            save_dir, 
            date.strftime("%H%M%S"), 
            f"duration-{dur_dt}dt_amp-{np.round(amp, 5)}.png"
        )
    )
    plt.close()

    pb_signals.append(pb_values)

pb_signals = np.array(pb_signals)
path_to_data_folder = os.path.join(
    data_folder, 
    date.strftime("%H%M%S")
)
if not os.path.isdir(path_to_data_folder):
    os.mkdir(path_to_data_folder)

np.savetxt(
    os.path.join(
        path_to_data_folder, 
        "signals.txt"
), pb_signals)
np.savetxt(
    os.path.join(
        path_to_data_folder, 
        "areas.txt"
), areas)
np.savetxt(
    os.path.join(
        path_to_data_folder, 
        "detunings.txt"
), detunings)

im = plt.imshow((pb_signals), cmap='viridis', origin="lower")
plt.xticks(np.arange(len(detunings))[::10], np.round(detunings * 100)[::10])
plt.yticks(np.arange(len(areas))[::10], np.round(areas, 3)[::10])
plt.xlabel(r"Detuning $\Delta$ [0.01 GHz]")
plt.ylabel(r"Pulse area $\Omega$")
plt.title("Lorentzian^2 Power Narrowing")
bar = plt.colorbar(im)
bar.set_label('Transition Probability')
plt.savefig(        
    os.path.join(
        save_dir, 
        f"lorentz2_pwr_nrw_duration-{dur_dt}dt_g-{G}.png"
    )
)
plt.show()