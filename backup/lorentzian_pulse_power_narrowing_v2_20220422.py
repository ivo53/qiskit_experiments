import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
# from qiskit.tools.jupyter import *
from qiskit import IBMQ
from qiskit import pulse                  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter      # This is Parameter Class for variable parameters.
from qiskit.pulse import library as pulse_lib
from qiskit.scheduler import measure
from qiskit import assemble
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

## create folder where plots are saved
file_dir = os.path.dirname(__file__)
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
save_dir = os.path.join(
    file_dir, 
    "plots", 
    "power_broadening (narrowing)",
    "lorentz_pulses",
    current_date
)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
power_broadening_folder = os.path.join(
    file_dir, 
    "data", 
    "power_broadening (narrowing)"
)
data_folder = os.path.join(
    power_broadening_folder, 
    "lorentz_pulses", 
    current_date
)
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)

folder_name = os.path.join(save_dir, date.strftime("%H%M%S"))
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)


provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_armonk")
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"

dt = backend_config.dt
print(f"Sampling time: {dt*1e9} ns")

backend_defaults = backend.defaults()

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc.)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
# We will find the qubit frequency for the following qubit.
qubit = 0
# We will define memory slot channel 0.
mem_slot = 0

# The sweep will be centered around the estimated qubit frequency.
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
                                                                    # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

# scale factor to remove factors of 10 from the data
scale_factor = 1e-14

frequency_span_Hz = 14 * MHz
frequency_step_Hz = (1/2) * MHz

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(frequency_min / GHz, 
                            frequency_max / GHz, 
                            frequency_step_Hz / GHz)

amplitudes = np.arange(0., 0.92, 0.02)

print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
in steps of {frequency_step_Hz / MHz} MHz.")

print(f"The amplitude will go from {amplitudes[0]} to {amplitudes[-1]}.")

frequencies_Hz = frequencies_GHz * GHz

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

# Convert seconds to dt
def get_dt_from(sec):
    return get_closest_multiple_of_16(sec/dt)


# Drive pulse parameters (us = microseconds)
# drive_sigma_sec = 0.1 * us                   
# drive_duration_sec = drive_sigma_sec            
dur_dt = 5152
t_dt = np.arange(dur_dt)
G = 250
# Create the base schedule
# Start with drive pulse acting on the drive channel
freq = Parameter('freq')
amp = Parameter('amp')
# lor = Parameter('lor')
lor_norm = G * G / ((t_dt - dur_dt / 2) ** 2 + G ** 2)
with pulse.build(backend=backend, default_alignment='sequential', name="sq_2d") as sched:
    
    # drive_duration = get_closest_multiple_of_16(
    #     pulse.seconds_to_samples(drive_duration_sec)
    # )
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(freq, drive_chan)
    pulse.play(
        pulse_lib.Waveform(
            samples=lor_norm,
            amp=amp,
            name='lor_pulse'
        ),
        drive_chan
    )
    pulse.measure(
        qubits=[qubit], 
        registers=[pulse.MemorySlot(mem_slot)]
    )
    # Create the frequency settings for the sweep (MUST BE IN HZ)
frequencies_Hz = frequencies_GHz*GHz
schedules = [
    sched.assign_parameters(
        {freq: f, amp: a}, 
        inplace=False
    ) for f in frequencies_Hz for a in amplitudes

]
schedules[-2].draw(backend=backend)



num_schedules = len(schedules)
num_shots = 1024
# n_max_exp = 300
# n_jobs = 1 + num_schedules // n_max_exp
# n_jobs

job_manager = IBMQJobManager()

jobs = job_manager.run(
    schedules,
    backend=backend, 
    shots=num_shots,
    name="Lorentzian Power Narrowing"
)
results = jobs.results()
transition_probability = []
for i in range(num_schedules):
    transition_probability.append(results.get_counts(i)["1"] / num_shots)
transition_probability = np.array(transition_probability).reshape(len(frequencies_Hz), len(amplitudes))
job_set_id = jobs.job_set_id()
print("JobsID:", job_set_id)

## save final data
path_to_data_folder = os.path.join(
    data_folder, 
    date.strftime("%H%M%S")
)
if not os.path.isdir(path_to_data_folder):
    os.mkdir(path_to_data_folder)

y, x = np.meshgrid(amplitudes, (frequencies_Hz - center_frequency_Hz)/10**6)
# z = np.reshape(transition_probability, (len(frequencies_Hz),len(amplitudes)))

with open(os.path.join(path_to_data_folder, "tr_prob.pkl"), 'wb') as f1:
    pickle.dump(transition_probability, f1)
with open(os.path.join(path_to_data_folder, "areas.pkl"), 'wb') as f2:
    pickle.dump(amplitudes, f2)
with open(os.path.join(path_to_data_folder, "detunings.pkl"), 'wb') as f3:
    pickle.dump((frequencies_Hz - center_frequency_Hz), f3)

fig, ax = plt.subplots(figsize=(5,4))

c = ax.pcolormesh(x, y, transition_probability, vmin=0, vmax=1)
# ax.set_title('Uni3')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
ax.set_ylabel('Rabi Freq. [a.u.]')
ax.set_xlabel('Detuning [MHz]')
plt.savefig(        
    os.path.join(
        save_dir, 
        f"lorentz_pwr_nrw_duration-{dur_dt}dt_g-{G}.png"
    )
)
plt.show()