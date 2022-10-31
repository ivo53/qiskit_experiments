import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
# from qiskit.tools.jupyter import *
from qiskit import IBMQ
from qiskit import pulse                  # This is where we access all of our Pulse features!
from qiskit.circuit import Parameter      # This is Parameter Class for variable parameters.
from qiskit.pulse import library as pulse_lib
from qiskit.scheduler import measure
from qiskit import assemble
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

backend_name = "armonk"
varied = "vary_width"
dur_dt = 2256 * 5
cutparam = .2 # %
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
file_dir = os.path.dirname(__file__)
area_cal_folder = os.path.join(file_dir, "data", backend_name, "calibration", varied, current_date)

data_files = os.listdir(area_cal_folder)
vals = []
for d in data_files:
    if d.startswith(f"{dur_dt}dt_cutparam-{cutparam}%_G"):
        if d.endswith("tr_prob.pkl"):
            vals.append(d)
assert len(vals) == 1, f"Exactly one file should be found, instead found {len(vals)}"
fitparams_folder = os.path.join(file_dir, "data", backend_name, "fits", "vary_width", current_date)
with open(os.path.join(fitparams_folder, vals[0].replace("_tr_prob", "") ), "rb") as f3:
    fitparams = pickle.load(f3)
_, l, p, _ = fitparams

## new fit function
a_pi = - np.log(1 - np.pi / l) / p
a_half = - np.log(1 - np.pi / (2 * l)) / p
a_5pi = - np.log(1 - 5 * np.pi / l) / p
a_3pi = - np.log(1 - 3 * np.pi / l) / p
##
## oldest fit function
# # a_pi = ((l - np.sqrt(l**2 + 4*k*np.pi)) / (2*k))**2
# # a_half = ((l - np.sqrt(l**2 + 2*k*np.pi)) / (2*k))**2
## old fit function
# a_pi = (-l * p + p * np.pi + k * lambertw(np.float64(l * p * np.exp(p * (l - np.pi) / k) / k, tol=1e-8))) / (k * p) #0.0508567
# a_half = (-2 * l * p + p * np.pi + 2 * k * lambertw(np.float64(l * p * np.exp(p * (2 * l - np.pi) / (2 * k)) / k, tol=1e-8))) / (2 * k * p)#0.0252243
##

## create folder where plots are saved
date = datetime.now()
current_date = date.strftime("%Y-%m-%d")
save_dir = os.path.join(
    file_dir,
    "plots", 
    backend_name, 
    "power_broadening (narrowing)",
    "lorentz_pulses",
    current_date
)
folder_name = os.path.join(
    save_dir,
    date.strftime("%H%M%S")
).replace("\\", "/")
data_folder = os.path.join(
    file_dir,
    "data",
    backend_name,
    "power_broadening (narrowing)",
    "lorentz_pulses",
    current_date,
    date.strftime("%H%M%S")
).replace("\\", "/")
make_all_dirs(data_folder)
make_all_dirs(folder_name)


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

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

# Convert seconds to dt
def get_dt_from(sec):
    return get_closest_multiple_of_16(sec/dt)

# The sweep will be centered around the estimated qubit frequency.
center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
                                                                    # warning: this will change in a future release
print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")


a_max = a_pi
num_amps = 3

frequency_span_Hz = 30 * MHz #5 * MHz #if cut_param < 1 else 1.25 * MHz
frequency_step_Hz = np.round(frequency_span_Hz / 99, 3) #(1/4) * MHz

max_experiments_per_job = 20

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(
    frequency_min / GHz, 
    frequency_max / GHz + 1e-5, 
    frequency_step_Hz / GHz
)

# amplitudes = np.linspace(0., a_max, num_amps).round(3)
amplitudes = np.array([0, a_half, a_pi, a_3pi, a_5pi])
# print(a_pi, a_half)
# S_pi = np.real(a_pi) * dur_pi
# S_half = np.real(a_half) * dur_pi
# # S_pi = dur_pi * np.pi / k

G = dur_dt / (2 * np.sqrt(100 / cutparam - 1))
idx_half = np.argmin(
    np.abs(amplitudes - a_half)
)
idx_pi = np.argmin(
    np.abs(amplitudes - a_pi)
)
assert idx_half != idx_pi
amplitudes[idx_half] = a_half
amplitudes[idx_pi] = a_pi

print(f"The gamma factor is G = {G} and cut param is {cutparam}")#, \
# compared to G_02 = {G_02} at cut param 0.2.")
print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
in steps of {frequency_step_Hz / MHz} MHz (total of {len(frequencies_GHz)} values).")
print(f"The amplitude will go from {amplitudes[0]} to {amplitudes[-1]} in approx steps of {a_max / (num_amps - 1)}:")
print(amplitudes)
frequencies_Hz = frequencies_GHz * GHz
# exit()
# Create the base schedule
# Start with drive pulse acting on the drive channel
freq = Parameter('freq')
amp = Parameter('amp')
with pulse.build(backend=backend, default_alignment='sequential', name="sq_2d") as sched:
    
    drive_chan = pulse.drive_channel(qubit)
    pulse.set_frequency(freq, drive_chan)
    pulse.play(
        pulse_lib.Lorentzian(
            duration=dur_dt,
            amp=amp,
            gamma=G,
            name='lor_pulse',
            zero_ends=False
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


num_schedules = len(schedules)
num_shots = 4096

job_manager = IBMQJobManager()

jobs = job_manager.run(
    schedules,
    backend=backend, 
    shots=num_shots,
    max_experiments_per_job=max_experiments_per_job,
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
freq_offset = (frequencies_Hz - center_frequency_Hz) / 10**6
y, x = np.meshgrid(amplitudes, freq_offset)
# z = np.reshape(transition_probability, (len(frequencies_Hz),len(amplitudes)))

with open(os.path.join(data_folder, f"{dur_dt}dt_G-{G}_tr_prob.pkl").replace("\\","/"), 'wb') as f1:
    pickle.dump(transition_probability, f1)
with open(os.path.join(data_folder, f"{dur_dt}dt_G-{G}_areas.pkl").replace("\\","/"), 'wb') as f2:
    pickle.dump(amplitudes, f2)
with open(os.path.join(data_folder, f"{dur_dt}dt_G-{G}_detunings.pkl").replace("\\","/"), 'wb') as f3:
    pickle.dump((frequencies_Hz - center_frequency_Hz), f3)

for i, am in enumerate(amplitudes):
    plt.figure(i)
    plt.plot(freq_offset, transition_probability[:, i], "bx")
    plt.xlabel("Detuning [MHz]")
    plt.ylabel("Transition Probability")
    plt.title(f"Lorentzian Freq Offset - Amplitude {am.round(3)}")
    if i == idx_half:
        plt.savefig(os.path.join(folder_name, f"lor_cutparam-{cutparam}_amp-{am.round(3)}_half-pi.png").replace("\\","/"))
    elif i == idx_pi:
        plt.savefig(os.path.join(folder_name, f"lor_cutparam-{cutparam}_amp-{am.round(3)}_pi.png").replace("\\","/"))
    else:
        plt.savefig(os.path.join(folder_name, f"lor_cutparam-{cutparam}_amp-{am.round(3)}.png").replace("\\","/"))
    plt.close()

fig, ax = plt.subplots(figsize=(5,4))

c = ax.pcolormesh(x, y, transition_probability, vmin=0, vmax=1)
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)
ax.set_ylabel('Rabi Freq. [a.u.]')
ax.set_xlabel('Detuning [MHz]')
plt.axhline(y = 0.5, color = 'w', linestyle = '--')
plt.axhline(y = 0.9, color = 'w', linestyle = '--')
plt.savefig(
    os.path.join(
        save_dir, 
        f"{date.strftime('%H%M%S')}_lorentz_pwr_nrw_duration-{dur_dt}dt_g-{G}_cutparam-{cutparam}.png"
    ).replace("\\","/")
)
plt.show()