import os
import pickle
from datetime import datetime, timedelta

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
# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)
def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_armonk")
backend_config = backend.configuration()
assert backend_config.open_pulse, "Backend doesn't support Pulse"

dt = backend_config.dt
print(f"Sampling time: {dt*1e9} ns")

backend_defaults = backend.defaults()
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
qubit = 0

backend_name = "armonk"
# cutparam = .2 # %
# G = 150
# dur_dt = 2 * G * np.sqrt(100 / cutparam - 1)
# dur_dt = get_closest_multiple_of_16(dur_dt)
cutparam = .6 # %
G = 400
dur_dt = 2 * G * np.sqrt(100 / cutparam - 1)
dur_dt = get_closest_multiple_of_16(dur_dt)

date = datetime.now() #- timedelta(days=2)
current_date = date.strftime("%Y-%m-%d")
file_dir = os.path.dirname(__file__)
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
make_all_dirs(folder_name)
make_all_dirs(data_folder)

varied = "vary_duration"
area_cal_folder = os.path.join(file_dir, "data", backend_name, "calibration", varied, current_date)

data_files = os.listdir(area_cal_folder)
vals = []
for d in data_files:
    if d.startswith(f"{dur_dt}dt_cutparam-{cutparam}%_G"):
        if d.endswith("tr_prob.pkl"):
            vals.append(d)
assert len(vals) == 1, "Only one file should be found"
fitparams_folder = os.path.join(file_dir, "data", backend_name, "fits", varied, current_date)
with open(os.path.join(fitparams_folder, vals[0].replace("_tr_prob", "") ), "rb") as f3:
    fitparams = pickle.load(f3)
_, l, p, _ = fitparams


## new fit function
a_pi = - np.log(1 - np.pi / l) / p
a_3pi = - np.log(1 - 3*np.pi / l) / p
a_5pi = - np.log(1 - 5*np.pi / l) / p
a_half = - np.log(1 - np.pi / (2 * l)) / p
print(a_3pi,a_5pi,a_pi, a_half, l, p)
print(dur_dt)

num_schedules = 1000
num_shots = 4096

center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]        # The default frequency is given in Hz
a_max = a_5pi
num_amps = 5

frequency_span_Hz = 50 * MHz #5 * MHz #if cut_param < 1 else 1.25 * MHz
frequency_step_Hz = np.round(frequency_span_Hz / 199, 3) #(1/4) * MHz

max_experiments_per_job = 25

# We will sweep 20 MHz above and 20 MHz below the estimated frequency
frequency_min = center_frequency_Hz - frequency_span_Hz / 2
frequency_max = center_frequency_Hz + frequency_span_Hz / 2
# Construct an np array of the frequencies for our experiment
frequencies_GHz = np.arange(
    frequency_min / GHz, 
    frequency_max / GHz + 1e-5, 
    frequency_step_Hz / GHz
)
frequencies_Hz = frequencies_GHz * GHz

# amplitudes = np.linspace(0., a_max, num_amps).round(3)
amplitudes = [0, a_half, a_pi, a_3pi, a_5pi]

idx_half = np.argmin(
    np.abs(amplitudes - a_half)
)
idx_pi = np.argmin(
    np.abs(amplitudes - a_pi)
)
idx_3pi = np.argmin(
    np.abs(amplitudes - a_3pi)
)
idx_5pi = np.argmin(
    np.abs(amplitudes - a_5pi)
)
assert idx_half != idx_pi
assert idx_pi != idx_3pi
assert idx_3pi != idx_5pi
amplitudes[idx_half] = a_half
amplitudes[idx_pi] = a_pi
amplitudes[idx_3pi] = a_3pi
amplitudes[idx_5pi] = a_5pi


jobs_id = "3232b979ca3d464cb209c3185d1d83cf-16559884561918826"

job_manager = IBMQJobManager()
jobs = job_manager.retrieve_job_set(jobs_id, provider=provider)

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
    plt.title(f"Lorentzian Freq Offset - Amplitude {np.round(am, 3)}")
    if i == idx_half:
        plt.savefig(os.path.join(folder_name, f"lor_cutparam-{cutparam}_amp-{np.round(am, 3)}_half-pi.png").replace("\\","/"))
    elif i == idx_pi:
        plt.savefig(os.path.join(folder_name, f"lor_cutparam-{cutparam}_amp-{np.round(am, 3)}_pi.png").replace("\\","/"))
    else:
        plt.savefig(os.path.join(folder_name, f"lor_cutparam-{cutparam}_amp-{np.round(am, 3)}.png").replace("\\","/"))
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