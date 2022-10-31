import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import curve_fit
from qiskit import pulse, IBMQ, execute           # This is where we access all of our Pulse features!
from qiskit.pulse import Delay,Play
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor
from qiskit.test.mock import FakeArmonk

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds
qubit = 0
scale_factor = 1.e-14

drive_chan = pulse.DriveChannel(qubit)
meas_chan = pulse.MeasureChannel(qubit)
acq_chan = pulse.AcquireChannel(qubit)

provider = IBMQ.load_account()
backend = provider.get_backend("ibmq_armonk")
# backend = FakeArmonk()
backend_name = str(backend)
print(f"Using {backend_name} backend.")
backend_defaults = backend.defaults()
backend_config = backend.configuration()
dt = backend_config.dt

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)
    
def get_job_data(job, average):
    """Retrieve data from a job that has already run.
    Args:
        job (Job): The job whose data you want.
        average (bool): If True, gets the data assuming data is an average.
                        If False, gets the data assuming it is for single shots.
    Return:
        list: List containing job result data. 
    """
    job_results = job.result(timeout=120) # timeout parameter set to 120 s
    result_data = []
    for i in range(len(job_results.results)):
        if average: # get avg data
            result_data.append(job_results.get_memory(i)[qubit]*scale_factor) 
        else: # get single data
            result_data.append(job_results.get_memory(i)[:, qubit]*scale_factor)  
    return result_data
 
# measurement 
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
drive_sigma = 0.05 # us
drive_samples = 8 * drive_sigma # us
drive_amp = 0.1 
drive_sigma = get_closest_multiple_of_16(drive_sigma * us / dt)
drive_samples = get_closest_multiple_of_16(drive_samples * us/ dt)

num_shots_per_exp = 1024

pi_pulse = pulse_lib.constant(
    duration=duration,
    amp=pi_amp,
    name='pi_square_pulse'
)

# sideband pulse
#num_freqs = 75
freq = rough_qubit_frequency + np.linspace(-400*MHz, 30*MHz, num_freqs)
t_samples = np.linspace(0, dt*drive_samples, drive_samples)
sine_pulse = np.sin(2*np.pi*(freq-rough_qubit_frequency)*t_samples) # no amp for the sine
sideband_pulse = pulse_lib.Waveform(
    np.multiply(
        np.real(
            pulse.samples
        ), 
        sine_pulse
    ),
    name='sideband_pulse'
)

# Ground state schedule
zero_schedule = pulse.Schedule(name="zero schedule")
zero_schedule |= measure

# Excited state schedule
one_schedule = pulse.Schedule(name="one schedule")
one_schedule |= Play(pi_pulse, drive_chan)
one_schedule |= measure << one_schedule.duration

# Third state schedule
two_schedule = pulse.Schedule(name="two schedule")
two_schedule |= Play(pi_pulse, drive_chan)
two_schedule |= Play(sideband_pulse, drive_chan)
two_schedule |= measure << two_schedule.duration

discr_job = execute(
    [zero_schedule, one_schedule],
    backend=backend,
    meas_level=1,
    meas_return='single',
    shots=num_shots_per_exp,
    schedule_los=[{drive_chan: rough_qubit_frequency}] * 2
)

# discr_job = backend.run(discr_qobj)
job_monitor(discr_job)

print(discr_job.job_id())

discr_data = get_job_data(discr_job, average=False)
zero_data = discr_data[0]
one_data = discr_data[1]
two_data = discr_data[2]
# print(zero_data, one_data)

def clusters_01_plot(x_min, x_max, y_min, y_max):
    """Helper function for plotting IQ plane for |0>, |1>. Limits of plot given
    as arguments."""
    # zero data plotted in blue
    plt.scatter(np.real(zero_data), np.imag(zero_data), 
                    s=5, cmap='viridis', c='blue', alpha=0.5, label=r'$|0\rangle$')
    # one data plotted in red
    plt.scatter(np.real(one_data), np.imag(one_data), 
                    s=5, cmap='viridis', c='red', alpha=0.5, label=r'$|1\rangle$')

    # Plot a large dot for the average result of the zero and one states.
    mean_zero = np.mean(zero_data) # takes mean of both real and imaginary parts
    mean_one = np.mean(one_data)
    plt.scatter(np.real(mean_zero), np.imag(mean_zero), 
                s=200, cmap='viridis', c='black',alpha=1.0)
    plt.scatter(np.real(mean_one), np.imag(mean_one), 
                s=200, cmap='viridis', c='black',alpha=1.0)
    
    plt.xlim(x_min, x_max)
    plt.ylim(y_min,y_max)
    plt.legend()
    plt.ylabel('I [a.u.]', fontsize=15)
    plt.xlabel('Q [a.u.]', fontsize=15)
    plt.title("0-1 discrimination", fontsize=15)

x_min = -20
x_max = 0
y_min = -20
y_max = 0
# clusters_01_plot(x_min, x_max, y_min, y_max)

# Create IQ vector (split real, imag parts)
zero_data_reshaped = np.array([zero_data.real, zero_data.imag]).T
one_data_reshaped = np.array([one_data.real, one_data.imag]).T


cluster_01_data = np.concatenate((zero_data_reshaped, one_data_reshaped))
print(cluster_01_data.shape) # verify data shape

# construct vector w/ 0's and 1's (for testing)
state_01 = np.zeros(num_shots_per_exp) # shots gives number of experiments
state_01 = np.concatenate((state_01, np.ones(num_shots_per_exp)))
print(len(state_01))

# Shuffle and split data into training and test sets
cluster_01_train, cluster_01_test, state_01_train, state_01_test = train_test_split(cluster_01_data, state_01, test_size=0.5)

# Set up the LDA
LDA_01 = LinearDiscriminantAnalysis()
LDA_01.fit(cluster_01_train, state_01_train)
print(LDA_01.predict([ [0,0], [10, 0] ]))

# Compute accuracy
score_01 = LDA_01.score(cluster_01_test, state_01_test)
print(score_01)

# Plot separatrix on top of scatter
def separatrixPlot(lda, x_min, x_max, y_min, y_max, shots):
    nx, ny = shots, shots

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='black')

clusters_01_plot(x_min, x_max, y_min, y_max)
separatrixPlot(LDA_01, x_min, x_max, y_min, y_max, num_shots_per_exp)
plt.show()

pkl_filename = "./cache/discriminator/lda.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(LDA_01, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
