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

center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
span = 0.008 * center_frequency_Hz
num_sweep_experiments = 50
frequencies = np.arange(center_frequency_Hz - span / 2, 
                        center_frequency_Hz + span / 2,
                        span / num_sweep_experiments)
dt = backend_config.dt
# measurement 
meas_map_idx = None
for i, measure_group in enumerate(backend_config.meas_map):
    if qubit in measure_group:
        meas_map_idx = i
        break
assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"
inst_sched_map = backend_defaults.instruction_schedule_map
measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

drive_sigma = 0.05 # us
drive_samples = 8 * drive_sigma # us
drive_amp = 0.1 
drive_sigma = get_closest_multiple_of_16(drive_sigma * us / dt)
drive_samples = get_closest_multiple_of_16(drive_samples * us/ dt)

gauss = pulse_lib.gaussian(duration=drive_samples,
                           sigma=drive_sigma,
                           amp=drive_amp,
                           name='freq_sweep_excitation_pulse')
schedule = pulse.Schedule(name='Frequency sweep')
schedule += Play(gauss, drive_chan)
schedule += measure << schedule.duration

schedule_frequencies = [{drive_chan: freq} for freq in frequencies]
# schedule.draw()

num_shots_per_frequency = 1024
frequency_sweep_program = assemble(schedule,
                                   backend=backend, 
                                   meas_level=1,
                                   meas_return='avg',
                                   shots=num_shots_per_frequency,
                                   schedule_los=schedule_frequencies)

job = backend.run(frequency_sweep_program)
job_monitor(job)
frequency_sweep_results = job.result(timeout=120)

sweep_values = []
for i in range(len(frequency_sweep_results.results)):
    # Get the results from the ith experiment
    res = frequency_sweep_results.get_memory(i)*1e-14
    # Get the results for `qubit` from this experiment
    sweep_values.append(res[qubit])

frequencies_GHz = frequencies / GHz

plt.figure(1)
plt.scatter(frequencies_GHz, np.real(sweep_values), color='black') # plot real part of sweep values
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured signal [a.u.]")
# plt.show()

## fit curve


def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)#, maxfev=100000)
    y_fit = function(x_values, *fitparams)
    
    return fitparams, y_fit

fit_params, y_fit = fit_function(frequencies_GHz,
                                 np.real(sweep_values), 
                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
                                 [0.3, 4.975, 0.1, -10.5] # initial parameters for curve_fit
                                )
plt.figure(2)
plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
plt.plot(frequencies_GHz, y_fit, color='red')
plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])

plt.xlabel("Frequency [GHz]")
plt.ylabel("Measured Signal [a.u.]")
# plt.show()

A, rough_qubit_frequency, B, C = fit_params
rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
print(f"We've updated our qubit frequency estimate from "
      f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")


min_amplitude, max_amplitude = 0.01, 0.5
num_exp = 100
amplitudes = np.linspace(
    min_amplitude, 
    max_amplitude, 
    num_exp
)
duration = get_closest_multiple_of_16(0.5 * us / dt)

schedules = []
for amp in amplitudes:
    sq_pulse = pulse_lib.constant(
        duration=duration,
        amp=amp,
        name='pi_sweep_square_pulse'
    )
    pi_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {amp}")
    pi_schedule += Play(sq_pulse, drive_chan)
    pi_schedule += measure << pi_schedule.duration
    schedules.append(pi_schedule)

num_shots_per_exp = 1024
rabi_pi_qobj = assemble(   
    schedules,
    backend=backend, 
    meas_level=1,
    meas_return='avg',
    shots=num_shots_per_exp,
    schedule_los=[{drive_chan: rough_qubit_frequency}] * num_exp
)

pi_job = backend.run(rabi_pi_qobj)
job_monitor(pi_job)

pi_sweep_results = pi_job.result(timeout=120)

pi_sweep_values = []
for i in range(len(pi_sweep_results.results)):
    # Get the results from the ith experiment
    results = pi_sweep_results.get_memory(i)*1e-14
    # Get the results for `qubit` from this experiment
    pi_sweep_values.append(results[qubit])


plt.figure(3)
plt.scatter(amplitudes, np.real(pi_sweep_values), color='black') # plot real part of sweep values
# plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Amplitudes [au]")
plt.ylabel("Measured signal [a.u.]")

print(amplitudes)
print(np.real(pi_sweep_values))
rabi_fit_params, rabi_y_fit = fit_function(
    amplitudes,
    np.real(pi_sweep_values), 
    lambda x, A, k, B: A * (np.cos(k * (x))) + B,
    [-1.8, 50, -8] # initial parameters for curve_fit
)

print(rabi_fit_params)
_, k, _ = rabi_fit_params
pi_amp = np.pi / (k)

plt.figure(4)
plt.scatter(amplitudes, np.real(pi_sweep_values), color='black')
plt.plot(amplitudes, rabi_y_fit, color='red')
plt.xlim([min(amplitudes), max(amplitudes)])

plt.xlabel("Amplitude [au]")
plt.ylabel("Measured Signal [a.u.]")
plt.show()

span = 0.001 * rough_qubit_frequency
num_ramsey_freq_exp = 100
ramsey_freq = np.linspace(
    rough_qubit_frequency - span / 2, 
    rough_qubit_frequency + span / 2,
    num_ramsey_freq_exp
)
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
ramsey_schedule = pulse.Schedule(name=f"Ramsey fringe experiment")
ramsey_schedule += Play(half_pi_pulse, drive_chan)
ramsey_schedule += Delay(duration, drive_chan)
ramsey_schedule += Play(half_pi_pulse, drive_chan)
ramsey_schedule += measure << ramsey_schedule.duration

ramsey_schedule_freq = [{drive_chan: freq} for freq in ramsey_freq]
ramsey_qobj = assemble(   
    ramsey_schedule,
    backend=backend,
    meas_level=1,
    meas_return='avg',
    shots=num_shots_per_exp,
    schedule_los=ramsey_schedule_freq
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



plt.figure(5)
plt.scatter(ramsey_freq, np.real(ramsey_values), color='black') # plot real part of sweep values
# plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
plt.xlabel("Frequency [au]")
plt.ylabel("Measured signal [a.u.]")

plt.show()