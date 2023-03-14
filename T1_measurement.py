import os
import pickle
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qiskit import pulse, IBMQ, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, Gate
# This Pulse module helps us build sampled pulses for common pulse shapes
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager


def measure_T1(backend_name):
    num_qubits_per_system = {
        "ibm_perth": 7,
        "ibm_lagos": 7,
        "ibm_nairobi": 7,
        "ibm_oslo": 7,
        "ibmq_jakarta": 7,
        "ibmq_manila": 5,
        "ibmq_quito": 5,
        "ibmq_belem": 5,
        "ibmq_lima": 5,
    }
    for b in num_qubits_per_system.keys():
        if backend_name in b:
            backend_full_name = b
            num_qubits = num_qubits_per_system[b]
            break

    GHz = 1.0e9 # Gigahertz
    MHz = 1.0e6 # Megahertz
    us = 1.0e-6 # Microseconds
    ns = 1.0e-9 # Nanoseconds
    qubit = 0
    mem_slot = 0

    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)

    provider = IBMQ.load_account()
    backend = provider.get_backend(backend_full_name)
    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    q_freq = [backend_defaults.qubit_freq_est[q] for q in range(num_qubits)]
    dt = backend_config.dt


    rough_qubit_frequency = center_frequency_Hz # 4962284031.287086 Hz

    # calibrate a pi pulse

    initial_amp, final_amp, num_exp = 0, .3, 100
    amplitudes = np.linspace(
        initial_amp,
        final_amp,
        num_exp
    )

    print(f"The resonant frequency is assumed to be {np.round(rough_qubit_frequency / GHz, 5)} GHz.")
    print(f"The area calibration will start from amp {amplitudes[0]} "
    f"and end at {amplitudes[-1]} with approx step {(final_amp - initial_amp)/num_exp}.")
    
    amp = Parameter('amp')
    freq = Parameter('freq')
    with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area") as sched:
        dur_dt = duration
        pulse.set_frequency(freq, drive_chan)
        if pulse_type == "sq" or "sin" in pulse_type:
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type
            )
        elif pulse_type == "gauss":
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type,
                sigma=sigma / np.sqrt(2),
                zero_ends=remove_bg
            )
        elif pulse_type in ["lor", "lor2", "lor3"]:
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type,
                gamma=sigma,
                zero_ends=remove_bg
            )
        else:
            pulse_played = pulse_dict[pulse_type](
                duration=dur_dt,
                amp=amp,
                name=pulse_type,
                sigma=sigma,
                zero_ends=remove_bg
            )
        pulse.play(pulse_played, drive_chan)

    # Create gate holder and append to base circuit
    base_circ = QuantumCircuit(num_qubits, num_qubits)
    for q in num_qubits:
        pi_gate = Gate("rabi", 1, [amp, freq])
        base_circ.append(pi_gate, [0])
        base_circ.measure(q, q)
        base_circ.add_calibration(pi_gate, (qubit,), sched, [amp, freq])
        circs = [
            base_circ.assign_parameters(
                    {amp: a, freq: q_freq[q]},
                    inplace=False
            ) for a in amplitudes]

    # T1 experiment parameters
    time_max_sec = 450 * us
    time_step_sec = 6.5 * us
    delay_times_sec = np.arange(1 * us, time_max_sec, time_step_sec)

    # We will use the same `pi_pulse` and qubit frequency that we calibrated and used before

    delay = Parameter('delay')
    qc_t1 = QuantumCircuit(num_qubits, num_qubits)


    qc_t1.x(0)
    qc_t1.delay(delay, 0)
    qc_t1.measure(0, 0)
    qc_t1.add_calibration("x", (0,), pi_pulse)

    exp_t1_circs = [qc_t1.assign_parameters({delay: get_dt_from(d)}, inplace=False) for d in delay_times_sec]

    sched_idx = -1
    t1_schedule = schedule(exp_t1_circs[sched_idx], backend)
    t1_schedule.draw(backend=backend)

    # Execution settings
    num_shots = 256

    job = backend.run(exp_t1_circs, 
                    meas_level=1, 
                    meas_return='single', 
                    shots=num_shots)

    job_monitor(job)

    t1_results = job.result(timeout=120)

    t1_values = []

    for i in range(len(delay_times_sec)):
        iq_data = t1_results.get_memory(i)[:,qubit] * scale_factor
        t1_values.append(sum(map(classify, iq_data)) / num_shots)

    plt.scatter(delay_times_sec/us, t1_values, color='black') 
    plt.title("$T_1$ Experiment", fontsize=15)
    plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
    plt.ylabel('Signal [a.u.]', fontsize=15)
    plt.show()

    # Fit the data
    fit_params, y_fit = fit_function(delay_times_sec/us, t1_values, 
                lambda x, A, C, T1: (A * np.exp(-x / T1) + C),
                [-3, 3, 100]
                )

    _, _, T1 = fit_params

    plt.scatter(delay_times_sec/us, t1_values, color='black')
    plt.plot(delay_times_sec/us, y_fit, color='red', label=f"T1 = {T1:.2f} us")
    plt.xlim(0, np.max(delay_times_sec/us))
    plt.title("$T_1$ Experiment", fontsize=15)
    plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
    plt.ylabel('Signal [a.u.]', fontsize=15)
    plt.legend()
    plt.show()
