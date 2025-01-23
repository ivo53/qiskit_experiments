import os
import sys
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

from qiskit import (
    QuantumCircuit, 
    pulse, 
    IBMQ
) 
# This is where we access all of our Pulse features!
from qiskit.circuit import Parameter, Gate
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

import common.pulse_types as pt
from utils.run_jobs import run_jobs

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

COMP_PARAMS = {
    "X": {
        1: [{"alpha": 1, "phases": [1/2]}],
        3: [{"alpha": 1, "phases": [1/6, 5/6]}],
        5: [{"alpha": 1, "phases": [0.0672, 0.3854, 1.1364]}],
        6: [{
                "alpha": [0.25897065, 0.31579605, 0.31907684, 0.32474782, 0.30207873, 0.27153638], 
                "phases": [0.64128025, 1.35410576, -0.63968337, -1.19841068, 1.00119089, 1.31535192],
            }], # for qubit 94 kyiv
        7: [{"alpha": 1, "phases": [0.2560, 1.6839, 0.5933, 0.8306]},
           {"alpha": [0.26216650072338854, 0.31654687577570756, 0.31654687577570756, 0.31654687577570756, 0.31654687577570756, 0.31654687577570756, 0.23991346939666508], 
            "phases": [-0.2681786687850911, 0.262007101043305, 0.1158282957504724, 0.058347259736134145, -0.2714955767202196, -0.1544419019664696, 0.38186913558140173]},
            {
                "alpha": [2.48679409e-01, 7.35181109e-02, 9.88250774e-02, 2.84600212e-01, 3.30750000e-01, 3.30750000e-01, 1.21189702e-01], 
                "phases": [-6.24175979e-01, 3.41266737e-01, 9.04041508e-01, 8.74381215e-01, 1.43158780e+00, -1.32953460e-04, -7.86870494e-01]
        }], # for qubit 80 sherbrooke
        9: [{"alpha": 1, "phases": [0.3951, 1.2211, 0.7806, 1.9335, 0.4580]}],
        11: [{"alpha": 1, "phases": [0.7016, 1.1218, 1.8453, 0.9018, 0.3117, 0.1699]}],
        13: [{"alpha": 1, "phases": [0.1200, 0.3952, 1.5643, 0.0183, 0.9219, 0.4975,1.1096]}],
        15: [{"alpha": 1, "phases": [0.5672, 1.4322, 0.9040, 0.2397, 0.9118, 0.5426, 1.6518, 0.1406]}],
        17: [{"alpha": 1, "phases": [0.3604, 1.1000, 0.7753, 1.6298, 1.2338, 0.2969, 0.6148, 1.9298, 0.4443]}]
    },
    "H": {
        3: [{"alpha": 0.6399, "phases": [1.8442, 1.0587]}],
        5: [{"alpha": 0.45, "phases": [1.9494, 0.5106, 1.3179]}],
        7: [{"alpha": 0.2769, "phases": [1.6803, 0.2724, 0.8255, 1.6624]}],
        9: [{"alpha": 0.2947, "phases": [1.2711, 0.1069, 0.5283, 1.1283, 1.9884]}],
        11: [{"alpha": 0.2985, "phases": [1.7377, 0.1651, 0.9147, 0.1510, 0.9331, 1.6415]}],
        13: [{"alpha": 0.5065, "phases": [0.0065, 1.7755, 0.7155, 0.5188, 0.2662, 1.2251, 1.3189]}],
        15: [{"alpha": 0.3132, "phases": [1.2316, 0.9204, 0.2043, 1.9199, 0.8910, 0.7381, 1.9612, 1.3649]}],
    }
}
pulse_dict = {
    "gauss": [pt.Gaussian, pt.LiftedGaussian],
    "lor": [pt.Lorentzian, pt.LiftedLorentzian],
    "lor2": [pt.Lorentzian2, pt.LiftedLorentzian2],
    "lor3": [pt.Lorentzian3, pt.LiftedLorentzian3],
    "lor2_3": [pt.Lorentzian2_3, pt.LiftedLorentzian2_3],
    "lor3_4": [pt.Lorentzian3_4, pt.LiftedLorentzian3_4],
    "lor3_5": [pt.Lorentzian3_5, pt.LiftedLorentzian3_5],
    "sq": [pt.Constant, pt.Constant],
    "sech": [pt.Sech, pt.LiftedSech],
    "sech2": [pt.Sech2, pt.LiftedSech2],
    "sin": [pt.Sine, pt.Sine],
    "sin2": [pt.Sine2, pt.Sine2],
    "sin3": [pt.Sine3, pt.Sine3],
    "sin4": [pt.Sine4, pt.Sine4],
    "sin5": [pt.Sine5, pt.Sine5],
    "demkov": [pt.Demkov, pt.LiftedDemkov],
    "ipN": [pt.InverseParabola, pt.InverseParabola],
    "fcq": [pt.FaceChangingQuadratic, pt.FaceChangingQuadratic],
    "lz1": [pt.LandauZener1, pt.LandauZener1],
    "lz4": [pt.LandauZener4, pt.LandauZener4],
    "lz8": [pt.LandauZener8, pt.LandauZener8],
    "ae1": [pt.AllenEberly1, pt.AllenEberly1],
    "ae4": [pt.AllenEberly4, pt.AllenEberly4],
    "ae8": [pt.AllenEberly8, pt.AllenEberly8],
    "dk2": [pt.DemkovKunike2, pt.DemkovKunike2],
    "comp": [pt.Composite, pt.Composite],
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-q", "--qubit", default=0, type=int,
        help="The number of the qubit to be used.")
    # parser.add_argument("-l", "--l", default=10, type=float,
    #     help="Parameter l in Rabi oscillations fit")
    # parser.add_argument("-p", "--p", default=0.5, type=float,
    #     help="Parameter p in Rabi oscillations fit")
    # parser.add_argument("-x0", "--x0", default=0.005, type=float,
    #     help="Parameter x0 in Rabi oscillations fit")
    parser.add_argument("-s", "--sigma", default=180, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-T", "--duration", default=2256, type=float,
        help="Lorentz duration parameter")
    parser.add_argument("-c", "--cutoff", default=0.5, type=float,
        help="Cutoff parameter in PERCENT of maximum amplitude of Lorentzian")
    parser.add_argument("-rb", "--remove_bg", default=1, type=int,
        help="Whether to drop the background (tail) of the pulse (0 or 1).")
    parser.add_argument("-cp", "--control_param", default="width", type=str,
        help="States whether width or duration is the controlled parameter")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=2048, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-ne", "--num_experiments", default=100, type=int,
        help="Number of experiments with different detuning each.")
    parser.add_argument("-b", "--backend", default="manila", type=str,
        help="The name of the backend to use in the experiment (one of perth, lagos, nairobi, \
        oslo, jakarta, manila, quito, belem, lima).")
    parser.add_argument("-sp", "--span", default=0.005, type=float,
        help="The span of the detuning sweep as a fraction of the driving frequency.")
    parser.add_argument("-N", "--N", default=1, type=int,
        help="The order of inverse parabola pulse(in case of inv. parabola).")
    parser.add_argument("-be", "--beta", default=0, type=float,
        help="The beta parameter for the face changing quadratic and Landau-Zener functions.")
    parser.add_argument("-tau", "--tau", default=100, type=float,
        help="The tau parameter for the Allen-Eberly and Demkov-Kunike-2 functions.")
    parser.add_argument("-v", "--variant", default=0, type=int,
        help="The variant of the composite pulses that are to be used.")
    parser.add_argument("-g", "--gate_type", default="X", type=str,
        help="The type of the composite gate that is played.")
    parser.add_argument("-np", "--num_pulses", default=5, type=int,
        help="Num composite pulses inside the composite gate that is played.")
    parser.add_argument("-nr", "--num_reps", default=1, type=int,
        help="Num repetitions of the composite gate that is played.")
    args = parser.parse_args()

    pulse_type = args.pulse_type
    # l = args.l
    # p = args.p
    # x0 = args.x0
    sigma = args.sigma
    duration = get_closest_multiple_of_16(round(args.duration))
    cutoff = args.cutoff
    remove_bg = args.remove_bg
    control_param = args.control_param
    max_experiments_per_job = args.max_experiments_per_job
    num_shots = args.num_shots
    span = args.span
    num_experiments = args.num_experiments
    qubit = args.qubit
    N = float(args.N)
    beta = args.beta
    tau = args.tau
    variant = args.variant
    gate_type = args.gate_type
    num_pulses= args.num_pulses
    num_reps= args.num_reps
    backend = args.backend
    backend_name = backend
    backend = "ibm_" + backend \
        if backend in ["perth", "lagos", "nairobi", "oslo", "kyoto", "brisbane"] \
            else "ibmq_" + backend
    
    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    current_date = date.strftime("%Y-%m-%d")
    calib_dir = os.path.join(file_dir, "calibrations", backend_name)
    save_dir = os.path.join(calib_dir, current_date)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    data_folder = os.path.join(file_dir, "data", backend_name, "calibration", current_date)
    # if not os.path.isdir(data_folder):
    #     os.mkdir(data_folder)
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    params_file = os.path.join(calib_dir, "actual_params.csv")
    print(params_file)
    if os.path.isfile(params_file):
        param_df = pd.read_csv(params_file)
    df = param_df[param_df.apply(
            lambda row: row["pulse_type"] == pulse_type and \
                row["duration"] == duration and \
                row["sigma"] == sigma and \
                row["rb"] == remove_bg, axis=1)]
    print(df)
    if df.shape[0] > 1:
        raise ValueError("More than one identical entry found!")
    elif df.shape[0] == 0:
        raise ValueError("No entries found!")
    ser = df.iloc[0]
    l = ser.at["l"]
    p = ser.at["p"]
    x0 = ser.at["x0"]
    print(l, p, x0)
    # unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
    GHz = 1.0e9 # Gigahertz
    MHz = 1.0e6 # Megahertz
    us = 1.0e-6 # Microseconds
    ns = 1.0e-9 # Nanoseconds
    mem_slot = 0

    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)

    # backend = IBMProvider().get_backend(backend)
    backend = QiskitRuntimeService(channel="ibm_quantum").backend(backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)

    print(f"Using {backend_name} backend.")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()
    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    print(center_frequency_Hz)
    num_qubits = backend_config.n_qubits
    if num_qubits <= qubit:
        raise ValueError("Qubit index out of range for this system.")
    span = span * center_frequency_Hz
    frequencies = np.linspace(center_frequency_Hz - span / 2, 
                              center_frequency_Hz + span / 2,
                              num_experiments)
    dt = backend_config.dt
    # print(dt)
    # amp = -np.log(1 - np.pi / l) / p + x0
    # amp = 0.06
    def add_circ(amp, duration, sigma, freq, qubit=0):
        # amp = Parameter("amp")
        # duration = 16 * 100
        # sigma = 192
        # freq_param = Parameter("freq")
        with pulse.build(backend=backend, default_alignment='sequential', name="analyze_detuning_spectrum") as sched:
            dur_dt = duration
            pulse.set_frequency(freq, drive_chan)
            if pulse_type == "sq" or "sin" in pulse_type:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type
                )
            elif pulse_type == "gauss":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type,
                    sigma=sigma
                )
            elif pulse_type in ["lor", "lor2", "lor3"]:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type,
                    sigma=sigma,
                )
            elif pulse_type == "comp":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    amps=COMP_PARAMS[gate_type][num_pulses][variant]["alpha"],
                    phases=COMP_PARAMS[gate_type][num_pulses][variant]["phases"],
                    name=pulse_type
                )
            elif pulse_type in ["fcq", "lz"]:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    beta=beta,
                    name=pulse_type
                )
            elif pulse_type in ["ae", "dk2"]:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    beta=beta,
                    tau=tau,
                    name=pulse_type
                )
            else:
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    name=pulse_type,
                    sigma=sigma,
                )
            for _ in range(num_reps):
                pulse.play(pulse_played, drive_chan)
        pi_gate = Gate("rabi", 1, [])
        base_circ = QuantumCircuit(5, 1)
        base_circ.append(pi_gate, [qubit])
        base_circ.measure(qubit, 0)
        base_circ.add_calibration(pi_gate, (qubit,), sched, [])
        return base_circ

    # freq = Parameter('freq')
    # with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_freq") as sched:
    #     dur_dt = duration
    #     pulse.set_frequency(freq, drive_chan)
    #     if pulse_type == "sq" or "sin" in pulse_type:
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=duration,
    #             amp=amp,
    #             name=pulse_type
    #         )
    #     elif pulse_type == "gauss":
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=duration,
    #             amp=amp,
    #             name=pulse_type,
    #             sigma=sigma / np.sqrt(2),
    #         )
    #     elif pulse_type in ["lor", "lor2", "lor3"]:
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=duration,
    #             amp=amp,
    #             name=pulse_type,
    #             sigma=sigma,
    #         )
    #     else:
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=duration,
    #             amp=amp,
    #             name=pulse_type,
    #             sigma=sigma,
    #         )
    #     pulse.play(pulse_played, drive_chan)

    # # Create gate holder and append to base circuit
    # custom_gate = Gate("pi_pulse", 1, [freq])
    # base_circ = QuantumCircuit(num_qubits, 1)
    # base_circ.append(custom_gate, [qubit])
    # base_circ.measure(qubit, 0)
    # base_circ.add_calibration(custom_gate, (qubit,), sched, [freq])
    # circs = [base_circ.assign_parameters(
    #         {freq: f},
    #         inplace=False
    #     ) for f in frequencies]

    circs = [add_circ(amp, duration, sigma, f, qubit=qubit) for f in frequencies]

    sweep_values, job_ids = run_jobs(circs, backend, duration, num_shots_per_exp=num_shots)

    frequencies_GHz = frequencies / GHz

    data = {"frequency_ghz": frequencies_GHz, "transition_probability": (sweep_values)}
    df = pd.DataFrame(data)
    df.to_csv(
        os.path.join(
            data_folder, 
            date.strftime("%H%M%S") + f"_calibration.csv"
        ),
        index=False
    )
    plt.figure(1)
    plt.scatter(frequencies_GHz, (sweep_values), color='black') # plot real part of sweep values
    plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    plt.title("Drive Frequency Calibration Curve")
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Transition Probability")
    plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_dur_{duration}_s_{int(sigma)}_frequency_sweep.png"))
    plt.show()

    ## fit curve


    def fit_function(x_values, y_values, function, init_params):
        fitparams, conv = curve_fit(function, x_values, y_values, init_params)#, maxfev=100000)
        y_fit = function(x_values, *fitparams)
        
        return fitparams, y_fit

    # fit_params, y_fit = fit_function(frequencies_GHz,
    #                                 np.real(sweep_values), 
    #                                 lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq)**2 + B**2)) + C,
    #                                 [0.3, 4.975, 0.2, 0] # initial parameters for curve_fit
    #                                 )
    # plt.figure(2)
    # plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
    # plt.plot(frequencies_GHz, y_fit, color='red')
    # plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
    # plt.title("Fitted Drive Frequency Calibration Curve")
    # plt.xlabel("Frequency [GHz]")
    # plt.ylabel("Measured Signal [a.u.]")
    # plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_frequency_sweep_fitted.png"))
    # plt.show()

    # A, rough_qubit_frequency, B, C = fit_params
    # rough_qubit_frequency = rough_qubit_frequency*GHz # make sure qubit freq is in Hz
    # print(f"We've updated our qubit frequency estimate from "
    #     f"{round(backend_defaults.qubit_freq_est[qubit] / GHz, 5)} GHz to {round(rough_qubit_frequency/GHz, 5)} GHz.")
