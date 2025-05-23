import os
import sys
import pickle
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, differential_evolution
from qiskit import pulse, QuantumCircuit, transpile
from qiskit.circuit import Parameter, Gate
# This Pulse module helps us build sampled pulses for common pulse shapes
# from qiskit.pulse import library as pulse_lib
# from qiskit.tools.monitor import job_monitor
# from qiskit.providers.ibmq.managed import IBMQJobManager
# from qiskit_ibm_provider import IBMProvider
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# current_dir = os.path.dirname(__file__)
# package_path = os.path.abspath(os.path.split(current_dir)[0])
# sys.path.insert(0, package_path)

from utils.run_jobs import run_jobs
import common.pulse_types as pt

def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")  
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

def add_entry_and_remove_duplicates(df, new_entry, cols_to_check=["pulse_type", "duration", "sigma", "rb", "N", "beta"]):
    # Define a function to check if two rows have the same values in the specified columns
    def rows_match(row1, row2, cols):
        for col in cols:
            if row1[col] != row2.at[0, col]:
                return False
        return True
    
    # Find rows in the DataFrame that have the same values in the specified columns as the new entry
    matching_rows = df.apply(lambda row: rows_match(row, new_entry, cols_to_check), axis=1)
    
    if matching_rows.any():
        # Remove old entries that have the same values in the specified columns as the new entry
        df = df[~matching_rows]
        
    # Add the new entry to the DataFrame
    df = pd.concat([df, new_entry], ignore_index=True)
    
    return df

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pt", "--pulse_type", default="gauss", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)")
    parser.add_argument("-s", "--sigma", default=180, type=float,
        help="Pulse width (sigma) parameter")    
    parser.add_argument("-T", "--duration", default=2256, type=float,
        help="Pulse duration parameter")
    parser.add_argument("-rb", "--remove_bg", default=0, type=int,
        help="Whether to drop the background (and thus discontinuities) "
            "of the pulse (0 or 1).")
    parser.add_argument("-epj", "--max_experiments_per_job", default=100, type=int,
        help="Maximum experiments per job")
    parser.add_argument("-ns", "--num_shots", default=2048, type=int,
        help="Number of shots per experiment (datapoint).")
    parser.add_argument("-b", "--backend", default="manila", type=str,
        help="The name of the backend to use in the experiment (one of nairobi, \
        oslo, manila, quito, belem, lima).")
    parser.add_argument("-ia", "--initial_amp", default=0.001, type=float,
        help="The initial amplitude of the area sweep.")
    parser.add_argument("-fa", "--final_amp", default=0.2, type=float,
        help="The final amplitude of the area sweep.")
    parser.add_argument("-ne", "--num_experiments", default=100, type=int,
        help="The number of amplitude datapoints to evaluate.")
    parser.add_argument("-l", "--l", default=1, type=float,
        help="The initial value of the l fit param.")
    parser.add_argument("-p", "--p", default=0.5, type=float,
        help="The initial value of the p fit param.")
    parser.add_argument("-x0", "--x0", default=0.01, type=float,
        help="The initial value of the x0 fit param.")
    parser.add_argument("-q", "--qubit", default=0, type=int,
        help="Number of qubit to use.")
    parser.add_argument("-N", "--N", default=1, type=int,
        help="The order of inverse parabola pulse(in case of inv. parabola).")
    parser.add_argument("-be", "--beta", default=0, type=float,
        help="The beta parameter for the face changing quadratic and Landau-Zener functions.")
    parser.add_argument("-tau", "--tau", default=100, type=float,
        help="The tau parameter for the Allen-Eberly and Demkov-Kunike-2 functions.")
    parser.add_argument("-sv", "--save", default=0, type=int,
        help="Whether to save the results from the fit (0 or 1).")
    parser.add_argument("-v", "--variant", default=0, type=int,
        help="The variant of the composite pulses that are to be used.")
    parser.add_argument("-g", "--gate_type", default="X", type=str,
        help="The type of the composite gate that is played.")
    parser.add_argument("-np", "--num_pulses", default=5, type=int,
        help="Num composite pulses inside the composite gate that is played.")
    parser.add_argument("-nr", "--num_reps", default=1, type=int,
        help="Num repetitions of the composite gate that is played.")
    args = parser.parse_args()

    # cutoff = args.cutoff
    duration = get_closest_multiple_of_16(round(args.duration))
    sigma = args.sigma
    initial_amp = args.initial_amp
    final_amp = args.final_amp
    num_shots_per_exp = args.num_shots
    num_exp = args.num_experiments
    backend = args.backend
    max_experiments_per_job = args.max_experiments_per_job
    remove_bg = int(args.remove_bg)
    pulse_type = args.pulse_type
    l = args.l
    p = args.p
    x0 = args.x0
    qubit = args.qubit
    N = float(args.N)
    beta = args.beta
    tau = args.tau
    variant = args.variant
    gate_type = args.gate_type
    num_pulses= args.num_pulses
    num_reps= args.num_reps
    save = bool(args.save)
    backend_name = backend
    backend = "ibm_" + backend
        # if backend in ["perth", "lagos", "nairobi", "oslo", "kyoto", "brisbane"] \
        #     else "ibmq_" + backend
    pulse_dict = {
        "gauss": [pt.Gaussian, pt.LiftedGaussian],
        "lor": [pt.Lorentzian, pt.LiftedLorentzian],
        "lor2": [pt.Lorentzian2, pt.LiftedLorentzian2],
        "lor3": [pt.Lorentzian3, pt.LiftedLorentzian3],
        "lor3_2": [pt.Lorentzian3_2, pt.LiftedLorentzian3_2],
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
        "drag": [pt.Drag2, pt.LiftedDrag2],
        "ipN": [pt.InverseParabola, pt.InverseParabola],
        "fcq": [pt.FaceChangingQuadratic, pt.FaceChangingQuadratic],
        "lz1": [pt.LandauZener1, pt.LandauZener1],
        "lz4": [pt.LandauZener4, pt.LandauZener4],
        "lz8": [pt.LandauZener8, pt.LandauZener8],
        "ae1": [pt.AllenEberly1, pt.AllenEberly1],
        "ae4": [pt.AllenEberly4, pt.AllenEberly4],
        "ae8": [pt.AllenEberly8, pt.AllenEberly8],
        "hlz1": [pt.HalfLandauZener1, pt.HalfLandauZener1],
        "hlz4": [pt.HalfLandauZener4, pt.HalfLandauZener4],
        "hlz8": [pt.HalfLandauZener8, pt.HalfLandauZener8],
        "hae1": [pt.HalfAllenEberly1, pt.HalfAllenEberly1],
        "hae4": [pt.HalfAllenEberly4, pt.HalfAllenEberly4],
        "hae8": [pt.HalfAllenEberly8, pt.HalfAllenEberly8],
        "bb": [pt.BambiniBerman, pt.BambiniBerman],
        "cs": [pt.CosSin, pt.CosSin],
        "comp": [pt.Composite, pt.Composite]
    }
    ## create folder where plots are saved
    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    date = datetime.now()
    time = date.strftime("%H%M%S")
    current_date = date.strftime("%Y-%m-%d")
    calib_dir = os.path.join(file_dir, "calibrations", backend_name, str(qubit))
    save_dir = os.path.join(calib_dir, current_date)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    data_folder = os.path.join(file_dir, "data", backend_name, str(qubit), "calibration", current_date)
    # if not os.path.isdir(data_folder):
    #     os.mkdir(data_folder)
    make_all_dirs(save_dir)
    make_all_dirs(data_folder)

    # unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
    GHz = 1.0e9 # Gigahertz
    MHz = 1.0e6 # Megahertz
    us = 1.0e-6 # Microseconds
    ns = 1.0e-9 # Nanoseconds
    mem_slot = 0

    drive_chan = pulse.DriveChannel(qubit)
    
    # # with real provider
    # provider = IBMQ.load_account()
    # backend = provider.get_backend(backend)
    
    # backend = IBMProvider().get_backend(backend)
    backend = QiskitRuntimeService(channel="ibm_quantum").backend(backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
    # #
    # # with Fake provider only
    # backend = FakeManilaV2()
    # # 
    print(f"Using {backend_name} backend.")
    # with real provider
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()
    
    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    dt = backend_config.dt
    print(dt, center_frequency_Hz)
    #
    # # with Fake provider only
    # center_frequency_Hz = backend.qubit_properties(qubit).frequency
    # print(center_frequency_Hz)
    # dt = backend.dt
    # print(dt)
    # #
    rough_qubit_frequency = center_frequency_Hz # 4962284031.287086 Hz
    
    ## set params
    fit_crop = 1#.8
    amplitudes = np.linspace(
        initial_amp, 
        final_amp, 
        num_exp
    )
    fit_crop_parameter = int(fit_crop * len(amplitudes))

    print(f"The resonant frequency is assumed to be {np.round(rough_qubit_frequency / GHz, 5)} GHz.")
    print(f"The area calibration will start from amp {amplitudes[0]} "
    f"and end at {amplitudes[-1]} with approx step {(final_amp - initial_amp)/num_exp}.")

    def add_circ(amp, duration, sigma, qubit=0):
        # amp = Parameter("amp")
        # duration = 16 * 100
        # sigma = 192
        # freq_param = Parameter("freq")
        with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area") as sched:
            dur_dt = duration
            pulse.set_frequency(rough_qubit_frequency, drive_chan)
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
            elif pulse_type == "drag":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    beta=beta,
                    name=pulse_type,
                    sigma=sigma,
                )
            elif pulse_type == "ipN":
                pulse_played = pulse_dict[pulse_type][remove_bg](
                    duration=dur_dt,
                    amp=amp,
                    N=N,
                    name=pulse_type
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
            pulse.play(pulse_played, drive_chan)
        pi_gate = Gate("rabi", 1, [])
        base_circ = QuantumCircuit(qubit+1, 1)
        base_circ.append(pi_gate, [qubit])
        base_circ.measure(qubit, 0)
        base_circ.add_calibration(pi_gate, (qubit,), sched, [])
        return base_circ

    # amp = Parameter('amp')
    # with pulse.build(backend=backend, default_alignment='sequential', name="calibrate_area") as sched:
    #     dur_dt = duration
    #     pulse.set_frequency(rough_qubit_frequency, drive_chan)
    #     if pulse_type == "sq" or "sin" in pulse_type:
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=dur_dt,
    #             amp=amp,
    #             name=pulse_type
    #         )
    #     elif pulse_type == "gauss":
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=dur_dt,
    #             amp=amp,
    #             name=pulse_type,
    #             sigma=sigma / np.sqrt(2)
    #         )
    #     elif pulse_type in ["lor", "lor2", "lor3"]:
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=dur_dt,
    #             amp=amp,
    #             name=pulse_type,
    #             sigma=sigma,
    #         )
    #     else:
    #         pulse_played = pulse_dict[pulse_type][remove_bg](
    #             duration=dur_dt,
    #             amp=amp,
    #             name=pulse_type,
    #             sigma=sigma,
    #         )
    #     pulse.play(pulse_played, drive_chan)

    # Create gate holder and append to base circuit
    # pi_gate = Gate("rabi", 1, [amp])
    # base_circ = QuantumCircuit(1, 1)
    # base_circ.append(pi_gate, [0])
    # base_circ.add_calibration(pi_gate, (qubit,), sched, [amp])
    # base_circ.measure(0, 0)
    # circs = [
    #     transpile(base_circ.assign_parameters(
    #             {amp: a},
    #             inplace=False
    #     ), backend=backend) for a in amplitudes]

    circs = [add_circ(a, duration, sigma, qubit=qubit) for a in amplitudes]
    pm_circs = pm.run(circs)
    # rabi_schedule = schedule(circs[-1], backend)
    # rabi_schedule.draw(backend=backend)

    # # with real provider
    # job_manager = IBMQJobManager()
    # pi_job = job_manager.run(
    #     circs,
    #     backend=backend,
    #     shots=num_shots_per_exp,
    #     max_experiments_per_job=max_experiments_per_job
    # )
    # pi_sweep_results = pi_job.results()
    # print("Job ID:", pi_job.job_set_id())
    # #
    # # with Fake provider only
    # pi_job = backend.run(
    #     circs,
    #     backend=backend,
    #     shots=num_shots_per_exp
    # )
    # print("Job ID:", pi_job.job_id())
    # pi_sweep_results = pi_job.result()
    # #

    values, job_ids = run_jobs(pm_circs, backend, duration, num_shots_per_exp=num_shots_per_exp, pm=pm)
   
    # print(amplitudes, np.real(pi_sweep_values))
    plt.figure(3)
    plt.scatter(amplitudes, np.real(values), color='black') # plot real part of sweep values
    plt.title("Rabi Calibration Curve")
    plt.xlabel("Amplitude [a.u.]")
    plt.ylabel("Transition Probability")
    if save:
        plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_dur_{duration}_s_{int(sigma)}_N_{N}_beta_{beta}_areacal.png"))
    datapoints = np.vstack((amplitudes, np.real(values)))
    with open(os.path.join(data_folder, f"area_calibration_{date.strftime('%H%M%S')}.pkl"), "wb") as f:
        pickle.dump(datapoints, f)
    ## fit curve
    def fit_function(x_values, y_values, function, init_params):
        try:
            fitparams, conv = curve_fit(
                function, 
                x_values, 
                y_values, 
                init_params, 
                maxfev=100000, 
                bounds=(
                    [-0.6, 1, 0, -0.025, 0.4], 
                    [-0.40, 1e4, 100, 0.025, 0.6]
                )
            )
        except ValueError:
            return 100000, 100000
        y_fit = function(x_values, *fitparams)
        
        return fitparams, y_fit

    def mae_function(x_values, y_values, function, init_params):
        return np.sum(np.abs(fit_function(x_values, y_values, function, init_params)[1] - y_values))

    # rabi_fit_params, _ = fit_function(
    #     amplitudes[: fit_crop_parameter],
    #     np.real(values[: fit_crop_parameter]), 
    #     lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * (x - x0))))) + B,
    #     [-0.47273362, l, p, 0, 0.47747625]
    #     # lambda x, A, k, B: A * (np.cos(k * x)) + B,
    #     # [-0.5, 50, 0.5]
    # )

    # bounds = list(zip([-0.47273363, 20, 0.49, 0, 0.47], [-.4727335, 1e4, 0.5, 0.01, 0.48]))
    
    # differential evolution
    # strategy = 'best1bin'  # The differential evolution strategy
    # population_size = 100  # The number of candidate solutions in each generation
    # maxiter = 10000  # The maximum number of iterations
    # mutation = (0.5, 1.0)  # The mutation factor range
    # recombination = 0.7  # The crossover probability range
    # result = differential_evolution(
    #     lambda init_params: mae_function(
    #         amplitudes[: fit_crop_parameter], 
    #         np.real(values[: fit_crop_parameter]), 
    #         lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * (x - x0))))) + B, 
    #         init_params 
    #     ), 
    #     bounds, strategy=strategy,
    #     popsize=population_size, maxiter=maxiter,
    #     mutation=mutation, recombination=recombination,
    #     x0=[-0.47273362, l, p, 0, 0.47747625]
    # )

    # optimal_params = result.x
    # min_error = result.fun

    max_l = 10000
    mae_threshold = 0.6
    ls = np.arange(1, 33)**2
    ls = ls[ls >= l]
    for current_l in ls:
        if mae_function(
            amplitudes[: fit_crop_parameter], 
            np.real(values[: fit_crop_parameter]), 
            lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * (x - x0))))) + B, 
            [-0.47273362, current_l, p, x0, 0.47747625]
        ) < mae_threshold:
            l = current_l
            break

    rabi_fit_params, _ = fit_function(            
        amplitudes[: fit_crop_parameter], 
        np.real(values[: fit_crop_parameter]), 
        lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * (x - x0))))) + B, 
        [-0.47273362, l, p, 0, 0.47747625] 
    )

    print(rabi_fit_params)
    A, l, p, x0, B = rabi_fit_params
    pi_amp = -np.log(1 - np.pi / l) / p + x0 #np.pi / (k)
    half_amp = -np.log(1 - np.pi / (2 * l)) / p + x0 #np.pi / (k)

    detailed_amps = np.arange(amplitudes[0], amplitudes[-1], amplitudes[-1] / 2000)
    extended_y_fit = A * (np.cos(l * (1 - np.exp(- p * (detailed_amps - x0))))) + B

    ## create pandas series to keep calibration info
    param_dict = {
        "date": [current_date],
        "time": [time],
        "pulse_type": [pulse_type],
        "A": [A],
        "l": [l],
        "p": [p],
        "x0": [x0],
        "B": [B],
        "pi_amp": [pi_amp],
        "half_amp": [half_amp],
        "drive_freq": [center_frequency_Hz],
        "duration": [duration],
        "sigma": [sigma],
        "rb": [int(remove_bg)],
        "N": [N],
        "beta": [beta],  
        "job_id": None # [",".join(job_ids)]
    }
    print(param_dict)
    if save:
        with open(os.path.join(data_folder, f"fit_params_area_cal_{date.strftime('%H%M%S')}.pkl"), "wb") as f:
            pickle.dump(param_dict, f)

        new_entry = pd.DataFrame(param_dict)
        params_file = os.path.join(calib_dir, "actual_params.csv")
        if os.path.isfile(params_file):
            param_df = pd.read_csv(params_file)
            param_df = add_entry_and_remove_duplicates(param_df, new_entry)
            # param_df = pd.concat([param_df, param_series.to_frame().T], ignore_index=True)
            param_df.to_csv(params_file, index=False)
        else:
            new_entry.to_csv(params_file, index=False)

    plt.figure(4)
    plt.scatter(amplitudes, np.real(values), color='black')
    plt.plot(detailed_amps, extended_y_fit, color='red')
    plt.xlim([min(amplitudes), max(amplitudes)])
    plt.title("Fitted Rabi Calibration Curve")
    plt.xlabel("Amplitude [a.u.]")
    plt.ylabel("Transition Probability")
    # plt.ylabel("Measured Signal [a.u.]")
    if save:
        plt.savefig(os.path.join(save_dir, date.strftime("%H%M%S") + f"_{pulse_type}_N_{N}_beta_{beta}_pi_amp_sweep_fitted.png"))

    plt.show()