import os
import pickle
import argparse
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

resonant_frequencies = {
    "manila": 4962284031.287086,
}
def make_all_dirs(path):
    path = path.replace("\\", "/")
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

def get_closest_multiple_of_16(num):
    return int(num + 8) - (int(num + 8) % 16)

def add_entry_and_remove_duplicates(df, new_entry, cols_to_check=["pulse_type", "duration", "sigma", "rb"]):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--backend", default="manila", type=str,
        help="Backend on which area was calibrated."
    )
    parser.add_argument(
        "-pt", "--pulse_type", default="lor", type=str,
        help="Pulse type (e.g. sq, gauss, sine, sech etc.)."
    )
    parser.add_argument(
        "-T", "--duration", default=1080, type=int,
        help="Pulse duration parameter"
    )
    parser.add_argument(
        "-s", "--sigma", default=180, type=int,
        help="Pulse width parameter"
    )
    parser.add_argument(
        "-rb", "--remove_bg", default=1, type=int,
        help="Whether background was removed."
    )
    parser.add_argument(
        "-d", "--date", default="2023-02-26", type=str,
        help="Date on which pulse area was calibrated."
    )
    parser.add_argument(
        "-l", "--l_init", default=100, type=float,
        help="Initial value for l."
    )
    parser.add_argument(
        "-p", "--p_init", default=0.2, type=float,
        help="Initial value for p."
    )
    # parser.add_argument(
    #     "-t", "--time", default="180032", type=str,
    #     help="Date on which pulse area was calibrated."
    # )
    parser.add_argument(
        "-sv", "--save", default=0, type=int,
        help="Whether to save the result (0 or 1)."
    )
    args = parser.parse_args()
    backend = args.backend
    pulse_type = args.pulse_type
    duration = args.duration
    sigma = args.sigma
    rb = args.remove_bg
    l_init = args.l_init
    p_init = args.p_init
    date = args.date
    # time = args.time
    save = bool(args.save)
    
    backend_name = backend
    backend = "ibm_" + backend \
        if backend in ["perth", "lagos", "nairobi", "oslo"] \
            else "ibmq_" + backend

    file_dir = os.path.dirname(__file__)
    file_dir = os.path.split(file_dir)[0]
    curr_date = datetime.now()
    current_date = curr_date.strftime("%Y-%m-%d")
    load_dir = os.path.join(file_dir, "data", backend_name, "calibration", date)
    calib_dir = os.path.join(file_dir, "calibrations")
    save_dir = os.path.join(calib_dir, date)

    for calib_file in os.listdir(os.path.join(calib_dir, date)):
        split_calib_file = calib_file.split("_")
        if split_calib_file[-1] == "areacal.png":
            if pulse_type == split_calib_file[1] and \
                duration == int(split_calib_file[3]) and \
                    int(sigma) == int(float(split_calib_file[5])):
                time = split_calib_file[0]
                break
    print(time)
    with open(os.path.join(load_dir, f"area_calibration_{time}.pkl"), "rb") as f:
        rabi_data = pickle.load(f)
    
    amps = rabi_data[0]
    tr_probs = rabi_data[1]

    def fit_function(x_values, y_values, function, init_params):
        fitparams, conv = curve_fit(
            function, 
            x_values, 
            y_values, 
            init_params, 
            maxfev=100000, 
            bounds=(
                [-0.51, 0, 0, -0.01, 0.45], 
                [-.45, 1e6, .5, 0.01, 0.55]
            )
        )
        y_fit = function(x_values, *fitparams)
        
        return fitparams, y_fit

    fit_crop_parameter = int(1 * len(amps))
    rabi_fit_params, _ = fit_function(
        amps[: fit_crop_parameter],
        tr_probs[: fit_crop_parameter], 
        lambda x, A, l, p, x0, B: A * (np.cos(l * (1 - np.exp(- p * (x - x0))))) + B,
        [-0.48273362, l_init, p_init, 0.005, 0.47747625]
        # lambda x, A, k, B: A * (np.cos(k * x)) + B,
        # [-0.5, 50, 0.5]
    )

    print(rabi_fit_params)
    A, l, p, x0, B = rabi_fit_params
    pi_amp = x0 - np.log(1 - np.pi / l) / p #((l - np.sqrt(l ** 2 + 4 * k * np.pi)) / (2 * k)) ** 2 #np.pi / (k)
    half_amp =  x0 - np.log(1 - np.pi / (2 * l)) / p #((l - np.sqrt(l ** 2 + 2 * k * np.pi)) / (2 * k)) ** 2 #np.pi / (k)

    detailed_amps = np.arange(amps[0], amps[-1], amps[-1] / 2000)
    extended_y_fit = A * (np.cos(l * (1 - np.exp(- p * (detailed_amps - x0))))) + B

    ## create pandas series to keep calibration info
    param_dict = {
        "date": [date],
        "time": [time],
        "pulse_type": [pulse_type],
        "A": [A],
        "l": [l],
        "p": [p],
        "x0": [x0],
        "B": [B],
        "pi_amp": [pi_amp],
        "half_amp": [half_amp],
        "drive_freq": [resonant_frequencies[backend_name]],
        "duration": [duration],
        "sigma": [sigma],
        "rb": [rb]
    }
    print(param_dict)

    plt.figure(4)
    plt.scatter(amps, tr_probs, color='black')
    plt.plot(detailed_amps, extended_y_fit, color='red')
    plt.xlim([min(amps), max(amps)])
    plt.title("Fitted Rabi Calibration Curve")
    plt.xlabel("Amplitude [a.u.]")
    plt.ylabel("Transition Probability")
    # plt.ylabel("Measured Signal [a.u.]")
    if save:
        plt.savefig(os.path.join(save_dir, time + f"_{pulse_type}_pi_amp_sweep_fitted.png"))
    plt.show()
    # exit()
    if save:
        new_entry = pd.DataFrame(param_dict)
        params_file = os.path.join(calib_dir, "actual_params.csv")
        if os.path.isfile(params_file):
            param_df = pd.read_csv(params_file)
            param_df = add_entry_and_remove_duplicates(param_df, new_entry)
            # param_df = pd.concat([param_df, param_series.to_frame().T], ignore_index=True)
            param_df.to_csv(params_file, index=False)
        else:
            new_entry.to_csv(params_file, index=False)
