import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from numba import jit
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

# Step 1: Generate initial random values
method = 'SLSQP'
dt_now = datetime.now()

# # Step 2: Interpolate between each pair of values using sine functions
# def sine_interpolation(t, t0, t1, y0, y1):
#     """Interpolate between points (t0, y0) and (t1, y1) using a sine function."""
#     return 0.5 * (y0 - y1) * (np.cos(np.pi * (t - t0) / (t1 - t0)) - 1) + y0

# def construct_rabi_pulse(initial_times, initial_values, num_interpolation_points=1000):
#     # Generate time values for interpolation
#     interpolated_times = np.linspace(0, 1, num_interpolation_points)
#     interpolated_values = np.zeros_like(interpolated_times)

#     # Interpolate values using sine functions between each pair of initial points
#     for i in range(num_points - 1): 
#         mask = (interpolated_times >= initial_times[i]) & (interpolated_times <= initial_times[i + 1])
#         interpolated_values[mask] = sine_interpolation(
#             interpolated_times[mask], initial_times[i], initial_times[i + 1], initial_values[i], initial_values[i + 1]
#         )
#     return interpolated_times, interpolated_values

# # Option 1: Use N points and interpolation with sinusoidal curves between them
# num_points = 20
# initial_times = np.linspace(0, 1, num_points)
# initial_values = np.random.rand(num_points)

# times, rabi_values = construct_rabi_pulse(initial_times, initial_values)

def make_all_dirs(path):
    folders = path.split("/")
    for i in range(2, len(folders) + 1):
        folder = "/".join(folders[:i])
        if not os.path.isdir(folder):
            os.mkdir(folder)

# Define the constant detuning
Delta = 0.5  # You can adjust this value as needed

# Define the Hamiltonian with detuning
def hamiltonian(t, Omega_t, Delta):
    return 0.5 * np.array([[-Delta, Omega_t(t)], [Omega_t(t), Delta]])

# Define the coupled differential equations for the two levels
def schrodinger(t, psi, Omega_t, Delta):
    psi1, psi2 = psi
    dpsi1_dt = -1j * (-0.5 * Delta * psi1 + 0.5 * Omega_t(t) * psi2)
    dpsi2_dt = -1j * (0.5 * Omega_t(t) * psi1 + 0.5 * Delta * psi2)
    return [dpsi1_dt, dpsi2_dt]

def calculate_probabilities(t, Omega_t, Delta, T):
    # Initial state (ground state)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    # Time points where the solution is computed
    t_span = (0, T)
    t_eval = np.linspace(0, T, 100)

    # Solve the Schrödinger equation
    sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval, args=(Omega_t, Delta), method='RK45')
    # Extract the probabilities
    P_excited = (np.abs(sol.y[1])**2)[-1]
    P_ground = (np.abs(sol.y[0])**2)[-1]

    # # Plot the results
    # plt.plot(sol.t, np.abs(sol.y[0])**2, label='Ground State')
    # plt.plot(sol.t, np.abs(sol.y[1])**2, label='Excited State')
    # plt.xlabel('Time')
    # plt.ylabel('Probability')
    # plt.legend()
    # plt.title('Time Evolution of a Two-Level System with Detuning')
    # plt.show()

    return P_excited

def loss(values, d_range, T, idx):
    # The loss is calculated as \Delta^{(5)}_{1/2} \times \Delta^{(3)}_{1/2} / (\Delta^{(1)}_{1/2})^2
    # freq_amps = freq_amps[:len(freq_amps)//2] + 1j * freq_amps[len(freq_amps)//2:]
    values = values[:len(values)//2] + 1j * values[len(values)//2:]
    # values = np.fft.ifft(freq_amps)
    times = np.linspace(0, T, len(values))
    raw_area = np.sum(np.abs(values[1:] * np.diff(times)))
    A = np.pi / raw_area # multiplier to correct the area
    def Omega_t(t, A, values):
        if isinstance(t, int) or isinstance(t, float):
            time_idx = np.argmin(
                np.abs(times - t)
            )
        elif isinstance(t, np.ndarray):
            time_idx = np.argmin(
                np.abs(times[:, None] - t[None], axis=0)
            )
        return A * values[time_idx]
    
    iqr_1_3_5pi = []
    for multiplier in [1,3,5]:
        v = []
        for d in d_range:
            v.append(calculate_probabilities(times, lambda t: Omega_t(t, multiplier * A, values), d, T))
        v = np.array(v)
        # plt.plot(v)
        # plt.show()
        iqr_metric = scipy.stats.iqr(v)
        iqr_1_3_5pi.append(iqr_metric)

    loss_value = np.abs(iqr_1_3_5pi[1] * iqr_1_3_5pi[2] / iqr_1_3_5pi[0]**2)

    if len(function_values) == 0:
        last_record.append(loss_value)

    if loss_value / last_record[-1] < 0.995:
        print(f"Function value at step {len(function_values) + 1}: {loss_value}")
        last_record.append(loss_value)
    function_values.append(loss_value)
    parameters.append(values)
    # Save every 1000 steps
    if function_values and len(function_values) % 1000 == 0:
        with open(os.path.join(current_dir, "pulse_optimisation", f"params_{method}_{date}_{time}_guess{idx}_step{len(function_values)}.pkl"), "wb") as f:
            pickle.dump(parameters, f)
        with open(os.path.join(current_dir, "pulse_optimisation", f"losses_{method}_{date}_{time}_guess{idx}_step{len(function_values)}.pkl"), "wb") as f:
            pickle.dump(function_values, f)

    return loss_value

N = 1024
d_min, d_max, num_d = -10, 10, 101
T = 10
d_range = np.linspace(d_min, d_max, num_d)

current_dir = os.path.dirname(__file__)
time = dt_now.strftime("%H%M%S")
date = dt_now.strftime("%Y-%m-%d")
make_all_dirs(os.path.join(current_dir, "pulse_optimisation"))

options = {"maxiter": 10000}

initial_guesses = []
A, sigma = 1, 10  # Standard deviation of the Lorentzian
z0 = np.fft.fft(A / (1 + ((np.arange(N) - N/2) / sigma)**2) ** 0.5)
x0 = np.concatenate([z0.real, z0.imag])

initial_guesses.append(x0)
for _ in range(999):
    x0 = np.random.uniform(-10, 10, 1024)
    initial_guesses.append(x0)

for idx, guess in enumerate(initial_guesses):
    # Create a list to store function values
    function_values = []
    last_record = []
    parameters = []
    result = minimize(
        loss, guess, 
        args=(d_range, T, idx), 
        method=method,
        options=options
    )


# Reconstruct the final complex parameters from the result
final_z = result.x[:len(result.x)//2] + 1j * result.x[len(result.x)//2:]