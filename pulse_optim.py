import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from numba import jit
import matplotlib.pyplot as plt

# Step 1: Generate initial random values
method = "fourier"

 
# Step 2: Interpolate between each pair of values using sine functions
def sine_interpolation(t, t0, t1, y0, y1):
    """Interpolate between points (t0, y0) and (t1, y1) using a sine function."""
    return 0.5 * (y0 - y1) * (np.cos(np.pi * (t - t0) / (t1 - t0)) - 1) + y0

def construct_rabi_pulse(initial_times, initial_values, num_interpolation_points=1000):
    # Generate time values for interpolation
    interpolated_times = np.linspace(0, 1, num_interpolation_points)
    interpolated_values = np.zeros_like(interpolated_times)

    # Interpolate values using sine functions between each pair of initial points
    for i in range(num_points - 1): 
        mask = (interpolated_times >= initial_times[i]) & (interpolated_times <= initial_times[i + 1])
        interpolated_values[mask] = sine_interpolation(
            interpolated_times[mask], initial_times[i], initial_times[i + 1], initial_values[i], initial_values[i + 1]
        )
    return interpolated_times, interpolated_values

def transform_fourier_amplitudes(fourier_amplitudes):

    # Step 2: Apply the Inverse FFT
    time_domain_signal = np.fft.ifft(amplitudes)

    return times, time_domain_signal

# Option 1: Use N points and interpolation with sinusoidal curves between them
num_points = 20
initial_times = np.linspace(0, 1, num_points)
initial_values = np.random.rand(num_points)

times, rabi_values = construct_rabi_pulse(initial_times, initial_values)

# Option 2: Simply choose the Fourier domain amplitudes and inverse transform into time domain
amplitudes = np.zeros(N, dtype=complex)
# Example: Initialize a Gaussian shape in the Fourier domain
A, sigma = 1, 10  # Standard deviation of the Lorentzian
amplitudes = np.fft.fft(A / (1 + ((np.arange(N) - N/2) / sigma)**2))

times, rabi_values = transform_fourier_amplitudes(fourier_amplitudes)


# Plot the results
plt.plot(interpolated_times, interpolated_values, label='Interpolated')
plt.scatter(initial_times, initial_values, color='red', label='Initial Points')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Rabi Frequency')
plt.title('Sine Interpolation of Rabi Frequency')
plt.show()


# Define the constant detuning
Delta = 0.5  # You can adjust this value as needed

# Define the Hamiltonian with detuning
def hamiltonian(t, Omega_t, Delta):
    return 0.5 * np.array([[-Delta, Omega_t(t)], [Omega_t(t), Delta]])

# def Omega_t(t):
#     if isinstance(t, int) or isinstance(t, float):
#         time_idx = np.argmin(
#             np.abs(initial_times - t)
#         )
#     elif isinstance(t, np.ndarray):
#         time_idx = np.argmin(
#             np.abs(initial_times[:, None] - t[None], axis=0)
#         )
#     if method == "interpolation":
#         return construct_rabi_pulse(initial_times, initial_values)[1][time_idx]
#     else:
#         return transform_fourier_amplitudes(np.fft.fft(initial_values))[1][time_idx]

# print(Omega_t(0.5), Omega_t(0.3))
# Define the coupled differential equations for the two levels
def schrodinger(t, psi, Omega_t, Delta):
    psi1, psi2 = psi
    dpsi1_dt = -1j * (-0.5 * Delta * psi1 + 0.5 * Omega_t(t) * psi2)
    dpsi2_dt = -1j * (0.5 * Omega_t(t) * psi1 + 0.5 * Delta * psi2)
    return [dpsi1_dt, dpsi2_dt]

def calculate_probabilities(t, Omega_t):
    # Initial state (ground state)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

    # Time points where the solution is computed
    t_span = (0, 1)
    t_eval = np.linspace(0, 1, 1000)

    # Solve the Schrödinger equation
    sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval, args=(Omega_t, Delta), method='RK45')

    # Extract the probabilities
    P_excited = np.abs(sol.y[1])**2 [-1]
    P_ground = np.abs(sol.y[0])**2 [-1]

    return P_excited

def loss(freq_amps, d_range, T):
    # The loss is calculated as \Delta^{(5)}_{1/2} \times \Delta^{(3)}_{1/2} / (\Delta^{(1)}_{1/2})^2
    values = np.fft.ifft(freq_amps)
    raw_area = np.sum(values[1:] * np.diff(t))
    A = np.pi / raw_area # multiplier to correct the area

    def Omega_t(t, A):
        if isinstance(t, int) or isinstance(t, float):
            time_idx = np.argmin(
                np.abs(initial_times - t)
            )
        elif isinstance(t, np.ndarray):
            time_idx = np.argmin(
                np.abs(initial_times[:, None] - t[None], axis=0)
            )
        return A * values[time_idx]
    
    iqr_1_3_5pi = []
    for multiplier in [1,3,5]:
        values = []
        for d in d_range:
            values.append(calculate_probabilities(t, lambda t: Omega_t(t, multiplier * A)))
        iqr_metric = scipy.stats.iqr(np.arange(values))
        iqr_1_3_5pi.append(iqr_metric)
    return iqr_1_3_5pi[1] * iqr_1_3_5pi[2] / iqr_1_3_5pi[0]**2


d_range = np.linspace(d_min, d_max, num_d)
  
result = minimize(loss, args=(d_range))
# Plot the results
plt.plot(sol.t, P_ground, label='Ground State')
plt.plot(sol.t, P_excited, label='Excited State')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.title('Time Evolution of a Two-Level System with Detuning')
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # Step 1: Initialize Fourier amplitudes
# N = 1024  # Number of points
# frequencies = np.fft.fftfreq(N)  # Frequency bins
# # plt.plot(np.arange(N), frequencies)
# # plt.show()
# amplitudes = np.zeros(N, dtype=complex)

# # Example: Initialize a Gaussian shape in the Fourier domain
# A, sigma = 1, 10  # Standard deviation of the Lorentzian
# amplitudes = np.fft.fft(A / (1 + ((np.arange(N) - N/2) / sigma)**2))

# # Step 2: Apply the Inverse FFT
# time_domain_signal = np.fft.ifft(amplitudes)

# # Step 3: Visualize the result
# plt.figure(figsize=(12, 6))

# # Plot real part of the time-domain signal
# plt.subplot(1, 2, 1)
# plt.plot(np.real(time_domain_signal))
# plt.title('Real Part of Time Domain Signal')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')

# # Plot imaginary part of the time-domain signal
# plt.subplot(1, 2, 2)
# plt.plot(np.imag(time_domain_signal))
# plt.title('Imaginary Part of Time Domain Signal')
# plt.xlabel('Sample')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()