import numpy as np
from scipy.integrate import solve_ivp
from numba import jit
import matplotlib.pyplot as plt

# Step 1: Generate initial random values
num_points = 20
initial_times = np.linspace(0, 1, num_points)
initial_values = np.random.rand(num_points)

 
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

interpolated_times, interpolated_values = construct_rabi_pulse(initial_times, initial_values)

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

def Omega_t(t):
    if isinstance(t, int) or isinstance(t, float):
        time_idx = np.argmin(
            np.abs(initial_times - t)
        )
    elif isinstance(t, np.ndarray):
        time_idx = np.argmin(
            np.abs(initial_times[:, None] - t[None], axis=0)
        )

    return construct_rabi_pulse(initial_times, initial_values)[1][time_idx]

print(Omega_t(0.5), Omega_t(0.3))
# Define the coupled differential equations for the two levels
def schrodinger(t, psi, Omega_t, Delta):
    psi1, psi2 = psi
    dpsi1_dt = -1j * (-0.5 * Delta * psi1 + 0.5 * Omega_t(t) * psi2)
    dpsi2_dt = -1j * (0.5 * Omega_t(t) * psi1 + 0.5 * Delta * psi2)
    return [dpsi1_dt, dpsi2_dt]

# Initial state (ground state)
psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])

# Time points where the solution is computed
t_span = (0, 1)
t_eval = np.linspace(0, 1, 1000)

# Solve the Schrödinger equation
sol = solve_ivp(schrodinger, t_span, psi0, t_eval=t_eval, args=(Omega_t, Delta), method='RK45')

# Extract the probabilities
P_excited = np.abs(sol.y[1])**2
P_ground = np.abs(sol.y[0])**2

# Plot the results
plt.plot(sol.t, P_ground, label='Ground State')
plt.plot(sol.t, P_excited, label='Excited State')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.title('Time Evolution of a Two-Level System with Detuning')
plt.show()