import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_ivp

def omega(t, omega_0, sigma, lamb=1):
    return omega_0 / (1 + (t / sigma) ** 2) ** lamb

def omega_dot(t, omega_0, sigma, lamb=1):
    return - lamb * omega_0 / (1 + (t / sigma) ** 2) ** (lamb + 1) * 2 * t / sigma

sigma = 24.8888888889e-9
T = 704e-9
omega_0 = np.pi / quad(lambda t: omega(t, 1, sigma), -T/2, T/2)[0]

def c2(y, t, delta=0):
    # Define the second-order differential equation
    dc2_2dt_2 = (omega_dot(t, omega_0, sigma) / omega(t, omega_0, sigma) - 1j * delta) * y[1] + (1j * omega_dot(t, omega_0, sigma) * delta / omega(t, omega_0, sigma) - omega(t, omega_0, sigma) ** 2 / 4) * y[0]
    return dc2_2dt_2



# Define the initial conditions
y0 = [0, 1]  # Example initial conditions: y(0) = 0, y'(0) = 1

# Define the time points at which you want to evaluate the solution
t = np.linspace(-10, 10, 100)  # Example time span: 0 to 10 with 100 points
det = np.linspace(-100e6, 100e6, 100)
# Solve the second-order differential equation
for d in det:
    y = solve_ivp(lambda y, t: c2(y, t, d), (-10, 10), [0])

# Plot the solution
plt.plot(t, y[:, 0], 'r', label='y(t)')  # Plotting y(t)
plt.plot(t, y[:, 1], 'b', label="y'(t)")  # Plotting y'(t)
plt.xlabel('Time')
plt.ylabel('Solution')
plt.legend()
plt.show()
