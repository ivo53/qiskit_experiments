import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.integrate import quad
from scipy.integrate import solve_ivp

def omega(t, omega_0, sigma, lamb=1):
    return omega_0 / (1 + (t / sigma) ** 2) ** lamb

def omega_dot(t, omega_0, sigma, lamb=1):
    return - lamb * omega_0 / (1 + (t / sigma) ** 2) ** (lamb + 1) * 2 * t / sigma

sigma = 24.8888888889e-9
T = 704e-9
omega_0 = np.pi / quad(lambda t: omega(t, 1, sigma), -T/2, T/2)[0]
# print(omega_0)
# def c2(t, y, d, omega_0, sigma, lamb):
#     # Define the second-order differential equation
#     dc2_2dt_2 = (omega_dot(t, omega_0, sigma, lamb=lamb) /\
#                  omega(t, omega_0, sigma, lamb=lamb) - 1j * d) * y[1] + \
#                  (1j * omega_dot(t, omega_0, sigma, lamb=lamb) * \
#                  d / omega(t, omega_0, sigma, lamb=lamb) - \
#                  omega(t, omega_0, sigma, lamb=lamb) ** 2 / 4) * y[0]
#     return y[2], np.array(dc2_2dt_2)

 

# Define the initial conditions
y0 = [0, 1]  # Example initial conditions: y(0) = 0, y'(0) = 1
l = 1

det = np.linspace(-100e6, 100e6, 100)
# Solve the second-order differential equation
c2_values = np.empty((len(det)), dtype=np.ndarray)
c2dt_values = np.empty((len(det)), dtype=np.ndarray)
c2dt2_values = np.empty((len(det)), dtype=np.ndarray)
t_values = np.empty((len(det)), dtype='float')
for i, d in enumerate(det):
    y = solve_ivp(lambda t, y: c2(t, y, d, omega_0, sigma, l), (-T/2, T/2), [0, 0, 0], max_step=1e-8)
    c2_values[i] = y.y[0][-1]
    c2dt_values[i] = y.y[1][-1]
    # c2dt2_values[i] = y.y[2][-1]
    t_values[i] = y.t[-1]

print(c2_values)
# Plot the solution
plt.plot(det, np.abs(c2_values) ** 2, 'r', label='y(t)')  # Plotting y(t)
plt.xlabel('Time')
plt.ylabel('Transition probability')
plt.legend()
plt.show()
