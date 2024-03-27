import numpy as np
from scipy.optimize import minimize
import symengine as sym
import matplotlib.pyplot as plt

def lor(t, args, n=1):
    return args[0] / (1 + ((t - args[1]/2) / args[2]) ** 2)**(1/2) * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)

def lorentzian_sum(t, args, n=10):
    Lsum = np.zeros_like(t)
    for i in range(n):
        Lsum += lor(t, args[3*i:3*i+3])
    Lsum += args[-1] * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)
    return Lsum

def comp(t, i, n, phase=0):
    assert phase >= 0 and phase <= 2 * np.pi 

def composites(t, args, n=10):
    comp_sum = np.zeros_like(t)
    for i in range(n):
        comp_sum += comp(t, i, n, phase=args[i])
    comp_sum += args[-1] * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)
    return Lsum

def generic_shape(t, args, num_pulses=10):
    return (args[0] / np.cosh((t - args[1] / 2) / args[2]) + \
        args[3] / (1 + ((t - args[4]/2) / args[5]) ** 2) + \
        args[6] * np.exp(-np.abs((t - args[7]/2) / args[8])) + \
        args[9] * np.exp(-np.abs((t - args[10]/2) / args[11]) ** 2) + args[12]) \
            * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)

def adiab_cond(t_, args, func, num_pulses):
    omega = func(t_, args, num_pulses)
    omega_dot = np.diff(omega) / np.diff(t_)
    theta1_dot = omega_dot * D / (omega[:-1] ** 2 + D ** 2)
    eps1 = (omega ** 2 + D ** 2) ** (1/2)
    adiabaticity = np.abs(theta1_dot / eps1[:-1])
    # Now calculate theta2 and thus the superadiabaticity
    eps1_dot = np.diff(eps1) / np.diff(t_)
    theta1_ddot = np.diff(theta1_dot) / np.diff(t_)[:-1]
    theta2_dot = (eps1[:-2] * theta1_ddot - eps1_dot[:-1] * theta1_dot[:-1]) / (eps1[:-2] ** 2 + 4 * theta1_dot[:-1] ** 2)
    eps2 = (eps1[:-1] ** 2 + 4 * theta1_dot ** 2) ** (1/2)
    superadiabaticity = np.abs(theta2_dot / eps2[:-1])

    return np.amax(superadiabaticity)

def find_nth_adiabaticity(t_, args, func, n=5):
    omega = func(t_, args)
    thetan = 0.5 * np.arctan(t)

num_pulses = 10
T, D = 5, 0.3
t_ = np.linspace(-T, T, 500)

# Define bounds for optimization
bounds = 0.5 * np.array(num_pulses * [(-1, 1), (-3, 3), (-1, 1)] + [(-1, 1)])

# Initialize loss value and iterator
res, it = 1e3, 1878
while res > 20:
    # Initial guess for optimization
    np.random.seed(it)
    gen = np.random.Generator(np.random.PCG64())
    x0 = 2 * gen.random(3 * num_pulses + 1) - 1
    # Perform optimization
    result = minimize(lambda args: adiab_cond(t_, args, lorentzian_sum, num_pulses), x0, bounds=bounds, options={"maxiter": 100000})
    res = result.fun
    it += 1

# Print the optimal value
print("Optimal value:", result.fun)
print(result.x)
print("Similar Lor value: ", adiab_cond(t_, [1,0,0.1], lor, num_pulses))
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(t_, lorentzian_sum(t_, result.x, num_pulses))
ax[1].plot(t_, lor(t_, [1,0,0.1]))
plt.show()
