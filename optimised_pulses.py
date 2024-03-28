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
    length = (t[-1] - t[0]) / (2 * n - 1)
    tmin = (2 * i) * length
    tmax = (2 * i) * (length + 1)
    return np.exp(1j * phase) * np.heaviside(t + tmin, 1) * np.heaviside(-t + tmax, 1)

def composites(t, args, n=10):
    comp_sum = np.zeros_like(t)
    for i in range(n):
        comp_sum += comp(t, i, n, phase=args[i])
    comp_sum += args[-1] * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)
    return comp_sum

def generic_shape(t, args, num_pulses=6):
    return (args[0] / np.cosh((t - args[1] / 2) / args[2]) + \
        args[3] / (1 + ((t - args[4]/2) / args[5]) ** 2) + \
        args[6] * np.exp(-np.abs((t - args[7]/2) / args[8])) + \
        args[9] * np.exp(-np.abs((t - args[10]/2) / args[11]) ** 2) + \
        # args[12] / (1 + ((t - args[13]/2) / args[14]) ** 2) ** (1/2) + \
        args[12] * (-np.abs(np.tanh((t - args[13]/2) / args[14])) + 1) + args[15]) \
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
    
    return np.amax(superadiabaticity) * (1 + np.abs(np.sum(omega) - np.sum(lor(t_, [1,0,0.1])))) * (1 + np.abs(np.amax(omega) - 1))

def find_nth_adiabaticity(t_, args, func, n=5):
    omega = func(t_, args)
    thetan = 0.5 * np.arctan(t_)

num_pulses = 5
multiples = 30
T, D = 5, 0.1
t_ = np.linspace(-T, T, 500)

# Define bounds for optimization
bounds = np.array(num_pulses * [(0.1, 5), (-0.2, 0.2), (0.1, 2)] + [(-1, 1)])

# Initialize loss value and iterator
first_res, res, it, max_iter = 0, 1e6, 0, 1e3
while res > first_res / multiples and it < max_iter :
    # Initial guess for optimization
    np.random.seed(it + 3222)
    gen = np.random.Generator(np.random.PCG64())
    x0 = np.array([4.9 * gen.random(num_pulses) + 0.1, 0.4 * gen.random(num_pulses) - 0.2, 1.9 * gen.random(num_pulses) + 0.1])
    x0 = np.append(x0.T.flatten(), 2 * gen.random() - 1)
    # Perform optimization
    result = minimize(lambda args: adiab_cond(t_, args, generic_shape, num_pulses), x0, bounds=bounds, options={"maxiter": 100000})
    if result.fun < res:
        print("New lowest value:", result.fun)
        best_result = result
        res = result.fun
    if it == 2:
        first_res = best_result.fun
        print("Our objective is", np.round(first_res / multiples, 0))
    it += 1
# Print the optimal value
print("Optimal value:", best_result.fun)
print("Seed value:", it+3222)
print(best_result.x)
print("Similar Lor value: ", adiab_cond(t_, [1,0,0.1], lor, num_pulses))
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(t_, generic_shape(t_, best_result.x, num_pulses))
ax[1].plot(t_, lor(t_, [1,0,0.1]))
plt.show()
