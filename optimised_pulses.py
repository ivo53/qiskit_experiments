import numpy as np
from scipy.optimize import minimize
import symengine as sym
import matplotlib.pyplot as plt

def lor(t, args):
    return args[0] / (1 + ((t - args[1]/2) / args[2]) ** 2) * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)

def generic_shape(t, args):
    return (args[0] / np.cosh((t - args[1] / 2) / args[2]) + \
        args[3] / (1 + ((t - args[4]/2) / args[5]) ** 2) + \
        args[6] * np.exp(-np.abs((t - args[7]/2) / args[8])) + \
        args[9] * np.exp(-np.abs((t - args[10]/2) / args[11]) ** 2) + args[12]) \
            * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)

def adiab_cond(t_, args, func):
    omega = func(t_, args)
    omega_dot = np.diff(omega)
    omega_dot = np.append(omega_dot, np.abs(omega[-1]))
    # adiabaticity = np.abs(omega_dot * D / ((omega ** 2 + D ** 2) ** (3/2)))
    theta1_dot = omega_dot * D / (omega ** 2 + D ** 2)
    eps1 = (omega ** 2 + D ** 2) ** (1/2)
    adiabaticity = np.abs(theta1_dot / eps1)
    # Now calculate theta2 and thus the superadiabaticity
    eps1_dot = np.diff(eps1)
    eps1_dot = np.append(eps1_dot, eps1[-1] - D)
    theta1_ddot = np.diff(theta1_dot)
    theta1_ddot = np.append(theta1_ddot, np.abs(theta1_dot[-1]))
    theta2_dot = (eps1 * theta1_ddot - eps1_dot * theta1_dot) / (eps1 ** 2 + 4 * theta1_dot ** 2)
    eps2 = (eps1 ** 2 + 4 * theta1_dot ** 2) ** (1/2)
    superadiabaticity = np.abs(theta2_dot / eps2)
    
    return np.amax(superadiabaticity)

def find_nth_adiabaticity(t_, args, func, n=5):
    omega = func(t_, args)
    thetan = 0.5 * arctan
T, D = 5, 0.1
t_ = np.linspace(-2*T,2*T, 500)
# Define bounds for optimization
bounds = 2 * np.array([(-1, 1), (-1, 1), (-1, 1), 
          (-1, 1), (-1, 1), (-1, 1), 
          (-1, 1), (-1, 1), (-1, 1), 
          (-1, 1), (-1, 1), (-1, 1), 
          (-1, 1)])

# Initial guess for optimization

res, it = 1, 1878
# Perform optimization
while res > 0.14:
    np.random.seed(it)
    x0 = np.random.random(13)
    result = minimize(lambda args: adiab_cond(t_, args, generic_shape), x0, options={"maxiter": 100000})
    res = result.fun
    it += 1

# Print the optimal value
print("Optimal value:", result.fun)
print(result.x)
print("Similar Lor value: ", adiab_cond(t_, [1,0,0.1], lor))
fig, ax = plt.subplots(1,2,figsize=(12,6))
ax[0].plot(t_, generic_shape(t_, result.x))
ax[1].plot(t_, lor(t_, [1,0,0.1]))
plt.show()
