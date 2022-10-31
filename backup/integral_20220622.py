import sympy
import numpy as np

delta = sympy.Symbol("D")
omega_0 = sympy.Symbol("O")
T = sympy.Symbol("T")
t = sympy.Symbol("t")
func = (delta * omega_0 * np.pi * sympy.cos(np.pi * t / T) / T) ** 2 / (2 * (omega_0 * sympy.sin(np.pi * t / T)) ** 5 + 5 * delta ** 2 * (omega_0 * sympy.sin(np.pi * t / T)) ** 3)
i = sympy.integrate(func, t)
print(i)