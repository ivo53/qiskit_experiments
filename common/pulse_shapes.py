import numpy as np
from scipy.integrate import quad_vec
# the equations of all pulse shapes


def integrate(f, a, b, n):
    values = np.linspace(a, b, n)
    return np.sum(f(values)) * (b - a) / n

def normalise(
    f, t, 
    T: float,
    sigma: float
):
    return (f(t, T, sigma) - f(0, T, sigma)) / (1 - f(0, T, sigma))

def rabi(   
    t, 
    T: float,
    sigma: float=None
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1)
    return f

def landau_zener1(
    t,
    T: float,
    beta: float,
    tau: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(1j * 0.5 * beta * tau * ((t - T/2) / tau) ** 2)
    return f

def landau_zener4(
    t,
    T: float,
    beta: float,
    tau: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(1j * 0.5 * beta * tau * np.arctan(np.sinh(((t - T/2) / tau))) ** 2) / np.cosh((t - T/2) / tau)
    return f

def allen_eberly1(
    t,
    T: float,
    beta: float,
    tau: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(-1j * beta * tau * np.log(np.cos((t - T/2) / tau)))
    return f

def allen_eberly5(
    t,
    T: float,
    beta: float,
    tau: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(1j * beta * tau * np.log(np.cosh((t - T/2) / tau))) / np.cosh((t - T/2) / tau)
    return f

def demkov_kunike_2(
    t,
    T: float,
    beta: float,
    tau: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(1j * beta * tau * np.log(np.cosh((t - T/2) / tau)))
    return f

def lorentz(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2)
    return f

def lorentz2(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2) ** 2
    return f

def lorentz3_2(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2) ** (3/2)
    return f

def lorentz3(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2) ** 3
    return f

def lorentz3_4(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2) ** (3/4)
    return f

def lorentz2_3(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2) ** (2/3)
    return f

def lorentz3_5(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / (1 + ((t - T/2) / sigma) ** 2) ** (3/5)
    return f

def sech(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / np.cosh((t - T/2) / sigma)
    return f

def sech2(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) / np.cosh((t - T/2) / sigma) ** 2
    return f

def gauss(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(-((t - T/2) / sigma) ** 2)
    return f

def demkov(
    t, 
    T: float,
    sigma: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.exp(-np.abs((t - T/2) / sigma))
    return f

def sin(
    t, 
    T: float,
    sigma: float=None
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.sin(np.pi * t / T)
    return f

def sin2(
    t, 
    T: float,
    sigma: float=None
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.sin(np.pi * t / T) ** 2
    return f

def sin3(
    t, 
    T: float,
    sigma: float=None
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.sin(np.pi * t / T) ** 3
    return f

def sin4(
    t, 
    T: float,
    sigma: float=None
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.sin(np.pi * t / T) ** 4
    return f

def sin5(
    t, 
    T: float,
    sigma: float=None
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * np.sin(np.pi * t / T) ** 5
    return f

def ipN(
    t,
    T: float,
    N: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * ((t - T/2) / (T/2)) ** (2*N)
    return f

def fcq(
    t,
    T: float,
    beta: float
):
    f = np.heaviside(t, 1) * np.heaviside(T - t, 1) * (1 + beta * (((t - T/2) / (T/2)) ** 2 - 1))
    return f

pulse_shapes = {
    "rabi": rabi,
    "lz": landau_zener,
    "ae": allen_eberly,
    "dk2": demkov_kunike_2,
    "lor": lorentz,
    "lor2": lorentz2,
    "lor3_2": lorentz3_2,
    "lor3": lorentz3,
    "lor3_4": lorentz3_4,
    "lor2_3": lorentz2_3,
    "lor3_5": lorentz3_5,
    "sech": sech,
    "rz": sech,
    "sech2": sech2,
    "gauss": gauss,
    "demkov": demkov,
    "sin": sin,
    "sin2": sin2,
    "sin3": sin3,
    "sin4": sin4,
    "sin5": sin5,
    "ip1": lambda t, T: ipN(t, T, 1),
    "ip2": lambda t, T: ipN(t, T, 2),
    "ip3": lambda t, T: ipN(t, T, 3),
    "ip4": lambda t, T: ipN(t, T, 4),
    "ip5": lambda t, T: ipN(t, T, 5),
    "ip6": lambda t, T: ipN(t, T, 6),
    "ip7": lambda t, T: ipN(t, T, 7),
    "ip8": lambda t, T: ipN(t, T, 8),
    "ip9": lambda t, T: ipN(t, T, 9),
    "ip10": lambda t, T: ipN(t, T, 10),
    "ipN": ipN,
    "fcq": fcq
}

def rabi_freq(t, omega_0, pulse_type: str, T: float, sigma: float=None, rb: bool=True):
    if not ("sin" in pulse_type or pulse_type == "rabi"):
        if sigma is None:
            raise ValueError(f"You need to provide pulse width for the {pulse_type} pulse type.")
    else:
        rb = False
    if rb:
        f = normalise(pulse_shapes[pulse_type], t, T, sigma)
    else:
        f = pulse_shapes[pulse_type](t, T, sigma)
    return omega_0 * f
    
def dimensionless_pulse_shape(t, pulse_type: str, T: float, sigma: float=None):
    if not ("sin" in pulse_type or pulse_type == "rabi"):
        if sigma is None:
            raise ValueError(f"You need to provide pulse width for the {pulse_type} pulse type.")
    else:
        rb = False
    if rb:
        f = normalise(pulse_shapes[pulse_type], t, T, sigma)
    else:
        f = pulse_shapes[pulse_type](t, T, sigma)
    return f

def find_pulse_width(pulse_type: str, T: float, sigma: float=None, rb: bool=True):
    if not ("sin" in pulse_type or pulse_type == "rabi"):
        if sigma is None:
            raise ValueError(f"You need to provide pulse width for the {pulse_type} pulse type.")
    else:
        rb = False
    if rb:
        tau = quad_vec(
            lambda t: normalise(pulse_shapes[pulse_type], t, T, sigma), 
            0, T, 
            epsabs=1e-6, epsrel=1e-6
        )[0]
        # tau = integrate(
        #     lambda t: normalise(pulse_shapes[pulse_type], t, T, sigma), 
        #     0, T, 1000000
        # )
    else:
        tau = quad_vec(
            lambda t: pulse_shapes[pulse_type](t, T, sigma), 
            0, T, 
            epsabs=1e-6, epsrel=1e-6
        )[0]
        # tau = integrate(
        #     lambda t: pulse_shapes[pulse_type](t, T, sigma), 
        #     0, T, 1000000
        # )

    return tau

def find_rabi_amp(pulse_type: str, T: float, sigma: float=None, pulse_area: float=np.pi, rb: bool=True):
    if "_" in pulse_type and len(pulse_type.split("_")[1]) > 2:
        pulse_type = pulse_type.split("_")[0]
    tau = find_pulse_width(pulse_type, T, sigma, rb)
    return pulse_area / tau