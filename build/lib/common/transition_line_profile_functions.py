import numpy as np
import mpmath as mp
import scipy.special as sp
from scipy.optimize import curve_fit, root
from scipy.integrate import quad, quad_vec

import common.pulse_shapes as ps

RABI_FREQ = {
    "constant": 6266474.70796,
    "rabi": 6266474.70796,
    "rz": 42874911.4203,
    "gauss": 25179780.7441,
    # "demkov": 31739880.846,
    "demkov": 28438933.238,
    "sech2": 35460561.388,
    "sin": 23131885.3151,
    "sin2": 25244940.9663,
    "sin3": 26023370.9794
}

Tt = {
    "constant": 1504e-9 / 3,
    "rabi": 1504e-9 / 3,
    "rz": (284 + 4/9) * 1e-9,
    "gauss": (398 + 2/9) * 1e-9,
    "demkov": (572 + 4/9) * 1e-9,
    "sech2": (284 + 4/9) * 1e-9,
    "sin": (213 + 1/3) * 1e-9,
    "sin2": (248 + 8/9) * 1e-9,
    "sin3": (284 + 4/9) * 1e-9,

}

SIGMA = {
    "rz": 23.39181 * 1e-9,
    "gauss": (49 + 7/9) * 1e-9,
    "demkov": (55 + 5/9) * 1e-9,
    "sech2": (44 + 4/9) * 1e-9
}

ALPHA = {
    "sech2": 0.4494679707017059,
    "sin": 0.840753969701287,
    "sin2": 0.8022417161585951,
    "sin3": 0.7776880847006185,
    "sin4": 0.74, # speculative
    "sin5": 0.72, # speculative
    "gauss": 0.6758103186913479,
    "demkov": 0.15786564335245298,
    "lor": 0.5,
    "lor2": -0.4773936486396947, # +- 0.01572690541716722
    "lor2_3": -0.4773936486396947, # +- 0.01572690541716722
    "lor3_4": -0.4773936486396947, # +- 0.01572690541716722
    "lor2": -0.4773936486396947, # +- 0.01572690541716722
    "lor2": -0.4773936486396947, # +- 0.01572690541716722
}

def fit_function(x_values, y_values, function, init_params, lower, higher, sigma=None, duration=None, time_interval=0.5e-9, remove_bg=True, area=np.pi):
    global s
    global dur
    global rb
    global a
    global dt
    s = sigma
    dur = sigma if duration is None else duration
    rb = remove_bg
    a = area
    dt = time_interval
    fitparams, conv = curve_fit(
        function, 
        x_values, y_values, init_params, 
        maxfev=1e6, 
        bounds=(lower, higher))
    y_fit = function(x_values, *fitparams)
    perr = np.sqrt(np.diag(conv))
    return fitparams, y_fit, perr

def post_process(P2, eps, delta):
    # eps = eps0 + 1/2 * (1 - eps1)
    # delta = 1/2 * (1 - eps1) * e^(-gamma * t)
    return eps + delta * (2 * P2 - 1)

def with_dephasing(P2, egamma):
    return P2 * egamma - egamma / 2 + 1 / 2

def lorentzian(x, s, A, q_freq, c):
    return A / (((x - q_freq) / s) ** 2 + 1) + c
    # return A / np.cosh(((x - q_freq) / s))**2 + c
    
def rz(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    O = ps.find_rabi_amp("rz", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    P2 = np.sin(0.5 * np.pi * O * sigma) ** 2 \
        / np.cosh(0.5 * np.pi * D * sigma) ** 2
    return post_process(P2, eps, delta)

def demkov(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("demkov", T, sigma, rb=rb, pulse_area=a)

    s_inf = np.pi * omega_0 * sigma
    D = (x - q_freq) * 1e6 * sigma / (2 * np.pi)
    bessel11 = np.array([complex(mp.besselj(1/2 + 1j * a / 2, s_inf / (2 * np.pi))) for a in D])
    bessel21 = np.array([complex(mp.besselj(-1/2 - 1j * a / 2, s_inf / (2 * np.pi))) for a in D])
    P2 = (s_inf / 4) ** 2 * np.abs(2 * np.real(bessel11 * bessel21)) ** 2 / np.cosh(D * np.pi / 2) ** 2
    return post_process(P2, eps, delta)

def sech_sq(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("sech2", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / np.cosh((t) / sigma) ** 2 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -T/2, T/2, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sech2"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sin(x, q_freq, delta, eps):
    T = dur * dt
    sigma = T / np.pi
    omega_0 = ps.find_rabi_amp("sin", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos(t / sigma) 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sin"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)


def double_approx(x, q_freq, delta, eps, tau, pulse_type):
    T = dur * dt
    sigma = s * dt
    omega_0 = ps.find_rabi_amp(pulse_type, T, sigma, pulse_area=a, rb=rb)
    sigma /= T
    omega_0 *= T
    D = (x - q_freq) * 1e6 * T
    omega = lambda t: ps.rabi_freq(t, omega_0, pulse_type, 1, sigma, rb=rb)
    beta = np.sqrt(np.pi * omega(tau))
    d = (D / (2 * beta))
    eta = np.abs(D) * sp.ellipeinc(np.pi, -omega_0**2 / D**2) / np.pi
    chi1 = d**2 / 2 + np.angle(sp.gamma(1/2 * (1 + 1j * d**2))) \
        - d**2 / 2 * np.log(d**2 / 2)
    chi2 = -np.pi / 4 - d**2 / 2 - np.angle(sp.gamma(1j * d**2 / 2)) \
        + d**2 / 2 * np.log(d**2 / 2)
    P2 = 1 / 4 * ((1 + np.exp(-np.pi * d**2)) * np.sin(eta / 2 - 2 * chi1)\
        - (1 - np.exp(-np.pi * d**2)) * np.sin(eta / 2 + 2 * chi2)) ** 2
    return post_process(P2, eps, delta)

def sin_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sin")
def sin2_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sin2")
def sin3_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sin3")
def sin4_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sin4")
def sin5_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sin5")
def sech_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sech")
def sech2_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sech2")
def gauss_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "gauss")
def demkov_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "demkov")
def lor_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "lor")
def lor2_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "lor2")
def lor3_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "lor3")

def rlzsm_approx(x, q_freq, delta, eps, tau, pulse_type):
    T = dur * dt
    sigma = s * dt
    # sigma = T / np.pi
    omega_0 = ps.find_rabi_amp(pulse_type, T, sigma, rb=rb, pulse_area=a)
    # sigma /= T 
    sigma /= T
    omega_0 *= T
    D = (x - q_freq) * 1e6 * T
    omega = lambda t: ps.rabi_freq(t, omega_0, pulse_type, 1, sigma, rb=rb)

    beta = np.sqrt(np.pi * omega(tau))
    d = (D / (2 * beta))
    alpha = beta * tau


    def ParabolicCylinderD(v, z, precision=30):
        m = np.arange(precision)
        Dv = 2**(-(v/2 + 1)) * np.exp(-z**2/4) / sp.gamma(-v) * np.sum((-1)**m[:, None] / sp.gamma(m[:, None] + 1) * \
            sp.gamma((m[:, None]-v[None])/2) * (np.sqrt(2) * z)**m[:, None], axis=0)
        return Dv

    def eta_(D, omega_0, tau):
        return (np.sqrt(D**2) * sp.ellipeinc(np.pi * (1 - tau), -(omega_0**2 / D**2))) / np.pi - \
            (np.sqrt(D**2) * sp.ellipeinc(np.pi * tau, -(omega_0**2 / D**2))) / np.pi

    def Uad(D, omega_0, tau):
        eta = eta_(D, omega_0, tau)
        return np.array([[np.cos(eta/2) + (1j * D * np.sin(eta/2))/np.sqrt(D**2 + omega(tau)**2),
                        -((1j * np.sin(eta/2) * omega(tau))/np.sqrt(D**2 + omega(tau)**2))],
                        [-((1j * np.sin(eta/2) * omega(tau))/np.sqrt(D**2 + omega(tau)**2)),
                        np.cos(eta/2) - (1j * D * np.sin(eta/2))/np.sqrt(D**2 + omega(tau)**2)]])

    def A(d, alpha):
        return (2**(1j * np.power(d,2)/2))/(2 * np.sqrt(np.pi)) * (sp.gamma(1/2 + (1j * d**2)/2)) * ((1 + np.exp(-np.pi * d**2)) * \
            ParabolicCylinderD(-1j * d**2, alpha * np.exp(1j * np.pi/4)) - (1j * np.sqrt(2 * np.pi))/(sp.gamma(1j * d**2)) * \
                np.exp(-np.pi * d**2/2) * ParabolicCylinderD(-1 + 1j * d**2, alpha * np.exp(-1j * np.pi/4)))

    def B(d, alpha):
        return (2**(1j * d**2/2) * np.exp(-1j * np.pi/4))/(d * np.sqrt(2 * np.pi)) * (sp.gamma(1 + (1j * d**2)/2)) * \
            ((1 - np.exp(-np.pi * d**2)) * ParabolicCylinderD(-1j * d**2, alpha * np.exp(1j * np.pi/4)) + \
                (1j * np.sqrt(2 * np.pi))/(sp.gamma(1j * d**2)) * np.exp(-np.pi * d**2/2) * \
                    ParabolicCylinderD(-1 + 1j * d**2, alpha * np.exp(-1j * np.pi/4)))
    
    def Ulin(u, v):
        return np.array(
            [
                [np.real(u) - 1j * np.imag(v), np.real(v) + 1j * np.imag(u)], 
                [-np.real(v) + 1j * np.imag(u), np.real(u) + 1j * np.imag(v)]
            ]
        )
    Ul = np.array(Ulin(A(d, alpha), B(d, alpha)).tolist(), dtype=complex)
    Ua = Uad(D, omega_0, tau)
    if tau < 0.5:
        U = np.einsum('jiz, jkz, klz -> ilz', Ul, Ua, Ul)
    else:
        U = np.einsum('jiz, ilz -> jlz', Ul, Ul)
    P2 = np.abs(U[0, 1]) ** 2
    return post_process(P2, eps, delta)

def sin_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sin")
def sin2_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sin2")
def sin3_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sin3")
def sin4_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sin4")
def sin5_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sin5")
def sech_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sech")
def sech2_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sech2")
def gauss_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "gauss")
def demkov_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "demkov")
def lor_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "lor")
def lor2_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "lor2")
def lor3_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "lor3")


def sin2(x, q_freq, delta, eps):
    T = dur * dt
    sigma = T / np.pi
    omega_0 = ps.find_rabi_amp("sin2", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 2 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-5)[0]
    # print(tau, G)
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sin2"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sin3(x, q_freq, delta, eps):
    T = dur * dt
    sigma = T / np.pi
    omega_0 = ps.find_rabi_amp("sin3", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 3
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-8)[0]
    G = quad_vec(fg_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-8)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sin3"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sin4(x, q_freq, delta, eps):
    T = dur * dt
    sigma = T / np.pi
    omega_0 = ps.find_rabi_amp("sin4", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 4
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-8)[0]
    G = quad_vec(fg_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-8)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sin4"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sin5(x, q_freq, delta, eps):
    T = dur * dt
    sigma = T / np.pi
    omega_0 = ps.find_rabi_amp("sin5", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 5
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-8)[0]
    G = quad_vec(fg_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-8)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sin5"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

# def gauss_sech2(x, q_freq, delta, eps):
#     O = RABI_FREQ["gauss"]
#     T = Tt["gauss"]
#     sigma = SIGMA["gauss"]
#     D = (x - q_freq) * 1e6
#     def f_(t):
#         return O * mp.exp(-0.5 * ((t - T/2) / sigma) ** 2)
#     S = np.float64(mp.quad(f_, [0, T]))
#     numerator = np.sin(0.5 * np.sqrt(np.pi) * O * sigma) ** 2
#     denomenator = np.cosh(np.pi * D * sigma / (4 * np.sqrt(np.log(O) - np.log(np.abs(D))))) ** 2
#     P2 = numerator / denomenator
#     return post_process(P2, eps, delta)

def gauss(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("gauss", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    alpha = np.abs(omega_0 / D)
    alpha[np.isnan(alpha)] = 10000000
    # print(alpha)
    m, mu, nu = (1.311468, 0.316193, 0.462350)

    ImD = 0.5 * D * sigma * np.sqrt(
        np.sqrt(
            4 * np.log(m * alpha) ** 2 + np.pi ** 2
        ) \
            - 2 * np.log(m * alpha)
    ) #/ np.sqrt(2)

    ReD =  D * sigma * (
        (np.sqrt(alpha ** 2 + 1) - 1) * \
            np.sqrt(
                0.5 * np.log(
                    alpha ** 2 / ((1 + nu * (np.sqrt(alpha ** 2 + 1) - 1)) ** 2 - 1)
                )
            ) + 0.5 * np.sqrt(
                np.sqrt(
                    np.log(alpha ** 2 / (mu * (2 - mu))) ** 2 + np.pi ** 2
                ) + np.log(alpha ** 2 / (mu * (2 - mu)))
            )
    ) #* np.sqrt(2)
    P2 = np.sin(ReD) ** 2 / np.cosh(ImD) ** 2
    return post_process(P2, eps, delta)

def gauss_rzconj(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("gauss", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.exp(- (t / sigma)**2)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -T/2, T/2, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["gauss"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)


def demkov_rzconj(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("demkov", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.exp(-np.abs(t / sigma))
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["demkov"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)


def lor_rzconj_amp(x, D, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = x # pulse_shapes.find_rabi_amp("lor", T, sigma, rb=rb)

    # D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["lor"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor_rzconj(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["lor"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor2_rzconj(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor2", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** 2
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T, T, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -T, T, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["lor"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor3_4_rzconj(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor3_4", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** (3/4)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T, T, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -T, T, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["lor"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor2_3_rzconj(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor2_3", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** (2/3)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T, T, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -T, T, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["lor"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor_narrowing(x, q_freq, a, b, d, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor", T, sigma, rb=rb, pulse_area=a)
    D = (x - q_freq) * 1e6
    # print(sigma, T, omega_0, D[0], D[-1])
    # P2 = a * np.abs(np.exp(-np.abs(D/b))) ** 2 + (1-a) / (1 + (D/d)**2)
    P2 = a / np.cosh(np.abs(D/b)) + (1-a) / (1 + (D/d)**2)

    return post_process(P2, eps, delta)

def lor2_narrowing(x, q_freq, a, b, d, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor2", T, sigma, rb=rb, pulse_area=a)
    D = (x - q_freq) * 1e6

    # P2 = a * np.exp(-np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.exp(-np.abs(D/b)**2) + (1-a) / (1 + (D/d)**2)
    P2 = a / np.cosh(np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.abs(np.exp(-D/b) * (-np.exp(2*D/b) * (-1 + D) * \
    #     np.heaviside(-D, 1) + (1 + D) * np.heaviside(D, 1))) ** 2 \
    #         + (1-a) / (1 + (D/d)**2)
    # 1/2 E^-w Sqrt[\[Pi]/2] (-E^(
    #  2 w) (-1 + w) HeavisideTheta[-w] + (1 + w) HeavisideTheta[w])
    return post_process(P2, eps, delta)

def lor3_2_narrowing(x, q_freq, a, b, d, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor3_2", T, sigma, rb=rb, pulse_area=a)
    D = (x - q_freq) * 1e6

    P2 = a / np.cosh(np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    return post_process(P2, eps, delta)

def lor3_4_narrowing(x, q_freq, a, b, d, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor3_4", T, sigma, rb=rb, pulse_area=a)
    D = (x - q_freq) * 1e6

    # P2 = a * np.exp(-np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.exp(-np.abs(D/b)**2) + (1-a) / (1 + (D/d)**2)
    P2 = a / np.cosh(np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.abs(np.abs(D/b)**(1/4) * sp.kv(-1/4, np.abs(D/b)) / sp.gamma(3/4)) ** 2 + \
    #     + (1-a) / (1 + (D/d)**2)

    # (2^(1/4) Abs[w]^(1/4) BesselK[-(1/4), Abs[w]])/Gamma[3/4]
    return post_process(P2, eps, delta)

def lor2_3_narrowing(x, q_freq, a, b, d, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor2_3", T, sigma, rb=rb, pulse_area=a)
    D = (x - q_freq) * 1e6

    # P2 = a * np.exp(-np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.exp(-np.abs(D/b)**2) + (1-a) / (1 + (D/d)**2)
    P2 = a / np.cosh(np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.abs(np.abs(D/b)**(1/6) * sp.kv(-1/6, np.abs(D/b)) / sp.gamma(2/3)) ** 2 + \
    #     + (1-a) / (1 + (D/d)**2)
    # (2^(1/3) Abs[w]^(1/6) BesselK[-(1/6), Abs[w]])/Gamma[2/3]
    return post_process(P2, eps, delta)

def lor3_5_narrowing(x, q_freq, a, b, d, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("lor3_5", T, sigma, rb=rb, pulse_area=a)
    D = (x - q_freq) * 1e6

    # P2 = a * np.exp(-np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.exp(-np.abs(D/b)**2) + (1-a) / (1 + (D/d)**2)
    P2 = a / np.cosh(np.abs(D/b)) + (1-a) / (1 + (D/d)**2)
    # P2 = a * np.abs(np.abs(D/b)**(1/10) * sp.kv(-1/10, np.abs(D/b)) / sp.gamma(3/5)) ** 2 + \
    #     + (1-a) / (1 + (D/d)**2)
    # (2^(2/5) Abs[w]^(1/10) BesselK[-(1/10), Abs[w]])/Gamma[3/5]

    return post_process(P2, eps, delta)


def sinc2(x, s, A, q_freq, c):
    return A * (np.sinc((x - q_freq) / s)) ** 2 + c

def rabi(x, q_freq, delta, eps):
    sigma = s * dt
    T = dur * dt
    omega_0 = ps.find_rabi_amp("rabi", T, sigma, rb=rb, pulse_area=a)

    D = (x - q_freq) * 1e6
    P2 = (omega_0 ** 2 / (omega_0**2 + (D) ** 2)) * \
        np.sin(0.5 * T * np.sqrt(omega_0**2 + (D) ** 2)) ** 2
    return post_process(P2, eps, delta)
