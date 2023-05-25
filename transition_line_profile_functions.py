import numpy as np
import mpmath as mp
import scipy.special as sp
from scipy.optimize import curve_fit, root
from scipy.integrate import quad, quad_vec

import pulse_shapes

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
    "demkov": 0.15786564335245298
}

def fit_function(x_values, y_values, function, init_params, lower, higher, sigma=None, duration=None, remove_bg=True):
    global s
    global dur
    global rb
    s = sigma
    dur = sigma if duration is None else duration
    rb = remove_bg
    fitparams, conv = curve_fit(
        function, 
        x_values, y_values, init_params, 
        maxfev=1e6, bounds=(lower, higher))
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
    sigma = s * 2e-9 / 9
    T = dur * 2e-9 / 9
    O = pulse_shapes.find_rabi_amp("rz", T, sigma, rb=rb)

    D = (x - q_freq) * 1e6
    P2 = np.sin(0.5 * np.pi * O * sigma) ** 2 \
        / np.cosh(0.5 * np.pi * D * sigma) ** 2
    return post_process(P2, eps, delta)

def demkov(x, q_freq, delta, eps):
    sigma = s * 2e-9 / 9
    T = dur * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp("demkov", T, sigma, rb=rb)

    s_inf = np.pi * omega_0 * sigma
    al = (x - q_freq) * 1e6 * sigma
    bessel11 = np.array([complex(mp.besselj(1/2 + 1j * a / 2, s_inf / (2 * np.pi))) for a in al])
    bessel21 = np.array([complex(mp.besselj(-1/2 - 1j * a / 2, s_inf / (2 * np.pi))) for a in al])
    P2 = (s_inf / 4) ** 2 * np.abs(2 * np.real(bessel11 * bessel21)) ** 2 / np.cosh(al * np.pi / 2) ** 2
    return post_process(P2, eps, delta)

def sech_sq(x, q_freq, delta, eps):
    sigma = s * 2e-9 / 9
    T = dur * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp("sech2", T, sigma, rb=rb)

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
    T = dur * 2e-9 / 9
    sigma = T / np.pi
    omega_0 = pulse_shapes.find_rabi_amp("sin", T, sigma, rb=rb)

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
    T = dur * 2e-9 / 9
    sigma = s * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp(pulse_type, T, sigma, rb=rb)
    sigma /= T
    omega_0 *= T
    D = (x - q_freq) * 1e6 * T
    omega = lambda t: pulse_shapes.rabi_freq(t, omega_0, pulse_type, 1, sigma, rb=rb)
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
def sech_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sech")
def sech2_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "sech2")
def gauss_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "gauss")
def lor_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "lor")
def lor2_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "lor2")
def lor3_dappr(x, q_freq, delta, eps, tau):
    return double_approx(x, q_freq, delta, eps, tau, "lor3")

def rlzsm_approx(x, q_freq, delta, eps, tau, pulse_type):
    T = dur * 2e-9 / 9
    sigma = s * 2e-9 / 9
    # sigma = T / np.pi
    omega_0 = pulse_shapes.find_rabi_amp(pulse_type, T, sigma, rb=rb)
    
    # sigma /= T
    sigma /= T
    omega_0 *= T
    D = (x - q_freq) * 1e6 * T
    omega = lambda t: pulse_shapes.rabi_freq(t, omega_0, pulse_type, 1, sigma, rb=rb)

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

    def a(d, alpha):
        return (2**(1j * np.power(d,2)/2))/(2 * np.sqrt(np.pi)) * (sp.gamma(1/2 + (1j * d**2)/2)) * ((1 + np.exp(-np.pi * d**2)) * \
            ParabolicCylinderD(-1j * d**2, alpha * np.exp(1j * np.pi/4)) - (1j * np.sqrt(2 * np.pi))/(sp.gamma(1j * d**2)) * \
                np.exp(-np.pi * d**2/2) * ParabolicCylinderD(-1 + 1j * d**2, alpha * np.exp(-1j * np.pi/4)))

    def b(d, alpha):
        return (2**(1j * d**2/2) * np.exp(-1j * np.pi/4))/(d * np.sqrt(2 * np.pi)) * (sp.gamma(1 + (1j * d**2)/2)) * \
            ((1 - np.exp(-np.pi * d**2)) * ParabolicCylinderD(-1j * d**2, alpha * np.exp(1j * np.pi/4)) + \
                (1j * np.sqrt(2 * np.pi))/(sp.gamma(1j * d**2)) * np.exp(-np.pi * d**2/2) * \
                    ParabolicCylinderD(-1 + 1j * d**2, alpha * np.exp(-1j * np.pi/4)))
    
    def Ulin(a, b):
        return np.array(
            [
                [np.real(a) - 1j * np.imag(b), np.real(b) + 1j * np.imag(a)], 
                [-np.real(b) + 1j * np.imag(a), np.real(a) + 1j * np.imag(b)]
            ]
        )
    Ul = np.array(Ulin(a(d, alpha), b(d, alpha)).tolist(), dtype=complex)
    Ua = Uad(D, omega_0, tau)
    U = np.einsum('jiz, jkz, klz -> ilz', Ul, Ua, Ul)
    P2 = np.abs(U[0, 1]) ** 2
    return post_process(P2, eps, delta)

def sin_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sin")
def sech_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sech")
def sech2_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "sech2")
def gauss_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "gauss")
def lor_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "lor")
def lor2_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "lor2")
def lor3_rlzsm(x, q_freq, delta, eps, tau):
    return rlzsm_approx(x, q_freq, delta, eps, tau, "lor3")


def sin2(x, q_freq, delta, eps):
    T = dur * 2e-9 / 9
    sigma = T / np.pi
    omega_0 = pulse_shapes.find_rabi_amp("sin2", T, sigma, rb=rb)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 2 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-6, 1e-6, epsabs=1e-13, epsrel=1e-5)[0]
    print(tau, G)
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["sin2"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sin3(x, q_freq, delta, eps):
    T = dur * 2e-9 / 9
    sigma = T / np.pi
    omega_0 = pulse_shapes.find_rabi_amp("sin3", T, sigma, rb=rb)

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
    T = dur * 2e-9 / 9
    sigma = T / np.pi
    omega_0 = pulse_shapes.find_rabi_amp("sin4", T, sigma, rb=rb)

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
    T = dur * 2e-9 / 9
    sigma = T / np.pi
    omega_0 = pulse_shapes.find_rabi_amp("sin5", T, sigma, rb=rb)

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
    sigma = s * 2e-9 / 9 / np.sqrt(2)
    T = dur * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp("gauss", T, sigma, rb=rb)

    D = (x - q_freq) * 1e6
    alpha = np.abs(omega_0 / D)
    alpha[np.isnan(alpha)] = 10000000
    # print(alpha)
    m, mu, nu = (1.311468, 0.316193, 0.462350)

    ImD = D * sigma * np.sqrt(
        np.sqrt(
            4 * np.log(m * alpha) ** 2 + np.pi ** 2
        ) \
            - 2 * np.log(m * alpha)
    ) / np.sqrt(2)

    ReD = np.sqrt(2) * D * sigma * (
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
    )
    P2 = np.sin(ReD) ** 2 / np.cosh(ImD) ** 2
    return post_process(P2, eps, delta)

def gauss_rzconj(x, q_freq, delta, eps):
    sigma = s * 2e-9 / 9 / np.sqrt(2)
    T = dur * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp("gauss", T, sigma, rb=rb)

    D = (x - q_freq) * 1e6
    def f_(t):
        return np.exp(-0.5 * (t / sigma)**2)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + ALPHA["gauss"] * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)


def demkov_rzconj(x, q_freq, delta, eps):
    sigma = s * 2e-9 / 9
    T = dur * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp("demkov", T, sigma, rb=rb)

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


def sinc2(x, s, A, q_freq, c):
    return A * (np.sinc((x - q_freq) / s)) ** 2 + c

def rabi(x, q_freq, delta, eps):
    sigma = s * 2e-9 / 9
    T = dur * 2e-9 / 9
    omega_0 = pulse_shapes.find_rabi_amp("rabi", T, sigma, rb=rb)

    D = (x - q_freq) * 1e6
    P2 = (omega_0 ** 2 / (omega_0**2 + (D) ** 2)) * \
        np.sin(0.5 * T * np.sqrt(omega_0**2 + (D) ** 2)) ** 2
    return post_process(P2, eps, delta)

