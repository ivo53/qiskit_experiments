import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import quad, quad_vec
import matplotlib.pyplot as plt

RABI_FREQ = {
    "constant": 6266474.70796,
    "rabi": 6266474.70796,
    "rz": 42874911.4203,
    "gauss": 25179780.7441,
    "demkov": 28438933.238,
    "sech2": 35460561.388,
    "sine": 23131885.3151,
    "sine2": 25244940.9663,
    "sine3": 26023370.9794,
    "lor": 4.20691e7,
    "lor2": 8.10088e7,
    "lor3_4": 6.45903e7,
    "lor2_3": 5.17585e7,
    "lor3_5": (7.11111111111) * 1e-9,
}

Tt = {
    "constant": 1504e-9 / 3,
    "rabi": 1504e-9 / 3,
    "rz": (284 + 4/9) * 1e-9,
    "gauss": (398 + 2/9) * 1e-9,
    "demkov": (572 + 4/9) * 1e-9,
    "sech2": (284 + 4/9) * 1e-9,
    "sine": (213 + 1/3) * 1e-9,
    "sine2": (248 + 8/9) * 1e-9,
    "sine3": (284 + 4/9) * 1e-9,
    "lor": (704) * 1e-9,
    "lor2": (181.333333333) * 1e-9,
    "lor3_4": (728.888888889) * 1e-9,
    "lor2_3": (1134.22222222) * 1e-9,
    "lor3_5": (1176.88888889) * 1e-9,
}

SIGMA = {
    "rz": 23.39181 * 1e-9,
    "gauss": (49 + 7/9) * 1e-9,
    "demkov": (55 + 5/9) * 1e-9,
    "sech2": (44 + 4/9) * 1e-9,
    "lor": (24.8888888889) * 1e-9,
    "lor2": (24.8888888889) * 1e-9,
    "lor3_4": (10.6666666667) * 1e-9,
    "lor2_3": (10.6666666667) * 1e-9,
    "lor3_5": (7.11111111111) * 1e-9,
}

pulse_type = "lor"
folder_name = "Power Narrowing" # "Pulse Shapes" if pulse_type == "sech2" else "Sine Pulses"
csv_file_address = f'C:/Users/Ivo/Documents/PhD Documents/{folder_name}/numerical_data/{pulse_type}.csv'
sim_data = np.genfromtxt(csv_file_address, delimiter=',')
sim_data[:,0] /= 1e6
sim_extended_freq = np.linspace(sim_data[0, 0], sim_data[-1, 0], 5000)

def fit_function(x_values, y_values, function, init_params, lower, higher):
    fitparams, conv = curve_fit(
        function, x_values, y_values, init_params, maxfev=1e6, bounds=(lower, higher)
    )
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

def sech_sq(x, q_freq, delta, eps, alpha):
    T = Tt["sech2"]
    sigma = SIGMA["sech2"]
    omega_0 = RABI_FREQ["sech2"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / np.cosh((t) / sigma) ** 2 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-4, 1e-4, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-4, 1e-4, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def gauss_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["gauss"]
    sigma = SIGMA["gauss"]
    omega_0 = RABI_FREQ["gauss"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return np.exp(-0.5 * (t / sigma)**2)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sine(x, q_freq, delta, eps, alpha):
    T = Tt["sine"]
    sigma = T / np.pi
    omega_0 = RABI_FREQ["sine"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos(t / sigma) 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sine2(x, q_freq, delta, eps, alpha):
    T = Tt["sine2"]
    sigma = T / np.pi
    omega_0 = RABI_FREQ["sine2"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 2 
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def sine3(x, q_freq, delta, eps, alpha):
    T = Tt["sine3"]
    sigma = T / np.pi
    omega_0 = RABI_FREQ["sine3"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return np.heaviside(t + T/2, 1) * np.heaviside(T/2 - t, 1) * np.cos((t) / sigma) ** 3
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-8)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-8)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def demkov_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["demkov"]
    sigma = SIGMA["demkov"]
    omega_0 = RABI_FREQ["demkov"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return np.exp(-np.abs(t / sigma))
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -1e-5, 1e-5, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["lor"]
    sigma = SIGMA["lor"]
    omega_0 = RABI_FREQ["lor"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2, epsabs=1e-13)[0]
    G = quad_vec(fg_, -T/2, T/2, epsabs=1e-13)[0] 
    S = np.array(omega_0 ** 2 + alpha * np.abs(D**3))
    P2 = np.abs(np.sin(0.5 * tau * np.sqrt(S))) ** 2 * np.abs(G / tau) ** 2
    return P2 # post_process(P2, eps, delta)

def lor2_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["lor2"]
    sigma = SIGMA["lor2"]
    omega_0 = RABI_FREQ["lor2"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** 2
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2, epsabs=1e-13, epsrel=1e-5)[0]
    G = quad_vec(fg_, -T/2, T/2, epsabs=1e-13, epsrel=1e-5)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

def lor3_4_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["lor3_4"]
    sigma = SIGMA["lor3_4"]
    omega_0 = RABI_FREQ["lor3_4"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** (3/4)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2)[0]
    G = quad_vec(fg_, -T/2, T/2)[0]
    P2 = np.abs(np.sin(0.5 * tau * np.sqrt(np.array(omega_0 ** 2 + alpha * D ** 2).astype("complex")))) ** 2 * np.abs(G / tau) ** 2 * omega_0 / np.array(omega_0 ** 2 + alpha * D ** 2).astype("complex")
    return post_process(P2, eps, delta)

def lor2_3_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["lor2_3"]
    sigma = SIGMA["lor2_3"]
    omega_0 = RABI_FREQ["lor2_3"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** (2/3)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2, epsabs=1e-13)[0]
    G = quad_vec(fg_, -T/2, T/2, epsabs=1e-13)[0]
    P2 = np.abs(np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2))) ** 2 * np.abs(G / tau) ** 2
    return P2 # post_process(P2, eps, delta)

def lor3_5_rzconj(x, q_freq, delta, eps, alpha):
    T = Tt["lor3_5"]
    sigma = SIGMA["lor3_5"]
    omega_0 = RABI_FREQ["lor3_5"]
    D = (x - q_freq) * 1e6
    def f_(t):
        return 1 / (1 + (t / sigma) ** 2) ** (3/5)
    def g_(t):
        return np.exp(1j * D * t)
    def fg_(t):
        return f_(t) * g_(t)
    tau = quad(f_, -T/2, T/2, epsabs=1e-13)[0]
    G = quad_vec(fg_, -T/2, T/2, epsabs=1e-13)[0]
    P2 = np.sin(0.5 * tau * np.sqrt(omega_0 ** 2 + alpha * D ** 2)) ** 2 * np.abs(G / tau) ** 2
    return post_process(P2, eps, delta)

FIT_FUNCTIONS = {
    "lorentzian": lorentzian,
    "sech2": sech_sq,
    "sine": sine,
    "sine2": sine2,
    "sine3": sine3,
    "gauss": gauss_rzconj,
    "demkov": demkov_rzconj,
    "lor": lor_rzconj,
    "lor2": lor2_rzconj,
    "lor3_4": lor3_4_rzconj,
    "lor2_3": lor2_3_rzconj,
    "lor3_5": lor3_5_rzconj,
}

baseline_fit_func = "lorentzian"

def fit_once(
    detuning, vals, fit_func,
    args, args_min, args_max,
    ef=None
):
    # initial = [0, 0.4, 0.4, 0.4] if fit_func in ["sim"] else [0.1, 0, 0]
    # initial_min = [-3, 0.3, 0.3, 0] if fit_func in ["sim"] else [-3, 0, 0]
    # initial_max = [3, 0.5, 0.6, 1] if fit_func in ["sim"] else [3, 0.5, 0.6]
    initial = [0.1, 0, 0]
    initial_min = [-3, 0, -1]
    initial_max = [3, 0.5, 1]
    fit_params, y_fit, err = fit_function(
        detuning,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)],
        vals,#[int(len(detuning) / 4):int(3 * len(detuning) / 4)], 
        FIT_FUNCTIONS[fit_func],
        args, args_min, args_max
    )
    y_fit = FIT_FUNCTIONS[fit_func](detuning, *fit_params)
    ##
    ##
    baseline_fit_params, baseline_y_fit, baseline_err = fit_function(
        detuning,
        vals, 
        FIT_FUNCTIONS[baseline_fit_func],
        [2, 0, 1, 0], # initial parameters for curve_fit
        [0, -10, 0, 0],
        [100, 10, 100, 0.5]

        # [1, 0, 1], # initial parameters for curve_fit
        # [0, -10, 0],
        # [10, 10, 1]
    )
    ##
    # print(fit_params, "\n", baseline_fit_params)
    if ef is not None:
        ef = sim_extended_freq
    else:
        ef = extended_freq
    extended_y_fit = FIT_FUNCTIONS[fit_func](ef, *fit_params)
    baseline_extended_y_fit = FIT_FUNCTIONS[baseline_fit_func](ef, *baseline_fit_params)

    similarity_idx = np.sum(np.abs(y_fit - vals))
    baseline_similarity_idx = np.sum(np.abs(baseline_y_fit - vals))

    overfitting_idx = np.mean(np.abs(np.diff(extended_y_fit)))
    baseline_overfitting_idx = np.mean(np.abs(np.diff(baseline_extended_y_fit)))
    overfitting = overfitting_idx > 0.1
    baseline_overfitting = baseline_overfitting_idx > 0.1
    # print(overfitting_idx, baseline_overfitting_idx)
    if overfitting:
        print("Strong overfitting present.")
        exit(1)
    return (similarity_idx, 
            y_fit, 
            extended_y_fit, 
            fit_params,
            err), \
           (baseline_similarity_idx, 
            baseline_y_fit, 
            baseline_extended_y_fit, 
            baseline_fit_params,
            baseline_err)

# fit, baseline = fit_once(
#     detuning, vals, fit_func,
#     args=[0, 1, 1e6], 
#     args_min=[-3, .99, 1e4],
#     args_max=[3, 1., 1e8]
# )

print(-RABI_FREQ[pulse_type] ** 2 / (sim_data[:, 0][0]*1e6) ** 2)

sim_fit, baseline_fit = fit_once(
    sim_data[:, 0], sim_data[:, 1], pulse_type,
    args=[0, 0.5, 0., -0.1],
    args_min=[-3, 0, 0., -RABI_FREQ[pulse_type] ** 2 / (sim_data[:, 0][0]*1e6) ** 2 + 0.01],
    args_max=[3, 1, 1, 1],
    ef=1
)
sim_similarity_idx, sim_y_fit, sim_extended_y_fit, sim_fit_params, sim_err = sim_fit
bs_similarity_idx, bs_y_fit, bs_extended_y_fit, bs_fit_params, bs_err = baseline_fit
print(sim_fit_params[-1], "+-", sim_err[-1])
print(sim_similarity_idx, "<", bs_similarity_idx)

fig = plt.figure(constrained_layout=True, figsize=(10,7))
gs = fig.add_gridspec(7, 1)
ax0 = fig.add_subplot(gs[:5, :])
ax0.scatter(sim_data[:, 0], sim_data[:, 1], color='black', marker="P")
ax0.plot(sim_extended_freq, bs_extended_y_fit, color='blue')
ax0.plot(sim_extended_freq, sim_extended_y_fit, color='red')
ax0.set_xlim(sim_extended_freq[0], -sim_extended_freq[0])
plt.show()
