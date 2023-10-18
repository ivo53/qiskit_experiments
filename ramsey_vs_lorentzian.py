import numpy as np
import qutip
import matplotlib.pyplot as plt


def omega_lor(t, args):
    return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

def omega_ramsey(t, args):
    return np.pi / (2 * args["T_peak"]) * (np.heaviside(t + 0.5 * args["T_gap"] + args["T_peak"], 1) * np.heaviside(- 0.5 * args["T_gap"] - t, 1) + \
                                          np.heaviside(t - 0.5 * args["T_gap"], 1) * np.heaviside(0.5 * args["T_gap"] + args["T_peak"] - t, 1))

d = 1
A = 9
D_max = 0.1
D_min = -D_max
T = 10000
options = qutip.Options()
options.nsteps = 5000
sigma = 1
lor_power = 3/5
T_peak = 500
gamma1 = 0.005
gamma2 = 0.005
T_gap = T - 2 * T_peak
detunings = np.linspace(D_min, D_max, 51)
tlist = np.linspace(-T/2, T/2, 100)
rho0 = qutip.Qobj([[1, 0], [0, 0]])
T1_effect = qutip.Qobj([[0, 1], [0, 0]])
T2_effect = qutip.Qobj([[1, 1], [-1, -1]])
H1 = qutip.sigmax() / 2
tr_probs_lor, tr_probs_ramsey = [], []
for d in detunings:
    H0 = - (d / 2) * qutip.sigmaz()
    result_lor = qutip.mesolve(
        [
            H0,
            [H1, omega_lor]
        ],
        rho0=rho0,
        tlist=tlist,
        c_ops=[gamma1 * T1_effect, 0.5 * gamma2 * T2_effect],
        e_ops=qutip.sigmaz(),
        args={"O": A, "sigma": sigma, "lor_power": lor_power},
        options=options
    )
    result_ramsey = qutip.mesolve(
        [
            H0,
            [H1, omega_ramsey]
        ],
        rho0=rho0,
        tlist=tlist,
        c_ops=[gamma1 * T1_effect, 0.5 * gamma2 * T2_effect],
        e_ops=qutip.sigmaz(),
        args={"T_peak": T_peak, "T_gap": T_gap},
        options=options
    )
    tr_probs_lor.append(-0.5 * (np.array(result_lor.expect[0])[-1] - 1.))
    tr_probs_ramsey.append(-0.5 * (np.array(result_ramsey.expect[0])[-1] - 1.))

fig, ax = plt.subplots()
ax.scatter(detunings, tr_probs_lor, c="r")
ax.scatter(detunings, tr_probs_ramsey, c="b")
ax.set_xlabel('Detunings')
ax.set_ylabel('Transition probability')
ax.legend(("Lorentzian", "Ramsey"))
plt.show()