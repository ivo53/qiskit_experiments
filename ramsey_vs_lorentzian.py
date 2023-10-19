import numpy as np
import qutip
import matplotlib.pyplot as plt

import numpy as np

def lindblad_rhs(rho, H, L_ops, t):
    """
    Calculate the right-hand side of the Lindblad equation.

    Parameters:
        rho (array): Density matrix.
        H (array): Hamiltonian.
        L_ops (list of arrays): List of Lindblad operators.

    Returns:
        drho_dt (array): Time derivative of the density matrix.
    """

    Ht = H(t)
    # Hamiltonian term
    commutator = np.dot(Ht, rho) - np.dot(rho, Ht)

    # Lindblad terms
    lindblad_terms = 0
    for L in L_ops:
        L_rho = np.dot(L, rho)
        rho_L = np.dot(rho, L)
        lindblad_terms += np.dot(L_rho, np.conj(L.T)) - 0.5 * np.dot(np.conj(L.T), L_rho) - 0.5 * np.dot(rho_L, np.conj(L.T))

    drho_dt = -1.0j * (commutator + lindblad_terms)

    return drho_dt

def run_lindblad(rho0, H, L_ops, t_list):
    """
    Solve the Lindblad equation for a given time list.

    Parameters:
        rho0 (array): Initial density matrix.
        H (array): Hamiltonian.
        L_ops (list of arrays): List of Lindblad operators.
        t_list (array): List of time points.

    Returns:
        rho_list (list of arrays): List of density matrices at each time point.
    """
    rho_list = np.empty((len(t_list),))
    rho_list[0] = rho0
    rho = rho0

    for idx, t in enumerate(t_list[1:]):
        dt = t - t_list[np.where(t_list == t)[0] - 1]
        drho_dt = lindblad_rhs(rho, H, L_ops, t)
        rho = rho + drho_dt * dt
        rho_list[idx + 1] = rho

    return rho_list


def omega_lor(t, args):
    return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

def omega_ramsey(t, args):
    return np.pi / (2 * args["T_peak"]) * (np.heaviside(t + 0.5 * args["T_gap"] + args["T_peak"], 1) * np.heaviside(- 0.5 * args["T_gap"] - t, 1) + \
                                          np.heaviside(t - 0.5 * args["T_gap"], 1) * np.heaviside(0.5 * args["T_gap"] + args["T_peak"] - t, 1))

d = 1
A = 9 * np.pi / 9.50252
D_max = 0.01
D_min = -D_max
T = 10000
options = qutip.Options()
options.nsteps = 100000
sigma = 1
lor_power = 3/5
T_peak = 500
gamma1 = 0.005
gamma2 = 0.005
T_gap = T - 2 * T_peak
detunings = np.linspace(D_min, D_max, 51)
tlist = np.linspace(-T/2, T/2, 10000)
rho0 = qutip.Qobj([[1, 0], [0, 0]])
T1_effect = qutip.Qobj([[0, 1], [0, 0]])
T2_effect = qutip.Qobj([[1, 1], [-1, -1]])
H1 = qutip.sigmax() / 2
tr_probs_lor, tr_probs_ramsey = [], []
# for d in detunings: 
#     H0 = - (d / 2) * qutip.sigmaz()
#     result_lor = qutip.mesolve(
#         [
#             H0,
#             [H1, omega_lor]
#         ],
#         rho0=rho0,
#         tlist=tlist,
#         c_ops=[gamma1 * T1_effect, 0.5 * gamma2 * T2_effect],
#         e_ops=qutip.sigmaz(),
#         args={"O": A, "sigma": sigma, "lor_power": lor_power},
#         options=options
#     )
#     result_ramsey = qutip.mesolve(
#         [
#             H0,
#             [H1, omega_ramsey]
#         ],
#         rho0=rho0,
#         tlist=tlist,
#         c_ops=[gamma1 * T1_effect, 0.5 * gamma2 * T2_effect],
#         e_ops=qutip.sigmaz(),
#         args={"T_peak": T_peak, "T_gap": T_gap},
#         options=options
#     )
#     tr_probs_lor.append(-0.5 * (np.array(result_lor.expect[0])[-1] - 1.))
#     tr_probs_ramsey.append(-0.5 * (np.array(result_ramsey.expect[0])[-1] - 1.))

# fig, ax = plt.subplots()
# ax.scatter(detunings, tr_probs_lor, c="r")
# ax.scatter(detunings, tr_probs_ramsey, c="b")
# ax.set_xlabel('Detunings')
# ax.set_ylabel('Transition probability')
# ax.legend(("Lorentzian", "Ramsey"))
# plt.show()

# Example usage
if __name__ == "__main__":
    d = 0.05
    # Define the time-dependent Hamiltonian function
    def time_dependent_Hamiltonian(t, d):
        return np.array([[-d, np.sin(t)], [np.sin(t), d]], dtype=complex)
    # Define the initial density matrix, Hamiltonian, and Lindblad operators
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    L_ops = [0.005 * np.array([[0, 1], [0, 0]], dtype=complex), 0.005 * np.array([[1, 1], [-1, -1]], dtype=complex)]

    # Time list
    t_list = np.linspace(0, 5, 100)

    # Solve the Lindblad equation
    rho_list = run_lindblad(rho0, time_dependent_Hamiltonian, L_ops, t_list)
    
    # Print the final density matrix
    print("Final Density Matrix:")
    probability = np.trace(rho_list[-1] @ np.array([[0, 0], [0, 1]]))
