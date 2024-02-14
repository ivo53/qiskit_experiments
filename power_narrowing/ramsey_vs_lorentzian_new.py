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
    rho_list = np.empty((len(t_list), 2, 2), dtype=complex)
    rho_list[0] = rho0
    rho = rho0

    for idx, t in enumerate(t_list[1:]):
        dt = t - t_list[idx]
        drho_dt = lindblad_rhs(rho, H, L_ops, t)
        rho = rho + drho_dt * dt
        rho_list[idx + 1] = rho

    return rho_list

def measure_dm(dm, meas_op):
    return np.trace(dm @ meas_op)

# Example usage
if __name__ == "__main__":
    # Define Rabi frequency
    def omega_lor(t, args):
        return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

    def omega_ramsey(t, args):
        return np.pi / (2 * args["T_peak"]) * (np.heaviside(t + 0.5 * args["T_gap"] + args["T_peak"], 1) * np.heaviside(- 0.5 * args["T_gap"] - t, 1) + \
                                               np.heaviside(t - 0.5 * args["T_gap"], 1) * np.heaviside(0.5 * args["T_gap"] + args["T_peak"] - t, 1))
    # Define the time-dependent Hamiltonian function
    def time_dependent_Hamiltonian(t, d, omega, args):
        return 0.5 * np.array([[-d, omega(t, args)], [omega(t, args), d]], dtype=complex)

    gamma1, gamma2 = 0.005, 0.005

    A = 9 * np.pi / 9.50252
    D_max = 0.01
    D_min = -D_max
    len_D = 51
    
    T = 10000
    sigma = 1
    lor_power = 3/5
    T_peak = 500
    T_gap = T - 2 * T_peak
    args_lor = {
        "O": A, "sigma": sigma, "lor_power": lor_power
    }
    args_ramsey = {
        "T_peak": T_peak, "T_gap": T_gap
    }
    detunings = np.linspace(D_min, D_max, len_D)
    t_list = np.linspace(-T/2, T/2, 50000)
    rho0 = np.array([[1, 0], [0, 0]], dtype=complex)
    T1_effect = np.array([[0, 1], [0, 0]], dtype=complex)
    T2_effect = np.array([[1, 1], [-1, -1]], dtype=complex)
    meas_op = np.array([[0, 0], [0, 1]])
    
    tr_probs_lor, tr_probs_ramsey = [], []
    for d in detunings:
        L_ops = [gamma1 * T1_effect, gamma2 * T2_effect]
        # Solve the Lindblad equation
        rho_lor_list = run_lindblad(
            rho0, 
            lambda t: time_dependent_Hamiltonian(t, d, omega_lor, args_lor),
            L_ops, t_list
        )
        rho_ramsey_list = run_lindblad(
            rho0, 
            lambda t: time_dependent_Hamiltonian(t, d, omega_ramsey, args_ramsey), 
            L_ops, t_list
        )
        tr_probs_lor.append(rho_lor_list[-1][1,1])
        tr_probs_ramsey.append(rho_ramsey_list[-1][1,1])

    fig, ax = plt.subplots()
    ax.scatter(detunings, tr_probs_lor, c="r")
    ax.scatter(detunings, tr_probs_ramsey, c="b")
    ax.set_xlabel('Detuning')
    ax.set_ylabel('Transition probability')
    ax.legend(("Lorentzian", "Ramsey"))
    plt.show()
