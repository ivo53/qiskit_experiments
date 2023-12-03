import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.integrate import quad

def ndsolve_lorentz_spectre(   
    sigma, T,
    d_start, d_num,
    d_end=None,
    num_t=1000,
    pulse_area=np.pi,
    lor_power=1
):
    d_end = d_end or -d_start
    
    pulse_type = "lorentz"

    def omega(t, args):
        return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

    
    O = pulse_area / quad(lambda t: omega(t, {"O": 1, "sigma": sigma, "lor_power": lor_power}), -T/2, T/2)[0]

    tlist = np.linspace(-T/2, T/2, num_t)
    options = qutip.Options()
    options.nsteps = 5000
    d_range = np.linspace(d_start, d_end, d_num) 
    total_num_exp = len(d_range)
    percentage = 0
    values = []
    for n_d, d in enumerate(d_range):
        last_percentage = percentage
        funcvalues = omega(tlist, {"O": O, "sigma": sigma, "lor_power": lor_power})
        n_exp = n_d + 1
        percentage = np.round(n_exp / total_num_exp, 3)
        if percentage > last_percentage:
            print(
                "[" + "-" * int(np.floor(percentage * 100)) + " " * int(np.ceil(100-percentage*100)) + "]",
                np.round(percentage * 100, 1), 
                "%"
            )
        H0 = - (d / 2) * qutip.sigmaz()
        H1 = qutip.sigmax() / 2
        output = qutip.sesolve(
            [
                H0, 
                [H1, omega]
            ],
            psi0=qutip.basis(2,0),
            args={"O": O, "sigma": sigma, "lor_power": lor_power},
            tlist=tlist,
            options=options
        )
        values.append(qutip.expect(qutip.sigmaz(), output.states)[-1])

    tr_probs = -0.5 * (np.array(values) - 1.)

    # # Plot the solution
    # plt.plot(d_range, tr_probs, 'r', label='y(t)')  # Plotting y(t)
    # plt.xlabel('Time')
    # plt.ylabel('Transition probability')
    # plt.legend()
    # plt.show()
    return d_range, tr_probs
    
def ndsolve_lorentz_rabi_osc(   
    sigma, T,
    A_start,
    A_num, d,
    A_end=None,
    num_t=1000,
    lor_power=1
):
    A_end = A_end or -A_start
    
    pulse_type = "lorentz"

    def omega(t, args):
        return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

    
    tlist = np.linspace(-T/2, T/2, num_t)
    options = qutip.Options()
    options.nsteps = 5000
    A_range = np.linspace(A_start, A_end, A_num) 
    total_num_exp = len(A_range)
    percentage = 0
    values = []
    for n_A, A in enumerate(A_range):
        last_percentage = percentage
        funcvalues = omega(tlist, {"O": A, "sigma": sigma, "lor_power": lor_power})
        n_exp = n_A + 1
        percentage = np.round(n_exp / total_num_exp, 3)
        if percentage > last_percentage:
            print(
                "[" + "-" * int(np.floor(percentage * 100)) + " " * int(np.ceil(100-percentage*100)) + "]",
                np.round(percentage * 100, 1), 
                "%"
            )
        H0 = - (d / 2) * qutip.sigmaz()
        H1 = qutip.sigmax() / 2
        output = qutip.sesolve(
            [
                H0, 
                [H1, omega]
            ],
            psi0=qutip.basis(2,0),
            args={"O": A, "sigma": sigma, "lor_power": lor_power},
            tlist=tlist,
            options=options
        )
        values.append(qutip.expect(qutip.sigmaz(), output.states)[-1])

    tr_probs = -0.5 * (np.array(values) - 1.)

    # # Plot the solution
    # plt.plot(A_range, tr_probs, 'r', label='y(t)')  # Plotting y(t)
    # plt.xlabel('Time')
    # plt.ylabel('Transition probability')
    # plt.legend()
    # plt.show()
    return A_range, tr_probs

def ndsolve_lorentz_map(   
    sigma, T,
    d_start, d_num,
    d_end=None,
    num_t=1000,
    A_start=0,
    A_num=100,
    max_pulse_area=10 * np.pi,
    lor_power=1
):
    d_end = d_end or -d_start
    
    pulse_type = "lorentz"

    def omega(t, args):
        return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

    
    A_end = max_pulse_area / quad(lambda t: omega(t, {"O": 1, "sigma": sigma, "lor_power": lor_power}), -T/2, T/2)[0]

    tlist = np.linspace(-T/2, T/2, num_t)
    options = qutip.Options()
    options.nsteps = 5000
    d_range = np.linspace(d_start, d_end, d_num)
    A_range = np.linspace(A_start, A_end, A_num) 
    total_num_exp = len(d_range) * len(A_range)
    percentage = 0
    tr_probs = []
    for n_d, d in enumerate(d_range):
        values = []
        for n_A, A in enumerate(A_range):
            last_percentage = percentage
            funcvalues = omega(tlist, {"O": A, "sigma": sigma, "lor_power": lor_power})
            n_exp = (n_d) * (A_num) + (n_A)
            percentage = np.round(n_exp / total_num_exp, 3)
            if percentage > last_percentage:
                print(
                    "[" + "-" * int(np.floor(percentage * 100)) + " " * int(np.ceil(100-percentage*100)) + "]",
                    np.round(percentage * 100, 1), 
                    "%"
                )
            H0 = - (d / 2) * qutip.sigmaz()
            H1 = qutip.sigmax() / 2
            output = qutip.sesolve(
                [
                    H0, 
                    [H1, omega]
                ],
                psi0=qutip.basis(2,0),
                args={"O": A, "sigma": sigma, "lor_power": lor_power},
                tlist=tlist,
                options=options
            )
            values.append(qutip.expect(qutip.sigmaz(), output.states)[-1])
        tr_probs.append(values)
    tr_probs = -0.5 * (np.array(tr_probs) - 1.)

    # # Plot the solution
    # plt.plot(d_range, tr_probs, 'r', label='y(t)')  # Plotting y(t)
    # plt.xlabel('Time')
    # plt.ylabel('Transition probability')
    # plt.legend()
    # plt.show()
    return d_range, A_range, tr_probs.T


# ndsolve_lorentz_rabi_osc(   
#     (24 + 8/9) * 1e-9, (704) * 1e-9,
#     0,
#     100, 2e6,
#     A_end=800e6,
#     num_t=1000,
#     lor_power=1
# )

d_range, A_range, tr_probs = ndsolve_lorentz_map(   
    (24 + 8/9) * 1e-9, (704) * 1e-9,
    -60e6, 61, 60e6,    
    num_t=1000,
    A_start=0,
    A_num=100,
    lor_power=1
)
fig, ax = plt.subplots(1,1)
cmap = plt.cm.get_cmap('cividis')  # Choose a colormap
im = ax.pcolormesh(d_range, A_range, tr_probs, vmin=0, vmax=1, cmap=cmap)
plt.show()