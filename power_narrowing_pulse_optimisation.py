import numpy as np
import qutip 
import sklearn

# Define the collapse operators (Lindblad operators)
collapse_ops = []
T = 1
num_t = 1000
sigma = 1
# Here you can add your own collapse operators if needed
# For example:
# collapse_ops.append(sqrt(gamma) * ops[0])

## Attempt optimisation using composite pulses
# def comp(t, i, n, phase=0):
#     assert phase >= 0 and phase <= 2 * np.pi
#     length = (t[-1] - t[0]) / (2 * n - 1)
#     tmin = (2 * i) * length
#     tmax = (2 * i) * (length + 1)
#     return np.exp(1j * phase) * np.heaviside(t + tmin, 1) * np.heaviside(-t + tmax, 1)
# def composites(t, args):
#     comp_sum = np.zeros_like(t)
#     for i in range(n):
#         comp_sum += comp(t, i, n, phase=args[i])
#     comp_sum += args[-1] * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)
#     return comp_sum
################

## Also try a Frankenstein sum of usual functions
def generic_shape(t, args):
    return (args["Osech"] / np.cosh((t - args["Tsech"] / 2) / args["sigmasech"]) + \
        args["Olor"] / (1 + ((t - args["Tlor"]/2) / args["sigmalor"]) ** 2) + \
        args["Oexp"] * np.exp(-np.abs((t - args["Texp"]/2) / args["sigmaexp"])) + \
        args["Ogauss"] * np.exp(-np.abs((t - args["Tgauss"]/2) / args["sigmagauss"]) ** 2) + \
        args["Otanh"] * (-np.abs(np.tanh((t - args["Ttanh"]/2) / args["sigmatanh"])) + 1) + args["const"]) \
            * np.heaviside(t + T/2, 1) * np.heaviside(-t + T/2, 1)
#################
# def detuning(n, t, assym, omega):
#     return assym * (n - 1) + (n - 1) * (n - 2) * omega(t) / 8
rho0 = qutip.Qobj([
    [1,0],
    [0,0]
])
options = qutip.Options()
# options.nsteps = 5000
tlist = np.linspace(0, sigma, num_t)
d_range = np.linspace(0.03125-0.01,0.03125+0.01,40)
# A_range = np.linspace(0, 10, 40)
tr_probs = []
for n_d, d in enumerate(d_range):
    values = []
    for n_A, A in enumerate([np.pi, 7 * np.pi]):
        
        # Define the Hamiltonian (you will need to add your own Hamiltonian terms)
        args={
            "Osech": Asech, "Tsech": Tsech, "sigmasech": sigmasech, 
            "Olor": Alor, "Tlor": Tlor, "sigmalor": sigmalor, 
            "Oexp": Aexp, "Texp": Texp, "sigmaexp": sigmaexp, 
            "Ogauss": Agauss, "Tgauss": Tgauss, "sigmagauss": sigmagauss, 
            "Otanh": Atanh, "Ttanh": Ttanh, "sigmatanh": sigmatanh, 
            "const": c
        }
        H0 = 0.5 * d * qutip.Qobj([[-1,0,],
                                   [0,1]])
        H1 = 0.5 * qutip.Qobj([[0,1],
                               [1,0]])
        output = qutip.mesolve(
            [
                H0, 
                [H1, generic_shape]
            ],
            rho0=rho0,
            args=args,
            tlist=tlist,
            options=options
        )
        values.append(output.states[-1])
    tr_probs.append(values)


for n in range(1000):
    sum = 0
    for i in range(3):
        sum += np.abs(values[-1].states[n].data[(i, i)])
        if i == 2:
            print(values[-1].states[n].data[(i,i)])


# Plot the results or do further analysis as needed