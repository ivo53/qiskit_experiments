import numpy as np
import qutip 

# Define the collapse operators (Lindblad operators)
collapse_ops = []
T = 1
num_t = 1000
sigma = 1
# Here you can add your own collapse operators if needed
# For example:
# collapse_ops.append(sqrt(gamma) * ops[0])
def omega(t, args):
    return args["O"] * np.sin(np.pi * t / args["sigma"]) * np.heaviside(t, 1) * np.heaviside(args["sigma"] - t, 1)

# def detuning(n, t, assym, omega):
#     return assym * (n - 1) + (n - 1) * (n - 2) * omega(t) / 8
rho0 = qutip.Qobj([
    [1,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
])
options = qutip.Options()
options.nsteps = 5000
tlist = np.linspace(0, sigma, num_t)
d_range = np.linspace(0.03125-0.01,0.03125+0.01,40)
A_range = np.linspace(0, 10, 40)
tr_probs = []
for n_d, d in enumerate(d_range):
    values = []
    for n_A, A in enumerate(A_range):
        
        # Define the Hamiltonian (you will need to add your own Hamiltonian terms)

        H0 = 0.5 * d * qutip.Qobj([[0,0,0,0,0,0,0,0],
                                   [0,0,0,0,0,0,0,0],
                                   [0,0,1,0,0,0,0,0],
                                   [0,0,0,2,0,0,0,0],
                                   [0,0,0,0,3,0,0,0],
                                   [0,0,0,0,0,4,0,0],
                                   [0,0,0,0,0,0,5,0],
                                   [0,0,0,0,0,0,0,6]])
        H1 = 0.5 * qutip.Qobj([[0,1,0,0,0,0,0,0],
                               [1,0,np.sqrt(2),0,0,0,0,0],
                               [0,np.sqrt(2),0,np.sqrt(3),0,0,0,0],
                               [0,0,np.sqrt(3),1/4,np.sqrt(4),0,0,0],
                               [0,0,0,np.sqrt(4),3/4,np.sqrt(5),0,0],
                               [0,0,0,0,np.sqrt(5),3/2,np.sqrt(6),0],
                               [0,0,0,0,0,np.sqrt(6),5/2,np.sqrt(7)],
                               [0,0,0,0,0,0,np.sqrt(7),15/4],])
        output = qutip.mesolve(
            [
                H0,
                [H1, omega]
            ],
            rho0=rho0,
            args={"O": A, "sigma": sigma},
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