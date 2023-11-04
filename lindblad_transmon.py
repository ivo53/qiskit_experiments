import matplotlib.pyplot as plt
import numpy as np
import qutip
from scipy.integrate import quad

def omega(t, args):
    return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]
sigma = 1
T = 10
d_start, d_num = -5, 11
d_end=5
num_t=1000
A_start=0
A_num=5
max_pulse_area=10 * np.pi
lor_power=1
A_end = max_pulse_area / quad(lambda t: omega(t, {"O": 1, "sigma": sigma, "lor_power": lor_power}), -T/2, T/2)[0]

tlist = np.linspace(-T/2, T/2, num_t)
options = qutip.Options()
options.nsteps = 5000
d_range = np.linspace(d_start, d_end, d_num)
A_range = np.linspace(A_start, A_end, A_num) 
total_num_exp = len(d_range) * len(A_range)
percentage = 0
rho0 = qutip.Qobj([[1,0,0],[0,0,0],[0,0,0]])
tr_probs = []
for n_d, d in enumerate(d_range):
    values = []
    for n_A, A in enumerate(A_range):

        H0 = d * qutip.Qobj([[0,0,0],[0,0,0],[0,0,1]])
        H1 = qutip.Qobj([[0,1,0],[1,0,np.sqrt(2)],[0,np.sqrt(2),0]]) / 2

        output = qutip.mesolve(
            [
                H0, 
                [H1, omega]
            ],
            rho0=rho0,
            args={"O": A, "sigma": sigma, "lor_power": lor_power},
            tlist=tlist,
            options=options
        )
        values.append(output)
# print(values[-1].states[-1].data[(0,0)])
for n in range(1000):
    sum = 0
    for i in range(3):
        sum += np.abs(values[-1].states[n].data[(i, i)])
        if i == 2:
            print(values[-1].states[n].data[(i,i)])
