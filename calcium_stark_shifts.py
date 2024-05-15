import numpy as np
from matplotlib import pyplot as plt
import pint
import qutip as qt
import atomphys as ap
from scipy.integrate import quad

# Ca = ap.Atom("Ca+")
# u = Ca._ureg
# print()
# S_1_2 = Ca.states[0]
# D_5_2 = Ca.states[2]
# other_states = [Ca.states[i] for i in np.arange(72)[~(np.arange(72)[:, None] == np.array([0, 2])[None]).any(1)]]
# T_729 = ap.Transition(S_1_2, D_5_2)
# print(T_729.to_dict())


wavenumbers = [
    13650.19, 13710.88, 25191.51, 25414.40, 52166.93, 56839.25, 56858.46, \
    60533.02, 60611.28, 68056.91, 68056.91, 70677.62, 72722.23, 72730.93, \
    74484.92, 74521.75, 78034.39, 78034.39, 78164.72, 78164.72, 79448.28, \
    80521.53, 80526.16, 83458.08, 83458.08, 83540.00, 83540.00, 84300.89, \
    84933.65, 84936.41, 86727.06, 86727.06, 86781.14, 86781.14, 87267.86, \
    87671.93, 87673.72, 88847.31, 88847.31, 88884.54, 88884.54, 88890.64, \
    88890.64, 89214.13, 89487.93, 89489.13, 90300.0, 90300.0, 90326.45, \
    90326.45, 90753.92, 90754.80, 91338.0, 91338.0, 91361.00, 91361.00, \
    91672.0, 91672.0, 92359.0, 92359.0, 92883.0, 92883.0, 93297.6, 93297.6, \
    93626.9, 93626.9, 93894.5, 93894.5
]
transitions_to = []
transitions_from = []
planck_constant = 6.62607015e-34  # Joule seconds (Js)
speed_of_light = 2.99792458e8     # meters per second (m/s)
cm_to_meters = 0.01               # centimeters to meters conversion
energy_conversion_factor = planck_constant * speed_of_light / cm_to_meters  # Joules
# Convert wavenumbers to energies (Joules)
energies_joules = [w * energy_conversion_factor for w in wavenumbers]

# Convert energies to frequencies (Hz)
frequencies_hz = [E / planck_constant for E in energies_joules]

print(np.array(frequencies_hz) * 10**-12)
natural_frequency = frequencies_hz[1]
detunings = np.delete(
    np.array(frequencies_hz) - natural_frequency, 1
) # D 5/2 is the excited state and does not cause Stark shift
frequencies_hz = np.delete(
    np.array(frequencies_hz), 1
) # D 5/2 is the excited state and does not cause Stark shift

# Define the lorentzian-shaped pulse
def omega(t, args):
    return args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]

def omega_sq(t, args):
    return (args["O"] / (1 + ((t) / args["sigma"]) ** 2) ** args["lor_power"]) ** 2

sigma = 1
T = 10
num_t = 1000
max_pulse_area = 10 * np.pi
lor_power = 1

d_start, d_num = -5000, 11
d_end = 50000
A_start, A_num = 0, 500000
A_end = max_pulse_area / quad(lambda t: omega(t, {"O": 1, "sigma": sigma, "lor_power": lor_power}), -T/2, T/2)[0]

tlist = np.linspace(-T/2, T/2, num_t)
options = qt.Options()
options.nsteps = 5000
d_range = np.linspace(d_start, d_end, d_num)
A_range = np.linspace(A_start, A_end, A_num) 

# Define the qubit states using basis vectors
ground_state = qt.basis(2, 0)
excited_state = qt.basis(2, 1)

for d in d_range:
    # Calculate the AC Stark shifts
    delta_g = sum(1 / (ord + d) + 1 / ((ord + d) + 2 * natural_frequency) for ord in detunings)
    delta_e = sum((1 / ((-1) ** int(f < natural_frequency) * ord + d) + 1 / (((-1) ** int(f < natural_frequency) * ord + d) + 2 * natural_frequency)) for f, ord in zip(frequencies_hz, detunings))
    print(delta_g, delta_e)
    for a in A_range:
    # Define the Hamiltonian
        H0_init = d * qt.sigmaz()/2
        H0_shift = delta_g * qt.basis(2, 0) * qt.basis(2, 0).dag() + delta_e * qt.basis(2, 1) * qt.basis(2, 1).dag()
        H0 = H0_init + H0_shift
        H1 = qt.sigmax() / 2
        H = [
                H0_init, 
                [H0_shift, omega_sq],
                [H1, omega]
            ],
        #execute mesolve
        times = np.linspace(0, 10, 100)  # Time from 0 to 10 arbitrary units
        psi0 = ground_state  # Initial state
        result = qt.mesolve(H, psi0, times, [], [], args={"O": a, "sigma": sigma, "lor_power": lor_power})

#visualize results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(times, result.expect[0], label='X Expectation')
plt.plot(times, result.expect[1], label='Y Expectation')
plt.plot(times, result.expect[2], label='Z Expectation')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Expectation values')
plt.title('Qubit Dynamics with AC Stark Shift')
plt.show()
