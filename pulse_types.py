## Define function that return different shape instances of the new SymbolicPulse type
import sympy
from qiskit.pulse.library import SymbolicPulse

# a sawtooth pulse
def Sawtooth(duration, amp, freq, name):
    t, amp, freq = sympy.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sawtooth",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=2 * amp * (freq * t - sympy.floor(1 / 2 + freq * t)),
        name=name,
    )

    return instance

# a Lorentzian pulse
def Lorentzian(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp / (1 + ((t - duration/2) / sigma) ** 2),
        name=name,
    )

    return instance

# a LiftedLorentzian pulse
def LiftedLorentzian(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp / (1 + ((t - duration/2) / sigma) ** 2)
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Lorentzian^2 pulse
def Lorentzian2(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp / (1 + ((t - duration/2) / sigma) ** 2) ** 2,
        name=name,
    )

    return instance

# a LiftedLorentzian^2 pulse
def LiftedLorentzian2(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp / (1 + ((t - duration/2) / sigma) ** 2) ** 2
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Lorentzian^3 pulse
def Lorentzian3(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^3",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp / (1 + ((t - duration/2) / sigma) ** 2) ** 3,
        name=name,
    )

    return instance

# a LiftedLorentzian^3 pulse
def LiftedLorentzian3(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp / (1 + ((t - duration/2) / sigma) ** 2) ** 3
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^3",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Gaussian pulse
def Gaussian(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Gaussian",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp / (1 + ((t - duration/2) / sigma) ** 2) ** 3,
        name=name,
    )

    return instance

# a LiftedGaussian pulse
def LiftedGaussian(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp / (1 + ((t - duration/2) / sigma) ** 2) ** 3
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedGaussian",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance
