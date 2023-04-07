## Define function that return different shape instances of the new SymbolicPulse type

import sympy
from qiskit.pulse.library import SymbolicPulse


# a constant pulse
def Constant(duration, amp, name):
    amp = sympy.symbols("amp")

    instance = SymbolicPulse(
        pulse_type="Constant",
        duration=duration,
        parameters={"amp": amp},
        envelope=amp,
        name=name,
    )

    return instance

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

# a Sine pulse
def Sine(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp, freq = sympy.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp * sympy.sin(sympy.pi * freq * t),
        name=name,
    )

    return instance

# a Sine^2 pulse
def Sine2(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp, freq = sympy.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^2",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp * sympy.sin(sympy.pi * freq * t) ** 2,
        name=name,
    )

    return instance

# a Sine^3 pulse
def Sine3(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp, freq = sympy.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^3",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp * sympy.sin(sympy.pi * freq * t) ** 3,
        name=name,
    )

    return instance

# a Sine^4 pulse
def Sine4(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp, freq = sympy.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^4",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp * sympy.sin(sympy.pi * freq * t) ** 4,
        name=name,
    )

    return instance

# a Sine^5 pulse
def Sine5(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp, freq = sympy.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^5",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp * sympy.sin(sympy.pi * freq * t) ** 5,
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
        envelope=amp * sympy.exp(- 0.5 * ((t - duration/2) / sigma) ** 2),
        name=name,
    )

    return instance

# a LiftedGaussian pulse
def LiftedGaussian(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp * sympy.exp(- 0.5 * ((t - duration/2) / sigma) ** 2)
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

# a Gaussian^2 pulse
def Gaussian2(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Gaussian^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp * sympy.exp(- 0.5 * ((t - duration/2) / sigma) ** 2) ** 2,
        name=name,
    )

    return instance

# a LiftedGaussian^2 pulse
def LiftedGaussian2(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp * sympy.exp(- 0.5 * ((t - duration/2) / sigma) ** 2) ** 2
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedGaussian^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Sech pulse
def Sech(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Sech",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp * sympy.sech((t - duration/2) / sigma),
        name=name,
    )

    return instance

# a LiftedSech pulse
def LiftedSech(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp * sympy.sech((t - duration/2) / sigma)
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedSech",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Sech^2 pulse
def Sech2(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Sech^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp * sympy.sech((t - duration/2) / sigma) ** 2,
        name=name,
    )

    return instance

# a LiftedSech^2 pulse
def LiftedSech2(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp * sympy.sech((t - duration/2) / sigma) ** 2
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedSech^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Sech^3 pulse
def Sech3(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Sech^3",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp * sympy.sech((t - duration/2) / sigma) ** 3,
        name=name,
    )

    return instance

# a LiftedSech^3 pulse
def LiftedSech3(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp * sympy.sech((t - duration/2) / sigma) ** 3
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedSech^3",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Demkov pulse
def Demkov(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Demkov",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp * sympy.exp(-sympy.Abs((t - duration/2) / sigma)),
        name=name,
    )

    return instance

# a LiftedDemkov pulse
def LiftedDemkov(duration, amp, sigma, name):
    t, amp, sigma = sympy.symbols("t, amp, sigma")

    envelope = amp * sympy.exp(-sympy.Abs((t - duration/2) / sigma))
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedDemkov",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Drag pulse
def Drag(duration, amp, sigma, beta, name):
    t, amp, sigma, beta = sympy.symbols("t, amp, sigma, beta")

    gaussian = amp * sympy.exp(- 0.5 * ((t - duration/2) / sigma) ** 2)
    gaussian_deriv = sympy.diff(gaussian, t)
    envelope = gaussian + 1j * beta * gaussian_deriv
    
    instance = SymbolicPulse(
        pulse_type="Drag",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma, "beta": beta},
        envelope=envelope,
        name=name,
    )

    return instance

# a LiftedDrag pulse
def LiftedDrag(duration, amp, sigma, beta, name):
    t, amp, sigma, beta = sympy.symbols("t, amp, sigma, beta")

    gaussian = amp * sympy.exp(- 0.5 * ((t - duration/2) / sigma) ** 2)
    gaussian_deriv = sympy.diff(gaussian, t)
    envelope = gaussian + 1j * beta * gaussian_deriv
    new_amp = amp / (amp - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))

    instance = SymbolicPulse(
        pulse_type="LiftedDrag",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma, "beta": beta},
        envelope=lifted_envelope,
        name=name,
    )

    return instance


