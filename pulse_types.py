## Define function that return different shape instances of the new SymbolicPulse type

import symengine as sym
from qiskit.pulse.library import SymbolicPulse


# a constant pulse
def Constant(duration, amp, name):
    t, duration_sym, amp_sym = sym.symbols("t, duration, amp")

    instance = SymbolicPulse(
        pulse_type="ConstantCustom",
        duration=duration,
        parameters={"amp": amp},
        envelope=amp_sym * sym.Piecewise((1, sym.And(t >= 0, t <= duration_sym)), (0, True)),
        name=name,
        valid_amp_conditions=sym.And(sym.Abs(amp_sym) >= 0, sym.Abs(amp_sym) <= 1),
    )

    return instance

# a sawtooth pulse
def Sawtooth(duration, amp, freq, name):
    t, amp_sym, freq_sym = sym.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sawtooth",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=2 * amp_sym * (freq_sym * t - sym.floor(1 / 2 + freq_sym * t)),
        name=name, 
    )

    return instance

# a Sine pulse
def Sine(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp_sym, freq_sym = sym.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp_sym * sym.sin(sym.pi * freq_sym * t),
        name=name,
    )

    return instance

# a Sine^2 pulse
def Sine2(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp_sym, freq_sym = sym.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^2",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp_sym * sym.sin(sym.pi * freq_sym * t) ** 2,
        name=name,
    )

    return instance

# a Sine^3 pulse
def Sine3(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp_sym, freq_sym = sym.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^3",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp_sym * sym.sin(sym.pi * freq_sym * t) ** 3,
        name=name,
    )

    return instance

# a Sine^4 pulse
def Sine4(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp_sym, freq_sym = sym.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^4",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp_sym * sym.sin(sym.pi * freq_sym * t) ** 4,
        name=name,
    )

    return instance

# a Sine^5 pulse
def Sine5(duration, amp, name, freq=None):
    if freq is None:
        freq = 1 / duration

    t, amp_sym, freq_sym = sym.symbols("t, amp, freq")

    instance = SymbolicPulse(
        pulse_type="Sine^5",
        duration=duration,
        parameters={"amp": amp, "freq": freq},
        envelope=amp_sym * sym.sin(sym.pi * freq_sym * t) ** 5,
        name=name,
    )

    return instance

# a Lorentzian pulse
def Lorentzian(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")
    
    instance = SymbolicPulse(
        pulse_type="Lorentzian",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2),
        name=name,
    )

    return instance

# a LiftedLorentzian pulse
def LiftedLorentzian(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Lorentzian^2 pulse
def Lorentzian2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** 2,
        name=name,
    )

    return instance

# a LiftedLorentzian^2 pulse
def LiftedLorentzian2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** 2
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Lorentzian^3 pulse
def Lorentzian3(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^3",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** 3,
        name=name,
    )

    return instance

# a LiftedLorentzian^3 pulse
def LiftedLorentzian3(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** 3
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^3",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Lorentzian^2/3 pulse
def Lorentzian2_3(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^(2/3)",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (2/3),
        name=name,
    )

    return instance

# a LiftedLorentzian^2/3 pulse
def LiftedLorentzian2_3(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (2/3)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^(2/3)",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Gaussian pulse
def Gaussian(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="GaussianCustom",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym * sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2),
        name=name,
    )

    return instance

# a LiftedGaussian pulse
def LiftedGaussian(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedGaussian",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Gaussian^2 pulse
def Gaussian2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Gaussian^2",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp_sym * sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2) ** 2,
        name=name,
    )

    return instance

# a LiftedGaussian^2 pulse
def LiftedGaussian2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2) ** 2
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedGaussian^2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Sech pulse
def Sech(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Sech",
        duration=duration,
        parameters={"amp": amp, "sigma": sigma},
        envelope=amp_sym * sym.sech((t - duration_sym/2) / sigma_sym),
        name=name,
    )

    return instance

# a LiftedSech pulse
def LiftedSech(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.sech((t - duration_sym/2) / sigma_sym)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedSech",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Sech^2 pulse
def Sech2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Sech^2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym * sym.sech((t - duration_sym/2) / sigma_sym) ** 2,
        name=name,
    )

    return instance

# a LiftedSech^2 pulse
def LiftedSech2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.sech((t - duration_sym/2) / sigma_sym) ** 2
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedSech^2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Sech^3 pulse
def Sech3(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Sech^3",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym * sym.sech((t - duration_sym/2) / sigma_sym) ** 3,
        name=name,
    )

    return instance

# a LiftedSech^3 pulse
def LiftedSech3(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.sech((t - duration_sym/2) / sigma_sym) ** 3
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedSech^3",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Demkov pulse
def Demkov(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Demkov",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym * sym.exp(-sym.Abs((t - duration_sym/2) / sigma_sym)),
        name=name,
    )

    return instance

# a LiftedDemkov pulse
def LiftedDemkov(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.exp(-sym.Abs((t - duration_sym/2) / sigma_sym))
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    
    instance = SymbolicPulse(
        pulse_type="LiftedDemkov",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance

# a Drag pulse
def Drag(duration, amp, sigma, beta, name):
    t, duration_sym, amp_sym, sigma_sym, beta_sym = sym.symbols("t, duration, amp, sigma, beta")

    gaussian = amp_sym * sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    gaussian_deriv = sym.diff(gaussian, t)
    envelope = gaussian + 1j * beta_sym * gaussian_deriv
    
    instance = SymbolicPulse(
        pulse_type="Drag",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma, "beta": beta},
        envelope=envelope,
        name=name,
    )

    return instance

# a LiftedDrag pulse
def LiftedDrag(duration, amp, sigma, beta, name):
    t, duration_sym, amp_sym, sigma_sym, beta_sym = sym.symbols("t, duration, amp, sigma, beta")

    gaussian = amp_sym * sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    gaussian_deriv = sym.diff(gaussian, t)
    envelope = gaussian + 1j * beta_sym * gaussian_deriv
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))

    instance = SymbolicPulse(
        pulse_type="LiftedDrag",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma, "beta": beta},
        envelope=lifted_envelope,
        name=name,
    )

    return instance


