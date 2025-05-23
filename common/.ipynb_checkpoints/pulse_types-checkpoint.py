## Define function that return different shape instances of the new SymbolicPulse type

import symengine as sym
from qiskit.pulse.library import SymbolicPulse


# a constant pulse
def Constant(duration, amp, name):
    t, duration_sym, amp_sym = sym.symbols("t, duration, amp")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.Piecewise((1, sym.And(t >= 0, t <= duration_sym)), (0, True))
    
    instance = SymbolicPulse(
        pulse_type="ConstantCustom",
        duration=duration,
        parameters={"duration": duration, "amp": amp},
        envelope=envelope,
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
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

# a t^{2N} pulse
def InverseParabola(duration, amp, N, name):
    t, duration_sym, amp_sym, N_sym = sym.symbols("t, duration, amp, N")
    
    instance = SymbolicPulse(
        pulse_type="InverseParabola",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "N": N},
        envelope=amp_sym * ((t - duration_sym/2) / (duration_sym/2)) ** (2*N_sym),
        name=name,
    )

    return instance

# a 1 + beta * t^2 pulse
def FaceChangingQuadratic(duration, amp, beta, name):
    t, duration_sym, amp_sym, beta_sym = sym.symbols("t, duration, amp, beta")
    
    instance = SymbolicPulse(
        pulse_type="FaceChangingQuadratic",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta},
        envelope=amp_sym * (1 + beta_sym * (((t - duration_sym / 2) / (duration_sym / 2)) ** 2 - 1)),
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

# a Lorentzian^3/2 pulse
def Lorentzian3_2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^3/2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (3/2),
        name=name,
    )

    return instance

# a LiftedLorentzian^3/2 pulse
def LiftedLorentzian3_2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (3/2)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^3/2",
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

# a Lorentzian^3/4 pulse
def Lorentzian3_4(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^(3/4)",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (3/4),
        name=name,
    )

    return instance

# a LiftedLorentzian^3/4 pulse
def LiftedLorentzian3_4(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (3/4)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^(3/4)",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=lifted_envelope,
        name=name,
    )

    return instance


# a Lorentzian^3/5 pulse
def Lorentzian3_5(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    instance = SymbolicPulse(
        pulse_type="Lorentzian^(3/5)",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma},
        envelope=amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (3/5),
        name=name,
    )

    return instance

# a LiftedLorentzian^3/5 pulse
def LiftedLorentzian3_5(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym / (1 + ((t - duration_sym/2) / sigma_sym) ** 2) ** (3/5)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    instance = SymbolicPulse(
        pulse_type="LiftedLorentzian^(3/5)",
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
        envelope=amp_sym * sym.exp(- ((t - duration_sym/2) / sigma_sym) ** 2),
        name=name,
    )

    return instance

# a LiftedGaussian pulse
def LiftedGaussian(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.exp(- ((t - duration_sym/2) / sigma_sym) ** 2)
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
        envelope=amp_sym * sym.exp(- ((t - duration_sym/2) / sigma_sym) ** 2) ** 2,
        name=name,
    )

    return instance

# a LiftedGaussian^2 pulse
def LiftedGaussian2(duration, amp, sigma, name):
    t, duration_sym, amp_sym, sigma_sym = sym.symbols("t, duration, amp, sigma")

    envelope = amp_sym * sym.exp(- ((t - duration_sym/2) / sigma_sym) ** 2) ** 2
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
    t, duration_sym, amp_sym, sigma_sym, beta_sym, angle_sym = sym.symbols("t, duration, amp, sigma, beta, angle")

    gaussian = sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    gaussian_deriv = -(t - duration_sym/2) / (sigma_sym**2) * gaussian
    envelope = amp_sym * (gaussian + sym.I * beta_sym * gaussian_deriv) * sym.exp(sym.I * angle_sym)

    consts_expr = _sigma > 0
    valid_amp_conditions_expr = sym.And(sym.Abs(_amp) <= 1.0, sym.Abs(_beta) < _sigma)

    instance = SymbolicPulse(
        pulse_type="Drag",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma, "beta": beta, "angle": angle},
        envelope=envelope,
        angle=angle,
        name=name,
        constraints=consts_expr,
        valid_amp_conditions=valid_amp_conditions_expr,
    )

    return instance

# a LiftedDrag pulse
def LiftedDrag(duration, amp, sigma, beta, name, angle=0.):
    t, duration_sym, amp_sym, sigma_sym, beta_sym, angle_sym = sym.symbols("t, duration, amp, sigma, beta, angle")

    gaussian = sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    gaussian_deriv = -(t - duration_sym/2) / (sigma_sym**2) * gaussian
    envelope = amp_sym * (gaussian + sym.I * beta_sym * gaussian_deriv) * sym.exp(sym.I * angle_sym)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    # lifted_envelope = [sym.re(lifted_envelope), sym.im(lifted_envelope)]
    consts_expr = sigma_sym > 0
    valid_amp_conditions_expr = sym.And(sym.Abs(amp_sym) <= 1.0, sym.Abs(beta_sym) < sigma_sym)

    instance = SymbolicPulse(
        pulse_type="LiftedDrag",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma, "beta": beta, "angle": angle},
        envelope=lifted_envelope,
        angle=angle,
        name=name,
        constraints=consts_expr,
        valid_amp_conditions=valid_amp_conditions_expr,
    )

    return instance


