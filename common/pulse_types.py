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

# a composite pulse
def Composite(duration, amp, amps, phases, name):
    def _heaviside(x):
        return (x + sym.Abs(x)) / (2 * sym.Abs(x) + 1e-300)
    assert len(amps) == len(phases)

    t, duration_sym, amp_sym = sym.symbols("t, duration, amp")

    num_pulses = len(amps)
    indices = range(num_pulses)

    # Create symbols for amplitude and phase for each pulse
    amps_sym = sym.symbols([f"amp{n}" for n in range(num_pulses)])
    phases_sym = sym.symbols([f"phase{n}" for n in range(num_pulses)])

    # Define parameters dictionary
    parameters = {
        "duration": duration,
        "amp": amp,
        **{str(a): val for a, val in zip(amps_sym, amps)},
        **{str(p): val for p, val in zip(phases_sym, phases)},
    }

    envelope = 0
    for n, amp, phase in zip(indices, amps, phases):

        envelope += amps_sym[n] * sym.exp(sym.I * phases_sym[n]) * _heaviside(t - duration_sym * n / num_pulses) * _heaviside(duration_sym * (n + 1) / num_pulses - t)
        
    instance = SymbolicPulse(
        pulse_type="Composite",
        duration=duration,
        parameters=parameters,
        envelope=amp_sym * envelope,
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance

# a constant pulse with Landau-Zener modulation 1
def LandauZener1(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.Piecewise((1, sym.And(t >= 0, t <= duration_sym)), (0, True))
    
    instance = SymbolicPulse(
        pulse_type="LandauZener1",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * 0.5 * beta_sym * tau_sym * ((t - duration_sym / 2) / tau_sym)**2),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with Landau-Zener modulation 8
def LandauZener8(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.sech((t - duration_sym / 2) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="LandauZener8",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * 0.5 * beta_sym * tau_sym * (sym.atan(sym.sinh((t - duration_sym / 2) / tau_sym))) ** 2),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with Landau-Zener modulation 4
def LandauZener4(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the cosine envelope
    envelope = amp_sym * 0.5 * sym.pi * sym.cos((t - duration_sym / 2) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="LandauZener4",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * beta_sym * tau_sym * 1/8 * sym.pi ** 2 * (sym.sin((t - duration_sym / 2) / tau_sym)) ** 2),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a sech pulse with Allen-Eberly modulation 8
def AllenEberly8(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the sech envelope
    envelope = amp_sym * sym.sech((t - duration_sym / 2) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="AllenEberly8",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * beta_sym * tau_sym * sym.log(sym.cosh((t - duration_sym / 2) / tau_sym))),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with Allen-Eberly modulation 1
def AllenEberly1(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.Piecewise((1, sym.And(t >= 0, t <= duration_sym)), (0, True))
    
    instance = SymbolicPulse(
        pulse_type="AllenEberly1",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(- sym.I * beta_sym * tau_sym * sym.log(sym.cos((t - duration_sym / 2) / tau_sym))), 
        name=name, 
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1, duration_sym / (2 * tau_sym) < sym.pi / 2),
    )

    return instance
# a cos pulse with Allen-Eberly modulation 4
def AllenEberly4(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the cosine envelope
    envelope = amp_sym * 0.5 * sym.pi * sym.cos((t - duration_sym / 2) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="AllenEberly4",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(- sym.I * beta_sym * tau_sym * sym.log(sym.cos(0.5 * sym.pi * sym.sin((t - duration_sym / 2) / tau_sym)))), 
        name=name, 
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1, duration_sym / (2 * tau_sym) < sym.pi / 2),
    )

    return instance
# a constant pulse with half Landau-Zener modulation 1
def HalfLandauZener1(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.Piecewise((1, sym.And(t >= 0, t <= duration_sym)), (0, True))
    
    instance = SymbolicPulse(
        pulse_type="LandauZener1",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * 0.5 * beta_sym * tau_sym * ((t - duration_sym) / tau_sym)**2),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with half Landau-Zener modulation 8
def HalfLandauZener8(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.sech((t - duration_sym) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="LandauZener8",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * 0.5 * beta_sym * tau_sym * (sym.atan(sym.sinh((t - duration_sym) / tau_sym))) ** 2),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with half Landau-Zener modulation 4
def HalfLandauZener4(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the cosine envelope
    envelope = amp_sym * 0.5 * sym.pi * sym.cos((t - duration_sym) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="LandauZener4",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * beta_sym * tau_sym * 1/8 * sym.pi ** 2 * (sym.sin((t - duration_sym) / tau_sym)) ** 2),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a sech pulse with half Allen-Eberly modulation 8
def HalfAllenEberly8(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the sech envelope
    envelope = amp_sym * sym.sech((t - duration_sym) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="AllenEberly8",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * beta_sym * tau_sym * sym.log(sym.cosh((t - duration_sym) / tau_sym))),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with half Allen-Eberly modulation 1
def HalfAllenEberly1(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.Piecewise((1, sym.And(t >= 0, t <= duration_sym)), (0, True))
    
    instance = SymbolicPulse(
        pulse_type="AllenEberly1",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(- sym.I * beta_sym * tau_sym * sym.log(sym.cos((t - duration_sym) / tau_sym))), 
        name=name, 
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1, duration_sym / (2 * tau_sym) < sym.pi / 2),
    )

    return instance
# a cos pulse with half Allen-Eberly modulation 4
def HalfAllenEberly4(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the cosine envelope
    envelope = amp_sym * 0.5 * sym.pi * sym.cos((t - duration_sym) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="AllenEberly4",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(- sym.I * beta_sym * tau_sym * sym.log(sym.cos(0.5 * sym.pi * sym.sin((t - duration_sym) / tau_sym)))), 
        name=name, 
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1, duration_sym / (2 * tau_sym) < sym.pi / 2),
    )

    return instance
# a constant pulse with BambiniBerman modulation
def BambiniBerman(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.sech((t - duration_sym / 2) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="BambiniBerman",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * beta_sym * (tau_sym * sym.log(sym.cosh((t - duration_sym / 2) / tau_sym)) + t)),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with Demkov-Kunike-2 modulation
def DemkovKunike2(duration, amp, beta, tau, delta_0, name):
    t, duration_sym, amp_sym, beta_sym, delta_0_sym, tau_sym = sym.symbols("t, duration, amp, beta, delta_0, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.sech((t - duration_sym / 2) / tau_sym)
    
    instance = SymbolicPulse(
        pulse_type="DemkovKunike2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "delta_0": delta_0, "tau": tau},
        envelope=envelope * sym.exp(sym.I * (beta_sym * tau_sym * sym.log(sym.cosh((t - duration_sym / 2) / tau_sym)) + delta_0_sym * t)),
        name=name,
        valid_amp_conditions=sym.And(amp_sym >= 0, amp_sym <= 1),
    )

    return instance
# a constant pulse with Demkov-Kunike-2 modulation
def CosSin(duration, amp, beta, tau, name):
    t, duration_sym, amp_sym, beta_sym, tau_sym = sym.symbols("t, duration, amp, beta, tau")
    
    # Define the constant envelope without Piecewise
    envelope = amp_sym * sym.cos(t / tau_sym) ** 2
    
    instance = SymbolicPulse(
        pulse_type="CosSin",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "beta": beta, "tau": tau},
        envelope=envelope * sym.exp(sym.I * 1/4 * beta_sym * (2 * t - tau_sym * sym.sin(2 * t / tau_sym))),
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
        name=name
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
        name=name
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

# a Drag2 pulse
def Drag2(duration, amp, sigma, beta, name):
    t, duration_sym, amp_sym, sigma_sym, beta_sym = sym.symbols("t, duration, amp, sigma, beta")

    gaussian = sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    gaussian_deriv = -(t - duration_sym/2) / (sigma_sym**2) * gaussian
    envelope = amp_sym * (gaussian + sym.I * beta_sym * gaussian_deriv)

    consts_expr = sigma_sym > 0
    valid_amp_conditions_expr = sym.And(sym.Abs(amp_sym) <= 1.0, sym.Abs(beta_sym) < sigma_sym)

    instance = SymbolicPulse(
        pulse_type="Drag2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma, "beta": beta},
        envelope=envelope,
        name=name,
        constraints=consts_expr,
        valid_amp_conditions=valid_amp_conditions_expr,
    )

    return instance

# a LiftedDrag2 pulse
def LiftedDrag2(duration, amp, sigma, beta, name):
    t, duration_sym, amp_sym, sigma_sym, beta_sym = sym.symbols("t, duration, amp, sigma, beta")

    gaussian = sym.exp(- 0.5 * ((t - duration_sym/2) / sigma_sym) ** 2)
    gaussian_deriv = -(t - duration_sym/2) / (sigma_sym**2) * gaussian
    envelope = amp_sym * (gaussian + sym.I * beta_sym * gaussian_deriv)
    new_amp = amp_sym / (amp_sym - envelope.subs(t, 0))
    lifted_envelope = new_amp * (envelope - envelope.subs(t, 0))
    # lifted_envelope = [sym.re(lifted_envelope), sym.im(lifted_envelope)]
    consts_expr = sigma_sym > 0
    valid_amp_conditions_expr = sym.And(sym.Abs(amp_sym) <= 1.0, sym.Abs(beta_sym) < sigma_sym)

    instance = SymbolicPulse(
        pulse_type="LiftedDrag2",
        duration=duration,
        parameters={"duration": duration, "amp": amp, "sigma": sigma, "beta": beta},
        envelope=lifted_envelope,
        name=name,
        constraints=consts_expr,
        valid_amp_conditions=valid_amp_conditions_expr,
    )

    return instance


