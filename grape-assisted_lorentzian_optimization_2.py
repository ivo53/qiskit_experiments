"""
GRAPE-style frequency-mask optimizer for power-narrowed lineshapes
=================================================================

Goal
----
Given a 2-level system with Hamiltonian
    H(t) = (Δ(t)/2) σ_z + (Ω(t)/2)[cos φ(t) σ_x + sin φ(t) σ_y],
we design controls Ω(t), φ(t) (and optionally Δ(t)) to:
  • Maximize transition probability P(Δ0) inside a target passband |Δ0| <= W
  • Minimize P(Δ0) outside (stopband)
  • Encourage **power narrowing**: FWHM decreases as the global amplitude scale rises

How to use
----------
1) Edit the three shape functions near the top: `omega_shape`, `phase_shape`, `delta_shape`.
   They map time array `t` (and Δ0) → shaped envelopes. Start with the provided examples
   (Lorentzian, flat phase, constant detuning) or replace them with your own.

2) Choose target settings in the `Config` dataclass (grid over Δ0, passband W, weights,
   amplitude scales for power-narrowing test, etc.).

3) Run the script. It will:
   - Build an initial control from the shapes (seed)
   - Optimize phase-only or full amplitude+phase samples (toggle in Config)
   - Print FWHM at two amplitude scales and show a quick plot of P(Δ0)

Dependencies
------------
• Python 3.10+
• JAX (CPU or GPU):  `pip install jax jaxlib`
• Matplotlib (for quick-look plots; optional)

Notes
-----
• The optimizer uses Adam on discretized control samples. Your analytic shape is used
  only to seed; the optimizer freely refines the samples (toggle which channels are free).
• If you want to **constrain** controls to a functional family during optimization, swap the
  free-sample parameterization with your own (see the TODO hook in `controls_to_profiles`).
• The FWHM term is wrapped in `jax.lax.stop_gradient` to avoid nondifferentiable kinks.

(c) 2025 — drop-in template prepared for pulse-shape exploration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import lax

# Optional plotting (skip if headless)
try:
    import matplotlib.pyplot as plt  # noqa: F401
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

# --------------------------------------------------------------------------------------
# User-editable pulse/detuning shapes (SEEDS)
# --------------------------------------------------------------------------------------

def omega_shape(t: jnp.ndarray, theta_amp: Tuple[float, float]) -> jnp.ndarray:
    """Seed amplitude envelope Ω_seed(t).
    Example: Lorentzian Ω0 / (1 + (t/T)^2) with (Ω0, T) = theta_amp.
    Replace this with any callable you like.
    """
    Omega0, T = theta_amp
    x = t / jnp.maximum(T, 1e-9)
    Om = Omega0 / (1.0 + x * x)
    return Om * (jnp.pi / (jnp.sum(Om) * (t[1] - t[0])))


def phase_shape(t: jnp.ndarray, theta_phase: Tuple[float, ...]) -> jnp.ndarray:
    """Seed phase φ_seed(t).
    Default: piecewise-constant with an optional central π-flip (Ramsey-like).
    Parameters: theta_phase = (phi0, do_flip∈{0,1})
    """
    phi0, do_flip = theta_phase
    # Simple example: flat phase with optional π flip at t=0
    return phi0 + jnp.pi * do_flip * (t >= 0.0)


def delta_shape(
    t: jnp.ndarray,
    delta0: float,
    theta_det: Tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Seed detuning Δ_seed(t) for a given Δ0.
    Default: constant Δ(t) = Δ0. To allow a shaped detuning, replace this.
    Example alternative (tanh): Δ(t) = Δ0 * tanh(beta * t / Tref)
    """
    if theta_det is None:
        return jnp.full_like(t, delta0)
    beta, Tref = theta_det
    return delta0 * jnp.ones_like(t)
    # return delta0 * jnp.tanh(beta * t / jnp.maximum(Tref, 1e-9))


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class Config:
    # Time grid
    T: float = 100.0  # total duration (arb. units)
    N: int = 600    # number of steps

    # Detuning scan grid Δ0
    delta0_max: float = 60.0
    num_delta: int = 241  # must be odd → includes 0

    # Pass/stop band settings (frequency-mask)
    W: float = 1.0               # half-width of passband around Δ0=0
    w_pass: float = 1.0          # weight inside passband (target P≈1)
    w_stop: float = 2.0          # weight outside passband (target P≈0)

    # Smoothing & amplitude constraints
    Omega_max: float = 50.0       # hard cap on |Ω|
    lam_smooth: float = 1e-3     # weights ∑|ΔΩ|^2 and ∑|Δφ|^2 penalties
    lam_det_smooth: float = 1.0  # set >0 if shaping Δ(t)

    lam_center: float = 5.0      # or whatever
    w_dark: float = 2.0          # emphasize the “dark” condition if you wish

    # Power-narrowing test: evaluate FWHM at two global amplitude scales
    amp_scales: Tuple[float, float, float] = (1.0, 2.0, 5.0)
    lam_pn: float = 1e-2  # penalty for positive slope of FWHM vs amplitude scale

    # Optimization
    steps: int = 800
    lr: float = 5e-2
    print_every: int = 50

    # Which channels are free to optimize (True → free samples, False → keep seeded)
    free_amp: bool = True
    free_phase: bool = True
    free_det: bool = True   # set True to optimize Δ(t) (requires lam_det_smooth > 0 for stability)

    # Seed parameters for shapes (only used to initialize the free samples)
    theta_amp: Tuple[float, float] = (3.0, 2.0)    # (Ω0, T) for Lorentzian seed
    theta_phase: Tuple[float, float] = (0.0, 0.0)  # (φ0, do_flip)
    theta_det: Tuple[float, float] | None = None   # None → constant detuning


# --------------------------------------------------------------------------------------
# Core math utilities
# --------------------------------------------------------------------------------------

# Pauli matrices
SX = jnp.array([[0.0, 1.0], [1.0, 0.0]])
SY = jnp.array([[0.0, -1.0j], [1.0j, 0.0]])
SZ = jnp.array([[1.0, 0.0], [0.0, -1.0]])
I2 = jnp.eye(2, dtype=jnp.complex64)


def hat_from_controls(Om: jnp.ndarray, Ph: jnp.ndarray, De: jnp.ndarray) -> jnp.ndarray:
    """Return sequence of Hamiltonians H_k with shape (N,2,2).
    H_k = (Δ_k/2) σ_z + (Ω_k/2)[cos φ_k σ_x + sin φ_k σ_y]
    """
    cx = jnp.cos(Ph)
    sx = jnp.sin(Ph)
    Hx = (Om * cx)[:, None, None] * SX[None, :, :]
    Hy = (Om * sx)[:, None, None] * SY[None, :, :]
    Hz = -(De)[:, None, None] * SZ[None, :, :]
    return 0.5 * (Hx + Hy + Hz)



@jit
def step_unitary(H: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Closed-form U = exp(-i H dt) for 2×2 Hermitian H.

    H is constructed as:
        H = (Δ/2) σ_z + (Ω/2)(cosφ σ_x + sinφ σ_y)
          = (1/2)(hx σ_x + hy σ_y + hz σ_z)

    With a = sqrt(hx^2+hy^2+hz^2) = |h|:
        exp(-i H dt) = cos(a dt/2) I - i sin(a dt/2) (hx σ_x + hy σ_y + hz σ_z)/a
                    = cos(a dt/2) I - i sin(a dt/2) (2H)/a
    """
    hx = jnp.real(jnp.trace(H @ SX))
    hy = jnp.real(jnp.trace(H @ SY))
    hz = jnp.real(jnp.trace(H @ SZ))
    a = jnp.sqrt(hx * hx + hy * hy + hz * hz)

    def small_norm(_):
        return I2 - 1j * H * dt

    def general(_):
        theta = 0.5 * a * dt
        nSigma = (2.0 * H) / jnp.maximum(a, 1e-12)
        return jnp.cos(theta) * I2 - 1j * jnp.sin(theta) * nSigma

    return lax.cond(a < 1e-9, small_norm, general, operand=None)


    def small_norm(_):
        return I2 - 1j * H * dt  # first-order; good enough when θ≪1

    def general(_):
        theta = 2.0 * hnorm * dt  # since H=1/2 h·σ → ||H|| = ||h||/2
        nH = H / jnp.maximum(2.0 * hnorm, 1e-12)  # normalize by ||h|| (not ||H||)
        return jnp.cos(theta) * I2 - 1j * jnp.sin(theta) * nH

    return lax.cond(hnorm < 1e-9, small_norm, general, operand=None)


def propagate(Om: jnp.ndarray, Ph: jnp.ndarray, De: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Compute final state |ψ(T)⟩ from |0⟩ under time-discretized controls.
    Returns the state vector (2,).
    """
    Hs = hat_from_controls(Om, Ph, De)

    def body(U, H):
        U_next = step_unitary(H, dt) @ U
        return U_next, None

    U0 = I2
    Uf, _ = lax.scan(body, U0, Hs)
    psi0 = jnp.array([1.0 + 0.0j, 0.0 + 0.0j])
    psiT = Uf @ psi0
    return psiT


# Vectorize propagation over Δ0 grid and amplitude scales

def line_shape_for_scale(
    Om_base: jnp.ndarray,
    Ph: jnp.ndarray,
    De_template: jnp.ndarray,
    dt: float,
    delta0_grid: jnp.ndarray,
    scale: float,
    Omega_cap: float,
) -> jnp.ndarray:
    """Compute P(Δ0) for a given global amplitude scale (scales Ω only)."""
    Om = Omega_cap * jnp.tanh((scale * Om_base) / jnp.maximum(Omega_cap, 1e-9))  # cap after scaling

    def run_one(delta0):
        # If De_template encodes shaped detuning relative to Δ0, we scale-add here.
        # Convention: De_template was built for delta0=1; here we multiply.
        # If De_template is constant, this equals jnp.full(N, delta0).
        De = delta0 * De_template
        psiT = propagate(Om, Ph, De, dt)
        # probability in |1>
        return jnp.abs(psiT[1]) ** 2

    return vmap(run_one)(delta0_grid)


# --------------------------------------------------------------------------------------
# Controls parameterization & initialization
# --------------------------------------------------------------------------------------

def seed_controls(cfg: Config) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build time grid and initial discrete controls from analytic seeds.
    Returns (t, dt, Ω_seed(t), φ_seed(t), Δ_seed(t; Δ0=1), Δ0_grid)
    The Δ_seed is normalized to Δ0=1 so we can later scale it by each Δ0.
    """
    t = jnp.linspace(-cfg.T / 2, cfg.T / 2, cfg.N)
    dt = cfg.T / (cfg.N - 1)

    Omega_seed = omega_shape(t, cfg.theta_amp)
    Phi_seed = phase_shape(t, cfg.theta_phase)

    # Build a normalized detuning template for Δ0=1
    De1 = delta_shape(t, 1.0, cfg.theta_det)

    # Δ0 grid (odd count ensures 0 is included)
    m = cfg.num_delta
    assert m % 2 == 1, "num_delta must be odd to include 0"
    delta0_grid = jnp.linspace(-cfg.delta0_max, cfg.delta0_max, m)

    # Clip Ω to [0, Ω_max] initially
    Omega_seed = jnp.clip(Omega_seed, 0.0, cfg.Omega_max)

    # Fix Omega area to PI
    Omega_seed *= jnp.pi / (jnp.sum(Omega_seed) * dt)  # normalize ∫Ω dt ≈ π

    return t, dt, Omega_seed, Phi_seed, De1, delta0_grid


@jax.tree_util.register_pytree_node_class
@dataclass
class Variables:
    """Free variables to optimize (discretized samples)."""
    Om: jnp.ndarray  # (N,)
    Ph: jnp.ndarray  # (N,)
    De: jnp.ndarray  # (N,) template for Δ0=1 (scale by Δ0 at evaluation)

    # Make this dataclass a JAX pytree so jit/grad work on it
    def tree_flatten(self):
        children = (self.Om, self.Ph, self.De)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        Om, Ph, De = children
        return cls(Om=Om, Ph=Ph, De=De)


def controls_from_seed(cfg: Config, Omega_seed, Phi_seed, De1) -> Variables:
    """Initialize Variables from seed; mark channels free/fixed via config flags."""
    Om = Omega_seed
    Ph = Phi_seed
    De = De1
    return Variables(Om=Om, Ph=Ph, De=De)


# --------------------------------------------------------------------------------------
# FWHM and loss construction
# --------------------------------------------------------------------------------------

# @jit
# def fwhm(delta0: jnp.ndarray, P: jnp.ndarray, center_idx: int) -> float:
#     """JAX-friendly FWHM using vectorized masks (no dynamic slicing).
#     Assumes a single central lobe around index `center_idx`.
#     """
#     n = P.size
#     half = 0.5 * jnp.max(P)
#     ge = P >= half  # boolean array

#     # Transitions ge[i]=True -> ge[i+1]=False indicate crossings of half-maximum
#     trans = ge[:-1] & (~ge[1:])  # shape (n-1,)
#     idx = jnp.arange(n - 1)

#     # Left crossing: last transition strictly left of center
#     left_candidates = jnp.where((trans) & (idx < center_idx), idx, -1)
#     left_idx = jnp.max(left_candidates)

#     # Right crossing: first transition at/after center
#     right_candidates = jnp.where((trans) & (idx >= center_idx), idx, n)
#     right_idx = jnp.min(right_candidates)

#     # Safe indices for interpolation (clip into valid range)
#     iL = jnp.clip(left_idx, 0, n - 2)
#     iR = jnp.clip(right_idx, 0, n - 2)

#     def interp(i):
#         x1 = jnp.take(delta0, i)
#         x2 = jnp.take(delta0, i + 1)
#         y1 = jnp.take(P, i)
#         y2 = jnp.take(P, i + 1)
#         denom = jnp.maximum(jnp.abs(y2 - y1), 1e-12)
#         w = jnp.clip((half - y1) / denom, 0.0, 1.0)
#         return x1 + w * (x2 - x1)

#     xL = interp(iL)
#     xR = interp(iR)
#     width = jnp.maximum(xR - xL, 0.0)

#     # Fallbacks: no crossing → either all above half (full range) or all below (zero)
#     all_true = jnp.all(ge)
#     all_false = jnp.all(~ge)
#     full_range = delta0[-1] - delta0[0]
#     width = jnp.where(all_true, full_range, width)
#     width = jnp.where(all_false, 0.0, width)
#     return width

@jit
def fwhm(delta0: jnp.ndarray, P: jnp.ndarray, eps: float=0.01) -> float:
    p_max = jnp.max(P)
    threshold = 0.5 * p_max
    mask = jax.nn.sigmoid((P - threshold) / eps)
    d_delta = delta0[1] - delta0[0]
    width = jnp.sum(mask) * d_delta
    return width

def fwhm_report(delta0: jnp.ndarray, P: jnp.ndarray) -> float:
    """Robust (non-jitted) FWHM for diagnostics and printing.
    Falls back to full range if P>=half everywhere, or 0 if never reaches half.
    """
    import numpy as _np
    d = _np.asarray(delta0)
    p = _np.asarray(P)
    peak = float(p.max())
    if peak <= 1e-12:
        return 0.0
    half = 0.5 * peak
    ge = p >= half
    if ge.all():
        return float(d[-1] - d[0])
    if (~ge).all(): # unnecessary
        return 0.0
    center = int(_np.argmin(_np.abs(d)))
    trans = ge[:-1] & (~ge[1:])
    left_idx = _np.max(_np.where((_np.arange(len(trans)) < center) & trans, _np.arange(len(trans)), -1))
    right_candidates = _np.where((_np.arange(len(trans)) >= center) & trans, _np.arange(len(trans)), len(trans)+1)
    right_idx = int(_np.min(right_candidates))
    # interpolate
    def interp(i):
        i = int(_np.clip(i, 0, len(d)-2))
        x1, x2 = d[i], d[i+1]
        y1, y2 = p[i], p[i+1]
        denom = max(abs(y2 - y1), 1e-12)
        w = min(max((half - y1) / denom, 0.0), 1.0)
        return x1 + w * (x2 - x1)
    if left_idx < 0 and right_idx >= len(trans):
        return 0.0
    xL = interp(left_idx)
    xR = interp(right_idx)
    return float(max(xR - xL, 0.0))



# def build_loss(cfg: Config, t, dt, delta0_grid):
#     """Return (loss_fn, diagnostics).

#     Key fix: do NOT use hard FWHM *inside* the loss (non-differentiable => ~zero gradients).
#     Instead we use a smooth 'soft-width' functional.

#     Default loss terms:
#       (A) Frequency-mask loss on P(Δ0): passband |Δ0|<=W -> 1, stopband -> 0
#       (B) Smoothness regularization on Ω and φ (and Δ template if enabled)
#       (C) Power narrowing: penalize width_soft(high) - width_soft(low) if positive

#     Optional:
#       (D) Center 1–0–1 constraint at Δ0=0 if you set cfg.lam_center>0 and provide 3 amp_scales.
#     """
#     global FWHM_DEBUG
#     FWHM_DEBUG = bool(getattr(cfg, "debug_fwhm", False))

#     pass_mask = (jnp.abs(delta0_grid) <= cfg.W).astype(delta0_grid.dtype)
#     stop_mask = 1.0 - pass_mask
#     center_idx = cfg.num_delta // 2
#     dΔ = delta0_grid[1] - delta0_grid[0]

#     def width_soft(P, eps):
#         peak = jnp.max(P)
#         half = 0.5 * peak
#         s = jax.nn.sigmoid((P - half) / eps)
#         return jnp.sum(s) * dΔ

#     @jit
#     def loss_fn(vars: Variables) -> float:
#         # Smooth positive Ω; then normalize ∫Ω dt = π (so scale=1 gives resonant π rotation)
#         Om = cfg.Omega_max * jax.nn.sigmoid(vars.Om)
#         Om = Om * (jnp.pi / (jnp.sum(Om) * dt))
#         Ph = vars.Ph
#         De_templ = vars.De

#         # Two scales for mask + narrowing
#         P_lo = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)
#         P_hi = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1], cfg.Omega_max)

#         pass_cost = cfg.w_pass * (jnp.sum(((1.0 - P_lo) ** 2) * pass_mask) + jnp.sum(((1.0 - P_hi) ** 2) * pass_mask)) * dΔ
#         stop_cost = cfg.w_stop * (jnp.sum((P_lo ** 2) * stop_mask) + jnp.sum((P_hi ** 2) * stop_mask)) * dΔ
#         mask_cost = 0.5 * (pass_cost + stop_cost)

#         # Smoothness penalties
#         dOm = jnp.diff(Om, prepend=Om[:1])
#         dPh = jnp.diff(Ph, prepend=Ph[:1])
#         dDe = jnp.diff(De_templ, prepend=De_templ[:1])
#         smooth = cfg.lam_smooth * (jnp.sum(dOm * dOm) + jnp.sum(dPh * dPh)) + cfg.lam_det_smooth * jnp.sum(dDe * dDe)

#         # Power narrowing via soft widths (differentiable)
#         eps = getattr(cfg, "width_eps", 0.02)
#         w_lo = width_soft(P_lo, eps)
#         w_hi = width_soft(P_hi, eps)
#         pn = cfg.lam_pn * jnp.maximum(w_hi - w_lo, 0.0)

#         # Optional center 1–0–1 constraint
#         center_loss = 0.0
#         if getattr(cfg, "lam_center", 0.0) > 0.0 and len(cfg.amp_scales) >= 3:
#             P1 = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)
#             P2 = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1], cfg.Omega_max)
#             P3 = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[2], cfg.Omega_max)
#             P1c, P2c, P3c = P1[center_idx], P2[center_idx], P3[center_idx]
#             center_loss = cfg.lam_center * ((P1c - 1.0) ** 2 + cfg.w_dark * (P2c - 0.0) ** 2 + (P3c - 1.0) ** 2)

#         if getattr(cfg, "debug_loss", False):
#             jdbg.print(
#                 "LOSS dbg: mask={mc:.3e} pass={pc:.3e} stop={sc:.3e} smooth={sm:.3e} pn={pn:.3e} center={cc:.3e}\n"
#                 "          P0(lo,hi)=({p0l:.3f},{p0h:.3f})  width_soft(lo,hi)=({wlo:.3f},{whi:.3f})",
#                 mc=mask_cost, pc=pass_cost, sc=stop_cost, sm=smooth, pn=pn, cc=center_loss,
#                 p0l=P_lo[center_idx], p0h=P_hi[center_idx], wlo=w_lo, whi=w_hi,
#             )

#         return mask_cost + smooth + pn + center_loss

#     def diagnostics(vars: Variables):
#         import numpy as _np
#         Om_raw = _np.asarray(vars.Om)
#         Om = cfg.Omega_max * (1.0 / (1.0 + _np.exp(-Om_raw / max(cfg.Omega_max, 1e-9))))
#         Om = Om * (jnp.pi / (jnp.sum(Om) * dt))
#         Ph = vars.Ph
#         De_templ = vars.De
#         P_lo = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)
#         P_hi = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1], cfg.Omega_max)
#         fw_lo = fwhm_report(delta0_grid, P_lo)
#         fw_hi = fwhm_report(delta0_grid, P_hi)

#         def width_soft_np(P, eps):
#             P = _np.asarray(P)
#             peak = P.max()
#             half = 0.5 * peak
#             s = 1.0 / (1.0 + _np.exp(-(P - half) / eps))
#             return float(s.sum() * float(delta0_grid[1] - delta0_grid[0]))

#         eps = float(getattr(cfg, "width_eps", 0.02))
#         wlo = width_soft_np(P_lo, eps)
#         whi = width_soft_np(P_hi, eps)

#         if getattr(cfg, "debug_arrays", False):
#             summary = dict(
#                 Om_min=float(_np.min(Om)),
#                 Om_max=float(_np.max(Om)),
#                 Om_mean=float(_np.mean(Om)),
#                 P0_lo=float(P_lo[center_idx]),
#                 P0_hi=float(P_hi[center_idx]),
#                 fw_lo=float(fw_lo),
#                 fw_hi=float(fw_hi),
#                 width_soft_lo=float(wlo),
#                 width_soft_hi=float(whi),
#             )
#             print("DIAG SUMMARY:"); pprint(summary)

#         return {"P_lo": P_lo, "P_hi": P_hi, "fw_lo": fw_lo, "fw_hi": fw_hi, "wlo": wlo, "whi": whi}

#     return loss_fn, diagnostics

def build_loss(cfg: Config, t, dt, delta0_grid):
    # Defining masks based on the configuration
    pass_mask = (jnp.abs(delta0_grid) <= cfg.W).astype(delta0_grid.dtype)
    stop_mask = 1.0 - pass_mask
    center_idx = cfg.num_delta // 2
    dΔ = delta0_grid[1] - delta0_grid[0]

    @jit
    def loss_fn(vars: Variables) -> float:
        # 1. Pulse Processing: Smooth & Area-Normalized
        Om = cfg.Omega_max * jax.nn.sigmoid(vars.Om)
        Om = Om * (jnp.pi / (jnp.sum(Om) * dt)) 
        Ph = vars.Ph
        De_templ = vars.De

        # 2. Simulate Lineshapes at Pi, 2Pi, and 3Pi scales
        P1 = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)
        P2 = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1], cfg.Omega_max)
        P3 = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[2], cfg.Omega_max)

        # 3. The 0-1-0 Resonant Constraint
        # Targets P(0)=1 for odd multiples and P(0)=0 for even multiples
        res_loss = (P1[center_idx] - 1.0)**2 + (P2[center_idx] - 0.0)**2 + (P3[center_idx] - 0.0)**2

        # 4. Use pass_mask and stop_mask for spectral purity
        # Penalize any transition probability in the stopband (off-resonance)
        stopband_cost = jnp.sum(P1**2 * stop_mask) + jnp.sum(P3**2 * stop_mask)

        # 5. Monotonicity Penalty (The "No Fringes" Term)
        # We penalize 'positive' slopes on the right side and 'negative' slopes on the left
        # to ensure there is only one central maximum.
        diff_right = jnp.diff(P1[center_idx:])
        diff_left = jnp.diff(P1[:center_idx+1])
        fringe_penalty = jnp.sum(jnp.maximum(0, diff_right)**2) + \
                         jnp.sum(jnp.maximum(0, -diff_left)**2)

        # 6. Power Narrowing Ratio
        eps_w = getattr(cfg, "width_eps", 0.05)
        w1 = fwhm(delta0_grid, P1, eps_w)
        w3 = fwhm(delta0_grid, P3, eps_w)
        pn_loss = w3 / jnp.maximum(w1, 1e-9) # Minimize the ratio of 5Pi-width to Pi-width

        # 7. Smoothness Regularization
        dOm = jnp.diff(Om)
        dPh = jnp.diff(Ph)
        smooth = (jnp.sum(dOm**2) + jnp.sum(dPh**2))

        return (cfg.lam_center * res_loss) + \
               (cfg.w_stop * (stopband_cost + fringe_penalty)) + \
               (cfg.lam_pn * pn_loss) + \
               cfg.lam_smooth * smooth

    def diagnostics(vars: Variables):
        import numpy as _np
        Om_raw = _np.asarray(vars.Om)
        Om = cfg.Omega_max * (1.0 / (1.0 + _np.exp(-Om_raw / max(cfg.Omega_max, 1e-9))))
        Om = Om * (jnp.pi / (jnp.sum(Om) * dt))
        Ph = vars.Ph
        De_templ = vars.De
        P_lo = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)
        P_hi = line_shape_for_scale(Om, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[2], cfg.Omega_max)
        fw_lo = fwhm(delta0_grid, P_lo)
        fw_hi = fwhm(delta0_grid, P_hi)

        def width_soft_np(P, eps):
            P = _np.asarray(P)
            peak = P.max()
            half = 0.5 * peak
            s = 1.0 / (1.0 + _np.exp(-(P - half) / eps))
            return float(s.sum() * float(delta0_grid[1] - delta0_grid[0]))

        eps = float(getattr(cfg, "width_eps", 0.02))
        wlo = width_soft_np(P_lo, eps)
        whi = width_soft_np(P_hi, eps)

        if getattr(cfg, "debug_arrays", False):
            summary = dict(
                Om_min=float(_np.min(Om)),
                Om_max=float(_np.max(Om)),
                Om_mean=float(_np.mean(Om)),
                P0_lo=float(P_lo[center_idx]),
                P0_hi=float(P_hi[center_idx]),
                fw_lo=float(fw_lo),
                fw_hi=float(fw_hi),
                width_soft_lo=float(wlo),
                width_soft_hi=float(whi),
            )
            print("DIAG SUMMARY:"); print(summary)

        return {"P_lo": P_lo, "P_hi": P_hi, "fw_lo": fw_lo, "fw_hi": fw_hi, "wlo": wlo, "whi": whi}

    return loss_fn, diagnostics
# --------------------------------------------------------------------------------------
# Simple Adam optimizer in pure JAX
# --------------------------------------------------------------------------------------

@dataclass
class AdamState:
    step: int
    m: Variables
    v: Variables


def zeros_like_vars(v: Variables) -> Variables:
    return Variables(jnp.zeros_like(v.Om), jnp.zeros_like(v.Ph), jnp.zeros_like(v.De))


def adam_update(vars: Variables, grads: Variables, state: AdamState, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m = Variables(*(b1 * state.m.Om + (1 - b1) * grads.Om,
                    b1 * state.m.Ph + (1 - b1) * grads.Ph,
                    b1 * state.m.De + (1 - b1) * grads.De))
    v = Variables(*(b2 * state.v.Om + (1 - b2) * (grads.Om ** 2),
                    b2 * state.v.Ph + (1 - b2) * (grads.Ph ** 2),
                    b2 * state.v.De + (1 - b2) * (grads.De ** 2)))
    t = state.step + 1
    mhat = Variables(*(m.Om / (1 - b1 ** t), m.Ph / (1 - b1 ** t), m.De / (1 - b1 ** t)))
    vhat = Variables(*(v.Om / (1 - b2 ** t), v.Ph / (1 - b2 ** t), v.De / (1 - b2 ** t)))

    def upd(x, mh, vh, free):
        upd_step = lr * mh / (jnp.sqrt(vh) + eps)
        return jnp.where(free, x - upd_step, x)

    new_vars = Variables(
        Om=upd(vars.Om, mhat.Om, vhat.Om, jnp.full_like(vars.Om, True)),
        Ph=upd(vars.Ph, mhat.Ph, vhat.Ph, jnp.full_like(vars.Ph, True)),
        De=upd(vars.De, mhat.De, vhat.De, jnp.full_like(vars.De, True)),
    )
    return new_vars, AdamState(t, m, v)


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------

def main(cfg: Config):
    # Build seed controls
    t, dt, Om_seed, Ph_seed, De1_seed, delta0_grid = seed_controls(cfg)

    # Initialize free variables (discretized samples)
    vars = controls_from_seed(cfg, Om_seed, Ph_seed, De1_seed)

    # Optionally fix some channels by zeroing their gradients in the loss wrapper
    loss_core, diag = build_loss(cfg, t, dt, delta0_grid)

    def masked_loss(v: Variables) -> float:
        # Route gradients to active channels only
        val = loss_core(v)
        return val

    grad_fn = jit(jax.grad(masked_loss))

    # Adam state
    st = AdamState(
        step=0,
        m=zeros_like_vars(vars),
        v=zeros_like_vars(vars),
    )

    
    def grad_norms(g: Variables):
        return (
            float(jnp.linalg.norm(g.Om)),
            float(jnp.linalg.norm(g.Ph)),
            float(jnp.linalg.norm(g.De)),
        )
    # Optimization loop
    for k in range(cfg.steps):
        g = grad_fn(vars)

        # Apply masks based on cfg.free_* flags (zero the gradient if not free)
        if not cfg.free_amp:
            g = Variables(Om=jnp.zeros_like(g.Om), Ph=g.Ph, De=g.De)
        if not cfg.free_phase:
            g = Variables(Om=g.Om, Ph=jnp.zeros_like(g.Ph), De=g.De)
        if not cfg.free_det:
            g = Variables(Om=g.Om, Ph=g.Ph, De=jnp.zeros_like(g.De))

        vars, st = adam_update(vars, g, st, lr=cfg.lr)

        if (k % cfg.print_every) == 0 or k == cfg.steps - 1:
            gn = grad_norms(g)
            d = diag(vars)
            print(
                f"iter {k:4d} | grad|| (Om,Ph,De) = ({gn[0]:.2e}, {gn[1]:.2e}, {gn[2]:.2e})\n"
                f"          | FWHM scales {cfg.amp_scales}: ("
                f"{d['fw_lo']:.4f}, {d['fw_hi']:.4f}) | P0: "
                f"({d['P_lo'][cfg.num_delta // 2]:.3f}, {d['P_hi'][cfg.num_delta // 2]:.3f}),"
                f" | P_max: ({jnp.amax(d['P_lo']):.3f}, {jnp.amax(d['P_hi']):.3f})"
            )

    # Final diagnostics & plot
    d = diag(vars)
    if HAVE_PLOT:
        fig, ax = plt.subplots()
        ax.plot(delta0_grid, d["P_lo"], label=f"scale={cfg.amp_scales[0]:.2f}")
        ax.plot(delta0_grid, d["P_hi"], label=f"scale={cfg.amp_scales[2]:.2f}")
        ax.axvline(-cfg.W, linestyle="--")
        ax.axvline(cfg.W, linestyle="--")
        ax.set_xlabel(r"$\Delta_0$")
        ax.set_ylabel("P(|1⟩)")
        ax.set_title("Optimized lineshape vs detuning amplitude")
        ax.legend()
        plt.show()

    return vars, d, (t, dt, delta0_grid)


if __name__ == "__main__":
    cfg = Config(
        T=10.0,
        N=600,
        delta0_max=6.0,
        num_delta=241,
        W=1.0,
        w_pass=1.0,
        w_stop=2.0,
        Omega_max=20.0,
        lam_smooth=1,
        lam_det_smooth=1e-2,#0.0,
        amp_scales=(1.0, 2.0, 5.0), 
        lam_center=10,
        w_dark=0.5,
        lam_pn=100.,
        steps=1000,
        lr=1e-3,
        print_every=50,
        free_amp=True,
        free_phase=True,
        free_det=False,
        theta_amp=(1.0, 0.01),     # Lorentzian seed (Ω0, T)
        theta_phase=(0.0, 0.0),   # flat, no π flip
        theta_det=None,           # constant detuning
    )
    main(cfg)
