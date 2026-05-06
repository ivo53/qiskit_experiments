"""
QuTiP scan: P(|1>) and leakage (1-P0-P1) vs (Omega_amp, Delta_amp)

- N-level transmon (Duffing ladder in rotating frame, RWA)
- Time-dependent drive Ω(t) and detuning Δ(t)
- Produces two pcolormesh plots:
    1) Final P1
    2) Final leakage = 1 - P0 - P1

Units:
- time in microseconds (µs)
- Ω, Δ, anharmonicity α in MHz (converted internally to angular 2π*MHz)
"""

import numpy as np
import matplotlib.pyplot as plt

from qutip import (
    destroy, qeye, basis, sesolve, mesolve,
    Options, expect
)

# ----------------------------
# Pulse shapes (edit as needed)
# ----------------------------
def raised_cosine_window(t, t_start, t_stop, ramp_frac=0.1):
    """
    Smooth turn-on/off window in [t_start, t_stop] with raised-cosine ramps.
    ramp_frac: fraction of total duration used for each ramp (0..0.5).
    """
    w = np.zeros_like(t, dtype=float)
    T = t_stop - t_start
    if T <= 0:
        return w
    ramp = max(0.0, min(0.5, ramp_frac)) * T

    # flat region
    flat_start = t_start + ramp
    flat_stop  = t_stop  - ramp

    # inside total window
    inside = (t >= t_start) & (t <= t_stop)
    w[inside] = 1.0

    if ramp > 0:
        # ramp up
        ru = (t >= t_start) & (t < flat_start)
        x = (t[ru] - t_start) / ramp
        w[ru] = 0.5 * (1 - np.cos(np.pi * x))

        # ramp down
        rd = (t > flat_stop) & (t <= t_stop)
        x = (t_stop - t[rd]) / ramp
        w[rd] = 0.5 * (1 - np.cos(np.pi * x))

    return w


def landau_zener_shapes(tlist, t_start, t_stop, ramp_frac=0.1):
    """
    LZ-like: Ω(t) ~ constant (with smooth edges), Δ(t) linear sweep from -1 to +1.
    Returned shapes are dimensionless and intended to be scaled by amplitudes:
        Ω(t) = Ω_amp * f(t)
        Δ(t) = Δ_amp * g(t)
    """
    w = raised_cosine_window(tlist, t_start, t_stop, ramp_frac=ramp_frac)

    # normalized time in [-1, +1] across the active window
    T = t_stop - t_start
    tau = (tlist - (t_start + t_stop) / 2) / (T / 2 + 1e-15)
    g = np.clip(tau, -1.0, 1.0) * w  # linear sweep, zero outside
    f = 1.0 * w                      # constant coupling, zero outside
    return f, g


# ----------------------------
# Transmon model (rot frame, RWA)
# ----------------------------
def build_transmon_hamiltonian(N, alpha_MHz):
    """
    N-level Duffing ladder Hamiltonian in the rotating frame at ω01.

    H(t) = H_anh + (-Δ(t))*n + (Ω(t)/2)*(a + a†)

    with H_anh = -(α/2) * n*(n - I)
    """
    TWOPI = 2 * np.pi
    a = destroy(N)
    n = a.dag() * a
    I = qeye(N)

    alpha = TWOPI * float(alpha_MHz)  # rad/µs

    H_anh = -0.5 * alpha * (n * (n - I))   # rad/µs
    H_det = -(n)                            # coefficient is Δ(t) in rad/µs
    H_drv = 0.5 * (a + a.dag())            # coefficient is Ω(t) in rad/µs

    return H_anh, H_det, H_drv


# ----------------------------
# Parameter scan
# ----------------------------
def scan_grid(
    N=8,
    alpha_MHz=250.0,
    T_us=0.2,
    n_steps=2000,
    ramp_frac=0.1,
    Omega_MHz_list=np.linspace(0.0, 80.0, 81),
    Delta_MHz_list=np.linspace(0.0, 200.0, 101),
    use_dissipation=False,
    T1_us=30.0,
    Tphi_us=None,
):
    """
    Returns:
        Omega_MHz_list, Delta_MHz_list, P1_grid, Leak_grid
    where grids have shape (len(Omega), len(Delta)).
    """
    TWOPI = 2 * np.pi

    # time grid
    tlist = np.linspace(0.0, T_us, int(n_steps))
    t_start, t_stop = 0.0, T_us

    # choose shapes (edit here to swap "siblings")
    f_shape, g_shape = landau_zener_shapes(tlist, t_start, t_stop, ramp_frac=ramp_frac)

    # operators
    H_anh, H_det, H_drv = build_transmon_hamiltonian(N, alpha_MHz)

    # projectors for P0 and P1
    ket0 = basis(N, 0)
    ket1 = basis(N, 1)
    P0_op = ket0 * ket0.dag()
    P1_op = ket1 * ket1.dag()

    # initial state
    psi0 = ket0

    # dissipation (optional)
    c_ops = []
    if use_dissipation:
        a = destroy(N)
        if T1_us is not None and T1_us > 0:
            c_ops.append(np.sqrt(1.0 / T1_us) * a)
        if Tphi_us is not None and Tphi_us > 0:
            # simple pure dephasing model ~ n operator
            n_op = a.dag() * a
            c_ops.append(np.sqrt(1.0 / Tphi_us) * n_op)

    opts = Options(
        store_states=False,
        nsteps=20000,
        atol=1e-9,
        rtol=1e-7,
    )

    # allocate outputs
    P1_grid = np.zeros((len(Omega_MHz_list), len(Delta_MHz_list)), dtype=float)
    Leak_grid = np.zeros_like(P1_grid)

    # main loops
    for i, Om_MHz in enumerate(Omega_MHz_list):
        Omega_t = (TWOPI * Om_MHz) * f_shape  # rad/µs

        for j, De_MHz in enumerate(Delta_MHz_list):
            Delta_t = (TWOPI * De_MHz) * g_shape  # rad/µs

            # time-dependent Hamiltonian
            H = [H_anh, [H_drv, Omega_t], [H_det, Delta_t]]

            if use_dissipation:
                res = mesolve(H, psi0, tlist, c_ops=c_ops, e_ops=[P0_op, P1_op], options=opts)
            else:
                res = sesolve(H, psi0, tlist, e_ops=[P0_op, P1_op], options=opts)

            P0_final = float(res.expect[0][-1])
            P1_final = float(res.expect[1][-1])

            leak = 1.0 - P0_final - P1_final
            # clip small numerical negatives
            if leak < 0 and leak > -1e-10:
                leak = 0.0

            P1_grid[i, j] = P1_final
            Leak_grid[i, j] = leak

    return Omega_MHz_list, Delta_MHz_list, P1_grid, Leak_grid


# ----------------------------
# Plotting
# ----------------------------
def plot_heatmaps(Omega_MHz, Delta_MHz, P1, Leak, title_prefix=""):
    """
    P1, Leak are arrays shaped (len(Omega), len(Delta)).
    Axes: x=Delta, y=Omega
    """
    # mesh for pcolormesh: need 2D grids
    DD, OO = np.meshgrid(Delta_MHz, Omega_MHz)

    # Figure 1: P1
    plt.figure(figsize=(7.2, 5.6))
    pcm1 = plt.pcolormesh(DD, OO, P1, shading="auto")
    plt.xlabel(r"Detuning amplitude $\Delta_\mathrm{amp}$ (MHz)")
    plt.ylabel(r"Rabi amplitude $\Omega_\mathrm{amp}$ (MHz)")
    plt.title((title_prefix + "Final $P_1$").strip())
    plt.colorbar(pcm1, label=r"$P_1$")
    plt.tight_layout()

    # Figure 2: Leakage
    plt.figure(figsize=(7.2, 5.6))
    pcm2 = plt.pcolormesh(DD, OO, Leak, shading="auto")
    plt.xlabel(r"Detuning amplitude $\Delta_\mathrm{amp}$ (MHz)")
    plt.ylabel(r"Rabi amplitude $\Omega_\mathrm{amp}$ (MHz)")
    plt.title((title_prefix + r"Leakage $1-P_0-P_1$").strip())
    plt.colorbar(pcm2, label=r"$1-P_0-P_1$")
    plt.tight_layout()

    plt.show()


# ----------------------------
# Run example
# ----------------------------
if __name__ == "__main__":
    # Keep the first run small-ish; increase grids once it works.
    Omega_list = np.linspace(0.0, 80.0, 81)     # MHz
    Delta_list = np.linspace(0.0, 200.0, 101)   # MHz

    Om, De, P1, Leak = scan_grid(
        N=8,
        alpha_MHz=250.0,     # |anharmonicity| in MHz (typical 200–300)
        T_us=0.2,            # total pulse length (µs)
        n_steps=2000,        # time steps
        ramp_frac=0.08,      # smooth edges
        Omega_MHz_list=Omega_list,
        Delta_MHz_list=Delta_list,
        use_dissipation=False,   # set True to include T1/Tphi
        T1_us=30.0,
        Tphi_us=None,
    )

    plot_heatmaps(Om, De, P1, Leak, title_prefix="LZ-like sweep, ")
