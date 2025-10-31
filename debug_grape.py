# We'll read the uploaded file, patch in debug prints and a scaling cap, and save a new debug version.
import re, json, os, textwrap, pathlib, sys

src_path = "./grape-assisted_lorentzian_optimization.py"
dst_path = "./grape-assisted_lorentzian_optimization_debug.py"

with open(src_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1) Import pprint and jax.debug
if "from pprint import pprint" not in code:
    code = code.replace(
        "from jax import lax",
        "from jax import lax\nfrom pprint import pprint\ntry:\n    from jax import debug as jdbg\nexcept Exception:\n    # Older JAX versions\n    import jax.debug as jdbg",
    )

# 2) Add debug flags into Config dataclass
cfg_pattern = r"@dataclass\s*\nclass Config:\n(.*?)\n\n"
m = re.search(cfg_pattern, code, flags=re.S)
if m and "debug_loss" not in m.group(1):
    insert = textwrap.dedent(
        """
            # Debugging
            debug_loss: bool = False      # jax-side prints from loss
            debug_fwhm: bool = False      # jax-side prints from fwhm
            debug_arrays: bool = False    # python-side pprint in diagnostics
        """
    )
    block = m.group(0).rstrip("\n")
    block_new = block[:-1] + insert + "\n\n"
    code = code.replace(block, block_new)

# 3) Cap amplitude after scaling in line_shape_for_scale by adding Omega_max param
code = code.replace(
    "def line_shape_for_scale(\n    Om_base: jnp.ndarray,\n    Ph: jnp.ndarray,\n    De_template: jnp.ndarray,\n    dt: float,\n    delta0_grid: jnp.ndarray,\n    scale: float,\n) -> jnp.ndarray:",
    "def line_shape_for_scale(\n    Om_base: jnp.ndarray,\n    Ph: jnp.ndarray,\n    De_template: jnp.ndarray,\n    dt: float,\n    delta0_grid: jnp.ndarray,\n    scale: float,\n    Omega_cap: float,\n) -> jnp.ndarray:"
)

code = code.replace(
    '    """Compute P(Δ0) for a given global amplitude scale (scales Ω only)."""',
    '    """Compute P(Δ0) for a given global amplitude scale (scales Ω only). Applies cap after scaling."""'
)

# Replace Om assignment to include cap via tanh
code = code.replace(
    "    Om = jnp.clip(scale * Om_base, a_min=0.0, a_max=None)",
    "    Om = Omega_cap * jnp.tanh((scale * jnp.clip(Om_base, a_min=0.0)) / max(Omega_cap, 1e-9))"
)

# 4) Update all calls to line_shape_for_scale to include cfg.Omega_max
code = code.replace(
    "P_lo = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0])",
    "P_lo = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)"
)
code = code.replace(
    "P_hi = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1])",
    "P_hi = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1], cfg.Omega_max)"
)

# Also in diagnostics section
code = code.replace(
    "P_lo = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0])",
    "P_lo = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[0], cfg.Omega_max)"
)
code = code.replace(
    "P_hi = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1])",
    "P_hi = line_shape_for_scale(Om_capped, Ph, De_templ, dt, delta0_grid, cfg.amp_scales[1], cfg.Omega_max)"
)

# 5) Add jax.debug.print statements inside loss_fn (guarded by cfg.debug_loss)
loss_insert = textwrap.dedent(
    """
        if cfg.debug_loss:
            jdbg.print(
                "LOSS dbg:\\n"
                "  dΔ={dD:.4f}\\n"
                "  pass_cost: lo={pcl:.3e} hi={pch:.3e}\\n"
                "  stop_cost: lo={scl:.3e} hi={sch:.3e}\\n"
                "  mask_cost={mc:.3e} smooth={sm:.3e} pn={pn:.3e}\\n"
                "  fw=(lo={fwlo:.4f}, hi={fwhi:.4f}) P0=(lo={p0l:.3f}, hi={p0h:.3f})\\n"
                "  Ω stats: min={om_min:.3f} max={om_max:.3f} mean={om_mean:.3f}",
                dD=dΔ,
                pcl=pass_cost_lo, pch=pass_cost_hi,
                scl=stop_cost_lo, sch=stop_cost_hi,
                mc=mask_cost, sm=smooth, pn=pn,
                fwlo=fw_lo, fwhi=fw_hi,
                p0l=P_lo[center_idx], p0h=P_hi[center_idx],
                om_min=jnp.min(Om_capped), om_max=jnp.max(Om_capped), om_mean=jnp.mean(Om_capped),
            )
    """
)

# place before "return mask_cost + smooth + pn"
code = code.replace(
    "        return mask_cost + smooth + pn",
    loss_insert + "\n        return mask_cost + smooth + pn"
)

# 6) Add jax.debug.print inside fwhm (guarded by a global flag we can read via closure? We used cfg in loss; here f is outside build_loss. We'll add a global module-level flag FWHM_DEBUG and set from Config via assignment in build_loss)
if "FWHM_DEBUG =" not in code:
    code = code.replace(
        "# --------------------------------------------------------------------------------------\n# FWHM and loss construction",
        "# --------------------------------------------------------------------------------------\n# FWHM and loss construction\nFWHM_DEBUG = False"
    )

# Add debug prints inside fwhm
fwhm_debug_insert = textwrap.dedent(
    """
    if FWHM_DEBUG:
        peak = jnp.max(P)
        jdbg.print(
            "FWHM dbg: peak={peak:.4f} half={half:.4f} left_idx={li} right_idx={ri} xL={xL:.4f} xR={xR:.4f} width={w:.4f} all_true={at} all_false={af}",
            peak=peak, half=half, li=left_idx, ri=right_idx, xL=xL, xR=xR, w=width,
            at=all_true, af=all_false
        )
    """
)
code = code.replace(
    "    width = jnp.where(all_false, 0.0, width)\n    return width",
    "    width = jnp.where(all_false, 0.0, width)\n" + fwhm_debug_insert + "\n    return width"
)

# 7) Set FWHM_DEBUG from Config in build_loss (so user can toggle via cfg)
code = code.replace(
    "def build_loss(cfg: Config, t, dt, delta0_grid):",
    "def build_loss(cfg: Config, t, dt, delta0_grid):"
)
# Insert assignment near top of function body
code = code.replace(
    'def build_loss(cfg: Config, t, dt, delta0_grid):\n    """Return a function loss(Variables) → scalar, plus a helper to compute diagnostics."""\n\n    # Pre-build passband mask',
    'def build_loss(cfg: Config, t, dt, delta0_grid):\n    """Return a function loss(Variables) → scalar, plus a helper to compute diagnostics."""\n\n    global FWHM_DEBUG\n    FWHM_DEBUG = bool(getattr(cfg, "debug_fwhm", False))\n\n    # Pre-build passband mask'
)

# 8) Add pprint diagnostics when cfg.debug_arrays True
diag_insert = textwrap.dedent(
    """
        if cfg.debug_arrays:
            summary = dict(
                Om_min=float(jnp.min(Om_capped)),
                Om_max=float(jnp.max(Om_capped)),
                Om_mean=float(jnp.mean(Om_capped)),
                P0_lo=float(P_lo[cfg.num_delta // 2]),
                P0_hi=float(P_hi[cfg.num_delta // 2]),
                fw_lo=float(fw_lo),
                fw_hi=float(fw_hi),
            )
            print("DIAG SUMMARY:"); pprint(summary)
    """
)
code = code.replace(
    "        return {\n            \"P_lo\": P_lo,\n            \"P_hi\": P_hi,\n            \"fw_lo\": fw_lo,\n            \"fw_hi\": fw_hi,\n        }",
    diag_insert + "\n        return {\n            \"P_lo\": P_lo,\n            \"P_hi\": P_hi,\n            \"fw_lo\": fw_lo,\n            \"fw_hi\": fw_hi,\n        }"
)

# 9) Default debug flags on at the bottom main cfg? Let's not change user's cfg; but we can add comment suggesting flipping flags.

with open(dst_path, "w", encoding="utf-8") as f:
    f.write(code)
