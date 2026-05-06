import equinox as eqx  # High-level JAX library for NNs

class PulseNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=3,    # [Amplitude, Detuning, Sigma]
            out_size=1,   # [Probability]
            width_size=64,
            depth=3,
            activation=jax.nn.softplus,
            key=key
        )

    def __call__(self, x):
        return jax.nn.sigmoid(self.mlp(x)) # Probability is [0, 1]