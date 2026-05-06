import jax
import jax.numpy as jnp
from diffrax import odeint, Tsit5, PIDController

def schrodinger_ode(t, y, args):
    # y is a complex vector [c0, c1]
    amp, detuning, sigma = args
    # Define pulse shape (Gaussian example)
    omega = amp * jnp.exp(-(t - 5)**2 / (2 * sigma**2))
    
    # Hamiltonian components
    H = 0.5 * jnp.array([[-detuning, omega], 
                         [omega, detuning]], dtype=jnp.complex64)
    
    # dpsi/dt = -i H psi
    return -1j * jnp.matmul(H, y)

def get_probability(amp, detuning, sigma):
    y0 = jnp.array([1.0, 0.0], dtype=jnp.complex64) # Start in ground state
    sol = odeint(schrodinger_ode, t0=0, t1=10, dt0=0.1, y0=y0, 
                 args=(amp, detuning, sigma), solver=Tsit5())
    psi_final = sol.ys[-1]
    return jnp.abs(psi_final[1])**2

if __name__ == "__main__":
    get_probability(amp, detuning, sigma)